#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Independent double-precision attention oracle. In addition to comparing
// outputs, require every gathered BF16 bit to equal its original cache row.
static bool check(ggml_backend_t backend, int nkv, int width, int nt, int tile, int d, int hq, bool wide) {
    constexpr int hkv = 2;
    const int ns = GGML_PAD(width, 256);
    ggml_init_params ip { 32 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, hq, nt);
    ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d*hkv, nkv);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d*hkv, nkv);
    ggml_tensor * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ns, nt);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nt);
    std::vector<float> qdata(d*hq*nt);
    std::vector<ggml_bf16_t> kd(size_t(d)*hkv*nkv), vd(kd.size());
    std::vector<int32_t> idata(ns*nt);
    std::vector<ggml_fp16_t> md(size_t(nkv)*nt);
    for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = std::sin(float(i % 173)*0.13f);
    for (size_t i = 0; i < kd.size(); ++i) {
        kd[i] = ggml_fp32_to_bf16(std::sin(float(i % 257)*0.19f));
        vd[i] = ggml_fp32_to_bf16(std::cos(float(i % 311)*0.11f) * (wide ? 131072.0f : 1.0f));
    }
    for (int t = 0; t < nt; ++t) {
        for (int j = 0; j < width; ++j) idata[t*ns+j] = (j*1543+t*977) % nkv;
        for (int j = width; j < ns; ++j) idata[t*ns+j] = idata[t*ns];
        for (int j = 0; j < nkv; ++j) {
            const float m = j >= nkv - nt + t ? -INFINITY : ((j % 11) == 0 ? -0.5f : 0.0f);
            md[size_t(t)*nkv+j] = ggml_fp32_to_fp16(m);
        }
    }

    ggml_tensor * output = nullptr;
    struct gathered { ggml_tensor * k; ggml_tensor * v; int start; int count; };
    std::vector<gathered> checks;
    const float scale = 1.0f/std::sqrt(float(d));
    for (int start = 0; start < nt; start += tile) {
        const int count = std::min(tile, nt-start);
        ggml_tensor * ids = ggml_view_1d(ctx, indices, ns*count, start*indices->nb[1]);
        ggml_tensor * ks = ggml_cast(ctx, ggml_get_rows(ctx, k, ids), GGML_TYPE_BF16);
        ggml_tensor * vs = ggml_cast(ctx, ggml_get_rows(ctx, v, ids), GGML_TYPE_BF16);
        checks.push_back({ks,vs,start,count});
        ggml_tensor * kg = ggml_view_4d(ctx, ks, d, ns, hkv, count, d*hkv*2, d*2, d*hkv*ns*2, 0);
        ggml_tensor * vg = ggml_view_4d(ctx, vs, d, ns, hkv, count, d*hkv*2, d*2, d*hkv*ns*2, 0);
        ggml_tensor * mt = ggml_view_4d(ctx, mask, 1, nkv, count, 1, 2, mask->nb[1], mask->nb[2], start*mask->nb[1]);
        ggml_tensor * iw = ggml_view_3d(ctx, indices, width, count, 1, indices->nb[1], indices->nb[2], start*indices->nb[1]);
        ggml_tensor * ms = ggml_get_rows(ctx, mt, iw);
        if (width < ns) {
            ggml_tensor * padding = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, ns-width, count), -INFINITY);
            ms = ggml_concat(ctx, ms, padding, 1);
        }
        ms = ggml_reshape_4d(ctx, ggml_cast(ctx, ms, GGML_TYPE_F16), ns, 1, 1, count);
        ggml_tensor * qt = ggml_view_3d(ctx, q, d, hq, count, q->nb[1], q->nb[2], start*q->nb[2]);
        qt = ggml_reshape_4d(ctx, ggml_cont(ctx, qt), d, 1, hq, count);
        ggml_tensor * part = ggml_flash_attn_ext(ctx, qt, kg, vg, ms, scale, 0, 0);
        ggml_flash_attn_ext_set_prec(part, GGML_PREC_F32);
        part = ggml_reshape_2d(ctx, part, d*hq, count);
        output = output ? ggml_concat(ctx, output, part, 1) : part;
        ggml_build_forward_expand(gf, output);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(q,qdata.data(),0,qdata.size()*sizeof(float));
    ggml_backend_tensor_set(k,kd.data(),0,kd.size()*sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(v,vd.data(),0,vd.size()*sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(indices,idata.data(),0,idata.size()*sizeof(int32_t));
    ggml_backend_tensor_set(mask,md.data(),0,md.size()*sizeof(ggml_fp16_t));
    bool ok = ggml_backend_graph_compute(backend,gf) == GGML_STATUS_SUCCESS;
    for (const auto & c : checks) {
        std::vector<ggml_bf16_t> got_k(size_t(d)*hkv*ns*c.count), got_v(got_k.size());
        ggml_backend_tensor_get(c.k,got_k.data(),0,got_k.size()*2);
        ggml_backend_tensor_get(c.v,got_v.data(),0,got_v.size()*2);
        for (int t = 0; t < c.count; ++t) for (int j = 0; j < ns; ++j) {
            const size_t source = size_t(idata[(c.start+t)*ns+j])*d*hkv;
            const size_t dest = size_t(t*ns+j)*d*hkv;
            ok &= std::memcmp(got_k.data()+dest,kd.data()+source,d*hkv*2) == 0;
            ok &= std::memcmp(got_v.data()+dest,vd.data()+source,d*hkv*2) == 0;
        }
    }
    std::vector<float> actual(qdata.size());
    ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
    double squared_error = 0, squared_reference = 0, max_scaled_error = 0;
    std::vector<double> scores(width), expected(d);
    for (int t = 0; t < nt; ++t) for (int h = 0; h < hq; ++h) {
        double largest = -INFINITY;
        const int kh = h/(hq/hkv);
        for (int j = 0; j < width; ++j) {
            const int row = idata[t*ns+j];
            double score = 0;
            for (int c = 0; c < d; ++c) score += double(qdata[(t*hq+h)*d+c])*ggml_bf16_to_fp32(kd[(size_t(row)*hkv+kh)*d+c]);
            scores[j] = score*scale + ggml_fp16_to_fp32(md[size_t(t)*nkv+row]);
            largest = std::max(largest,scores[j]);
        }
        double sum = 0;
        std::fill(expected.begin(),expected.end(),0);
        for (int j = 0; j < width; ++j) {
            const double prob = std::exp(scores[j]-largest);
            const int row = idata[t*ns+j];
            sum += prob;
            for (int c = 0; c < d; ++c) expected[c] += prob*ggml_bf16_to_fp32(vd[(size_t(row)*hkv+kh)*d+c]);
        }
        for (int c = 0; c < d; ++c) {
            const double ref = expected[c]/sum;
            const double got = actual[(t*hq+h)*d+c];
            ok &= std::isfinite(got);
            squared_error += (got-ref)*(got-ref);
            squared_reference += ref*ref;
            max_scaled_error = std::max(max_scaled_error,std::abs(got-ref)/(wide ? 131072.0 : 1.0));
        }
    }
    const double nmse = squared_error/std::max(squared_reference,1e-30);
    ok &= nmse < 1e-8 && max_scaled_error < 3e-4;
    std::printf("{\"context\":%d,\"selected\":%d,\"queries\":%d,\"tile\":%d,\"dimension\":%d,\"heads\":%d,\"wideBf16\":%s,\"nmse\":%.12g,\"maxScaledError\":%.12g,\"passed\":%s}\n",
            nkv,width,nt,tile,d,hq,wide?"true":"false",nmse,max_scaled_error,ok?"true":"false");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("Vulkan0",nullptr);
    if (!backend) return 2;
    const char * device = ggml_backend_dev_description(ggml_backend_get_device(backend));
    std::printf("Device: %s\n",device);
    if (!std::strstr(device,"RX 7900 XTX")) { ggml_backend_free(backend); return 3; }
    bool ok = true;
    ok &= check(backend,4096,257,3,2,256,24,false);
    ok &= check(backend,8192,2051,17,16,256,24,false);
    ok &= check(backend,131072,2051,2,1,256,24,false);
    ok &= check(backend,8192,17,65,64,32,4,true);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
