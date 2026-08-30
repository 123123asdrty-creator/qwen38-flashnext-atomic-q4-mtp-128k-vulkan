// hotmoe-profile - measure MoE expert routing skew and temporal locality.
//
// Captures every `ffn_moe_topk-<il>` tensor via the scheduler eval callback and
// accumulates, per layer, how often each expert is selected. Also records the
// full per-token selection trace during decode so that cache hit rates for a
// given VRAM budget can be simulated offline.
//
// Output: JSON to $HOTMOE_OUT (default hotmoe-profile.json).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>
#include <map>

struct prof_state {
    int n_expert = 0;

    bool capture   = false;
    int  seg       = 0;   // which prompt
    int  step      = 0;   // decode step within the prompt
    bool is_decode = false;

    // counts[il] -> [n_expert]
    std::map<int, std::vector<uint64_t>> counts_prefill;
    std::map<int, std::vector<uint64_t>> counts_decode;

    // decode trace: one record per (seg, step, layer)
    std::vector<int32_t> tr_seg, tr_step, tr_layer;
    std::vector<int32_t> tr_ids;   // flattened, n_expert_used per record
    int n_expert_used = 0;

    std::vector<int32_t> tmp;
    int max_layer = -1;
};

static bool prof_cb(struct ggml_tensor * t, bool ask, void * ud) {
    prof_state * st = (prof_state *) ud;

    if (ask) {
        return strncmp(t->name, "ffn_moe_topk", 12) == 0;
    }

    if (!st->capture) {
        return true;
    }
    if (t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
        return true;
    }

    const char * dash = strrchr(t->name, '-');
    if (!dash) {
        return true;
    }
    const int il = atoi(dash + 1);
    if (il < 0) {
        return true;
    }
    if (il > st->max_layer) {
        st->max_layer = il;
    }

    const int neu = (int) t->ne[0];   // n_expert_used
    const int ntk = (int) t->ne[1];   // n_tokens
    st->n_expert_used = neu;

    const size_t nb = ggml_nbytes(t);
    st->tmp.resize(nb / sizeof(int32_t));
    ggml_backend_tensor_get(t, st->tmp.data(), 0, nb);

    auto & cmap = st->is_decode ? st->counts_decode : st->counts_prefill;
    auto it = cmap.find(il);
    if (it == cmap.end()) {
        it = cmap.emplace(il, std::vector<uint64_t>(st->n_expert, 0)).first;
    }
    auto & cnt = it->second;

    for (int tk = 0; tk < ntk; ++tk) {
        for (int e = 0; e < neu; ++e) {
            const int32_t id = st->tmp[(size_t) tk * neu + e];
            if (id >= 0 && id < st->n_expert) {
                cnt[id]++;
            }
        }
    }

    // full trace only for decode (n_tokens == 1), keeps the file small
    if (st->is_decode && ntk == 1) {
        st->tr_seg.push_back(st->seg);
        st->tr_step.push_back(st->step);
        st->tr_layer.push_back(il);
        for (int e = 0; e < neu; ++e) {
            st->tr_ids.push_back(st->tmp[e]);
        }
    }

    return true;
}

static std::vector<std::string> split_prompts(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find('\n', i);
        std::string line = (j == std::string::npos) ? s.substr(i) : s.substr(i, j - i);
        while (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "%%%") {
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
        } else {
            cur += line;
            cur += "\n";
        }
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    prof_state st;

    params.cb_eval           = prof_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * ctx   = llama_init ? llama_init->context() : nullptr;
    if (!model || !ctx) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // expert count from model metadata
    {
        char arch[128] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        char key[256];
        snprintf(key, sizeof(key), "%s.expert_count", arch);
        char val[64] = {0};
        if (llama_model_meta_val_str(model, key, val, sizeof(val)) > 0) {
            st.n_expert = atoi(val);
        }
        LOG_INF("%s: arch=%s n_expert=%d\n", __func__, arch, st.n_expert);
    }
    if (st.n_expert <= 0) {
        LOG_ERR("%s: could not determine expert count\n", __func__);
        return 1;
    }

    std::vector<std::string> prompts = split_prompts(params.prompt);
    if (prompts.empty()) {
        prompts.push_back("Hello, how are you doing today?\n");
    }
    LOG_INF("%s: %zu prompt segment(s), n_predict=%d\n", __func__, prompts.size(), params.n_predict);

    common_sampler * smpl = llama_init->sampler(0);

    for (size_t p = 0; p < prompts.size(); ++p) {
        st.seg  = (int) p;
        st.step = 0;

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_init->reset_samplers();
        smpl = llama_init->sampler(0);

        std::vector<llama_token> toks = common_tokenize(ctx, prompts[p], true, true);
        if (toks.empty()) {
            continue;
        }

        LOG_INF("=== segment %zu: %zu prompt tokens ===\n", p, toks.size());

        // prefill
        st.capture   = true;
        st.is_decode = false;
        const int nb_ub = params.n_batch > 0 ? (int) params.n_batch : 512;
        for (size_t i = 0; i < toks.size(); i += (size_t) nb_ub) {
            const int n = (int) std::min((size_t) nb_ub, toks.size() - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, n);
            if (llama_decode(ctx, b)) {
                LOG_ERR("prefill failed\n");
                return 1;
            }
        }

        // decode
        st.is_decode = true;
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        for (int k = 0; k < params.n_predict; ++k) {
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }
            st.step = k;

            llama_batch b = llama_batch_get_one(&id, 1);
            if (llama_decode(ctx, b)) {
                LOG_ERR("decode failed\n");
                break;
            }
            id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);

            char buf[128];
            int nch = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (nch > 0) {
                fwrite(buf, 1, nch, stderr);
                fflush(stderr);
            }
        }
        fprintf(stderr, "\n");
        st.capture = false;
    }

    // ---- write JSON ----
    const char * outp = getenv("HOTMOE_OUT");
    std::string out = outp ? outp : "hotmoe-profile.json";
    std::ofstream f(out, std::ios::binary);
    f << "{\n";
    f << "  \"n_expert\": " << st.n_expert << ",\n";
    f << "  \"n_expert_used\": " << st.n_expert_used << ",\n";
    f << "  \"max_layer\": " << st.max_layer << ",\n";

    auto dump_counts = [&](const char * label, std::map<int, std::vector<uint64_t>> & m) {
        f << "  \"" << label << "\": {\n";
        bool first = true;
        for (auto & kv : m) {
            if (!first) {
                f << ",\n";
            }
            first = false;
            f << "    \"" << kv.first << "\": [";
            for (size_t i = 0; i < kv.second.size(); ++i) {
                if (i) {
                    f << ",";
                }
                f << kv.second[i];
            }
            f << "]";
        }
        f << "\n  },\n";
    };
    dump_counts("counts_prefill", st.counts_prefill);
    dump_counts("counts_decode",  st.counts_decode);

    f << "  \"trace\": {\n";
    f << "    \"n_rec\": " << st.tr_layer.size() << ",\n";
    f << "    \"neu\": " << st.n_expert_used << ",\n";
    auto dump_i32 = [&](const char * label, std::vector<int32_t> & v, bool last) {
        f << "    \"" << label << "\": [";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) {
                f << ",";
            }
            f << v[i];
        }
        f << "]" << (last ? "\n" : ",\n");
    };
    dump_i32("seg",   st.tr_seg,   false);
    dump_i32("step",  st.tr_step,  false);
    dump_i32("layer", st.tr_layer, false);
    dump_i32("ids",   st.tr_ids,   true);
    f << "  }\n}\n";
    f.close();

    LOG_INF("%s: wrote %s (%zu decode records)\n", __func__, out.c_str(), st.tr_layer.size());

    llama_backend_free();
    return 0;
}
