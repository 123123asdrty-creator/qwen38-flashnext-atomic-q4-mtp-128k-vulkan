#include "llama-hotmoe.h"

#include "ggml-alloc.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <memory>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

struct hotmoe_dynamic_layer {
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * down_src = nullptr;
    std::vector<uint64_t> last_seen;
    std::vector<uint32_t> frequency;
};

struct hotmoe_dynamic_state {
    int n_expert_used = 0;
    int swaps_per_token = 0;
    int trace_per_token = 4;
    bool async_fetch = false;
    std::vector<int32_t> pending_ids;
    bool pending = false;
    uint64_t tick = 0;
    size_t update_cursor = 0;
    size_t trace_cursor = 0;
    std::vector<int> trace_layers;
    std::unordered_map<int, hotmoe_dynamic_layer> layers;
};

static std::mutex g_hotmoe_dynamic_mutex;
static std::unordered_map<const llama_hotmoe *, std::unique_ptr<hotmoe_dynamic_state>> g_hotmoe_dynamic;

static hotmoe_dynamic_state * hotmoe_dynamic_find(const llama_hotmoe * hm) {
    std::lock_guard<std::mutex> lock(g_hotmoe_dynamic_mutex);
    const auto it = g_hotmoe_dynamic.find(hm);
    return it == g_hotmoe_dynamic.end() ? nullptr : it->second.get();
}

llama_hotmoe::~llama_hotmoe() {
    reset();
}

void llama_hotmoe::reset() {
    {
        std::lock_guard<std::mutex> lock(g_hotmoe_dynamic_mutex);
        g_hotmoe_dynamic.erase(this);
    }
    for (auto * b : bufs) {
        if (b) {
            ggml_backend_buffer_free(b);
        }
    }
    for (auto * c : ctxs) {
        if (c) {
            ggml_free(c);
        }
    }
    bufs.clear();
    ctxs.clear();
    layers.clear();
    enabled = false;
    host_direct = false;
    init_attempted = false;
    n_slots = 0;
    n_expert = 0;
    vram_bytes = 0;
    seed_coverage = 0.0;
    ++generation;
}

// ---------------------------------------------------------------------------
// rank file: produced by tools/hotmoe/rank.py from a hotmoe-profile.json
//
//   n_expert <N>
//   L <il> <e0> <e1> ... <eN-1>      experts ranked hottest first
// ---------------------------------------------------------------------------
struct hotmoe_ranks {
    int n_expert = 0;
    std::unordered_map<int, std::vector<int32_t>> order;
};

static bool hotmoe_load_ranks(const std::string & path, hotmoe_ranks & out) {
    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_WARN("%s: could not open HotMoE profile '%s'\n", __func__, path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "n_expert") {
            is >> out.n_expert;
        } else if (tag == "L") {
            int il = -1;
            is >> il;
            if (il < 0) {
                continue;
            }
            std::vector<int32_t> v;
            v.reserve(out.n_expert > 0 ? out.n_expert : 512);
            int e;
            while (is >> e) {
                v.push_back(e);
            }
            out.order[il] = std::move(v);
        }
    }
    return !out.order.empty();
}

// ---------------------------------------------------------------------------

static int hotmoe_env_int(const char * name, int def) {
    const char * s = getenv(name);
    if (!s || !*s) {
        return def;
    }
    return atoi(s);
}

void llama_hotmoe_init(llama_model & model, bool force) {
    llama_hotmoe & hm = model.hotmoe;

    const bool host_direct_req = hotmoe_env_int("LLAMA_HOSTMOE", 0) > 0;

    if (hm.enabled) {
        return;
    }
    if (force && hm.init_attempted) {
        return;
    }
    if (!host_direct_req && !force && hotmoe_env_int("LLAMA_HOTMOE_DEFER", 0) > 0) {
        LLAMA_LOG_INFO("%s: HotMoE deferred until all model and context allocations are resident\n", __func__);
        return;
    }
    if (!host_direct_req && !force && hotmoe_env_int("LLAMA_HOTMOE_PHASED", 0) > 0) {
        LLAMA_LOG_INFO("%s: HotMoE phased mode - deferring cache allocation until decode\n", __func__);
        return;
    }

    int n_slots_req = hotmoe_env_int("LLAMA_HOTMOE", 0);
    const char * profile_out = getenv("LLAMA_HOTMOE_PROFILE_OUT");
    const bool profile_req = profile_out && *profile_out;
    if (n_slots_req <= 0 && !host_direct_req && !profile_req) {
        return;
    }
    hm.init_attempted = force;
    hm.max_tokens = std::max(1, hotmoe_env_int(
            host_direct_req ? "LLAMA_HOSTMOE_MAX_TOK" : "LLAMA_HOTMOE_MAX_TOK", 1));
    const int swaps_per_token = std::max(0, hotmoe_env_int("LLAMA_HOTMOE_DYNAMIC", 0));
    auto dynamic = std::make_unique<hotmoe_dynamic_state>();
    dynamic->swaps_per_token = swaps_per_token;
    dynamic->trace_per_token = std::max(1, hotmoe_env_int("LLAMA_HOTMOE_TRACE", 4));
    dynamic->async_fetch = hotmoe_env_int("LLAMA_HOTMOE_ASYNC", 0) > 0;

    // pick the first non-meta GPU device to host the cache
    ggml_backend_dev_t dev = nullptr;
    for (const auto & d : model.devices) {
        if (d.is_meta || d.dev == nullptr) {
            continue;
        }
        if (ggml_backend_dev_type(d.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = d.dev;
            break;
        }
    }
    if (!dev) {
        LLAMA_LOG_WARN("%s: HotMoE requested but no GPU device is available - disabled\n", __func__);
        return;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // collect the layers that have host-resident routed experts
    struct cand { int il; ggml_tensor * gate; ggml_tensor * up; ggml_tensor * down; };
    std::vector<cand> cands;

    for (int il = 0; il < (int) model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (!l.ffn_gate_exps || !l.ffn_up_exps || !l.ffn_down_exps) {
            continue;
        }
        if (l.ffn_gate_up_exps) {
            continue;   // merged gate/up layout is not handled yet
        }
        if (l.ffn_up_exps_s || l.ffn_gate_exps_s || l.ffn_down_exps_s) {
            continue;   // per-expert scales are not handled yet
        }
        // only worth caching when the source actually lives in host memory
        if (!l.ffn_gate_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_gate_exps->buffer)) {
            continue;
        }
        if (!l.ffn_up_exps->buffer   || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) ||
            !l.ffn_down_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_down_exps->buffer)) {
            continue;
        }
        cands.push_back({ il, l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps });
    }

    if (cands.empty()) {
        LLAMA_LOG_WARN("%s: HotMoE found no CPU-resident expert tensors - disabled "
                       "(did you pass -ncmoe / --cpu-moe?)\n", __func__);
        return;
    }

    hm.n_expert = (int) cands[0].gate->ne[2];

    if (profile_req && !hm.profile_enabled) {
        hm.profile_enabled = true;
        hm.profile_path = profile_out;
        hm.profile_max_tokens = std::max(1, hotmoe_env_int("LLAMA_HOTMOE_PROFILE_MAX_TOK", 1));
        for (const auto & c : cands) {
            hm.profile_counts[c.il].assign(hm.n_expert, 0);
        }
        LLAMA_LOG_INFO("%s: recording HotMoE routes for batches of at most %d tokens to '%s'\n",
                __func__, hm.profile_max_tokens, hm.profile_path.c_str());
    }

    if (n_slots_req <= 0 && !host_direct_req) {
        return;
    }

    // Auto-size only after the target, MTP sidecar, KV caches, and scheduler
    // buffers are resident (LLAMA_HOTMOE_DEFER=1). This uses the backend's
    // live per-process Vulkan budget and ggml's exact allocation-size routine,
    // including tensor and buffer alignment; no empirical slot probing is
    // involved. The reserve covers the larger first HotMoE execution graph.
    if (!host_direct_req && hotmoe_env_int("LLAMA_HOTMOE_AUTO", 0) > 0) {
        const int upper = std::min(std::max(1, n_slots_req), hm.n_expert - 1);
        const size_t n_tensors = cands.size() * 5 + 8;

        auto cache_bytes_for_slots = [&](int slots) -> size_t {
            ggml_init_params test_ip = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * test_ctx = ggml_init(test_ip);
            GGML_ASSERT(test_ctx != nullptr);
            for (const auto & c : cands) {
                ggml_new_tensor_3d(test_ctx, c.gate->type, c.gate->ne[0], c.gate->ne[1], slots + 1);
                ggml_new_tensor_3d(test_ctx, c.up->type,   c.up->ne[0],   c.up->ne[1],   slots + 1);
                ggml_new_tensor_3d(test_ctx, c.down->type, c.down->ne[0], c.down->ne[1], slots + 1);
                ggml_new_tensor_2d(test_ctx, GGML_TYPE_I32, 1, hm.n_expert);
                ggml_new_tensor_2d(test_ctx, GGML_TYPE_I32, 1, hm.n_expert);
            }
            const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(test_ctx, buft);
            ggml_free(test_ctx);
            return bytes;
        };

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        const size_t reserve_bytes = (size_t) std::max(0, hotmoe_env_int("LLAMA_HOTMOE_RESERVE_MIB", 512)) * 1024 * 1024;

        auto max_slots_for = [&](size_t budget) {
            int lo = 0;
            int hi = upper;
            while (lo < hi) {
                const int mid = lo + (hi - lo + 1) / 2;
                if (cache_bytes_for_slots(mid) <= budget) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            return lo;
        };

        const int raw_max = max_slots_for(free_bytes);
        const size_t usable_bytes = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
        n_slots_req = max_slots_for(usable_bytes);
        const size_t chosen_bytes = n_slots_req > 0 ? cache_bytes_for_slots(n_slots_req) : 0;

        LLAMA_LOG_INFO("%s: HotMoE exact auto-size: free %.2f MiB / total %.2f MiB, "
                       "raw max %d slots, reserve %.2f MiB, selected %d slots (%.2f MiB)\n",
                __func__, free_bytes / 1024.0 / 1024.0, total_bytes / 1024.0 / 1024.0,
                raw_max, reserve_bytes / 1024.0 / 1024.0, n_slots_req,
                chosen_bytes / 1024.0 / 1024.0);
        if (n_slots_req <= 0) {
            LLAMA_LOG_WARN("%s: no HotMoE slot fits after the requested reserve\n", __func__);
            return;
        }
    }

    if (host_direct_req) {
        // The RX 7900 XTX reports a 4096-byte imported-host-pointer alignment.
        // Keep this overrideable so another driver can state a larger value.
        size_t import_alignment = (size_t) std::max(4096,
                hotmoe_env_int("LLAMA_HOSTMOE_ALIGNMENT", 4096));
        if ((import_alignment & (import_alignment - 1)) != 0) {
            LLAMA_LOG_ERROR("%s: LLAMA_HOSTMOE_ALIGNMENT must be a power of two\n", __func__);
            return;
        }

        const size_t n_tensors = cands.size() * 3 + 8;
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: HostMoE ggml_init failed\n", __func__);
            return;
        }
        hm.ctxs.push_back(ctx);

        size_t imported_bytes = 0;


        auto import_tensor = [&](ggml_tensor * src, const char * kind, int il) -> ggml_tensor * {
            const uintptr_t data_begin = reinterpret_cast<uintptr_t>(src->data);
            const uintptr_t data_end = data_begin + ggml_nbytes(src);
            uintptr_t map_begin = data_begin & ~(uintptr_t) (import_alignment - 1);
            uintptr_t map_end = (data_end + import_alignment - 1) & ~(uintptr_t) (import_alignment - 1);

#ifdef _WIN32
            MEMORY_BASIC_INFORMATION mbi = {};
            if (VirtualQuery(src->data, &mbi, sizeof(mbi)) == 0) {
                LLAMA_LOG_ERROR("%s: VirtualQuery failed for %s layer %d\n", __func__, kind, il);
                return nullptr;
            }
            // Check containment against the mapped VIEW (AllocationBase), not against one
            // protection region. Relaxing an earlier tensor's range splits the view into several
            // regions, so a region-based check would reject every tensor after the first — which
            // is exactly what happened once the per-tensor copy-on-write fallback started firing.
            // What actually matters is that the aligned range never leaves this file mapping.
            MEMORY_BASIC_INFORMATION edge = {};
            const bool begin_ok = VirtualQuery(reinterpret_cast<const void *>(map_begin), &edge, sizeof(edge)) != 0
                && edge.AllocationBase == mbi.AllocationBase && edge.State == MEM_COMMIT;
            const bool end_ok = VirtualQuery(reinterpret_cast<const void *>(map_end - 1), &edge, sizeof(edge)) != 0
                && edge.AllocationBase == mbi.AllocationBase && edge.State == MEM_COMMIT;
            if (!begin_ok || !end_ok) {
                LLAMA_LOG_ERROR("%s: aligned %s layer %d range leaves its mapped view\n", __func__, kind, il);
                return nullptr;
            }
#endif

            const size_t map_size = map_end - map_begin;

#ifdef _WIN32
            // Relax THIS tensor's range to copy-on-write before importing it.
            //
            // The driver rejects read-only pages outright (ErrorInvalidExternalHandle), so this is
            // required, not a fallback -- and it has to happen before the first import attempt.
            // Per tensor rather than per view because copy-on-write charges Windows commit for
            // every page it covers: over the whole 91.66 GiB mapping that is an instant
            // ERROR_COMMITMENT_LIMIT, while per tensor it charges only the ranges actually
            // imported. Nothing ever writes these pages, so they stay shared with the file mapping
            // and cost no physical memory.
            {
                DWORD old_protect = 0;
                if (!VirtualProtect(reinterpret_cast<void *>(map_begin), map_size, PAGE_WRITECOPY, &old_protect)) {
                    LLAMA_LOG_ERROR("%s: could not relax the %s tensor for layer %d to copy-on-write "
                                    "(%zu MiB, VirtualProtect %lu)\n",
                            __func__, kind, il, map_size / 1024 / 1024, (unsigned long) GetLastError());
                    return nullptr;
                }
            }
#endif

            ggml_backend_buffer_t buffer = ggml_backend_dev_buffer_from_host_ptr(
                    dev, reinterpret_cast<void *>(map_begin), map_size, ggml_nbytes(src));

            if (!buffer) {
                LLAMA_LOG_ERROR("%s: Vulkan refused the RAM-backed %s tensor for layer %d "
                                "(%zu MiB, alignment %zu)\n",
                        __func__, kind, il, map_size / 1024 / 1024, import_alignment);
                return nullptr;
            }
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            hm.bufs.push_back(buffer);

            ggml_tensor * alias = ggml_dup_tensor(ctx, src);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                alias->nb[d] = src->nb[d];
            }
            ggml_format_name(alias, "hostmoe.%s.%d", kind, il);

            const size_t offset = data_begin - map_begin;
            void * alias_addr = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + offset;
            if (ggml_backend_tensor_alloc(buffer, alias, alias_addr) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: could not bind the RAM-backed %s tensor for layer %d\n",
                        __func__, kind, il);
                return nullptr;
            }
            imported_bytes += map_size;
            return alias;
        };

        // How many RAM-resident expert layers the GPU takes over, counted from the LAST CPU layer
        // backwards. Default: all of them.
        //
        // This is a budget knob, not a correctness one. Importing a range requires relaxing it to
        // copy-on-write (the driver rejects read-only pages with ErrorInvalidExternalHandle), and
        // Windows charges commit for every copy-on-write page even though these are only ever
        // read. At ~1.45 GiB per layer, importing all 37 CPU layers charges ~54 GiB of commit --
        // survivable on a large pagefile, but worth being able to bound. Layers that are not
        // imported simply keep the ordinary CPU expert path, so this degrades smoothly.
        int host_layers = hotmoe_env_int("LLAMA_HOSTMOE_LAYERS", (int) cands.size());
        if (host_layers <= 0 || host_layers > (int) cands.size()) {
            host_layers = (int) cands.size();
        }
        // Prefer the layers nearest the output: they run last, so keeping them off the CPU shortens
        // the tail of each token's critical path.
        const size_t first = cands.size() - (size_t) host_layers;

        for (size_t i = first; i < cands.size(); ++i) {
            const auto & c = cands[i];
            llama_hotmoe_layer hl;
            hl.host_gate = import_tensor(c.gate, "gate", c.il);
            hl.host_up   = import_tensor(c.up,   "up",   c.il);
            hl.host_down = import_tensor(c.down, "down", c.il);
            if (!hl.host_gate || !hl.host_up || !hl.host_down) {
                if (hm.layers.empty()) {
                    LLAMA_LOG_WARN("%s: HostMoE import failed; leaving the normal CPU expert path active\n", __func__);
                    hm.reset();
                    return;
                }
                // Partial success is still a win: stop here and let the remaining layers run on
                // the CPU rather than throwing away the layers that did import.
                LLAMA_LOG_WARN("%s: HostMoE imported %zu of %d layers; the rest stay on the CPU\n",
                        __func__, hm.layers.size(), host_layers);
                break;
            }
            hm.layers[c.il] = hl;
        }

        hm.host_direct = true;
        hm.enabled = true;
        ++hm.generation;
        LLAMA_LOG_INFO("%s: HostMoE enabled - GPU kernels directly read %.2f GiB of expert weights "
                       "from system RAM across PCIe on %zu layers\n",
                __func__, imported_bytes / 1024.0 / 1024.0 / 1024.0, hm.layers.size());
        LLAMA_LOG_INFO("%s: HostMoE active for batches of at most %d tokens; larger prompts keep the CPU path\n",
                __func__, hm.max_tokens);
        if (n_slots_req > 0) {
            LLAMA_LOG_INFO("%s: LLAMA_HOSTMOE takes precedence over the VRAM HotMoE cache for this run\n", __func__);
        }
        return;
    }

    hm.n_slots  = std::min(n_slots_req, hm.n_expert - 1);
    if (hm.n_slots <= 0) {
        return;
    }

    // optional profile-guided seeding
    hotmoe_ranks ranks;
    const char * prof = getenv("LLAMA_HOTMOE_PROFILE");
    bool have_ranks = false;
    if (prof && *prof) {
        have_ranks = hotmoe_load_ranks(prof, ranks);
    }

    // ---- create the tensors ----
    const size_t n_tensors = cands.size() * 5 + 8;
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: HotMoE ggml_init failed\n", __func__);
        return;
    }
    hm.ctxs.push_back(ctx);

    const int K = hm.n_slots;

    for (const auto & c : cands) {
        llama_hotmoe_layer hl;

        auto mk_cache = [&](ggml_tensor * src, const char * name) {
            ggml_tensor * t = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], K + 1);
            ggml_format_name(t, "hotmoe.%s.%d", name, c.il);
            return t;
        };
        hl.gate = mk_cache(c.gate, "gate");
        hl.up   = mk_cache(c.up,   "up");
        hl.down = mk_cache(c.down, "down");
        hl.lut_hot  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, hm.n_expert);
        hl.lut_cold = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, hm.n_expert);
        ggml_format_name(hl.lut_hot,  "hotmoe.lut_hot.%d",  c.il);
        ggml_format_name(hl.lut_cold, "hotmoe.lut_cold.%d", c.il);

        hm.layers[c.il] = hl;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        LLAMA_LOG_ERROR("%s: HotMoE failed to allocate %d slots/layer on %s - "
                        "lower LLAMA_HOTMOE\n", __func__, K, ggml_backend_buft_name(buft));
        hm.layers.clear();
        return;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    hm.bufs.push_back(buf);
    hm.vram_bytes = ggml_backend_buffer_get_size(buf);

    // ---- populate ----
    std::vector<uint8_t> packed;
    std::vector<int32_t> lut_h(hm.n_expert), lut_c(hm.n_expert);

    size_t n_seeded = 0;

    for (const auto & c : cands) {
        llama_hotmoe_layer & hl = hm.layers[c.il];

        // which experts go in the cache
        std::vector<int32_t> chosen;
        chosen.reserve(K);
        auto it = ranks.order.find(c.il);
        if (have_ranks && it != ranks.order.end()) {
            for (int32_t e : it->second) {
                if ((int) chosen.size() >= K) {
                    break;
                }
                if (e >= 0 && e < hm.n_expert) {
                    chosen.push_back(e);
                }
            }
            n_seeded++;
        }
        for (int32_t e = 0; (int) chosen.size() < K && e < hm.n_expert; ++e) {
            if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
                chosen.push_back(e);
            }
        }

        hl.slot_of.assign(hm.n_expert, -1);
        hl.expert_of.assign(K, -1);
        auto & dl = dynamic->layers[c.il];
        dynamic->trace_layers.push_back(c.il);
        dl.gate_src = c.gate;
        dl.up_src   = c.up;
        dl.down_src = c.down;
        dl.last_seen.assign(hm.n_expert, 0);
        dl.frequency.assign(hm.n_expert, 0);
        for (int s = 0; s < K; ++s) {
            hl.expert_of[s]           = chosen[s];
            hl.slot_of[chosen[s]]     = s;
            dl.last_seen[chosen[s]]   = K - s;
            dl.frequency[chosen[s]]   = 1;
        }

        // Copy each packed tensor in one upload. Large caches otherwise issue
        // one backend transfer per expert, projection, and layer.
        struct pair { ggml_tensor * dst; ggml_tensor * src; };
        const pair pairs[3] = { { hl.gate, c.gate }, { hl.up, c.up }, { hl.down, c.down } };
        for (const auto & p : pairs) {
            const size_t stride = p.src->nb[2];   // bytes for one expert
            GGML_ASSERT(p.dst->nb[2] == stride);
            packed.resize((size_t) (K + 1) * stride);
            for (int s = 0; s < K; ++s) {
                const char * src_e = (const char *) p.src->data + (size_t) hl.expert_of[s] * stride;
                memcpy(packed.data() + (size_t) s * stride, src_e, stride);
            }
            memset(packed.data() + (size_t) K * stride, 0, stride);
            ggml_backend_tensor_set(p.dst, packed.data(), 0, packed.size());
        }

        // lookup tables
        for (int e = 0; e < hm.n_expert; ++e) {
            const int s = hl.slot_of[e];
            lut_h[e] = (s >= 0) ? s : K;    // K == the zero expert
            lut_c[e] = (s >= 0) ? -1 : e;   // -1 == skipped by the CPU kernel
        }
        ggml_backend_tensor_set(hl.lut_hot,  lut_h.data(), 0, ggml_nbytes(hl.lut_hot));
        ggml_backend_tensor_set(hl.lut_cold, lut_c.data(), 0, ggml_nbytes(hl.lut_cold));
    }

    hm.enabled = true;
    ++hm.generation;
    dynamic->tick = K;
    {
        std::lock_guard<std::mutex> lock(g_hotmoe_dynamic_mutex);
        g_hotmoe_dynamic[&hm] = std::move(dynamic);
    }

    LLAMA_LOG_INFO("%s: HotMoE enabled - %d slots/layer of %d experts (%.1f%%) on %zu layers, "
                   "%.2f GiB VRAM, seeded from profile: %zu/%zu layers\n",
                   __func__, K, hm.n_expert, 100.0 * K / hm.n_expert, hm.layers.size(),
                   hm.vram_bytes / 1024.0 / 1024.0 / 1024.0, n_seeded, cands.size());
    LLAMA_LOG_INFO("%s: HotMoE active for batches of at most %d tokens\n", __func__, hm.max_tokens);
    if (swaps_per_token > 0) {
        dynamic = nullptr;
        const auto * state = hotmoe_dynamic_find(&hm);
        LLAMA_LOG_INFO("%s: HotMoE live cache enabled, at most %d expert swaps/token, tracing %d/%zu layers/token\n",
                __func__, swaps_per_token, state->trace_per_token, state->trace_layers.size());
        if (state->async_fetch) {
            LLAMA_LOG_INFO("%s: HotMoE predictive expert uploads are queued asynchronously\n", __func__);
        }
    }
}

void llama_hotmoe_maybe_phase(llama_model & model, int n_tokens, ggml_backend_sched_t sched) {
    if (hotmoe_env_int("LLAMA_HOTMOE_DEFER", 0) > 0) {
        if (!model.hotmoe.enabled) {
            LLAMA_LOG_INFO("%s: sizing and allocating HotMoE after fixed runtime allocations\n", __func__);
            llama_hotmoe_init(model, true);
        }
        return;
    }
    if (hotmoe_env_int("LLAMA_HOTMOE_PHASED", 0) <= 0) {
        return;
    }
    llama_hotmoe & hm = model.hotmoe;
    if (hm.host_direct) {
        return;
    }
    const int prompt_min = std::max(2, hotmoe_env_int("LLAMA_HOTMOE_PHASE_MIN", 512));
    if (n_tokens >= prompt_min && hm.enabled) {
        LLAMA_LOG_INFO("%s: releasing HotMoE cache for a %d-token prompt batch\n", __func__, n_tokens);
        ggml_backend_sched_synchronize(sched);
        hm.reset();
    } else {
        const int decode_max = std::max(1, hotmoe_env_int("LLAMA_HOTMOE_MAX_TOK", 1));
        if (n_tokens > decode_max || hm.enabled || hm.init_attempted) {
            return;
        }
        LLAMA_LOG_INFO("%s: allocating HotMoE cache for decode\n", __func__);
        ggml_backend_sched_synchronize(sched);
        llama_hotmoe_init(model, true);
    }
}

void llama_hotmoe::capture_profile(const std::vector<ggml_tensor *> & ids, ggml_backend_sched_t sched) {
    if (!profile_enabled || profile_counts.empty()) {
        return;
    }

    struct capture {
        int il;
        size_t offset;
        size_t count;
    };
    std::vector<capture> captures;
    std::vector<int32_t> values;
    uint64_t n_tokens = 0;

    size_t total_count = 0;
    for (size_t il = 0; il < ids.size(); ++il) {
        ggml_tensor * t = ids[il];
        if (t != nullptr && profile_counts.find((int) il) != profile_counts.end()) {
            total_count += ggml_nelements(t);
        }
    }
    values.resize(total_count);
    captures.reserve(profile_counts.size());

    size_t offset = 0;
    for (size_t il = 0; il < ids.size(); ++il) {
        ggml_tensor * t = ids[il];
        if (t == nullptr || profile_counts.find((int) il) == profile_counts.end()) {
            continue;
        }
        const size_t count = ggml_nelements(t);
        captures.push_back({ (int) il, offset, count });
        n_tokens = std::max<uint64_t>(n_tokens, t->ne[1]);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, values.data() + offset, 0, count * sizeof(int32_t));
        offset += count;
    }
    if (captures.empty()) {
        return;
    }

    ggml_backend_sched_synchronize(sched);
    for (const auto & c : captures) {
        auto & counts = profile_counts.at(c.il);
        for (size_t i = 0; i < c.count; ++i) {
            const int32_t expert = values[c.offset + i];
            if (expert >= 0 && expert < (int32_t) counts.size()) {
                ++counts[expert];
            }
        }
    }
    profile_tokens += n_tokens;

    std::ofstream out(profile_path, std::ios::trunc);
    if (!out) {
        LLAMA_LOG_WARN("%s: could not write HotMoE profile '%s'\n", __func__, profile_path.c_str());
        return;
    }
    out << "# Atomic HotMoE route profile; tokens " << profile_tokens << "\n";
    out << "n_expert " << n_expert << "\n";
    for (const auto & [il, counts] : profile_counts) {
        std::vector<int32_t> order(counts.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return counts[a] > counts[b];
        });
        out << "L " << il;
        for (int32_t expert : order) {
            out << ' ' << expert;
        }
        out << '\n';
    }
}

int llama_hotmoe::dynamic_swaps() const {
    const auto * dynamic = hotmoe_dynamic_find(this);
    return dynamic ? dynamic->swaps_per_token : 0;
}

bool llama_hotmoe::trace_layer(int il) const {
    const auto * dynamic = hotmoe_dynamic_find(this);
    if (dynamic == nullptr || dynamic->swaps_per_token <= 0 || dynamic->trace_layers.empty()) {
        return false;
    }
    const size_t count = std::min((size_t) dynamic->trace_per_token, dynamic->trace_layers.size());
    for (size_t i = 0; i < count; ++i) {
        if (dynamic->trace_layers[(dynamic->trace_cursor + i) % dynamic->trace_layers.size()] == il) {
            return true;
        }
    }
    return false;
}

void llama_hotmoe::capture_async(const std::vector<ggml_tensor *> & ids, ggml_backend_sched_t sched) {
    auto * dynamic = hotmoe_dynamic_find(this);
    if (!enabled || dynamic == nullptr || dynamic->swaps_per_token <= 0 || dynamic->pending) {
        return;
    }

    int neu = 0;
    for (auto * t : ids) {
        if (t != nullptr) {
            if (t->ne[1] != 1) {
                return;
            }
            neu = (int) t->ne[0];
            break;
        }
    }
    if (neu <= 0) {
        return;
    }

    dynamic->n_expert_used = neu;
    dynamic->pending_ids.assign(ids.size() * neu, -1);

    bool copied = false;
    for (size_t il = 0; il < ids.size(); ++il) {
        ggml_tensor * t = ids[il];
        if (t == nullptr) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        if (backend == nullptr) {
            continue;
        }
        ggml_backend_tensor_get_async(backend, t, dynamic->pending_ids.data() + il * neu, 0, neu * sizeof(int32_t));
        copied = true;
    }
    dynamic->pending = copied;
    if (copied && !dynamic->trace_layers.empty()) {
        dynamic->trace_cursor = (dynamic->trace_cursor +
                std::min((size_t) dynamic->trace_per_token, dynamic->trace_layers.size())) % dynamic->trace_layers.size();
    }
}

void llama_hotmoe::apply_pending(ggml_backend_sched_t sched) {
    auto * dynamic = hotmoe_dynamic_find(this);
    if (dynamic == nullptr || !dynamic->pending) {
        return;
    }

    ggml_backend_sched_synchronize(sched);
    dynamic->pending = false;
    ++dynamic->tick;

    for (auto & kv : layers) {
        const int il = kv.first;
        auto & dl = dynamic->layers.at(il);
        for (int i = 0; i < dynamic->n_expert_used; ++i) {
            const int32_t e = dynamic->pending_ids[(size_t) il * dynamic->n_expert_used + i];
            if (e >= 0 && e < n_expert) {
                dl.last_seen[e] = dynamic->tick;
                if (dl.frequency[e] != UINT32_MAX) {
                    ++dl.frequency[e];
                }
            }
        }
    }

    int budget = dynamic->swaps_per_token;
    int layer_span = 0;
    for (const auto & kv : layers) {
        layer_span = std::max(layer_span, kv.first + 1);
    }
    for (int scan = 0; scan < layer_span && budget > 0; ++scan) {
        const int il = (int) ((dynamic->update_cursor + scan) % layer_span);
        auto it = layers.find(il);
        if (it == layers.end()) {
            continue;
        }
        auto & l = it->second;
        auto & dl = dynamic->layers.at(il);

        int32_t candidate = -1;
        for (int i = 0; i < dynamic->n_expert_used; ++i) {
            const int32_t e = dynamic->pending_ids[(size_t) il * dynamic->n_expert_used + i];
            if (e >= 0 && e < n_expert && l.slot_of[e] < 0 &&
                (candidate < 0 || dl.frequency[e] > dl.frequency[candidate] ||
                 (dl.frequency[e] == dl.frequency[candidate] && dl.last_seen[e] > dl.last_seen[candidate]))) {
                candidate = e;
            }
        }
        if (candidate < 0) {
            continue;
        }

        int victim_slot = 0;
        for (int s = 1; s < n_slots; ++s) {
            if (dl.frequency[l.expert_of[s]] < dl.frequency[l.expert_of[victim_slot]] ||
                (dl.frequency[l.expert_of[s]] == dl.frequency[l.expert_of[victim_slot]] &&
                 dl.last_seen[l.expert_of[s]] < dl.last_seen[l.expert_of[victim_slot]])) {
                victim_slot = s;
            }
        }
        const int32_t victim = l.expert_of[victim_slot];
        if (dl.frequency[candidate] <= dl.frequency[victim] + 1) {
            continue;
        }

        struct pair { ggml_tensor * dst; ggml_tensor * src; };
        const pair pairs[3] = { { l.gate, dl.gate_src }, { l.up, dl.up_src }, { l.down, dl.down_src } };
        for (const auto & p : pairs) {
            const size_t stride = p.src->nb[2];
            const char * src_e = (const char *) p.src->data + (size_t) candidate * stride;
            ggml_backend_t backend = dynamic->async_fetch
                ? ggml_backend_sched_get_tensor_backend(sched, p.dst)
                : nullptr;
            if (backend != nullptr) {
                ggml_backend_tensor_set_async(backend, p.dst, src_e, (size_t) victim_slot * stride, stride);
            } else {
                ggml_backend_tensor_set(p.dst, src_e, (size_t) victim_slot * stride, stride);
            }
        }

        l.slot_of[victim] = -1;
        l.slot_of[candidate] = victim_slot;
        l.expert_of[victim_slot] = candidate;

        const int32_t hot_victim = n_slots;
        const int32_t cold_victim = victim;
        const int32_t hot_candidate = victim_slot;
        const int32_t cold_candidate = -1;
        ggml_backend_tensor_set(l.lut_hot,  &hot_victim,     victim * sizeof(int32_t),    sizeof(int32_t));
        ggml_backend_tensor_set(l.lut_cold, &cold_victim,    victim * sizeof(int32_t),    sizeof(int32_t));
        ggml_backend_tensor_set(l.lut_hot,  &hot_candidate,  candidate * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_set(l.lut_cold, &cold_candidate, candidate * sizeof(int32_t), sizeof(int32_t));

        --budget;
    }
    if ((dynamic->tick & 511) == 0) {
        for (auto & kv : dynamic->layers) {
            for (auto & value : kv.second.frequency) {
                value = (value + 1) / 2;
            }
        }
    }
    dynamic->update_cursor = (dynamic->update_cursor + std::max(1, dynamic->swaps_per_token)) % layer_span;
}
