#pragma once

// HotMoE - a resident cache of the most-frequently-routed MoE experts in VRAM.
//
// Motivation
// ----------
// With `-ncmoe N` the expert weights of the first N layers are kept in host RAM
// and the rest live in VRAM. Because every layer routes the same number of
// tokens, that placement covers exactly (n_layer - N)/n_layer of all expert
// reads - it is blind to *which* experts are actually used.
//
// Expert selection is heavily skewed. Measured on Qwen3.8-Flash-Next
// (512 experts, 10 used per token) the busiest 25% of experts in a layer absorb
// ~81% of all routed selections. HotMoE spends the same VRAM budget on the hot
// experts of *every* layer instead of on all experts of *some* layers, which
// turns a 25% hit rate into ~81% for free.
//
// Mechanism
// ---------
// The full expert tensors stay where they are (host RAM, mmap'd). For each MoE
// layer we additionally allocate a VRAM cache of `n_slots + 1` experts. The
// extra slot is filled with zeroes and acts as a no-op expert.
//
// Two small i32 lookup tables map an expert id onto the two paths:
//
//   lut_hot [e] = slot index of e, or `n_slots` (the zero expert) if not cached
//   lut_cold[e] = e if e is not cached, or -1 if it is
//
// The MoE matmul is then evaluated twice and summed:
//
//   out = mul_mat_id(cache, x, lut_hot [ids])     // on the GPU
//       + mul_mat_id(full,  x, lut_cold[ids])     // on the CPU
//
// Cached experts contribute through the GPU term and read zeroes on the CPU
// term; uncached experts do the reverse. The GPU side needs no kernel change -
// the zero expert stays resident in cache and costs almost nothing. The CPU
// side skips rows whose id is negative (see ggml_compute_forward_mul_mat_id),
// so uncached-only work is all it pays for.
//
// Configuration is by environment variable so that no CLI or public API
// surface changes:
//
//   LLAMA_HOTMOE          slots per layer; 0/unset disables HotMoE
//   LLAMA_HOTMOE_PROFILE  hotmoe-profile.json to seed the cache from
//   LLAMA_HOTMOE_MAX_TOK  max tokens per ubatch that use the cache (default 1)
//   LLAMA_HOSTMOE         expose host-resident experts directly to the GPU
//                         through Vulkan external host memory (experimental)
//   LLAMA_HOSTMOE_MIN_TOK min tokens per ubatch that use HostMoE (default 1)
//   LLAMA_HOSTMOE_MAX_TOK max tokens per ubatch that use HostMoE (default 1)
//   LLAMA_HOSTMOE_STAGE   copy imported tensors into committed RAM first (default 1)

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;

struct llama_hotmoe_layer {
    // VRAM cache, [n_embd, n_ff, n_slots + 1] laid out like the source tensor
    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;

    // Stable Vulkan aliases of the full host-resident expert tensors. The
    // storage remains in system RAM; only the selected expert ranges cross
    // PCIe when the Vulkan mul_mat_id kernels read them.
    ggml_tensor * host_gate = nullptr;
    ggml_tensor * host_up   = nullptr;
    ggml_tensor * host_down = nullptr;

    // Non-owning CPU views of the same staged storage used by the Vulkan views.
    ggml_tensor * cpu_gate = nullptr;
    ggml_tensor * cpu_up   = nullptr;
    ggml_tensor * cpu_down = nullptr;

    // [1, n_expert] expert-to-cache lookup tables
    ggml_tensor * lut_hot  = nullptr;
    ggml_tensor * lut_cold = nullptr;

    std::vector<int32_t> slot_of;    // expert -> slot, or -1 when not cached
    std::vector<int32_t> expert_of;  // slot   -> expert, or -1 when empty
};

struct llama_hotmoe {
    bool enabled     = false;
    bool host_direct = false;
    bool init_attempted = false;
    bool profile_enabled = false;
    int  n_slots    = 0;
    int  n_expert   = 0;
    int  min_tokens = 1;
    int  max_tokens = 1;
    int  host_phase = -1; // -1 unknown, 0 normal/decode graph, 1 HostMoE prompt graph
    int  profile_max_tokens = 1;

    size_t vram_bytes = 0;
    double seed_coverage = 0.0;   // fraction of profiled selections the seed covers
    uint64_t generation = 0;

    std::string profile_path;
    uint64_t profile_tokens = 0;
    std::unordered_map<int, std::vector<uint64_t>> profile_counts;

    std::unordered_map<int, llama_hotmoe_layer> layers;

    // owned resources
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<void *>                host_allocs;

    ~llama_hotmoe();
    void reset();

    const llama_hotmoe_layer * get(int il) const {
        if (!enabled) {
            return nullptr;
        }
        auto it = layers.find(il);
        return it == layers.end() ? nullptr : &it->second;
    }

    void capture_async(const std::vector<ggml_tensor *> & ids, ggml_backend_sched_t sched);
    void capture_profile(const std::vector<ggml_tensor *> & ids, ggml_backend_sched_t sched);
    void apply_pending(ggml_backend_sched_t sched);
    int dynamic_swaps() const;
    bool trace_layer(int il) const;
};

// Allocate and populate the cache. Safe to call on any model: it is a no-op
// unless LLAMA_HOTMOE is set and the model has CPU-resident expert tensors.
void llama_hotmoe_init(llama_model & model, bool force = false);

// Release the cache for large prompt batches and rebuild it for decode when
// LLAMA_HOTMOE_PHASED is enabled.
void llama_hotmoe_maybe_phase(llama_model & model, int n_tokens, ggml_backend_sched_t sched);
