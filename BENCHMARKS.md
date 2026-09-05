# Measured performance

## September 4, 2026 release

| Loaded context | Prompt tok/s | Decode tok/s |
|---|---:|---:|
| 8K | 173.27 | 13.73 |
| 32K | 215.79 | 23.74 |
| 48K | 214.74 | 22.23 |
| 64K | 210.81 | 21.45 |
| 96K | 200.87 | 20.32 |
| 124K | 196.49 | 20.39 |

Target: exact AtomicChat AD-4.27bpw Q4_K_M-M64, all 33 shards. Hardware: RX 7900 XTX 24 GB, Ryzen 9 7900X, 64 GB RAM, Windows/Vulkan. Capacity: 131,072 tokens. Placement: 40 CPU expert layers. Batch/microbatch/draft microbatch: 4096/1536/256. MTP: depth 3, minimum probability 0.7. Target K/V and selector: BF16. Helper K/V: Q8. Sparse prompt gather tile: 64. Staged HostMoE: four shared CPU/RAM layers, 512–4096-token prompt batches.

This completed append-only synthetic code workload measures approximately 4,096 fresh prompt tokens followed by 128 greedy generated tokens at each depth. Intervening fill requests and cache counts are retained in [structured evidence](evidence/throughput-20260904.json). Prompt speed excludes cached tokens; decode is the native generated-token timing. Diagnostic phase synchronization was enabled. Desktop activity and page residency affect results.

The preceding 39-CPU-layer candidate reached 211.92 PP / 21.33 decode at 64K but fell to 102.08 PP / 21.56 decode at 96K. Moving one more expert layer to the CPU recovered 962.5 MiB of GPU headroom and produced the completed result above. It costs some decode speed, while preventing the long-context prompt collapse in this run.

The original configuration reserved 25,434.87 MiB of GPU buffers, exceeding the card's physical 24,560 MiB before desktop overhead. It measured 60.18 PP / 15.29 decode at 32K and 49.57 PP / 14.59 decode at 64K on the same fixture. These are sequential controlled runs, not randomized or repeated statistical trials.

## Correctness scope

The memory-only revision matched all 128 greedy reference tokens at 8K, 32K, 48K and 64K. With sparse attention and the 40-layer profile, 32K/48K/64K matched; 8K differed. There is no dense reference above 64K. Sparse attention preserves selection and BF16 cache values, but changes FP32 reduction order.

The independent `test-qsa-bf16-gather` GPU test passed four cases against double-precision CPU attention, spanning tile boundaries, causal masking, padded selections, a 128K cache, and BF16 values beyond F16 range. Maximum scaled error was at most 1.45e-7; NMSE was approximately 2.6e-14–4.8e-14. TOP_K passed 252 native checks. These checks do not establish a full-model KLD or general intelligence result.

Historical measurements reported Atomic weights versus original BF16 mean KLD 0.084216, and corrected target verification versus serial target mean KLD 0.048002 (failing the strict 0.01 incremental gate). Separate KLD results cannot be added. Those results predate this release and are retained, with their original limitations, in [previous measurements](docs/PREVIOUS-BENCHMARKS.md).
