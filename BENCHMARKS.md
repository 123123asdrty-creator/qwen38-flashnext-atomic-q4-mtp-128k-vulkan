# Benchmark and quality summary

All local rows used the exact 33-shard AtomicChat Q4_K_M-M64 model and proved
`AMD Radeon RX 7900 XTX` plus a 15595.52 MiB Vulkan model buffer.

| Test | Context | Prompt tok/s | Decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| warmed short check | 8192 allocated / about 2K loaded | 243.4153 | 23.8584 | 63/67 |
| final depth check | 131072 allocated / 30000 loaded | 126.2350 | 18.4065 | 191/191 |

The final depth check generated 256 tokens. It completed successfully and was
accepted by the user, but it does not meet the earlier 200 prompt / 20 decode
floor. The short-context result must not be presented as long-context speed.

## Quality evidence

| Comparison | PPL ratio | Mean KLD | Same top |
|---|---:|---:|---:|
| Atomic Q4 weights vs original BF16, publisher | 1.025657 | 0.084216 | 89.487% |
| current BF16-cache runtime vs prior same-Q4 runtime | 1.001369 ± 0.006981 | 0.024927 ± 0.002932 | 95.988% |
| MTP target-verify graph vs serial target graph | 1.015908 ± 0.017304 | 0.048002 ± 0.005825 | 93.725% |
| Q8_0 target KV vs BF16 target KV | — | 0.155274 | rejected |

The MTP row failed the deliberately strict incremental limits of KLD at most
0.01 and PPL-ratio point estimate at most 1.005. Its mean KLD is below 0.1,
but separate KLD experiments are not additive; no end-to-end BF16 claim is
made. BF16 target K/V remains the production default.

The selected correction uses rollback-ring state only for true multi-token
target verification batches. Prompt and serial paths retain the ordinary graph.
Checkpoint state now includes the MTP hidden-row position and is restored on
rejection. Automatic context checkpoints are disabled with `-ctxcp 0`.

Compact machine-readable values are in `evidence/final-results.json`.
