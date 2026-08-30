# Controlled tuning results

All rows use the same AtomicChat model, Vulkan build, 131072 allocated context,
f16 K/V, 38 CPU-MoE layers, batch 4096, 22 effective threads, no MTP, and the
same three coding prompts. These are short-context decode tests, not a filled
128K-context claim.

| Hot slots | Dynamic swaps | Ubatch | Warm decode | Decode 1 | Decode 2 | Decode 3 | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 2 | 4096 | 7.30 | 8.94 | 9.27 | 9.53 | allocator fallback |
| 10 | 2 | 2048 | 11.62 | 15.12 | 15.75 | 16.34 | stable |
| 16 | 2 | 2048 | 12.18 | 17.18 | 18.81 | 18.27 | stable |
| 24 | 2 | 2048 | 12.81 | 16.51 | 19.67 | 19.78 | selected |
| 24 | 3 | 2048 | 12.96 | 16.41 | 19.56 | 18.69 | slower than 2 swaps |

On an exact 12,133-token controlled prompt, the unmodified mmap path measured
80.72 tok/s. Exact-page PLE prefetch improved the first 4K block from 76.69 to
82.37 tok/s (about 7.4%), but did not reach the requested 200 tok/s. A separate
prompt-only placement with six more GPU layers fit in VRAM but stalled for over
two minutes before producing a checkpoint, so it is not shipped. The package
therefore keeps the selected decode-first profile and labels prompt throughput
as degraded rather than claiming the 450 tok/s planning target.

MTP is intentionally disabled. Earlier native-MTP tests did not deliver a
stable greater-than-1.2x gain, and the f16 128K allocation has priority.
