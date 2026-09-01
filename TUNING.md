# Tuning

The launcher exposes `-Model`, `-DraftModel`, `-Server`, `-Context`,
`-Batch`, `-UBatch`, `-DraftNMax`, `-DraftPMin`, and `-Port` directly.

| Parameter | Default | Guidance |
|---|---:|---|
| `Context` | 131072 | Keep for the production profile. |
| `Batch` / `UBatch` | 4096 / 1024 | Reduce UBatch to 512 for allocation trouble; retest 2048 for prompt speed. |
| `Threads` | 22 | Tuned for the Ryzen 9 7900X. |
| `CpuMoeLayers` | 38 | Lower needs more VRAM; higher is generally slower. |
| `DraftNMax` | 3 | Try 2 only in a controlled long-context A/B. |
| `DraftPMin` | 0.7 | Higher is more conservative; lower verifies more drafts. |
| `CacheK` / `CacheV` | bf16 / bf16 | Quality reference. Q8 target KV is rejected. |
| `DisableMtp` | off | Explicit non-MTP control; DraftModel is then optional. |
| `DisablePromptCache` | off | Clean independent-request benchmark mode. |
| `DisableQsaKeyOnly` | off | Diagnostic full-indexer mode; costs about 768 MiB at 128K. |
| `GpuDeviceIndex` | 1 | Tested XTX index on the dual-GPU system. |

Fixed quality controls: BF16 QSA routing key, dense bypass off, gather off,
strided-add off, on-device speculative checkpoints off, HotMoE/HostMoE off,
`-ctxcp 0`, `--fit off`, mmap on, and context shifting off.

Change one variable per run. Restart between MTP and non-MTP profiles, run the
same prompt twice, and treat the first row as page-cache warm-up.
