# Qwen3.8 Flash-Next AtomicChat · 128K Vulkan Runtime

A local Windows runtime for the existing **AtomicChat Q4_K_M** model on a **Radeon RX 7900 XTX with 64 GB RAM**. Download the normal model separately and point the launcher at its first shard. The launcher downloads the matching MTP helper automatically.

With MTP enabled by default, the September profile measured roughly **196–216 prompt tokens/s from 32K to 124K loaded context**, with **20–24 decode tokens/s** on the included synthetic workload.

| Loaded context | Prompt tok/s | Decode tok/s |
|---|---:|---:|
| 8K | 173.27 | 13.73 |
| 32K | 215.79 | 23.74 |
| 48K | 214.74 | 22.23 |
| 64K | 210.81 | 21.45 |
| 96K | 200.87 | 20.32 |
| 124K | 196.49 | 20.39 |

All measurements allocate 131,072 context tokens. Each measured step adds about 4,096 fresh tokens and generates 128 greedy tokens. The final prompt depth is 126,976 tokens; this is not a completely filled 128K measurement. See [benchmark evidence](BENCHMARKS.md).

## Start locally

Requirements: Windows 11, PowerShell 7, AMD Vulkan drivers, an RX 7900 XTX 24 GB, Ryzen 9 7900X-class CPU, 64 GB RAM, and a fast SSD. This is a profile for this hardware, not a universal GPU preset.

1. Clone this repository onto an SSD.
2. Place all 33 AtomicChat shards together. The first must be named:
   `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`.
3. Run in PowerShell 7:

```powershell
.\Launch-AtomicQ4-MTP.ps1 `
  -Model "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
```

Connect your existing OpenAI-compatible client to **http://127.0.0.1:8080/v1**. Add `-CheckOnly` to validate files and the selected GPU without loading the model. The launcher verifies runtime hashes, target shard sizes, and native GPU placement.

**The matching MTP helper is included in [Releases](https://github.com/123123asdrty-creator/Qwen3.8-Flash-Next-AtomicChat-128K-Vulkan/releases/tag/mtp-2026.09.04).** On first launch, `scripts/Get-Mtp.ps1` downloads its two parts, joins them into `models/mtp/mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf`, and verifies the complete SHA-256: `41ef1d94ee9249d4140de494d1ad6de4441860e1b50e31cf4cceb0971f8ddf12` (2,362,007,744 bytes). This is the exact converted helper used for the measurements above.

To prepare it ahead of time, run `.\scripts\Get-Mtp.ps1`. Downloads resume after interruption. Allow about 4.8 GB of free space during assembly; the downloaded parts are removed after verification. For an existing copy, pass `-DraftModel "X:\models\mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf"`. `-DisableMtp` is available for explicit control runs, whose decode speeds differ from the MTP results.

The server keeps two recurrent-state checkpoints so successive requests can reuse their unchanged prefix. Clients should send the complete message sequence with each request. A new conversation or changed prefix still requires prompt processing.

## What changed

- **Bounded sparse prompt attention:** gathers selected positions in groups of 64 while preserving BF16 target K/V and selector storage.
- **More room for long context:** 1,536-token target batches, a separate 256-token MTP batch, 40 CPU expert layers, shared draft output projection, and smaller MTP catch-up reservations.
- **Shared expert memory:** CPU decode and staged GPU prompt reads use the same committed RAM copies for four expert layers.
- **Faster selection:** an eight-bank Vulkan TOP_K histogram reduces contention on wide prompt rows.
- **Conversation continuity:** recurrent checkpoints and explicit prefix caching replace the old checkpoint-disabled launch.

The runtime allocates **20,663 MiB of GPU buffers at startup**. Its target compute workspace grows from 1,217 to 3,501 MiB in the long-context run, bringing the modeled live buffer total to about **22,947 MiB** before desktop and driver overhead. The [memory map](docs/MEMORY-MAP.md) accounts for all 1,258 tensor payloads, native buffers, staged RAM, and mapped storage.

## Precision and validation

The exact Atomic Q4_K_M weights remain selected. Target K/V and selector caches stay **BF16**; recurrent states stay FP32. Only the MTP helper caches use Q8.

Four independent GPU attention checks passed against a double-precision CPU oracle, including a 128K cache and BF16 values outside F16 range. Gathered BF16 cache bits matched exactly. The new reduction order can change generated text: 32K, 48K, and 64K matched the reference on this fixture; 8K differed. This release has not established full-model KLD equivalence or unchanged general intelligence. Historical quality results and their limits remain in [benchmark documentation](BENCHMARKS.md).

## Package contents

The repository includes the server and perplexity binaries, DLL dependencies and hashes, model validation, the source patch and additions, and synthetic benchmark/memory evidence. Releases provide the matching MTP helper. The regular AtomicChat target model is downloaded separately.

See [tuning controls](TUNING.md), [build instructions](SOURCE.md), and the machine-readable [release profile](config/release-profile.json). Use `-DisableMtp`, `-DisableHostMoe`, or `-DisableSparsePrefill` for explicit control runs; their speeds are not the numbers above.
