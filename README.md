# Qwen3.8 Flash-Next Atomic Q4 MTP 128K Vulkan Runtime

An exact, portable Windows runtime profile for one model and one machine class:

- AtomicChat `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`, all 33 shards
- the matching converted Q4_0 MTP sidecar listed below
- AMD Radeon RX 7900 XTX 24 GB
- AMD Ryzen 9 7900X
- 64 GB system RAM
- Windows 11, a fast SSD, and current AMD Vulkan drivers

The profile keeps the target model and its QSA routing cache at BF16, uses the
RX 7900 XTX for Vulkan-resident model buffers, and uses MTP depth 3 for
speculative decode. It does not select a smaller model or another quantization
when memory is tight.

This repository replaces the former HotMoE package. HotMoE, HostMoE, KVarN,
Q8 target KV, dense bypass, and experimental gather paths are disabled in the
quality-first profile.

## Current measured status

| Gate | Result | Status |
|---|---:|---|
| Published Atomic Q4 weights vs original BF16 mean KLD | 0.084216 | reference |
| Corrected target-verify graph vs serial target mean KLD | 0.048002 | measured; strict 0.01 incremental gate failed |
| Warm 2K prompt processing | 243.42 tok/s | passes 200 tok/s short-context target |
| Warm 2K MTP decode | 23.86 tok/s | passes 20 tok/s short-context target |
| MTP draft acceptance | 63 / 67 (94.03%) | measured |
| 128K allocation, 30K loaded, 256 generated | 126.24 prompt / 18.41 decode tok/s; 191/191 accepted | completed; below the original 200/20 targets |

KLD values from separate comparisons are not additive. This package does not
claim that the complete MTP path is under 0.1 versus original BF16 until a
direct end-to-end comparison proves it. See [BENCHMARKS.md](BENCHMARKS.md) for
the exact interpretation and limitations.

## What is included

- a self-contained `llama-server` Vulkan runtime;
- `Launch-AtomicQ4-MTP.ps1` with practical tuning parameters exposed;
- exact model and MTP-sidecar validation;
- a runtime integrity manifest and package test;
- compact endpoint-speed and fixed-prefix KLD evidence;
- compact benchmark evidence and the source patch identity.

Model files and temporary KLD-logit files are intentionally not redistributed.

## Install

1. Clone or download this repository to an SSD.
2. Put all 33 AtomicChat model shards in one directory. Select shard 1:

   `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`

3. Obtain the matching converted MTP sidecar:

   `mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf`

   Required size: `2,362,007,744` bytes.

   SHA-256: `41ef1d94ee9249d4140de494d1ad6de4441860e1b50e31cf4cceb0971f8ddf12`

4. Open PowerShell 7 in this repository and run:

```powershell
.\Launch-AtomicQ4-MTP.ps1 `
  -ModelPath "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf" `
  -DraftModelPath "X:\models\mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf" `
  -CheckOnly
```

5. Run the same command without `-CheckOnly` to start the server.

The Web UI is `http://127.0.0.1:8080`. OpenAI-compatible clients use
`http://127.0.0.1:8080/v1`.

## Default launch

```powershell
.\Launch-AtomicQ4-MTP.ps1 `
  -ModelPath "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf" `
  -DraftModelPath "X:\models\mtp-Qwen3.8-Flash-Next-Q4_0-qwen4exp-fast.gguf"
```

The default profile is:

- 131,072 allocated context, one slot;
- target K/V and QSA routing key in BF16;
- batch 4096, micro-batch 3072, 22 threads;
- 38 CPU-MoE layers with all remaining eligible layers on Vulkan;
- MTP depth 3, `p-min 0.7`, Q8_0 draft cache;
- QSA key-only storage on;
- automatic context checkpoints off;
- prompt cache on, cache shifting/reuse off;
- Flash Attention and mmap on;
- Vulkan adapter restricted to device index 1 for the tested dual-GPU layout.

At startup, the native log must name `AMD Radeon RX 7900 XTX` and report:

```text
Vulkan0 model buffer size = 15595.52 MiB
```

If the log names an integrated GPU, reports no usable GPU, or omits the Vulkan
model buffer, stop the server. Do not accept CPU-only generation as valid.
The 38 CPU-MoE placement and CPU-mapped PLE tensor are intentional parts of the
hybrid design; they do not mean the server fell back to CPU-only inference.

## Tuning and verification

- [TUNING.md](TUNING.md) explains each exposed speed, memory, and quality knob.
- [BENCHMARKS.md](BENCHMARKS.md) records accepted and rejected results.
- `.\scripts\Test-Package.ps1` verifies the bundled runtime.
- `evidence/final-results.json` is the compact result manifest.

## Important limits

- Do not use `--no-mmap` or `--mlock`. The 36+ GB PLE tensor must remain
  SSD-backed/mmap-backed on this 64 GB machine.
- Do not change target K/V to Q8_0 for production. That candidate measured mean
  KLD 0.155274 against BF16 target caches.
- Do not enable automatic context checkpoints. Chunk-shape drift was observed
  with the default count; the launcher fixes `-ctxcp 0`.
- MTP acceptance is not an intelligence proof. KLD/PPL and deterministic task
  checks remain the quality evidence.
- Close other Vulkan inference programs and GPU-heavy applications before
  launch. The launcher refuses to start if another llama inference process is
  already running.

## Source and redistribution

The bundled runtime is derived from llama.cpp under the included MIT license.
See [SOURCE.md](SOURCE.md) for the base revision, full experimental-tree patch,
build identity, and reproducibility limits. The bundled binary is the supported
release path; source reproduction is provisional until the patch is curated
down to the selected profile. No weights, private prompts, credentials,
application data, or giant benchmark-logit files are included.
