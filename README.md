# Qwen3.8-Flash-Next 128K Vulkan HotMoE Runtime

Portable Windows runtime profile for an RX 7900 XTX (24 GB), Ryzen 9 7900X,
and 64 GB system RAM. It is locked to the AtomicChat
`AD-4.27bpw-Q4_K_M-M64` 33-shard model, 131072 context, f16 K/V caches,
Vulkan, mmap PLE storage, exact-page PLE prompt prefetch, and the included
HotMoE implementation. The model weights are not redistributed.

## Install

1. Download or clone this repository on Windows 11. No compiler, Python,
   package manager, or global llama.cpp installation is required.
2. Download all 33 shards of AtomicChat
   `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`. Keep them together in one
   directory; shard 1 must end in `00001-of-00033.gguf`.
3. Open PowerShell in the repository directory and run the launch command
   below, changing only `ModelPath`.
4. Open `http://127.0.0.1:8080` or point any OpenAI-compatible client at
   `http://127.0.0.1:8080/v1`.

The launcher verifies the model name, shard count, and exact total byte size
before allocating memory. Use `-CheckOnly` first if the files were copied from
another machine.

## Launch

Open PowerShell in this folder and pass only the path to shard 1:

```powershell
.\Launch-Qwen128K.ps1 -ModelPath "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
```

The OpenAI-compatible server listens only on `127.0.0.1:8080`. Use `-Port`
to choose another local port.

To validate the package and model without allocating the model:

```powershell
.\Launch-Qwen128K.ps1 -ModelPath "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf" -CheckOnly
```

For a slow, complete checksum pass:

```powershell
.\scripts\Test-ModelFiles.ps1 -ModelPath "X:\models\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf" -FullHash
```

## Locked production settings

- Build: `b200-c589f0e`, Vulkan HotMoE fork
- Context: `131072`, one sequence
- KV: f16 K and f16 V
- MoE: 38 CPU-resident expert layers, 24 Vulkan hot slots per layer
- Dynamic cache: 2 replacements per decoded token
- Batch / ubatch: 4096 / 2048
- Threads: 22, leaving two logical CPUs to Windows
- PLE tensor: CPU mmap with lazy reads
- PLE prompt optimization: exact-page Windows prefetch, decode path untouched
- Flash attention: on
- Auto-fit, HostMoE, MTP/speculation, and context shifting: off

Do not add `GGML_VULKAN_MEMORY_HEAP_SIZE=23500`; this build has no code that
reads that variable. Do not raise ubatch to 4096: it requested a 9080 MiB
Vulkan compute buffer and fell back after an out-of-device-memory allocation.

## Acceptance status

The package has passed model-load, 131072-context allocation, f16-cache, PLE
mmap, HotMoE initialization, and repeat decode tests on the target machine.
The best short-context run reached 19.67-19.78 tok/s after warm-up. A prompt
that actually fills all 131072 positions is still required before claiming
20 tok/s at full depth. The controlled 12,133-token prompt measured 80.72
tok/s without prefetch and 82.37 tok/s on the prefetched first block; the
requested 200 tok/s was not achieved. See [MEMORY-MAP.md](MEMORY-MAP.md) and
[BENCHMARKS.md](BENCHMARKS.md).

The model is not included. Obtain the exact 33-shard AtomicChat release from
its publisher; the launcher rejects other quantizations and incomplete sets.

## Redistribution and source

The bundled runtime is derived from llama.cpp under the included MIT license.
`patches/hotmoe-qwen4exp-tracked.patch` contains tracked source changes against
base commit `c589f0ed10c643678c4707dd160c21ac7633ebc0`; new source files are under
`patches/source-additions`. 
