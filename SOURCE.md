# Build from source

The runtime is based on llama.cpp commit `843d5750579a15ed4a42d73eb862855c271021ac` (build 10682). The packaged patch is a 44-file snapshot of the custom implementation, accompanied by the HostMoE sources and independent BF16 gather test. It includes earlier runtime fixes as well as the September memory and attention changes.

1. Clone llama.cpp and check out that exact base commit in a clean source directory.
2. Install Visual Studio C++ build tools, CMake, and the Vulkan SDK.
3. From this package, run:

```powershell
.\scripts\Build-PatchedRuntime.ps1 -SourceDirectory "X:\src\llama.cpp"
```

The script checks the base identity, applies the patch exactly once, copies all source additions, and builds the server and perplexity binaries. Do not apply the patch manually before running it. The runtime exposes the HTTP inference API; the embedded UI build is disabled.

The patch was verified to apply to the exact clean base index. The native build completed the 124K-depth run and numerical checks described in [BENCHMARKS.md](BENCHMARKS.md). Packaged source-location strings are normalized; every executable section was verified byte-for-byte unchanged. A separate clean-tree rebuild has not been timed or compared byte-for-byte. Compiler and Vulkan SDK versions can change binary hashes; `config/runtime.sha256` identifies the distributed artifacts.

To build the attention oracle, run CMake's `test-qsa-bf16-gather` target with tests enabled. Select the discrete GPU with `GGML_VK_VISIBLE_DEVICES=1` on the measured dual-AMD machine and verify the adapter name. The test downloads gathered cache values for exact bit comparison and independently computes masked sparse attention on the CPU.

Only `config/release-profile.json` describes the measured shipping configuration. Other experimental controls remain available in the source without a performance or quality claim.
