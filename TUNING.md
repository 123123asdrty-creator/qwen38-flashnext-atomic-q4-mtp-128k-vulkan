# Runtime controls

| Parameter | Default | Effect |
|---|---:|---|
| Context | 131072 | Allocated context capacity; 124K loaded is the deepest measured prompt |
| Batch | 4096 | Logical prompt batch |
| UBatch | 1536 | Bounds the growing sparse attention workspace |
| DraftUBatch | 256 | Separate MTP catch-up workspace; capped to target UBatch |
| CpuMoeLayers | 40 | Eight expert layers remain in dedicated VRAM |
| Threads | 22 | Leaves two logical CPUs for the desktop on the measured machine |
| DraftNMax | 3 | Maximum speculative draft length |
| DraftPMin | 0.7 | Minimum MTP draft probability |
| ContextCheckpoints | 2 | Retains recent recurrent prefix boundaries in RAM |
| CacheK / CacheV | bf16 | Target cache precision; other values are outside this measured profile |
| DraftCache | q8_0 | MTP helper cache precision |
| GpuDeviceIndex | 1 | Native preflight must identify RX 7900 XTX as Vulkan0 |
| Port | 8080 | Localhost-only inference API |

MTP is enabled when a matching `-DraftModel` is supplied; the model alone runs with MTP off. `-DisableMtp`, `-DisableHostMoe`, `-DisableSparsePrefill`, `-DisablePromptCache`, and `-DisableQsaKeyOnly` provide explicit off paths. Larger microbatches can grow GPU scratch enough to trigger a severe prompt-speed drop at depth. Recheck memory and throughput when changing context, placement, cache types, or the desktop workload.

The launch clears inherited LLAMA_/GGML_ overrides, then sets the declared profile. CPU-mapped PLE remains mmap-backed. Four expert banks are staged in committed system RAM and shared by CPU decode and Vulkan prompt reads. Duplicate original mappings remain valid fallback storage. Native phase profiling and allocation auditing are off for normal use.

Prefix reuse requires an unchanged earlier token sequence. Two recurrent checkpoints retain usable boundaries; setting the count to zero can cause full reprocessing between turns. Arbitrary middle edits and context relocation cannot reuse the recurrent prefix. No automatic context shifting is enabled.
