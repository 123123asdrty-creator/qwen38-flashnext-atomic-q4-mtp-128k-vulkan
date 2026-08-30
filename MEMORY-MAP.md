# Verified 128K memory map

Values below are emitted by build `b200-c589f0e` while loading the exact
AtomicChat Q4_K_M-M64 model with the production profile. A Vulkan compute
reservation is not additive to all persistent buffers because the scheduler
reuses/aliases allocations, so summing every reported number overstates live
VRAM. Use `scripts/Watch-Vram.ps1` for the process-resident reading.

| Allocation | Verified value |
|---|---:|
| Vulkan model buffers (`-ncmoe 38`) | 15595.52 MiB |
| HotMoE duplicate cache (24 x 38) | 1.78 GiB |
| Regular QSA f16 KV | 3072.00 MiB |
| Selector/rotation cache | 1152.00 MiB |
| Recurrent GDN state | 112.57 MiB |
| Vulkan compute reservation (`-ub 2048`) | 4492.28 MiB |
| Vulkan-host compute reservation | 2007.71 MiB |
| Isolated PLE CPU-mapped shard | 36621.27 MiB |
| Windows dedicated-VRAM counter after load | 21.77 GiB |

The measured cache total is therefore 4224 MiB, not 3.0 GB. The 3072 MiB
figure covers the conventional f16 K/V cache; this runtime also allocates the
1152 MiB QSA selector/rotation cache. With `-ub 4096`, the compute reservation
rose to 9080.56 MiB and produced an allocator warning, so 2048 is mandatory.

At clean shutdown the runtime reported `24424 MiB self = 15595 model + 4336
context + 4492 compute`, plus 135 MiB unaccounted. Its internal category table
does not list the separate HotMoE allocation even though HotMoE reports 1.78
GiB, so that table cannot be summed with HotMoE as if every reservation were
simultaneously resident. The Windows per-process counter is the live-residency
measurement.

HotMoE's measured live hit rate reached 20.4% with 24 slots during the short
coding benchmark. The earlier 80% estimate corresponds to a much larger
top-expert set and is not attainable with this 24 GB allocation.
