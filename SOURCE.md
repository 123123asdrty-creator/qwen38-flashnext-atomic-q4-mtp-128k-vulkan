# Build from source

The bundled runtime reports build 10682 at llama.cpp commit
`843d5750579a15ed4a42d73eb862855c271021ac`.

1. Clone llama.cpp and check out that exact commit.
2. Apply `patches/qwen4exp-atomic-mtp-runtime.patch`.
3. Copy `patches/source-additions/src/llama-hotmoe.cpp` and
   `llama-hotmoe.h` into the source tree's `src` directory.
4. Run `scripts/Build-PatchedRuntime.ps1 -SourceDirectory X:\src\llama.cpp`.
5. Copy the resulting Release server/perplexity executables and DLL dependency
   set from the build `bin` directory.

The 118 KB patch is a full experimental-tree snapshot, not a minimal release
patch. It includes the current key-only QSA path,
selective rollback-ring predicates, MTP transient-state serialization/restoration,
request MTP controls, and server integration. The additions are required by the
patched CMake source list. Only `config/release-profile.json` is claimed as the
accepted runtime profile. The prebuilt, hash-verified runtime is the supported
path; rebuilding from this broad snapshot is provisional until a minimal patch
is curated and revalidated.
