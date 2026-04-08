## Library target: `namespace rokoko` + `rokoko.h`

Rokoko needs to be embeddable as a static library in the [orchestrator](../orchestrator/) project, which links both paraketto (STT) and rokoko (TTS) into a single binary. Both projects define `struct Weights`, `CUDA_CHECK`, `GpuArena` and other common names — linking them together without namespaces produces symbol collisions.

### Changes

1. **Namespace wrapping**: `weights.h` (GpuArena, Weights, constants), `kernels.h` (kernel wrappers), `rokoko_common.h` (buffer structs, WAV I/O), `g2p.h` (G2PModelCuda struct only — extern "C" Cutlass declarations and __global__ kernels stay global). `CUDA_CHECK` and `vlog` macros + `extern bool g_verbose` stay global. `normalize.h` already has its own `text_norm` namespace — left as-is.

2. **Source files**: `rokoko.cpp` and `rokoko_f16.cpp` wrap their content in `namespace rokoko { }` after the `extern "C"` Cutlass declarations. `kernels.cu` and `weights.cpp` similarly wrapped/using. Cutlass GEMM/Conv `.cu` files untouched (extern "C" linkage, no namespace needed). `default_weights_filename()` stays in global namespace (called from main.cu which declares it extern without namespace).

3. **`rokoko.h`**: New public header with `TtsPipeline`, `TtsContext`, voice loading, vocab/tokenization, chunking, and download helpers — all extracted from `main.cu`. `TtsContext` wraps the entire init sequence (weight prefetch, CUDA init, upload, G2P load, voice mmap, arena allocation, warmup) into a single `.init()` call.

4. **`verbose.cpp`**: Provides `bool g_verbose = false;` definition for library builds (main.cu still defines it for CLI builds).

5. **`server.h`**: Uses `using namespace rokoko;` for the same two-phase template lookup reason as paraketto.

### Verification

Both build variants (FP32, FP16) compile and produce identical synthesis output. Benchmark unchanged:

```
short:  229x RTFx (median)
medium: 298x RTFx (median)
long:   349x RTFx (median)
```
