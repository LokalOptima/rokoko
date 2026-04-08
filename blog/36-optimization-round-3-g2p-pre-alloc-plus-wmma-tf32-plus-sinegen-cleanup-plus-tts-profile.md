## Optimization Round 3: G2P Pre-alloc + WMMA TF32 + SineGen Cleanup + TTS Profile (2026-03-14)

Four optimizations in one round: eliminating CUDA graph invalidation, tensor core acceleration for the fused FFN, memory reclamation for SineGen, and profiling TTS to find next bottlenecks.

### Item 1: Pre-allocate G2P workspace at max T

**Problem**: When a text longer than any previous input arrived, the G2P workspace grew via `cudaFree` + `cudaMalloc`. This changed the base pointer, invalidating all cached CUDA graphs (whose pointers were baked in during capture). Every cached graph had to be re-captured.

**Fix**: Allocate workspace at `max_pos_` (2048) size during `load_from_file_()`. The workspace is 89.7 MB (dominated by `h*T*T` attention scores at 64 MB for T=2048). Trivial next to TTS's 327 MB. The grow-and-invalidate block in `infer()` becomes a simple bounds check that should never fire.

**Result**: Graphs are never invalidated regardless of input sequence. Cached replay confirmed at 2.7ms.

### Item 2: WMMA TF32 GEMV in fused FFN kernel (reverted — no improvement)

**Problem**: The fused FFN kernel's scalar GEMV (gate+up and down projections) was suspected to be slower than cuBLAS tensor cores (1.1ms vs 0.3ms in earlier testing — 3.7x gap).

**Approach**: Replace scalar dot-product loops with WMMA 16x16x8 TF32 matrix-multiply fragments. Each warp processes 16-wide output tiles, accumulating over the input dimension in chunks of 8.

**Challenge — Opaque fragment layout**: The WMMA API treats fragment element mappings as opaque. Can't manually fill `b_frag.x[i]` — `i % 8` doesn't map to row index. First attempt also tried `store_matrix_sync` to thread-local arrays (a warp-cooperative op that requires shared memory). Rewrote using per-warp shared memory broadcast buffers (128 floats per warp for B fragment, 256 floats for accumulator store).

**A/B benchmark result**: Both scalar and WMMA achieve **1.1ms cached replay**. WMMA wasted 15/16 of tensor core compute broadcasting the input vector across 16 columns of the B fragment. The broadcast buffer fill (128 floats per warp per k-chunk) and accumulator store/extract added overhead that negated any tensor core throughput advantage. The scalar version with coalesced memory access (from the weight transpose) was already memory-bandwidth-limited, not compute-limited — tensor cores couldn't help.

**Decision**: Reverted WMMA. Kept scalar FFN kernel (simpler, same speed, full FP32 precision). The `-arch=native` Makefile change was kept since it generates correct code for our Blackwell GPU regardless.

**Lesson**: WMMA is designed for GEMM, not GEMV. For matrix-vector products, the 16-wide N dimension is pure waste. The earlier "cuBLAS 0.3ms" measurement was likely a batched GEMM (multiple columns), not a single-column GEMV. For true GEMV on small matrices (256×2048), scalar with coalesced access is hard to beat.

### Item 3: TTS nsys profile

**Approach**: Profile-only, no code changes. Run nsys to identify where time actually goes.

**Key findings** (49 tokens, T=49, medium-length text, 28.2ms total):

| Kernel Category | Time (ms) | % | Instances | Avg (us) |
|---|---|---|---|---|
| im2col + simt SGEMM (conv) | 6.2 | 31% | 150 | 41 |
| Cutlass tensorop GEMM | 4.7 | 23% | 42 | 112 |
| Instance norm (AdaIN) | 2.3 | 11% | 70 | 32 |
| cuBLAS GEMV | 1.6 | 8% | 765 | 2.1 |
| LSTM gates | 1.1 | 5% | 754 | 1.4 |
| Weight norm | 0.8 | 4% | 89 | 8.7 |
| Channel bias add | 0.5 | 3% | 76 | 7.1 |
| Snake activation | 0.4 | 2% | 48 | 9.3 |
| Everything else | 2.6 | 13% | — | — |

**Observations**:
- **im2col + simt SGEMM (31%)**: Convolutions using `im2col` unfold + non-tensor-core SGEMM (`align1`). The `align1` variant means weight layout doesn't meet tensor core alignment requirements. Aligning weights to 4-float boundaries would unlock tensor core GEMM and potentially 3-4x speedup for these ops.
- **LSTM gates (754 instances!)**: Tiny kernels (1.4 us each) with massive launch count. Each bidirectional LSTM timestep is a separate kernel. Fusing all timesteps or using CUDA graphs for the LSTM section could eliminate launch overhead.
- **cuBLAS GEMV (765 instances)**: Similarly tiny. These are the LSTM matrix-vector products.
- **Launch overhead**: Total kernel time ~20ms, total wall time 28ms → ~8ms in launch overhead + CPU work. This is ~29% overhead, confirming that kernel fusion / CUDA graphs would help significantly.
- **Encode-phase graph** is feasible: ~100 kernels before the L sync point. Cache per-T like G2P.

### Item 4: SineGen save/restore

**Problem**: SineGen allocated `d_phase_low` (L2*9 floats), `d_har_source` (T_audio floats), and `d_rand_ini` (9 floats) from `decode_arena` but never reclaimed them. Wasted ~0.5-2.5 MB depending on text length.

**Fix**: Wrap in `decode_arena.save()` / `decode_arena.restore()` around the SineGen temporaries (keeping `d_gen_har` outside since it must persist). Updated `compute_decode_bytes()` to take `max(sinegen_temps, generator_pool)` instead of summing both, since they now overlap.

**Safety**: All ops are on the same CUDA stream — stream ordering guarantees SineGen kernels complete before any subsequent kernel writes to the reclaimed region. Same pattern used for F0/N chains at line 995.

### Summary

| Change | Effect |
|---|---|
| G2P workspace pre-alloc | Graphs never invalidated, 89.7 MB upfront |
| WMMA TF32 FFN | No improvement — reverted. Scalar GEMV already at 1.1ms |
| SineGen save/restore | ~0.5-2.5 MB reclaimed per inference |
| TTS profile | Identified im2col+SGEMM (31%) and LSTM launch overhead (13%) as top targets |

### Files changed

| File | Changes |
|---|---|
| `src/g2p.h` | Pre-allocate workspace at max_pos_ in `load_from_file_()`, replace grow-and-invalidate with bounds check in `infer()` |
| `src/tts.cpp` | SineGen save/restore around temporaries, updated `compute_decode_bytes()` to overlap SineGen temps with generator pool |
| `Makefile` | Added `-arch=native` to NVFLAGS |
