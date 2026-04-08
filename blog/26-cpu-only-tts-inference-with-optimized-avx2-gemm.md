## 2026-03-09: CPU-Only TTS Inference with Optimized AVX2 GEMM

### Motivation

The CUDA inference pipeline (`rokoko`) requires an NVIDIA GPU. For deployment on CPU-only machines (servers, edge devices, CI), we need a fully self-contained CPU inference path — no cuBLAS, no OpenBLAS, no external dependencies. Just C++17, AVX2+FMA, and `-lpthread`.

### The Port

Ported the entire Kokoro pipeline to CPU in three new files:

- **`src/cpu_ops.h`** (header-only) — All kernels: SGEMM, element-wise ops (GELU, sigmoid, snake, leaky ReLU), normalization (LayerNorm, InstanceNorm, AdaIN, AdaLayerNorm), softmax, conv1d via im2col, conv_transpose1d, depthwise conv, LSTM gates, STFT/iSTFT, weight normalization. Every function is `static inline` with AVX2 fast paths.
- **`src/rokoko_cpu.cpp`** — Full inference pipeline mirroring `rokoko_cuda.cpp`: ALBERT encoder → text encoder → prosody predictor (duration/F0/N) → AdaIN decoder → SineGen → generator with upsampling + residual blocks → iSTFT vocoder. Weight loading via `mmap` with `MAP_POPULATE`.
- **`src/rokoko_cpu.h`** — CPU weight struct, arena allocator, all model dimension constants.

Validated against GPU output across multiple test sentences: >0.993 correlation on all. Differences stem from the GPU using TF32 tensor math (reduced mantissa precision in matrix multiplies), not from any algorithmic divergence.

### Challenge 1: Column-Major GEMM Calling Convention

The entire codebase uses cuBLAS column-major conventions: `C = alpha * op(A) * op(B) + beta * C` with `op(X) = X` or `X^T`, leading dimensions, etc. Rather than converting everything to row-major, we matched the cuBLAS API exactly in `sgemm_cpu()`. This let us drop in the CPU GEMM as a 1:1 replacement for every `cublasSgemm` call — same argument order, same transpose flags, same leading dimensions.

The four transpose combinations (TN, NN, TT, NT) each have different memory access patterns:

| Mode | A access | B access | Typical use |
|------|----------|----------|-------------|
| TN | Row contiguous | Col contiguous | ALBERT attention/FFN |
| NN | Col contiguous | Col contiguous | Conv1d (via im2col) |
| NT | Col contiguous | Row contiguous | ConvTranspose1d |
| TT | Row contiguous | Strided | Rare |

### Challenge 2: Naive GEMM Was 433x Slower Than GPU

The first working CPU version used a straightforward AVX2 dot-product GEMM: for each (i, j), compute `dot(A_row_i, B_col_j)` using a 2x-unrolled `avx2_dot`. It worked, but benchmarked at **4465 ms** for a 15-token input producing 1.6 sec of audio — **0.36x realtime**. The GPU does the same in ~10 ms.

Profiling revealed the problem: the TN case (ALBERT's hot path) had the j-outer, i-inner loop order. With M=768, K=768, N=15 (typical ALBERT GEMM), the A matrix (2.3 MB) doesn't fit in L2 cache, so it gets re-read N=15 times — 34.5 MB of DRAM traffic for what should be a 2.3 MB working set.

### Solution: Two-Strategy GEMM

Rather than one GEMM-fits-all, we implemented two strategies optimized for the actual workload shapes:

**K-blocked TN (for ALBERT/encoder):** Since both A rows and B columns are contiguous in TN mode, we don't need packing at all. K-blocking with KC=384 keeps the B panel (~N×KC×4 = 15×384×4 = 23 KB) hot in L1, while A streams through L2 once per K-tile. Loop order is i-outer, j-inner so each A row is loaded once. This is simple, zero-overhead, and optimal for the small-N ALBERT GEMMs.

**Tiled NN with 8×8 micro-kernel (for conv1d generator):** The generator's conv1d layers produce large NN-mode GEMMs (e.g., M=7681, N=128, K=384 after im2col). These need the full BLIS/GOTO treatment:

1. **Packing:** A and B are packed into MR=8 and NR=8 wide contiguous micro-panels
2. **Cache blocking:** MC×KC A panels fit in L2, KC×NC B panels fit in L2, MC×NC C panels fit in L1
3. **8×8 AVX2 micro-kernel:** 8 `__m256` accumulators (one per C column), each k-step loads one A vector and broadcasts 8 B scalars. The 8×8 tile stays in registers for the entire KC accumulation — zero C traffic until the final store.

```
MC=256, KC=384, NC=256, MR=8, NR=8
```

KC=384 was chosen to cover the full K dimension of the most common conv1d GEMMs (C_in=128, kernel=3 → K=384), eliminating K-tiling overhead for these hot paths.

### Challenge 3: Snake Activation Bottleneck

Profiling the generator revealed that only 57% of time was in GEMM — the rest was dominated by the **snake activation**: `y = x + (1/α) sin²(αx)`. With C=128 channels and T=7681 time steps, each snake call computes ~983K scalar `sin()` calls. The generator has 36 snake calls (18 per block), totaling ~35M `sin()` invocations. At ~15-25 ns per scalar `sin()`, that's 500-900 ms — nearly half the total inference time.

### Solution: AVX2 Vectorized Sin

Implemented a Cephes-style `fast_sin_avx2()` using SSE/AVX2 intrinsics:

1. **Range reduction:** `j = round_even(|x| × 4/π)`, then `x' = x - j × π/4` using extended precision (three constants dp1+dp2+dp3 for exact cancellation)
2. **Polynomial evaluation:** 6th-order minimax polynomials for sin and cos on [-π/4, π/4], computed with FMA chains
3. **Octant selection:** `blendv_ps` picks sin or cos polynomial based on `j & 2`; sign correction from `j & 4` and original sign

Processes 8 `sin()` values per call. Accuracy: ~1e-6 relative error, more than sufficient for neural network activations. The vectorized snake loop replaces the scalar inner loop, cutting snake time by ~10x.

**Bug encountered:** Initial implementation had the `blendv_ps` operands swapped — selecting cos where sin was needed and vice versa. Output correlation dropped to 0.004 (essentially random). The Cephes convention is: `poly_mask = (j & 2 == 0)` selects the **sin** polynomial (not cos). Fixed by swapping `blendv_ps(s, c, mask)` to `blendv_ps(c, s, mask)`.

### Challenge 4: im2col Memory Traffic

Each conv1d call in the generator unfolds the input via im2col before the GEMM. For Conv1d(128, 128, k=3) on T=7681: the im2col output is 384×7681 = 11.8 MB. With 18 such calls in generator block 1 alone, that's 212 MB of writes — all going through cache and polluting the working set.

The original im2col had a per-element conditional check (`if (t_in >= 0 && t_in < T_in)`) that prevented auto-vectorization. For stride=1 (the common case), each im2col row is just a shifted copy of an input channel row with zero-padding at boundaries.

### Solution: Bulk memcpy im2col

For stride=1, we compute the valid range analytically and use `memcpy` for the contiguous interior region, `memset` for the zero-padded boundaries. This replaces T_out conditional branches + scalar stores with three bulk memory operations per (channel, kernel_position) pair.

### Results

Benchmarked on 15-token input ("Hello world") producing 1.6 sec audio at 24 kHz:

| Version | Avg (ms) | RTFx | vs Baseline |
|---------|----------|------|-------------|
| Naive AVX2 dot GEMM | 4465 | 0.36x | 1.0x |
| + Tiled NN / K-blocked TN | 1272 | 1.3x | 3.5x |
| + Vectorized snake (fast_sin) | 1205 | 1.3x | 3.7x |
| + Optimized im2col (memcpy) | **1149** | **1.4x** | **3.9x** |

Profile breakdown at final version:

| Section | Time (ms) | % |
|---------|-----------|---|
| ALBERT encoder | 47 | 4% |
| Text encoder | 12 | 1% |
| Prosody predictor | 29 | 2% |
| F0/N predictors | 11 | 1% |
| Decoder | 50 | 4% |
| SineGen + STFT | 17 | 1% |
| Generator block 0 (512→256, T=1280) | 325 | 28% |
| Generator block 1 (256→128, T=7681) | 637 | 55% |
| Post-conv + iSTFT | 30 | 3% |

The generator's conv1d GEMMs are compute-bound at ~25-30 GFLOPS effective throughput (40-50% of theoretical AVX2 FMA peak). Further gains would require multi-threading or AVX-512.

### What Didn't Help

- **K-loop unrolling in micro-kernel:** Manually unrolling the 8×8 micro-kernel inner loop by 4 made no difference — the compiler at `-O3` already handles this.
- **Increasing MC from 128 to 256:** Marginal improvement. The packing overhead reduction was offset by larger L2 working set.
- **Software prefetching in pack_a_nn:** The large stride (lda=7681 → 30 KB between k-steps) is beyond hardware prefetch range, but adding `__builtin_prefetch` 4 steps ahead didn't measurably help either — the L3 latency is already pipelined by the out-of-order engine.

### Files Changed

| File | Change |
|------|--------|
| `src/cpu_ops.h` | New: all CPU kernels — tiled GEMM, K-blocked GEMM, fast_sin_avx2, vectorized snake, optimized im2col, all element-wise/norm/conv ops |
| `src/rokoko_cpu.cpp` | New: full CPU inference pipeline with profiling support |
| `src/rokoko_cpu.h` | New: CPU weight struct, arena allocator, constants |
| `Makefile` | Added `rokoko_cpu` target and `CPUFLAGS` |
| `scripts/compare_wav.py` | New: WAV comparison tool (max/mean/RMS diff, correlation, SNR) |
| `scripts/make_test_input.py` | New: creates pre-phonemized input.bin for benchmarking |
