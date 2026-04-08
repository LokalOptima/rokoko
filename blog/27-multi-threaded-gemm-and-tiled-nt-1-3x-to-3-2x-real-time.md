## 2026-03-09: Multi-Threaded GEMM and Tiled NT — 1.3x → 3.2x Real-Time

After the initial CPU port with optimized AVX2 GEMM reached **1.4x real-time** (single-threaded), the natural next step was multi-threading the hot GEMM paths and filling the remaining gap in transpose-case coverage.

### The landscape

Research into other CPU Kokoro implementations revealed that nobody has written a dedicated hand-tuned C++ inference engine. Everyone wraps ONNX Runtime or GGML. Best published CPU result: **5x RT** on a 32-vCPU AMD EPYC (PyTorch/ONNX, multi-threaded). Our single-threaded 1.4x RT was already competitive with multi-threaded ONNX on similar hardware. Key finding: **int8 quantization hurts** Kokoro on CPU (slower than fp32 across multiple sources) — the decoder ops don't map well to quantized SIMD.

### Step 1: OpenMP threading on tiled GEMM

The generator's conv1d GEMMs have M=7681, N=128, K=384 — the i0 tile loop in `sgemm_tiled_nn` has `ceil(7681/256) = 30` tiles, plenty of work for multiple threads.

**Key design decisions:**

1. **Shared packed_B, thread-local packed_A.** The B panel is packed once per (k0, j0) block into a static shared buffer, then read by all threads. Each thread packs its own A tile into `tls_packed_A` (already thread-local). This avoids redundant B packing across threads.

2. **Size gate to avoid OpenMP overhead on small GEMMs.** The `#pragma omp parallel for` uses `if(n_i_tiles >= 4)` for NN and `if(M >= 32)` for TN. ALBERT's tiny M=15 GEMMs skip threading entirely — the barrier cost would exceed the compute time.

3. **Dynamic scheduling.** `schedule(dynamic)` handles uneven tile sizes at the M boundary (last tile is partial).

**Challenge: choosing the optimal thread count.** Benchmarked 1-16 threads:

| Threads | Avg (ms) | RTFx | Gen block 0 | Gen block 1 |
|---------|----------|------|-------------|-------------|
| 1       | 1200     | 1.3x | 342ms       | 729ms       |
| 4       | 686      | 2.3x | 191ms       | 315ms       |
| **8**   | **595**  | **2.7x** | **160ms** | **284ms**   |
| 10      | 609      | 2.6x | 163ms       | 322ms       |
| 16      | 608      | 2.6x | 178ms       | 262ms       |

8 threads is the sweet spot — gen block 1 scales 2.6x (729 → 284ms) but serial sections (snake activations, adain normalization, residual adds) limit further scaling. Beyond 8 threads, synchronization overhead increases without proportional compute gains.

### Step 2: Tiled NT GEMM for conv_transpose1d

The `conv_transpose1d` operation uses `sgemm('N', 'T', ...)` which was falling through to a naive rank-1 update path — iterating over K, broadcasting B values, doing column-wise AXPY updates on C. For block 1's conv_transpose (M=1280, N=1536, K=256), the C matrix is 7.5MB — every rank-1 update thrashes it through cache.

**Solution:** After packing, NT becomes identical to NN at the micro-kernel level. Both A and B panels have K as the outermost stride. I only needed a new `pack_b_t` function that gathers from B's transposed layout (elements at `B[j + k*ldb]`, NR-contiguous per k-step — ideal for AVX2 loads) and reused the existing 8×8 micro-kernel and OpenMP-parallelized tile dispatch.

**The result surprised me** — the impact was larger than the 3% of compute that conv_transpose1d represents:

| Component | Before (8T) | After (8T) | Change |
|-----------|-------------|------------|--------|
| Gen block 0 | 160ms | 114ms | -29% |
| Gen block 1 | 284ms | 213ms | -25% |
| **Total** | **595ms** | **493ms** | **-17%** |

Block 0's conv_transpose (M=128, N=5120, K=512) was particularly affected — the naive NT path had zero cache blocking and zero threading. The tiled version with multi-threading turned a serial cache-thrashing operation into a parallel cache-friendly one.

### What didn't help (analysis)

**Wider micro-kernel (8×12).** Analysis showed that FLOPs/cycle is the same as 8×8 — both are FMA-bound at 32 FLOPs/cycle on the AVX2+FMA pipeline. The only benefit is ~31% fewer tile dispatches, saving perhaps 800 cycles per GEMM. With each GEMM taking ~40K+ cycles, the savings are <2%. Additionally, NR=12 creates edge tiles (128/12 = 10r8) that need a separate edge kernel, adding complexity.

### Numerical validation

All optimizations preserve the safety gate: correlation ≥ 0.995 vs GPU reference, max absolute diff < 0.06. The multi-threaded output is bit-identical to single-threaded (same FMA operations, different thread assignment, with deterministic tile boundaries).

### Results summary

| Version | Time (ms) | RTFx | Speedup |
|---------|-----------|------|---------|
| Initial CPU port (1 thread) | 1200 | 1.3x | baseline |
| + OpenMP threading (8T) | 595 | 2.7x | 2.0x |
| + Tiled NT GEMM (8T) | 493 | 3.2x | 2.4x |

**3.2x real-time on CPU** — competitive with PyTorch on similar hardware, with no external dependencies (no BLAS, no ONNX Runtime, no Python).

### Profile breakdown (final, 8 threads)

| Section | Time | % |
|---------|------|---|
| ALBERT encoder | 17ms | 3% |
| Text encoder | 13ms | 3% |
| Prosody predictor | 28ms | 6% |
| F0/N predictors | 12ms | 2% |
| Decoder | 57ms | 12% |
| SineGen + STFT | 18ms | 4% |
| **Gen block 0** (512→256, T=1280) | **114ms** | **23%** |
| **Gen block 1** (256→128, T=7681) | **213ms** | **43%** |
| Post-conv + iSTFT | 23ms | 5% |

Generator remains 66% of total time. The 18 sequential conv1d calls per generator stage (3 resblocks × 3 dilated branches × 2 convs) are individually multi-threaded but the serial chain limits total scaling.

### Step 2b: Dual dispatch — j0-parallel for small-M, large-N GEMMs

After profiling, I noticed the **decoder was running single-threaded** on large GEMMs. The decoder's conv1d operations have M=128 (short sequences) but N=1024 (many channels), K=1542-3270. With M < MC=256, there's only 1 i0 tile, and the `if(n_i_tiles >= 4)` gate disabled all threading.

**Solution:** Added a second parallel path that distributes work over j0 tiles instead. When n_i_tiles < 4 but n_j_tiles ≥ 4, pack A once into a shared buffer and let each thread pack and compute its own j0 tile:

- **i0-parallel** (path 1): shared packed_B, thread-local packed_A — for generator (M=7681)
- **j0-parallel** (path 2): shared packed_A, thread-local packed_B — for decoder (M=128, N=1024)
- **Sequential** (path 3): small GEMMs where threading overhead would dominate

Result: decoder dropped from 57ms → 26ms (2.2x speedup). Total: **493ms → 442ms, 3.6x real-time**.

### Step 3: Precomputed twiddle factors + vectorized SineGen

The iSTFT was spending 19ms computing 3.4M scalar sin/cos calls for DFT twiddle factors. With n_fft=20, there are only 20×20=400 unique twiddle values. Precomputing them into lookup tables eliminated all trig from the inner loop: **19ms → 1.5ms** (12.7x faster). Same optimization applied to the forward STFT.

The SineGen also had 345K scalar sinf calls for harmonic generation. Replaced with `fast_sin_avx2` (our Cephes-style vectorized sin from the snake activation work): **SineGen+STFT dropped from 18ms → 11ms**.

### Updated results

| Version | Time (ms) | RTFx | Speedup |
|---------|-----------|------|---------|
| Initial CPU port (1 thread) | 1200 | 1.3x | baseline |
| + OpenMP threading (8T) | 595 | 2.7x | 2.0x |
| + Tiled NT GEMM (8T) | 493 | 3.2x | 2.4x |
| + j0-parallel dispatch (8T) | 442 | 3.6x | 2.7x |
| + Precomputed twiddles + vec SineGen (8T) | 410 | 3.9x | 2.9x |

### Step 4: Fused im2col+GEMM for conv1d (implicit GEMM)

The biggest remaining inefficiency was the separate im2col step before every conv1d GEMM. For block 1's 18 conv1d calls (C_in=128, K=3, T=7681), each im2col materializes a `[384, 7681]` matrix = 11.3MB. That's 11.3MB written and 11.3MB read back during A-packing — 22.6MB of redundant data movement per call, **407MB total** for block 1 alone.

**The insight:** For stride=1 convolutions, consecutive output time steps map to consecutive input positions within each channel. The im2col matrix is just the input tensor viewed through (channel, kernel_position) offsets. Instead of materializing this matrix, we can pack A directly from the input tensor during the GEMM's packing phase.

**Implementation — `pack_a_conv1d`:** Replaces `pack_a_nn` in the tiled GEMM loop. For each micro-panel of 8 time steps:
- Computes the (channel, kernel_position) pair using increment-and-wrap instead of integer division
- **Hot path:** When all 8 time steps fall within valid input range, does a single `_mm256_loadu_ps` directly from `x[c * T_in + t_in]`
- **Boundary path:** For the first/last few time steps near padding boundaries, element-wise load with zero-fill

The hot path handles >95% of elements (only `dilation * (K-1)` time positions per channel hit the boundary path, vs `T_out` total).

**`sgemm_fused_conv1d_nn`:** Same triple-dispatch structure as `sgemm_tiled_nn` (i0-parallel, j0-parallel, sequential) but calls `pack_a_conv1d` instead of `pack_a_nn`. B packing unchanged.

**Result:**

| Component | Before (8T) | After (8T) | Change |
|-----------|-------------|------------|--------|
| Gen block 0 (T=1280) | ~107ms | 84ms | -21% |
| Gen block 1 (T=7681) | ~170ms | 124ms | -27% |
| **Total** | **410ms** | **338ms** | **-17.5%** |

Block 1 benefited more because its larger T creates proportionally more im2col waste. Safety check: correlation 0.9964 vs GPU reference, well within the ≥0.995 gate.

### Updated results

| Version | Time (ms) | RTFx | Speedup |
|---------|-----------|------|---------|
| Initial CPU port (1 thread) | 1200 | 1.3x | baseline |
| + OpenMP threading (8T) | 595 | 2.7x | 2.0x |
| + Tiled NT GEMM (8T) | 493 | 3.2x | 2.4x |
| + j0-parallel dispatch (8T) | 442 | 3.6x | 2.7x |
| + Precomputed twiddles + vec SineGen (8T) | 410 | 3.9x | 2.9x |
| + Fused im2col+GEMM (8T) | 338 | 4.7x | 3.5x |

### Profile breakdown (current, 8 threads)

| Section | Time | % |
|---------|------|---|
| ALBERT encoder | 16ms | 5% |
| Text encoder | 15ms | 4% |
| Prosody predictor | 27ms | 8% |
| F0/N predictors | 12ms | 4% |
| Decoder | 25ms | 7% |
| SineGen + STFT | 11ms | 3% |
| **Gen block 0** (512→256, T=1280) | **84ms** | **25%** |
| **Gen block 1** (256→128, T=7681) | **124ms** | **37%** |
| Post-conv + iSTFT | 4ms | 1% |

Generator is still 62% of total. Practical ceiling analysis puts the target at ~220ms (7x RT). We're at 338ms — still 1.5x away from practical ceiling.

### Files changed

| File | Changes |
|------|---------|
| `src/cpu_ops.h` | Dual dispatch (i0/j0 parallel) in `sgemm_tiled_nn` and `sgemm_tiled_nt`, `gemm_tile_dispatch` helper, shared_packed_A + tls_packed_B buffers, parallel i loop in `sgemm_kblocked_tn`, `pack_b_t` for NT case, `pack_a_conv1d` + `sgemm_fused_conv1d_nn` for implicit GEMM, fused path in `conv1d_cpu` |
| `Makefile` | Added `-fopenmp` to CPUFLAGS |
| `plan.md` | New: optimization plan with steps, safety gates, expected speedups |
