## Round 5: [C,T] → [T,C] Layout Migration + Cutlass + Coalesced Instance Norm

### Motivation

cuBLAS was selecting slow `align1` Cutlass kernels because `T_out` (variable, often unaligned) was the SGEMM leading dimension in the [C,T] layout. A padding approach was tried and reverted — overhead negated savings. The solution: switch to [T,C] (channels-last) layout where SGEMM leading dimensions become model constants (C_out, CK) — always aligned. This also enables Cutlass implicit GEMM which eliminates im2col.

### Phase 1: [C,T] → [T,C] Layout Migration

Changed all 16 layout-dependent CUDA kernels from `c*T+t` → `t*C+c` indexing. Added `concat_channels_f32`, `concat3_channels_f32`, and `concat4_channels_f32` kernels for [T,C] channel concatenation (replacing zero-cost memcpy concat that worked in [C,T]).

SGEMM reformulation:
- `gemm_conv1d`: `cublasSgemm(OP_T, OP_N, C_out, T_out, CK, w, CK, col, CK, y, C_out)` — lda=CK, ldb=CK, ldc=C_out, all model constants
- `gemm_conv_transpose1d`: similar, all leading dims are model constants
- Alignment SGEMMs: `alignment^T × encoder` for [T,C] output

Eliminated ~18 transpose kernels from encode/decode paths (data is now natively [T,C] = [T,H] which is what LSTM needs). Added 2 transposes at STFT boundary (STFT inherently operates in [freq_bins, frames] = [C,T]).

Added CUDA graph cache for the encode phase — all arena allocations moved upfront for deterministic graph replay.

**Challenge:** The concat operations that were zero-cost memcpy in [C,T] (channels are contiguous blocks) became interleaved copies in [T,C]. Initial implementation used chained `concat_channels_f32` calls (3 kernels for 4-way concat). Replaced with single `concat4_channels_f32` kernel.

**Challenge:** CUDA graph capture requires deterministic memory addresses. The initial instance_norm optimization used a `static` persistent reduction buffer — this worked outside graphs but broke during graph replay because the `cudaMemsetAsync` only happens during capture, not replay. Fixed by passing workspace as a parameter from the arena.

### Phase 2: Cutlass Implicit GEMM Integration

Replaced `im2col + cublasSgemm` with Cutlass FP32 SIMT Conv2d implicit GEMM (Conv2d with H=1 for 1D). Uses `OpClassSimt` on `Sm80` (forward-compatible with SM120/Blackwell consumer). Fuses bias into `LinearCombination` epilogue.

Required NHWC weight layout [C_out, K, C_in] vs original [C_out, C_in, K]. Added `cutlass_reshape_weights` kernel and `s_w_nhwc` map to store reshaped copies alongside originals.

**Challenge:** Cutlass SIMT implicit GEMM was **not faster** than im2col + cuBLAS. nsys profiling showed:
- Cutlass implicit GEMM (75 calls): 48.3ms
- im2col (75 calls) + cuBLAS SGEMM: 19.7ms + 29ms = 48.8ms

The implicit GEMM doesn't eliminate work — it moves im2col into the GEMM kernel itself. For our small 1D convolutions, the overhead is comparable. The Cutlass path is kept for future TensorOp optimization but currently doesn't provide a speedup.

### Phase 3: Coalesced Instance Norm (the big win)

nsys profiling revealed `instance_norm_style_affine` was 35.2% of total GPU time (46ms, 70 calls). ncu deep dive showed:
- DRAM throughput: **3.35%** (catastrophic)
- SM throughput: **1.31%**
- Mem Busy: 64.63%
- Diagnosis: **latency-bound** — strided access pattern with [T,C] layout

The old kernel used 1 warp (32 threads) per channel, reading `x[t*C+c]` with stride C. For C=512, each read is 2048 bytes apart — zero coalescing.

**Solution:** Two-pass coalesced kernel inspired by NVIDIA Apex's group norm NHWC implementation:
- Pass 1 (`instnorm_sum_kernel`): Grid(C_tiles, T_tiles). Each thread handles one channel, iterates over T tile with coalesced reads along C. Accumulates sum + sum_sq, writes via atomicAdd to [2,C] reduction buffer.
- Pass 2 (`instnorm_style_norm_kernel`): Same grid. Reads reduction buffer for mean/var, normalizes with coalesced reads and writes, fuses style affine transform.

**Result:** instance_norm dropped from **46.0ms → 6.9ms** (6.7x speedup). Total GPU time: **130.7ms → 91.6ms** (1.43x).

### Benchmark Results

```
=== After all optimizations (bench.sh, 10 warmup, 30 timed) ===

--- short (1.60s audio, n=30) ---
           median       p95       min       max
  TTS:      17.15     19.51     16.99     21.52  ms
  RTFx:        88x        77x  (median / p95)

--- medium (5.72s audio, n=30) ---
  TTS:      38.56     39.27     38.43     39.75  ms
  RTFx:       141x       138x  (median / p95)

--- long (18.82s audio, n=30) ---
  TTS:      95.67     96.15     95.40     96.32  ms
  RTFx:       188x       187x  (median / p95)

STT: short PASS, medium PASS, long PASS
```

Comparison (long text TTS median):

| Version | TTS (ms) | RTFx | vs Baseline |
|---------|----------|------|-------------|
| Pre-layout-switch baseline | 120.9 | 150x | 1.0x |
| [T,C] + graphs + concat4 (no Cutlass) | 131.1 | 138x | 0.92x |
| + Cutlass implicit GEMM | 134.7 | 135x | 0.90x |
| + Coalesced instance norm | 95.7 | 188x | **1.26x** |

### nsys Kernel Breakdown (long text, after all optimizations)

```
Kernel                                          Total(us)      %
CUTLASS GEMM (F32 out) [implicit conv]            48313    52.8%
LSTM SGEMV (cuBLAS)                                9546    10.4%
LSTM gates                                         6211     6.8%
cuBLAS SGEMM                                       5195     5.7%
instnorm_style_norm_kernel                         3754     4.1%
snake_kernel                                       3214     3.5%
instnorm_sum_kernel                                3150     3.4%
add_kernel                                         3086     3.4%
TOTAL                                             91570
```

### Lessons Learned

1. **Profile before optimizing.** Intuition said Cutlass implicit GEMM would be a big win (eliminating im2col). Profiling showed it was a wash — the same work moved inside the kernel.

2. **Layout changes cascade.** Switching from [C,T] to [T,C] touched 16 kernels, 2 SGEMM formulations, ~20 transpose elimination sites, concatenation operations, and arena sizing. But the payoff (aligned SGEMM + enabled coalesced norm) justified the complexity.

3. **Memory coalescing dominates.** The instance norm kernel went from 3.35% to ~80% DRAM throughput just by changing which dimension threads iterate over. 6.7x speedup from a pure access pattern change, no algorithmic change.

4. **Static buffers break CUDA graphs.** Any `cudaMalloc`/`cudaMemset` in a statically-allocated buffer will only execute during graph capture, not replay. Pass workspace from the arena instead.

### Files Changed

| File | Changes |
|---|---|
| `src/kernels.cu` | 16 kernel [C,T]→[T,C] index changes, new concat/concat3/concat4 kernels, two-pass coalesced instance_norm + instance_norm_style_affine |
| `src/kernels.h` | Updated all doc comments to [T,C], added concat3/concat4/workspace params |
| `src/tts.cpp` | SGEMM reformulation, ~18 transpose eliminations, STFT boundary, concat changes, CUDA graph cache, arena sizing, norm workspace plumbing |
| `src/cutlass_conv.cu` | New: Cutlass FP32 SIMT Conv2d implicit GEMM + weight reshape kernel |
| `Makefile` | Added cutlass_conv.o build rule, CUTLASS include path |
| `third_party/.gitignore` | Ignore Cutlass headers (downloaded at build time) |
