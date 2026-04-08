## 2026-03-18: Dropping cuBLAS — Cutlass + Custom Kernels Only

### Motivation

With Cutlass implicit GEMM for convolutions and Cutlass batched GEMM for G2P attention already in place, cuBLAS was only used for:
- `cublasSgemm` for TTS linear layers (ALBERT attention, text encoder projections, predictor)
- `cublasSgemv` for small matrix-vector products (style FC layers, [D, 128] × [128])
- `cublasSgemmBatched` for G2P attention (already migrated in the GEMM work)

The goal: eliminate the last `libcublas`/`libcublasLt` dependency entirely.

### Cutlass GEMM: The Full Tiered Architecture

Created `cutlass_gemm.cu` with three layout combinations (TN, NT, NN) × four fallback tiers:

1. **TF32 Large** (128×128×16): high per-CTA throughput for large problems
2. **TF32 Small** (64×64×16): 4× more CTAs for small problems where Large under-fills the GPU
3. **TF32 Align1**: handles M not divisible by 4 (SIMT epilogue with align-1 stores, but TF32 MMA compute)
4. **SIMT**: no alignment requirements at all, for K not divisible by 4

Plus batched variants (TN, NN) for G2P multi-head attention, and `cutlass_gemm_tn_bias` with stride-0 C source for fused bias broadcast.

The tier selection is automatic — each `cutlass_gemm_*` function tries tiers in order and falls back on `can_implement()` failure. All use the same operator caching pattern from Round 6.

### Custom GEMV Kernel

For N=1 cases (style FC: [D, 128] × [128] → [D]), cuBLAS `Sgemv` was replaced with a custom `gemv_tn_f32` kernel. One warp per output row, 8 rows per block (256 threads total), warp-shuffle reduction over K. This handles the ~100 small matrix-vector products in the predictor.

### The Alignment Bug

Initial Cutlass GEMM integration produced NaN outputs and verification failures. Root cause: G2P workspace pointers were only 4-byte aligned, but Cutlass TensorOp requires 256-byte alignment for vectorized 128-bit loads. Fixed by adding `align256()` to all workspace pointer assignments in `g2p.h`.

### Bias Fusion

Added `cutlass_gemm_tn_bias()` using stride-0 C source layout — the same technique as `cutlass_conv.cu` residual fusion. The bias vector [M] is broadcast across all N columns via `LayoutCM(0)` (zero stride = same column repeated). This eliminates ~100 separate `channel_bias_add_f32` kernel launches per TTS inference.

### G2P Graph Capture Fix

Cutlass `initialize()` allocates internal state and can't run inside CUDA graph capture. Solution: first call for each input length T runs kernels directly (populates operator caches), graph capture deferred to the second call when all operators hit cache. Third+ calls replay the graph.

### Results

Performance matches the cuBLAS baseline within measurement noise — the goal was dependency elimination, not speedup:

| Metric | cuBLAS | Cutlass |
|--------|--------|---------|
| Short (1.6s audio) | 8.13ms / 173x | ~8ms / ~175x |
| Long (18.8s audio) | 60.79ms / 289x | ~60ms / ~290x |
| Binary dependencies | libcublas + libcublasLt | none (Cutlass is header-only) |

### What We Removed

- `libcublas`, `libcublasLt` from linker flags
- All `cublas*.h` includes
- `cublasHandle_t` creation/destruction in `main.cu`
- 2 shared libraries (~120 MB on disk) no longer loaded at runtime

### Files Changed

| File | Changes |
|---|---|
| `src/cutlass_gemm.cu` | New — TN/NT/NN × 4 tiers + batched + bias fusion |
| `src/kernels.cu` | Added `gemv_tn_f32` kernel |
| `src/kernels.h` | Added GEMV declaration |
| `src/tts.cpp` | Replaced cuBLAS wrappers with Cutlass GEMM calls |
| `src/g2p.h` | 256-byte workspace alignment, deferred graph capture |
| `Makefile` | Removed `-lcublas -lcublasLt` |
