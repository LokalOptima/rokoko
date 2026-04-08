## 2026-03-15: Optimization Round 6 — Cutlass Operator Caching

### Context

After Round 5, the Cutlass implicit GEMM was functionally correct but slower than cuBLAS im2col+SGEMM (17.2ms vs 9.5ms on short text, 95.7ms vs 92.3ms on long text). The short-vs-long gap pointed to per-call overhead rather than kernel throughput.

### The Problem: `initialize()` on Every Call

Cutlass `ImplicitGemmConvolution::initialize()` does substantial host-side work: it computes tiling parameters, grid dimensions, iterator configurations, and packs everything into a `Params` struct. We were calling this on **every single convolution** — dozens of times per inference.

But Cutlass also has an `update()` method that does exactly one thing: swap the device pointers (input, weights, bias, output) without recomputing any of the tiling/grid state. This is ~100x cheaper than `initialize()`.

### The Fix: Cache by Problem Shape

The key insight: most convolutions in the model share the same `(C_in, C_out, K, stride, padding, dilation, T_in)` shape. For example, the 5 text encoder layers all run (512, 512, K=5) with the same T. The generator resblocks run dozens of (C, C, K=3/7/11) convolutions at the same T.

We cache an initialized operator per unique problem shape in an `unordered_map`. First call for a shape: full `initialize()`. All subsequent calls: `update()` (pointer swap) + `operator()` (kernel launch).

```cpp
struct ConvKey {
    int C_in, C_out, T_in, K, stride, padding, dilation;
    bool operator==(const ConvKey& o) const { /* field-by-field */ }
};

static std::unordered_map<ConvKey, ImplicitGemm, ConvKeyHash> s_op_cache;

// In cutlass_conv1d_fprop:
auto it = s_op_cache.find(key);
if (it != s_op_cache.end()) {
    it->second.update(arguments, workspace);  // just swap pointers
    it->second(stream);                        // launch kernel
} else {
    conv_op.initialize(arguments, workspace, stream);  // full init
    conv_op(stream);
    s_op_cache[key] = conv_op;  // cache for next time
}
```

### Results

Operator caching brought Cutlass to **exact parity** with cuBLAS:

```
=== Cutlass + Operator Caching (bench.sh, 30 timed runs, exclusive GPU lock) ===

--- short (1.60s audio, n=30) ---
           median       p95       min       max
  TTS:       9.61     12.21      9.45     12.25  ms
  RTFx:       148x       116x  (median / p95)

--- medium (5.72s audio, n=30) ---
           median       p95       min       max
  TTS:      29.74     31.77     29.52     32.67  ms
  RTFx:       181x       169x  (median / p95)

--- long (18.82s audio, n=30) ---
           median       p95       min       max
  TTS:      92.36     93.17     92.22     96.64  ms
  RTFx:       194x       192x  (median / p95)

STT: short PASS, medium PASS, long PASS
```

Comparison (long text TTS median):

| Version | TTS (ms) | RTFx | vs cuBLAS |
|---------|----------|------|-----------|
| cuBLAS im2col+SGEMM (main branch) | 92.3 | 194x | 1.00x |
| Cutlass implicit GEMM (no caching) | 95.7 | 188x | 0.97x |
| **Cutlass + operator caching** | **92.4** | **194x** | **1.00x** |

Short text improved the most: 17.2ms → 9.6ms (1.79x), confirming that `initialize()` overhead dominated small-problem performance.

### What This Means

Cutlass implicit GEMM now matches cuBLAS while:
- **Eliminating im2col entirely** — no separate im2col kernel or workspace needed
- **Fusing bias** into the GEMM epilogue — one fewer kernel launch per conv
- **Setting up for TensorOp** — switching from `OpClassSimt` to `OpClassTensorOp` (TF32) could push throughput beyond cuBLAS

The SIMT path uses scalar loads (`AlignedArray<float,1>`), matching cuBLAS SGEMM throughput. TF32 TensorOp would use 128-bit vectorized loads + tensor core math — the potential upside.

### Lessons Learned

1. **`initialize()` is expensive, `update()` is free.** Cutlass operators are designed to be initialized once per problem shape, then reused with pointer swaps. Calling `initialize()` per-inference is an anti-pattern.

2. **Profile the right thing.** The gap between short (1.8x slower) and long (1.04x slower) text immediately pointed to fixed per-call overhead, not kernel throughput. The fix was obvious once we looked at the pattern.

### Files Changed

| File | Changes |
|---|---|
| `src/cutlass_conv.cu` | Added `ConvKey`/`ConvKeyHash`, `s_op_cache` map, cache-hit path with `update()`, cache-miss path with `initialize()` + cache store |
