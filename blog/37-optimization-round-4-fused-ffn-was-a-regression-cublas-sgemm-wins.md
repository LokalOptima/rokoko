## Optimization Round 4: Fused FFN was a regression — cuBLAS SGEMM wins (2026-03-15)

### Discovery: the fused kernel was slower, not faster

Built a proper benchmark harness (`bench.sh`) to measure G2P and TTS timing with 10 warmup runs and 30 timed runs, reporting median/p95/min/max. First benchmark revealed the fused FFN scalar kernel (introduced in Round 2) was actually a **regression** compared to the original cuBLAS SGEMM calls.

The key insight: the fused kernel processes each of T columns independently as scalar GEMV, doing `O(d × ff)` scalar multiply-adds per thread block. cuBLAS processes all T columns simultaneously as a single SGEMM, using tensor cores. For T>1 (which is always true for real text), batched GEMM is fundamentally more efficient than T separate GEMVs.

**A/B benchmark results:**

| Variant | G2P short | G2P medium | Notes |
|---|---|---|---|
| Baseline (last commit, no CUDA graphs, cuBLAS FFN) | 0.73ms | 0.91ms | Separate gate/up SGEMM + SwiGLU + down SGEMM |
| Fused FFN kernel (scalar GEMV, with CUDA graphs) | 1.10ms | 1.36ms | 1.5x slower despite graphs! |
| cuBLAS SGEMM FFN (with CUDA graphs) | 0.34ms | 0.48ms | **2x faster than baseline** |

The fused kernel's per-column scalar GEMV was so slow it negated the entire benefit of CUDA graphs. Replacing it with cuBLAS SGEMM *and* keeping CUDA graphs gave a 2x improvement over the original.

### Implementation

Replaced the single `g2p_fused_ffn_kernel` call with 5 operations per layer:

1. `g2p_bias_rms_norm_kernel` — fused out_bias + RMSNorm → normed
2. `cublasSgemm` — gate+up GEMM: `ffn_out[2*ff, T] = gate_up_w[2*ff, d] × normed[d, T]`
3. `g2p_swiglu_bias_kernel` — fused bias + SwiGLU on interleaved layout
4. `cublasSgemm` — down GEMM with `beta=1` for fused residual: `X += down_w[d, ff] × ffn_out[ff, T]`
5. `g2p_bias_kernel` — down bias

The gate_up_w weight transpose from Round 2 was already correct for this: stored as `[d, 2*ff]` row-major = `[2*ff, d]` col-major, matching cuBLAS `CUBLAS_OP_N` with `lda=2*ff`. For the down GEMM, `ldb=2*ff` lets cuBLAS read only the first ff rows of the interleaved ffn_out buffer (SwiGLU output sits in the gate half).

Added `ffn_out[2*ff, T]` to workspace layout and pre-alloc formula. Workspace grew from 89.7 MB to 105.7 MB.

### Benchmark harness

Added `bench.sh` — starts server, warms up graph caches, runs N timed requests per text (short/medium/long), reports median/p95 timing and RTFx. Takes its own `flock --exclusive /tmp/gpu.lock` for the entire run.

Added STT verification: after benchmarking, saves one audio per text and runs it through paraketto STT. Compares normalized transcription against input text. All three lengths pass.

### Final benchmark results

```
=== Rokoko Benchmark ===
Warmup: 10 | Timed runs: 30

--- short (1.60s audio) ---
           median       p95       min       max
  G2P:       0.34      0.40      0.33      0.43  ms
  TTS:      10.58     15.11     10.41     15.71  ms
  Total:    11.61     17.17     11.38     17.82  ms
  RTFx:       138x        93x  (median / p95)

--- medium (5.72s audio) ---
  G2P:       0.48      0.54      0.46      0.55  ms
  TTS:      36.36     42.55     36.29     43.55  ms
  Total:    38.36     45.10     38.07     45.74  ms
  RTFx:       149x       127x  (median / p95)

--- long (18.82s audio) ---
  G2P:       0.70      0.73      0.64      0.77  ms
  TTS:     121.06    121.50    120.92    121.93  ms
  Total:   125.43    126.03    125.07    126.62  ms
  RTFx:       150x       149x  (median / p95)

=== STT Verification (paraketto) ===
  short: PASS
  medium: PASS
  long: PASS
```

### Lesson learned

Kernel fusion is not always a win. Fusing 5 kernel launches into 1 sounds great for reducing launch overhead, but if the fused kernel uses a fundamentally worse algorithm (per-column scalar GEMV vs batched GEMM with tensor cores), the compute regression dominates. The fused kernel's 1.1ms per-layer execution (memory-bandwidth-limited scalar dot products) dwarfed the ~0.1ms of launch overhead it saved.

**Rule of thumb**: if cuBLAS can process the full batch in a single SGEMM call, don't try to beat it with hand-written GEMV — even inside a fused kernel.

### Files changed

| File | Changes |
|---|---|
| `src/g2p.h` | Replace fused FFN kernel with cuBLAS SGEMM + small fused kernels, add ffn_out to workspace, update pre-alloc formula |
| `bench.sh` | New benchmark harness with warmup, median/p95 stats, STT verification via paraketto |
