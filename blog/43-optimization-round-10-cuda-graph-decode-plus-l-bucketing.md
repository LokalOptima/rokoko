## 2026-03-15: Optimization Round 10 — CUDA Graph Decode + L-Bucketing

### Decode Graph

Wrapped the entire decode phase in a CUDA graph — captured on first inference, replayed on subsequent calls with the same `(T, L_bucketed)` key. L is rounded up to the nearest multiple of 32 so different utterances that produce similar frame counts share a single cached graph.

Key changes:
- Decode graph cache keyed by `(T, L_padded)`, with arena-base invalidation when the decode arena grows
- SineGen `rand_ini` moved to a persistent device buffer (`seed=42`) to avoid per-call randomness that prevents graph replay
- Async `cudaMemcpyAsync` for host→device token upload, overlapped with graph dispatch

### Results

```
=== CUDA Graph Decode (bench.sh, 30 runs) ===
  Short:   8.13ms / 173x RTFx
  Medium: 23.81ms / 222x RTFx
  Long:   60.79ms / 289x RTFx
```

Compared to Round 8 (dual-tile): 1.2x faster on short, 1.1x on medium, 1.17x on long. The biggest win is on short text where launch overhead was a larger fraction of total time.
