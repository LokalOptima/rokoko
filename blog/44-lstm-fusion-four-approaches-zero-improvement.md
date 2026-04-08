## 2026-03-17: LSTM Fusion — Four Approaches, Zero Improvement

### Motivation

nsys profiling (node-level, long text) showed LSTM consuming 25% of total GPU kernel time:

```
LSTM SGEMV (cuBLAS):    9.49ms  (4,647 calls, 2.04μs avg)
lstm_gates_kernel:      6.10ms  (4,636 calls, 1.32μs avg)
Total LSTM:            15.59ms  (9,283 kernel instances)
```

The model has 5 BiLSTMs (H=256): 1 in the text encoder, 3 in the duration encoder, 1 for duration prediction, and 1 shared LSTM for F0/N prediction. Each BiLSTM runs 2 directions × T timesteps × 2 kernels (SGEMV + gates) = 4T kernel launches. With T≈310 for long text, that's ~6,200 kernel instances — the overwhelming majority of graph nodes.

The current implementation: cuBLAS `cublasSgemv` for the [1024, 256] hidden-to-gate GEMV, plus a separate `lstm_gates_f32` kernel for sigmoid/tanh nonlinearities. Both are captured inside CUDA graphs (encode + decode), so CPU-side launch overhead is zero. The question: can we do better than ~9,000 graph nodes of tiny kernels?

### Approach 1: Single-Block Fused (H=256 threads) — Already Failed

This was tried in Phase 3.9 and produced catastrophic 37.7x RTFx. One CUDA block with H=256 threads, each thread computing 4 gate dot products of length 256 sequentially. Only 1 SM active (out of ~60), uncoalesced Whh reads (stride H=256 between threads reading different rows), only 8 warps for latency hiding. cuBLAS SGEMV distributes the same work across the full GPU.

### Approach 2: Single-Block Fused (4H=1024 threads, transposed Whh)

**Idea:** Fix the old kernel's problems — use 4H=1024 threads (one per gate output), transpose Whh from [4H, H] to [H, 4H] at load time for coalesced access. All 1024 outputs computed in parallel, then threads 0-255 apply gates and write h.

**Implementation:** Added `whh_fwd_T`/`whh_rev_T` fields to `BiLSTMWeights`, transposed at load time with `transpose_f32`. New kernel uses shared memory for `s_h[H]` (hidden state broadcast) and `s_gemv[4H]` (GEMV output). Inner loop reads `Whh_T[k * G + tid]` — adjacent threads read adjacent memory (coalesced).

**Result:**

```
=== Single-Block 1024 Threads (bench.sh, 10 runs) ===
  short:  10.73ms TTS   132x RTFx  (baseline: 8.13ms, 173x)
  medium: 36.58ms TTS   148x RTFx  (baseline: 23.81ms, 222x)
  long:  101.56ms TTS   178x RTFx  (baseline: 60.79ms, 289x)
```

**Why it failed:** Still only 1 SM. The Whh GEMV reads 1MB per timestep. A single SM gets ~16 GB/s of the ~960 GB/s total memory bandwidth — 60x less than cuBLAS which distributes across ~30 SMs. The 4x improvement over Approach 1 (from 37.7x to 178x) came from coalesced access + 32 warps for latency hiding, but single-SM bandwidth is the hard limit.

### Approach 3: Multi-SM Cooperative (naive, 2 grid.sync()/step)

**Idea:** Use `cudaLaunchCooperativeKernel` with `cooperative_groups::grid::sync()` for inter-block synchronization. Multiple SMs compute the GEMV cooperatively, then sync, then apply gates, then sync, then next timestep.

First verified cooperative launches work inside CUDA graph capture (they do, CUDA 12+).

**Implementation:** `gridDim.x = max_cooperative_blocks`, each thread handles `ceil(G / total_threads)` GEMV outputs. After GEMV: `grid.sync()`. Then each thread handles `ceil(H / total_threads)` gate computations. After gates: `grid.sync()`. Required `-rdc=true` + device link step in Makefile.

**Result:**

```
=== Cooperative Naive (bench.sh, 10 runs) ===
  short:  14.68ms TTS   102x RTFx
  medium: 57.73ms TTS    96x RTFx
  long:  169.11ms TTS   108x RTFx
```

**Why it failed:** Two problems:
1. **Thread utilization:** With ~60 SMs × 256 threads = 15,360 total threads but only G=1024 GEMV outputs, 93% of threads were idle during the GEMV phase.
2. **Sync overhead:** 2 `grid.sync()` calls per timestep × 310 timesteps × 2 directions × 5 BiLSTMs ≈ 6,200 syncs. At ~5-10μs per grid sync, that's 31-62ms of pure synchronization overhead.

### Approach 4: Multi-SM Cooperative (warp-per-unit, 1 grid.sync()/step)

**Idea:** Fix both problems from Approach 3. Assign 1 warp (32 threads) per hidden unit. Each warp cooperatively computes all 4 gate dot products for its hidden unit using warp shuffle reduction, then immediately applies gate nonlinearities — no inter-block dependency within a timestep. Only 1 `grid.sync()` needed (to ensure all h values are written before the next timestep).

**Key insight for coalescing:** With 32 lanes in a warp all working on the same Whh row, lane `l` reads `Whh[row * H + l]` — adjacent lanes read adjacent elements. Perfectly coalesced WITHOUT transposing Whh.

**Implementation:** 32 blocks × 8 warps/block = 256 warps = 256 hidden units (one per warp). Each warp: 32 lanes handle 8 elements each of the 256-element dot product, warp shuffle reduction (`__shfl_down_sync`), lane 0 applies sigmoid/tanh and writes c, h.

**Result:**

```
=== Cooperative Warp-Per-Unit (bench.sh, 30 runs) ===
  short:   8.15ms TTS   173x RTFx  (baseline: 8.13ms, 173x)
  medium: 23.66ms TTS   224x RTFx  (baseline: 23.81ms, 222x)
  long:   60.25ms TTS   291x RTFx  (baseline: 60.79ms, 289x)
```

STT: short PASS, medium PASS, long PASS.

**This matched the baseline exactly** — within measurement noise. The GEMV computation matched cuBLAS speed, but the grid.sync() overhead (~1-2μs per sync × ~3,100 syncs = ~3-6ms) cancelled out the savings from eliminating ~9,000 graph nodes.

### Why LSTM Fusion Doesn't Help

The fundamental reason: **CUDA graph node dispatch overhead on RTX 5070 Ti is negligible** (~0.3μs per node). Total kernel time for all ~10,000 graph nodes was 61.85ms; wall time was ~65ms. The overhead of dispatching 9,283 LSTM graph nodes is only ~3ms — not enough headroom for any fusion approach to pay for itself.

| Approach | Blocks | Threads | Syncs/step | Short | Long | Problem |
|----------|--------|---------|------------|-------|------|---------|
| Baseline (cuBLAS) | many | many | — | 8.13ms | 60.79ms | — |
| 1-block H threads | 1 | 256 | 0 | — | — | 1 SM, uncoalesced |
| 1-block 4H threads | 1 | 1024 | 0 | 10.73ms | 101.56ms | 1 SM bandwidth |
| Coop naive | ~60 | 256 | 2 | 14.68ms | 169.11ms | 93% idle + 2 syncs |
| Coop warp/unit | 32 | 256 | 1 | 8.15ms | 60.25ms | Grid.sync ≈ savings |

### Lessons Learned

1. **Profile the overhead, not just the kernels.** I initially estimated ~3μs per graph node dispatch based on kernel time vs wall time analysis. The actual overhead was ~0.3μs — a 10x overestimate that motivated three approaches before proper A/B testing revealed the truth.

2. **The GEMV bandwidth wall is real.** For a [1024, 256] GEMV reading 1MB of Whh per timestep, single-SM bandwidth (16 GB/s) versus multi-SM (960 GB/s) is a 60x disadvantage. No amount of kernel fusion can overcome having 1/60th the memory bus.

3. **Grid.sync() is not free.** Even at ~1-2μs per sync, thousands of syncs per inference add up. The cost of inter-SM synchronization must be weighed against the savings from reduced graph nodes.

4. **CUDA graphs on modern GPUs have very low per-node overhead.** The RTX 5070 Ti (SM120/Blackwell) dispatches graph nodes at ~0.3μs each. This makes "reduce graph node count" a much weaker optimization lever than expected.

5. **Always A/B test against the actual baseline.** Comparing against stale benchmark numbers can produce phantom improvements. The proper comparison revealed our Approach 4 was a wash, not the "1.5x speedup" initially measured against outdated numbers.
