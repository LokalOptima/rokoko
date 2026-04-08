## 2026-03-02: Phase 3.7 — Performance Optimization

### The Hunt for Speed

At 7.3x realtime with naive CUDA kernels, we were far below PyTorch's 149x. Time to profile.

### Attempt 1: GPU Arena Allocator

Hypothesized that ~120 `cudaMalloc` + 77 `cudaFree` per inference was the bottleneck. Implemented a `GpuArena` — single 256 MB allocation with bump pointer, reset per inference.

Result: **6.6x realtime** — essentially no change. Modern CUDA driver already caches allocations. The arena stays because it's cleaner code, but it wasn't the bottleneck.

### Attempt 2: Fused LSTM Kernel

Wrote a fused LSTM kernel — one block, H=256 threads, all timesteps in one launch. Each thread handles one hidden unit across all timesteps, using shared memory for the hidden state.

Result: **6.5x realtime** — slightly *worse* than cuBLAS per-step! The fused kernel's inner loop does a naive O(H) dot product per gate per timestep. For H=256, that's 1024 sequential multiply-accumulates per thread per timestep. cuBLAS SGEMV is much better at exploiting the full GPU for this.

### nsys Profile: The Real Bottleneck

```
conv1d_general_kernel:   932ms  (66%)  312 calls
conv_transpose1d_kernel: 222ms  (16%)  12 calls
fused_lstm_kernel:       156ms  (11%)  72 calls
conv1d_kernel:            78ms   (6%)  216 calls
```

**93% of GPU time was in naive convolution kernels.** Meanwhile PyTorch uses cuDNN's `sm80_xmma_fprop_implicit_gemm` — NVIDIA's proprietary, heavily optimized implicit GEMM kernel for convolutions.

### cuDNN Integration

The fix was obvious: link cuDNN and use its kernels instead of our naive ones. Added `cudnn_conv1d()` and `cudnn_conv_transpose1d()` wrappers using cuDNN's 4D tensor descriptors (treating 1D conv as 2D with H=1).

Result: **35.9x realtime** — 5.4x speedup from cuDNN alone.

### Back to cuBLAS SGEMV for LSTM

With convolutions fast, the fused LSTM kernel was now **75% of remaining GPU time**. Reverted to the cuBLAS SGEMV per-timestep approach: batch-compute input gates via SGEMM, then per-timestep `cublasSgemv` for Whh@h + `lstm_gates_f32` kernel for gate activations.

Result: **73.8x realtime** — another 2.1x from fixing the LSTM.

### Precomputed Weight Norms

Weight normalization (`w = g * v / ||v||`) was computed every inference for ~100 convolution weights. Since the weights are constant, precompute once at init time by overwriting `wv` in-place.

Result: **76.6x realtime** — small but free improvement.

### Performance Progression

| Optimization | RTFx | Speedup |
|---|---|---|
| Naive CUDA kernels | 6.5x | baseline |
| GpuArena allocator | 6.6x | +1.5% |
| cuDNN convolutions | 35.9x | +5.4x |
| cuBLAS SGEMV LSTM | 73.8x | +2.1x |
| Precomputed weight norms | 76.6x | +4% |

### Current GPU Kernel Profile (nsys)

After all optimizations:
```
cuDNN conv (cutlass fprop):  30.6%  — the real work
NCHW↔NHWC conversion:        9.8%  — cuDNN format overhead
cuBLAS SGEMV (LSTM):          6.8%  — per-timestep hidden-to-gate
weight_norm_kernel:            7.7%  — (now eliminated via precompute)
instance_norm:                 6.8%  — decoder AdaIN normalization
lstm_gates:                    4.1%  — sigmoid/tanh gate activations
snake_kernel:                  2.5%  — generator activation
```

### vs PyTorch

PyTorch achieves 149x realtime on the same GPU — roughly 2x our current speed. Their advantage:
- cuDNN is amortized across larger batch operations (we process single tokens)
- CUDA graphs eliminate kernel launch overhead
- Fused attention kernels (we use separate Q/K/V SGEMMs + softmax)
- Their LSTM likely uses cuDNN's native implementation

There's still headroom. The NCHW↔NHWC conversion overhead (9.8%) could be eliminated by using NHWC format natively. The per-timestep LSTM SGEMV could potentially be replaced with cuDNN's LSTM. And FP16 would halve memory bandwidth requirements.

### All 64 Validation Checks Still Pass

Every optimization was verified against PyTorch reference activations. Max diff remains under 0.001 across the entire pipeline.
