## 2026-03-02: Phase 3.8 — cuDNN Elimination + im2col

### cuDNN BiLSTM: Fast but Fragile

At 76.6x RTF, cuDNN was handling all convolutions. The next bottleneck was our per-timestep cuBLAS SGEMV approach for LSTMs. cuDNN offers a native LSTM implementation that batches all timesteps internally, so we tried it.

cuDNN BiLSTMs brought us to **84.2x RTF**. Combined with TF32 tensor core math mode (`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`) and fusing the InstanceNorm+StyleAffine into a single kernel, we hit **87x RTF**.

But cuDNN's convenience came with hidden costs:
- **NCHW↔NHWC layout conversions**: cuDNN's optimized conv kernels require NHWC format, but our pipeline is channels-first. The driver silently inserts conversion kernels that consumed ~10% of GPU time.
- **Descriptor ceremony**: Creating and destroying `cudnnTensorDescriptor`, `cudnnFilterDescriptor`, `cudnnConvolutionDescriptor` for every convolution call.
- **Opaque kernel selection**: cuDNN picks its own algorithm. Sometimes it's a highly-tuned implicit GEMM; other times it's a less efficient fallback.

### im2col + cuBLAS: Transparent and Faster

The insight from profiling was that our Conv1d workloads are small (C=128-512, T=64-7681, K=3-7). For these sizes, an explicit im2col approach — unfolding the convolution into a GEMM — is competitive with cuDNN and eliminates all layout conversion overhead:

1. `im2col_1d_f32`: unfold input `[C_in, T]` → column matrix `[C_in*K, T_out]`
2. `cublasSgemm`: multiply weight matrix `[C_out, C_in*K]` × column matrix
3. For K=1: skip im2col entirely — the input IS the column matrix

This replaced cuDNN convolutions entirely. Similarly, ConvTranspose1d uses GEMM + `col2im_1d_f32` (with atomicAdd for overlapping positions).

Result: **91x RTF** with zero cuDNN dependency for convolutions. The cuDNN library was still linked only for BiLSTM at this point.

### Kernel Tracing: Mapping PyTorch to CUDA

To understand remaining optimization opportunities, we built a PyTorch kernel tracer that hooks into `torch.ops.aten` to log every CUDA kernel PyTorch launches during inference. PyTorch fires **5,831 kernels** for a single 15-token utterance. Our C++ implementation consolidates these into ~350 kernel launches through:
- Weight norm precomputation (eliminates 89 kernels)
- Layout conversion elimination (eliminates 229 kernels)
- Fused InstanceNorm+StyleAffine (eliminates ~150 kernels)
- im2col batching (1 GEMM per conv instead of cuDNN's multi-kernel pipeline)

### Performance Summary

| Change | RTFx |
|--------|------|
| Baseline (naive CUDA + cuBLAS LSTM) | 76.6x |
| + cuDNN BiLSTM + TF32 | 87x |
| + im2col replacing cuDNN conv | 91x |
