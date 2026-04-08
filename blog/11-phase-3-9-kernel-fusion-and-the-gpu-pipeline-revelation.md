## 2026-03-03: Phase 3.9 — Kernel Fusion & the GPU Pipeline Revelation

### The Premise

At 91x RTF, we had a clean codebase with no cuDNN dependency for convolutions but still used cuDNN for BiLSTMs. The plan: fuse more kernels, eliminate remaining cuDNN, and squeeze out the last bits of performance.

The plan had 5 optimization steps:
1. Replace cuDNN BiLSTM with cuBLAS SGEMV
2. Precompute LSTM biases at load time
3. GPU SineGen (replace CPU harmonic source generation)
4. Fused residual + LayerNorm kernel for ALBERT
5. cuBLASLt GEMM+Bias fusion

What we learned was completely unexpected.

### Step 1: cuDNN BiLSTM Removal (91x → 87.5x)

Replaced cuDNN's LSTM with our own approach: one cuBLAS SGEMM for all input gates, then per-timestep `cublasSgemv` for the hidden-to-gate projection plus a small `lstm_gates_f32` kernel for sigmoid/tanh activations.

We also tried a `fused_lstm_f32` kernel that runs all timesteps in a single launch (one CUDA block, H=256 threads each doing sequential gate computation). **Catastrophic result: 37.7x RTF.** The kernel is register-pressure-bound at H=256 — each thread does 4 gates × 256 multiply-accumulates per timestep, all hitting global memory. cuBLAS SGEMV distributes this work across the full GPU.

After reverting to the SGEMV approach: **87.5x RTF** — a small regression from losing cuDNN's internal optimizations, but now we have zero cuDNN dependency.

### Steps 2 & 4: Micro-Fusions (87.5x → 88.0x)

**Precomputed LSTM biases**: Each BiLSTM has two bias vectors (bih, bhh) that get added at runtime. Precomputing `bih + bhh` once at init time saves 12 kernel launches per inference. Effect: 87.5x → 87.9x.

**Fused residual + LayerNorm**: ALBERT has 12 layers, each with 2 residual-add-then-LayerNorm operations (24 total). A new `residual_layer_norm_f32` kernel does both in one pass — loads both inputs, computes their sum, normalizes, and writes the result. This halves the memory traffic vs separate `add_f32` + `layer_norm_f32`. Effect: 87.9x → 88.0x.

### Step 5: cuBLASLt Bias Fusion (88.0x → 88.0x)

cuBLASLt's `CUBLASLT_EPILOGUE_BIAS` promises to fuse bias addition into the GEMM epilogue. We applied it to all 72 ALBERT projections (Q/K/V/dense/FFN × 12 layers) plus the bert_enc and dur_proj projections, eliminating ~74 separate `bias_add_f32` kernel launches.

**Result: zero measurable improvement.** The `cublasLtMatmulDesc` creation/destruction overhead (8 API calls per GEMM: create matmulDesc + 3 layouts, execute, destroy 4 objects) perfectly offset the saved `bias_add` kernel launches. For small matrices (768×15), the bias_add kernel is essentially free — it barely touches the GPU.

We also couldn't use `CUBLASLT_EPILOGUE_GELU_BIAS` to fuse GELU into the FFN projections because ALBERT uses `gelu_new` (tanh approximation) while cuBLASLt implements exact GELU (erf-based). The numerical difference compounds through 12 layers.

### Step 3: GPU SineGen — The Revelation (88.0x → 150.3x)

This was supposed to be a minor optimization. The CPU SineGen generates harmonic source waveforms from F0 predictions — about 19,200 samples across 9 harmonics. The computation itself is trivial.

The implementation:
- **sinegen_phase_f32**: 9 threads (one per harmonic), each doing a sequential cumulative sum over L2=64 elements. Runs as 1 CUDA block.
- **sinegen_source_f32**: T_audio=19,200 threads doing parallel phase interpolation, sin(), UV masking, hash-based Box-Muller noise, linear combination, and tanh.

**Result: 88.0x → 150.3x RTF.** A 70% speedup from replacing a function that takes microseconds on CPU.

### Why CPU Sync Points Kill GPU Performance

The CPU SineGen wasn't slow because of computation. It was slow because of `cudaStreamSynchronize(stream)`.

Here's what happens during inference without the sync:
```
GPU timeline: [...ALBERT...][...TextEnc...][...Prosody...][...Decoder...][...Generator...]
```
All kernels are queued asynchronously. The GPU executes them back-to-back with zero idle time. The CPU races ahead, queuing work faster than the GPU can execute it.

With the CPU SineGen sync:
```
GPU:  [...ALBERT...TextEnc...Prosody...][====IDLE====][..Decoder..Generator..]
CPU:  [queue queue queue queue] [SYNC] [sinegen_cpu] [queue queue queue]
```
The `cudaStreamSynchronize` forces the CPU to wait for ALL previously queued GPU work to complete. Then the CPU computes SineGen. Then it copies the result back to GPU. Then it starts queuing decoder work. During the sync + CPU computation + memcpy, the GPU sits completely idle.

But it's worse than that. Before the sync, the CPU was queueing work *ahead* of the GPU — filling the GPU's command queue so it always had work ready. After the sync, that pipeline buffer is drained. The GPU has to wait for each kernel to be individually queued by the CPU, adding launch latency between kernels.

The GPU SineGen isn't faster because GPUs are better at computing sine functions. It's faster because **it never stops the pipeline.** The phase computation (9 threads, trivially small) and source generation (19K threads, ~0.1ms) are just two more kernels in the stream — queued and executed without any CPU intervention.

### The Lesson

**Kernel count reduction ≠ performance.** We eliminated ~100 kernel launches through bias fusion, residual+LN fusion, and bias precomputation. Total impact: 0.5x RTFx. We eliminated one CPU sync point. Impact: 62x RTFx.

The performance hierarchy for GPU inference:
1. **Pipeline stalls** (CPU sync points, host↔device transfers) — 10-100x impact
2. **Algorithm choice** (cuBLAS SGEMM vs naive kernel, im2col vs cuDNN) — 2-10x impact
3. **Kernel fusion** (combining element-wise ops) — 1-5% impact at this scale

For small-model inference where individual kernels take microseconds, keeping the GPU pipeline full matters far more than reducing the number of kernels. The GPU can fire hundreds of tiny kernels with negligible overhead — as long as they're all on the same stream and the CPU never blocks.

### Performance Progression (Full History)

| Phase | Optimization | RTFx | vs Previous |
|-------|-------------|------|-------------|
| 3.6 | Naive CUDA kernels | 6.5x | baseline |
| 3.7 | cuDNN conv + cuBLAS LSTM + precomputed WN | 76.6x | +11.8x |
| 3.8 | cuDNN BiLSTM + TF32 | 87x | +1.14x |
| 3.8 | im2col + cuBLAS replacing cuDNN conv | 91x | +1.05x |
| 3.9 | cuDNN removal + kernel fusion | 88x | -3% (cuDNN overhead was hiding real cost) |
| **3.9** | **GPU SineGen (eliminate CPU sync)** | **150x** | **+1.70x** |

### vs PyTorch

We started this project with PyTorch at 149.5x RTF as the ceiling to beat. Our C++/CUDA implementation now matches it at **150.3x RTF** — with a codebase that:
- Has zero Python dependencies
- Compiles to a single static binary
- Uses no cuDNN (only cuBLAS + cuBLASLt + custom kernels)
- Runs in 327 MB GPU memory (model weights only)
- Initializes in ~110ms (vs PyTorch's multi-second startup)

And we haven't touched FP16 yet.
