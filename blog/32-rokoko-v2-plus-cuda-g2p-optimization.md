## Day 12: rokoko_v2 + CUDA G2P optimization (2024-03-11)

### rokoko_v2: Neural G2P + TTS in one binary

Created `rokoko_v2` — a new binary that replaces the 3000-line dictionary phonemizer with the neural G2P model. The pipeline is dead simple: text → preprocess → G2P forward pass → tokenize → TTS infer → WAV.

Data files reduced from 6+ (gold dict, silver dict, POS tagger, espeak fallback, G2P model, CMU extra) to just 2 (weights.bin + g2p_v8_model.bin).

The key trick was copying only the VOCAB table (90 entries) and `to_tokens()` function (~15 lines) from phonemize.cpp, avoiding linking the full phonemizer. The `.cu` file is compiled by nvcc (for G2P CUDA kernels), while `rokoko_cuda.cpp` is compiled by g++ (for AVX2 headers in phonemize.h), then linked together.

### CUDA G2P: 2.24x speedup (now 1.55x faster than torch.compile)

Profiled the CUDA G2P inference and found the bottleneck: **massive per-operation overhead from cublasLt descriptors and per-head attention loops**.

The original code had ~200 GPU operations per inference (8 layers × 25 ops/layer):
- 13 cublasLt calls per layer, each creating/destroying 4 descriptors (matmul desc + 3 matrix layouts) = **52 descriptor API calls per layer**
- 4 separate per-head GEMMs for attention scores + 4 for value weighted sums = 8 tiny GEMMs on [64×T] matrices
- Separate scale kernel and per-head softmax launches

**Optimizations applied:**

| Change | Ops eliminated per layer |
|--------|------------------------|
| cublasSgemm instead of cublasLt | -52 descriptor create/destroy API calls |
| cublasSgemmStridedBatched for attention | 8 GEMM calls → 2 batched |
| Scale folded into GEMM alpha | -1 scale kernel |
| Batched softmax across heads | 4 launches → 1 |
| Fused residual add via beta=1 | -2 add kernels |

**Challenge**: The attention matrices in the QKV buffer have a non-trivial strided layout. QKV is column-major [3d, T] where each head h occupies rows [h*dk, (h+1)*dk). The stride between heads is just `dk` (=64), which maps perfectly to cublasSgemmStridedBatched's strideA/strideB. The output buffer `attn_out[d, T]` also has stride `dk` between heads. The critical insight was computing K^T*Q (not Q^T*K) so softmax operates on contiguous rows — this layout is preserved naturally in the batched version.

**Challenge**: Replacing cublasLt loses the EPILOGUE_BIAS fusion (bias add inside the GEMM). Solution: a simple `g2p_bias_kernel` that adds bias to each column of a column-major matrix — one block per column, cheap to launch.

**Results:**

| Implementation | µs/word | Speedup vs torch.compile |
|---------------|---------|--------------------------|
| PyTorch CUDA (eager) | 2078 | 0.47x |
| **torch.compile** | **980** | **1.0x** |
| C++ CUDA (old, cublasLt) | 1414 | 0.69x |
| **C++ CUDA (optimized)** | **632** | **1.55x** |

### Files changed

| File | Changes |
|------|---------|
| `src/rokoko_v2.cu` | New: neural G2P + TTS pipeline in one binary |
| `src/g2p_model_cuda.h` | Replaced cublasLt with cublasSgemm, batched attention, fused residuals |
| `Makefile` | Added rokoko_v2 build target |
