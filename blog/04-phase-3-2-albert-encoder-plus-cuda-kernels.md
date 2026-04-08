## 2026-03-02: Phase 3.2 — ALBERT Encoder + CUDA Kernels

### FP32 First, FP16 Later

Initially tried to implement everything in FP16 (following parakeet's pattern). Spent hours debugging mysterious GEMM divergences — max_diff=11.0 between our output and PyTorch's reference. After much debugging, the user wisely pointed out: **get it working in FP32 first, then optimize to FP16**. This was absolutely the right call.

Switching to FP32 immediately separated precision issues from logic bugs. The weight export now stores FP32 tensors (327 MB instead of 164 MB), and all kernels/GEMMs use `cublasSgemm` and FP32 arithmetic.

### Custom CUDA Kernels

Created `src/kernels.cu` with 9 kernels:

| Kernel | Operation | Design |
|--------|-----------|--------|
| `embedding_gather` | y[i,:] = table[ids[i],:] | 1 block per row, 256 threads |
| `add_f32` | y = a + b | 256 threads per block |
| `layer_norm_f32` | LayerNorm | 1 warp (32 threads) per row, 3-pass |
| `gelu_f32` | GELU (tanh approx) | 256 threads per block |
| `leaky_relu_f32` | LeakyReLU | 256 threads per block |
| `sigmoid_f32` | Sigmoid | 256 threads per block |
| `softmax_f32` | Softmax over last dim | 1 warp per row |
| `bias_add_f32` | y[n,d] = x[n,d] + bias[d] | 256 threads per block |
| `transpose_f32` | 2D transpose | 32x32 tiles, shared memory |

### ALBERT Forward Pass

Implemented the full ALBERT encoder in ~100 lines of C++:
1. **Embedding lookup**: word + position + token_type → [T, 128]
2. **LayerNorm**: normalize embeddings
3. **Projection**: Linear(128→768) via cuBLAS SGEMM
4. **12x shared layer** (ALBERT reuses one set of weights):
   - Q/K/V projections (3 cuBLAS SGEMMs)
   - Multi-head attention (batched SGEMM for scores + context)
   - Softmax
   - Dense projection
   - Residual + LayerNorm
   - FFN: Linear(768→2048) + GELU + Linear(2048→768)
   - Residual + LayerNorm

### Validation Against PyTorch

Created `scripts/dump_activations.py` to dump PyTorch intermediate tensors, then compare against CUDA output at each stage.

**Bugs found and fixed:**

1. **Validation structure bug**: Initially compared `buf.hidden` (post-12-layers) against intermediate references like `bert_proj.bin`. The buffer gets overwritten each layer, so all comparisons were against the final output. Fixed by running the forward pass step-by-step during validation.

2. **Missing GELU activation in dump script**: PyTorch's `albert_layer.ffn()` is just the Linear — the activation function is `albert_layer.activation()`, called separately. The dump script was missing this, producing wrong references.

3. **Wrong GELU variant**: ALBERT uses `gelu_new` (tanh approximation), not the exact erf-based GELU. Per-element difference of 0.0004 compounds to 0.004 across 12 layers. Switching to `tanhf`-based GELU brought the error down 1000x.

**Final validation results (FP32):**
```
bert_emb_ln           max_diff=0.000000  (exact match)
bert_proj (128->768)  max_diff=0.000001
bert_layer_0          max_diff=0.000003
bert_layer_1          max_diff=0.000006
bert_layer_2          max_diff=0.000005
bert_layer_11         max_diff=0.000006
bert_output           max_diff=0.000006
bert_encoder_out      max_diff=0.000012
d_en (transposed)     max_diff=0.000012
```

For reference, PyTorch's own CPU vs GPU divergence after 12 layers is max_diff=0.000004. Our max_diff=0.000006 is in the same ballpark — essentially exact.

### Next
Step 3.3 — Text Encoder (3 Conv1d blocks + bidirectional LSTM).
