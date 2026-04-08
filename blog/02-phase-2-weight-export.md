## 2026-03-02: Phase 2 — Weight Export

### The Export Script

Created `scripts/export_weights.py` following the same binary format as parakeet's `export_weights.py`. Key difference: parakeet exported from ONNX models (which had anonymized tensor names like `onnx::MatMul_6382` requiring graph tracing to resolve), while kokoro exports directly from PyTorch's `state_dict()` — clean, semantic names out of the box.

The binary format:
```
[4B "KOKO" magic][4B version=1][8B header_len]
[text header: one line per tensor with name, offset, size, dtype, dims]
[padding to 4096-byte boundary]
[tensor data, each 256-byte aligned]
```

### Weight Structure

688 tensors, 81.8M parameters total:

| Component | Tensors | FP32 | FP16 | Description |
|-----------|---------|------|------|-------------|
| `bert` | 25 | 25.2 MB | 12.6 MB | ALBERT encoder (shared layers) |
| `bert_encoder` | 2 | 1.6 MB | 0.8 MB | Linear 768→512 |
| `text_encoder` | 24 | 22.4 MB | 11.2 MB | Conv blocks + bidir LSTM |
| `predictor` | 146 | 64.8 MB | 32.4 MB | Duration/F0/noise prediction |
| `decoder` | 491 | 213.3 MB | 106.6 MB | ISTFTNet (AdaIN + upsampling) |
| **Total** | **688** | **327.2 MB** | **163.7 MB** | |

The decoder dominates at 65% of total weight size. It has 491 tensors — lots of `weight_g`/`weight_v` pairs from weight normalization, plus AdaIN norm parameters.

Interesting detail: the model uses **weight normalization** (`weight_g` and `weight_v`) rather than standard `weight` tensors for most convolutions. In the C++ implementation we'll need to either:
1. Pre-compute `weight = weight_g * weight_v / ||weight_v||` at load time, or
2. Apply weight norm in the kernels

Option 1 is simpler and has zero runtime cost.

### Verification

Round-trip verification passes: write all 688 tensors to `weights.bin`, reload, compare byte-by-byte against originals. All match exactly.

```
weights.bin: 163,675,648 bytes (163.7 MB)
  688 tensors, all FP16
  Verification: OK, all 688 tensors match exactly
```

### Next
Phase 3 — C++/CUDA implementation.
