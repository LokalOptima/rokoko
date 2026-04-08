## 2026-03-02: Phase 3.3 — Text Encoder

### Architecture

The text encoder converts token IDs into contextualized 512-dim representations:

```
input_ids [T]
  → Embedding [T, 512]
  → Transpose [512, T]   (channels-first for CNN)
  → 3x { Conv1d(512,512,k=5,pad=2) → LayerNorm → LeakyReLU(0.2) }
  → Transpose [T, 512]   (batch-first for LSTM)
  → Bidirectional LSTM(input=512, hidden=256)  → [T, 512]
  → Transpose [512, T]   (channels-first output)
```

### New CUDA Kernels

Three new kernels added:

| Kernel | Operation | Notes |
|--------|-----------|-------|
| `conv1d_f32` | Standard Conv1d | 1 warp per (channel, time), accumulates C_in*K=2560 products |
| `weight_norm_f32` | w = g * v / ‖v‖ | 1 warp per output channel, computes L2 norm of 2560-element vector |
| `layer_norm_channels_first_f32` | LN across channels at each time position | Unlike row-major LN in ALBERT, this normalizes across C=512 at each fixed time t |

### Weight Normalization

The Conv1d layers use PyTorch's `weight_norm` which stores `weight_g` [512,1,1] and `weight_v` [512,512,5] separately. At inference time we precompute the materialized weight: `w = weight_g * weight_v / ‖weight_v‖_2` (per output channel L2 norm over the 2560-element fan-in vector). This is done once per forward pass with a dedicated kernel.

### BiLSTM: CPU Implementation First

The bidirectional LSTM runs on CPU for now — download weights and input, run forward+reverse passes sequentially, upload the concatenated output. This is simple and correct. For the 15-token test input, each LSTM step processes a single 512-dim vector through 4 gates with 256 hidden units. Total: 30 steps (15 forward + 15 reverse).

The CPU LSTM will be moved to GPU later (CUDA LSTM kernel), but for correctness validation this is ideal — zero ambiguity about gate ordering, no cuBLAS tricks needed.

### Validation Results

```
text_encoder_embed   max_diff=0.000000  (exact match)
text_encoder_cnn_0   max_diff=0.000002
text_encoder_cnn_1   max_diff=0.000001
text_encoder_cnn_2   max_diff=0.000002
text_encoder_lstm    max_diff=0.000001
```

Every stage matches PyTorch to within 2e-6. The channels-first LayerNorm, weight normalization, and Conv1d all work correctly on the first try.

### Next
Step 3.4 — Prosody Predictor (duration/F0/noise prediction).
