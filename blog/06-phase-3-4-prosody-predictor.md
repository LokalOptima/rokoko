## 2026-03-02: Phase 3.4 — Prosody Predictor

### The Beast

The prosody predictor is the most complex component of Kokoro — it's responsible for converting the encoded text into timing and pitch information that drives the decoder. It has **146 weight tensors** and chains together:

1. **DurationEncoder**: 3 rounds of BiLSTM + AdaLayerNorm + style concatenation
2. **Duration LSTM + projection**: predicts how many frames each phoneme lasts
3. **Alignment expansion**: uses predicted durations to stretch encoder output
4. **Shared BiLSTM**: processes expanded features for both F0 and noise
5. **F0 chain**: 3 AdainResBlk1d blocks with instance norm, style conditioning, and 2x time upsampling
6. **Noise chain**: identical architecture to F0

### New CUDA Kernels

8 new kernels for this phase:

| Kernel | Purpose |
|--------|---------|
| `instance_norm_1d_f32` | Per-channel normalization across time (for InstanceNorm1d) |
| `style_affine_1d_f32` | `(1+gamma)*x + beta` with per-channel style params |
| `ada_layer_norm_f32` | LayerNorm + style conditioning in one pass |
| `conv_transpose1d_depthwise_f32` | Depthwise transposed conv for 2x upsampling |
| `upsample_nearest_1d_2x_f32` | Nearest-neighbor 2x upsampling (shortcut path) |
| `scale_f32` | Element-wise scalar multiply (for `1/sqrt(2)`) |
| `sigmoid_sum_f32` | Fused sigmoid + sum-reduce (duration prediction) |
| `tile_1d_f32` | Broadcast vector to matrix (style expansion) |

### AdainResBlk1d — The Key Building Block

The F0 and noise prediction chains use `AdainResBlk1d` — a residual block with:
- **Residual path**: AdaIN1d → LeakyReLU → [optional ConvTranspose1d] → weight-normed Conv1d → AdaIN1d → LeakyReLU → weight-normed Conv1d
- **Shortcut path**: [optional nearest-neighbor upsample] → [optional 1x1 Conv1d]
- **Combine**: `(residual + shortcut) / sqrt(2)`

AdaIN1d itself is InstanceNorm (with learned affine) + a Linear layer that transforms the 128-dim style vector into per-channel scale and bias.

Block[1] in each chain does the 2x time upsampling (ConvTranspose1d depthwise, stride=2, k=3) and halves channels from 512 to 256. The upsample produces F0/noise predictions at 2x the frame rate, which the decoder later downsamples with a stride-2 Conv1d.

### Architecture Pattern

The generalized `bilstm_cpu()` helper replaced all the duplicated LSTM boilerplate. There are 5 BiLSTMs in the predictor alone (3 in DurationEncoder + 1 duration + 1 shared), all with identical dimensions (input=640, hidden=256).

### Validation Results

```
dur_enc_lstm_0      max_diff=0.000018
dur_enc_aln_0       max_diff=0.000042
dur_enc_lstm_1      max_diff=0.000014
dur_enc_aln_1       max_diff=0.000038
dur_enc_lstm_2      max_diff=0.000008
dur_enc_aln_2       max_diff=0.000013
dur_enc_output      max_diff=0.000013
dur_lstm_output     max_diff=0.000008
dur_proj_raw        max_diff=0.000029
pred_duration       max_diff=0.000004
pred_alignment      max_diff=0.000000  (exact match)
pred_en (expanded)  max_diff=0.000013
shared_lstm_output  max_diff=0.000003
f0_block_0          max_diff=0.000007
f0_block_1          max_diff=0.000038
f0_block_2          max_diff=0.000029
f0_pred             max_diff=0.000305
n_block_0           max_diff=0.000006
n_block_1           max_diff=0.000010
n_block_2           max_diff=0.000007
n_pred              max_diff=0.000005
```

All 21 predictor stages validate. Predicted durations `[17,2,2,2,2,3,2,1,2,3,4,3,13,7,1]` match PyTorch exactly (L=64 frames total). The F0 prediction has the highest error (max_diff=0.000305) due to error accumulation through 3 residual blocks and the InstanceNorm computation, but mean_diff is only 0.000023.

### Next
Step 3.5 — ISTFTNet Decoder (the final component, generating audio from F0/noise/text features).
