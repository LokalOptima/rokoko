## Round 12: Pad Unaligned Decoder Channels

**Problem**: 14 decoder conv1 calls with C_in=514 or C_in=1090 can't use FP16 TensorOp (requires C_in % 8 == 0). They fall back to TF32 Cutlass, costing 3.9ms on medium text.

**Fix**: Pad weights at startup and activations at cast time:
- New `pad_blocks_f32` kernel: zero-pads conv weight C_in from 514→520, 1090→1096 (one-time at init)
- New `cast_f32_to_f16_pad` kernel: casts `[T, C_old]` FP32 → `[T, C_new]` FP16 with zero-padded channels
- `gemm_conv1d` gains a padded FP16 path: when C_in % 8 ≠ 0 and a padded NHWC FP16 weight exists, uses the pad+cast path
- `s_w_nhwc_f16_padded` map: stores padded NHWC FP16 weights alongside C_in_pad

Only the conv1 weights in each block need padding (conv2 inputs are already aligned, shortcut is K=1 GEMM).

### bench.sh Results

| Text | Before | After | Delta |
|------|--------|-------|-------|
| Short (1.6s) | 8.07ms | 6.45ms | −1.62ms (−20%) |
| Medium (5.7s) | 18.25ms | 16.57ms | −1.68ms (−9%) |
| Long (18.8s) | 52.77ms | 51.43ms | −1.34ms (−3%) |

STT 3/3 PASS.
