## 2026-03-18: Code Cleanup

Removed ~400 lines of dead code accumulated from optimization iterations:

**Dead kernels** (9 removed from `kernels.cu` + `kernels.h`):
- `fused_lstm_f32` — failed LSTM fusion attempt (Round LSTM)
- `sigmoid_f32` — replaced by fused `sigmoid_sum_f32`
- `cast_i64_to_i32` — token IDs are int32 now
- `instance_norm_1d_f32` — replaced by `instance_norm_style_affine_f32`
- `style_affine_1d_f32` — fused into instance_norm
- `snake_f32` — fused into instance_norm via snake_alpha (Round 9)
- `conv_transpose1d_f32` — replaced by `gemm_conv_transpose1d`
- `upsample_nearest_f32` — replaced by `upsample_nearest_1d_2x_f32`
- `tanh_f32` — no callers

**Dead Cutlass types** (2 removed from `cutlass_gemm.cu`):
- `GemmBatchedTN_SIMT`, `GemmBatchedNN_SIMT` — batched SIMT fallback never instantiated (fallback uses loop of single GEMMs instead)
- Saves ~30s compile time (2 fewer Cutlass template instantiations)

**Stale comments**: Updated cuBLAS references across `tts.cpp`, `weights.h`, `kernels.h`, `cutlass_conv.cu`, `cutlass_gemm.cu`, `main.cu`.
