## 2026-03-15: Optimization Round 9 — Residual Fusion + Snake Fusion

### Residual Add → Cutlass Epilogue

In generator resblocks, the pattern was: `conv → add(conv_out, residual)`. Extended `cutlass_conv1d_fprop` with a `residual` pointer parameter — Cutlass accumulates directly into the residual buffer (`C=residual, beta=1`), then `channel_bias_add` handles bias separately. Eliminates 27 `add_f32` kernels per inference and 33% less memory traffic per fused site.

### Snake → Instance Norm Kernel

Snake activation (`x + sin²(αx)/α`) always followed the AdaIN instance norm in generator resblocks. Added optional `snake_alpha` parameter to `instnorm_style_norm_kernel` — computes norm + snake in a single memory pass instead of separate write + read/write. Eliminates 48 kernel launches.

### Results

```
Long: 68.2ms → 65.6ms (269x RTFx, +41% vs cuBLAS baseline)
STT: short PASS, medium PASS, long PASS
```
