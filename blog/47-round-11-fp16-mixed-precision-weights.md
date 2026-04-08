## Round 11: FP16 Mixed-Precision Weights

**Goal**: Use FP16 Tensor Cores (175.8 TFLOPS on RTX 5070 Ti) instead of TF32 (43.9 TFLOPS). Convert weight matrices to half precision; keep activations/accumulator/output in FP32.

### Approach

1. After weight norms, cast 191 weight matrices to FP16 (~95 MB extra GPU memory)
2. Before each GEMM/Conv, cast FP32 activations to FP16 into a 16 MB staging buffer
3. Use FP16 MMA instruction `GemmShape<16,8,16>` (vs TF32 `<16,8,8>`)
4. Accumulator and output stay FP32 — no quality loss in non-linear ops
5. GEMV (LSTM Whh) reads FP16 weights with FP32 x directly — memory-bound, no cast needed

**Graceful fallback**: Convolutions with unaligned channels (C_in % 8 ≠ 0: 1090, 514, 22) automatically fall back to TF32 path.

### New files

| File | Purpose |
|------|---------|
| `src/cutlass_gemm_f16.cu` | FP16 GEMM: TN, NN, TN+bias, tiered dispatch (Large/Small/Align1/SIMT) |
| `src/cutlass_conv_f16.cu` | FP16 implicit GEMM Conv1d + fused reshape+cast kernel for NHWC weights |
| `scripts/compare_wav.py` | Mel spectrogram SNR comparison (80-bin log mel, n_fft=1024, hop=256) |

### nsys Profiling (medium text, 5.7s audio)

| Kernel category | FP32 time | FP16 time | Change |
|----------------|-----------|-----------|--------|
| Cutlass Conv Large (48 calls) | 8.7ms | 4.5ms | **−48%** |
| Cutlass Conv SIMT→FP16 Small (88 calls) | 3.4ms | 1.8ms | **−47%** |
| Cutlass GEMM (FP16-eligible) | 4.3ms | 2.6ms | **−40%** |
| cast_f32_to_f16 overhead | — | +1.7ms | 519 calls × 3.2μs |
| GEMV FP16 (LSTM Whh, 1826 calls) | 2.5ms | 2.3ms | −8% |
| **Net kernel savings** | | | **5.8ms** |

**Still on TF32** (unaligned channels): 3.5ms of Conv/GEMM can't use FP16 due to C_in=1090/514/22.

### Verification

- **Paraketto STT**: 3/3 texts PASS (short/medium/long) — identical transcriptions
- **Mel spectrogram SNR**: 29.2 dB (positive control noise floor: 35 dB, negative control: −1.7 dB)

### bench.sh Results

| Text | FP32 TTS | FP16 TTS | Speedup | FP32 RTFx | FP16 RTFx |
|------|----------|----------|---------|-----------|-----------|
| Short (1.6s) | 9.48ms | 8.07ms | 15% | 150x | 159x |
| Medium (5.7s) | 22.73ms | 18.25ms | 20% | 229x | 271x |
| Long (18.8s) | 58.08ms | 52.77ms | 9% | 296x | 324x |

### Waste identified (next steps)

1. **cast_f32_to_f16: 1.7ms** (519 calls) — 30% of savings eaten by activation casting. Fix: keep activations in FP16 throughout pipeline, or fuse cast into preceding kernels.
2. **TF32 fallback for unaligned channels: 3.5ms** — decoder bottleneck at 1090/514 channels. Fix: pad channels to alignment boundary.
3. **GEMV FP16: marginal** — 1826 LSTM Whh calls at 1.3μs each are launch-overhead-bound. FP16 only saves 8%. Not worth optimizing further.
