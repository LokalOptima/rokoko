## 2026-03-02: Phase 3.5 — ISTFTNet Decoder

### The Largest Component

The decoder has 491 weight tensors (65% of the model) and transforms encoded text + prosody predictions into raw audio waveforms. It has two main parts:

**Pre-generator** (AdainResBlk1d blocks):
```
[asr_aligned(512,64), F0_down(1,64), N_down(1,64)]  → cat → [514, 64]
  → encode: AdainResBlk1d(514→1024)  → [1024, 64]
  → 4x decode: cat(x, asr_res(64), F0, N) → AdainResBlk1d(1090→1024 or →512)
  → decode[3]: upsample 2x → [512, 128]
```

**Generator** (ISTFTNet with harmonic source):
```
for i in [0, 1]:
  noise_convs[i](har) → noise_res[i](AdaINResBlock1)  → x_source
  ups[i](x, ConvTranspose1d stride=10/6)  → x
  x = x + x_source
  x = avg(resblock[j](x) for j in 3 kernels)  → Snake activation, dilated convs
LeakyReLU(0.01) → conv_post(128→22, k=7) → exp(spec) + sin(phase) → iSTFT
```

### 11 New CUDA Kernels

| Kernel | Operation | Notes |
|--------|-----------|-------|
| `conv1d_general_f32` | Conv1d with stride, dilation, padding | Generalizes the original conv1d kernel |
| `conv_transpose1d_f32` | Non-depthwise ConvTranspose1d | For generator upsampling (stride=10, 6) |
| `snake_f32` | `x + (1/a)*sin(a*x)^2` | Generator resblock activation |
| `upsample_nearest_f32` | Arbitrary factor upsampling | For pre-generator shortcut paths |
| `reflection_pad_1d_f32` | ReflectionPad1d | Generator last upsample stage |
| `exp_f32` | Element-wise exp | STFT magnitude recovery |
| `sin_f32` | Element-wise sin | Phase reconstruction |
| `tanh_f32` | Element-wise tanh | General activation |
| `stft_f32` | DFT-based STFT | n_fft=20, hop=5, Hann window, center padding |
| `istft_f32` | Overlap-add iSTFT | Hann window, normalization by window sum |

### Bugs Found and Fixed

**1. In-place Conv1d race condition** (noise_res diverged with max_diff=6.38):

The `adain_resblock1_forward` helper ran `conv1d_general_f32(xt_buf, ..., xt_buf)` — reading and writing the same buffer. In a convolution, each output position depends on a neighborhood of input positions. When different CUDA threads write outputs while other threads read inputs from the same buffer, results become nondeterministic. Fix: use a separate `conv_out_buf` to avoid aliasing.

**2. Buffer overflow in ConvTranspose1d** (ups[0] caused illegal memory access):

The `d_gen_cw` weight buffer was allocated for `256*256*11 = 720K floats`, but the generator's upsampling layer needs `512*256*20 = 2.6M floats`. The weight norm materialization wrote past the buffer end. Fix: size the buffer for the largest weight (ups[0]).

**3. Wrong LeakyReLU slope** (conv_post diverged with max_diff=7.67):

The generator's upsample loop uses `F.leaky_relu(x, LRELU_SLOPE)` with `LRELU_SLOPE=0.1`, but the final activation before conv_post uses `F.leaky_relu(x)` — no slope argument, so PyTorch defaults to **0.01**. Subtle difference in the source code, massive impact on output.

### AdaINResBlock1 vs AdainResBlk1d

Two different resblock architectures in the model:

| Feature | AdainResBlk1d (pre-gen) | AdaINResBlock1 (generator) |
|---------|------------------------|---------------------------|
| Activation | LeakyReLU(0.2) | Snake: `x + (1/a)*sin(a*x)^2` |
| Convolutions | 2 per block | 6 per block (3 rounds) |
| Dilations | None | 1, 3, 5 (per round) |
| AdaIN | 2 per block | 6 per block |
| Shortcut | Optional 1x1 conv | Identity (same dims) |
| Upsample | Optional ConvTranspose1d | None |

### Harmonic Source: SineGen Skip

The generator's harmonic noise source (`SineGen`) involves random phase initialization and noise, making it non-deterministic. For validation, we dump the post-STFT harmonic tensor `[22, 7681]` from PyTorch and load it as a reference in C++. The SineGen itself will be implemented for end-to-end inference but isn't needed for component-level validation.

### Validation Results

All 64 checks pass across the full model:

```
--- Decoder Validation ---
dec_asr                  max_diff=0.000001
dec_f0_down              max_diff=0.000016
dec_n_down               max_diff=0.000005
dec_cat                  max_diff=0.000016
dec_encode               max_diff=0.000005
dec_asr_res              max_diff=0.000001
dec_decode_0_output      max_diff=0.000005
dec_decode_1_output      max_diff=0.000006
dec_decode_2_output      max_diff=0.000008
dec_decode_3_output      max_diff=0.000050
dec_gen_input            max_diff=0.000050

--- Generator Validation ---
gen_noise_conv_0         max_diff=0.000002
gen_noise_res_0          max_diff=0.000003
gen_ups_0                max_diff=0.000020
gen_merge_0              max_diff=0.000021
gen_resblocks_0          max_diff=0.000069
gen_noise_conv_1         max_diff=0.000000
gen_noise_res_1          max_diff=0.000001
gen_ups_1                max_diff=0.000008
gen_refl_pad             max_diff=0.000008
gen_merge_1              max_diff=0.000008
gen_resblocks_1          max_diff=0.000061
gen_conv_post            max_diff=0.000053
gen_spec                 max_diff=0.000084
gen_phase                max_diff=0.000016
gen_audio (38400 samples) max_diff=0.000007
```

The full pipeline — from ALBERT encoder through text encoder, prosody predictor, decoder pre-generator, and generator — produces audio that matches PyTorch to within **7 millionths** of a sample value. At 24kHz, those 38,400 samples represent 1.6 seconds of speech.

### What's Left

The model is now fully validated component-by-component against PyTorch. What remains:
1. **SineGen**: Implement harmonic source generation (F0 → sinusoidal waveforms)
2. **End-to-end pipeline**: Wire together all components (phoneme text → audio)
3. **Phoneme tokenization**: Port the text→phoneme conversion (or call it from Python)
4. **Benchmarking**: Time the full C++ pipeline vs PyTorch/ONNX
5. **FP16 optimization**: Switch kernels to half precision for speed
