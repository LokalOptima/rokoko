## 2026-03-02: Phase 3.6 — End-to-End Pipeline + SineGen

### SineGen: CPU Harmonic Source

The generator's harmonic-plus-noise source model (`SineGen`) converts F0 predictions into time-domain sinusoidal waveforms. It works at the full audio sample rate (24kHz) — each F0 frame maps to 300 audio samples (upsample ratio = 300).

For each of 9 harmonics: accumulate phase from instantaneous frequency, apply voiced/unvoiced masking from F0, then project through a learnable Linear(9→1) layer. Add Gaussian noise weighted by an unvoiced indicator. The result is a `[22, T_audio]` tensor (22 = n_fft+2 channels) that the generator uses as its harmonic source.

CPU implementation is fine here — the computation is inherently sequential (phase accumulation) and the data is small relative to GPU kernel launch overhead.

### End-to-End Inference

Connected the full pipeline: `input.bin` (token IDs + style vector from `scripts/phonemize.py`) → ALBERT → text encoder → prosody predictor → decoder → generator → `output.wav`.

The `scripts/phonemize.py` helper runs Python-side text→phoneme conversion and voice style lookup, outputting a binary file that the C++ binary consumes. This keeps the C++ side pure inference with zero Python dependencies.

### First Performance Numbers

With all naive CUDA kernels (no cuDNN, CPU LSTMs):
- **3.5-3.9x realtime** — CPU BiLSTMs are the bottleneck

After moving BiLSTMs to GPU (cuBLAS SGEMM per timestep + gate kernel):
- **7.2-7.3x realtime**
