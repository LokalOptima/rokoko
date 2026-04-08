## 2026-03-02: Phase 1 — PyTorch Reference Setup

### Context

We started with an ONNX Runtime setup that ran Kokoro at ~52x realtime on GPU. The plan is to follow the same path as the [parakeet](https://github.com/lapo/parakeet) ASR project: get a PyTorch reference working, export weights to a flat binary format, then write custom CUDA kernels to outperform everything.

### The Python Setup Saga

**Python version**: Kokoro requires Python `<3.13`. We were on 3.13, so downgraded to 3.12. Quick `uv venv --python 3.12 && uv sync`.

**Dependencies**: The `kokoro` PyTorch package (v0.9.4) pulls in a surprisingly large tree:
- `misaki` — G2P (grapheme-to-phoneme) library
- `spacy` + `en_core_web_sm` — NLP tokenization for text chunking
- `espeakng-loader` — espeak-ng speech synthesizer for OOD word fallback
- `torch 2.10.0+cu128` — the neural network runtime
- Total: ~110 packages

**The espeak-ng rabbit hole**: First run just... silently exited with code 1. No traceback, no error message. Turns out `misaki` uses `espeak-ng` for out-of-vocabulary word pronunciation. The library (`libespeak-ng`) was installed system-wide, but the Python integration (`espeakng-loader`) handled it. The *real* problem was `spacy` needing its English model (`en_core_web_sm`).

Normally you'd run `python -m spacy download en_core_web_sm`, but that internally calls `pip install` — and `uv` venvs don't have `pip`. The fix:
```bash
uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

This is a good reminder of why we're doing the C++ rewrite — the Python dependency chain is absurd for what should be a simple text→audio pipeline.

### The API

Kokoro's PyTorch API is clean. `KPipeline` handles everything:

```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cuda')
for graphemes, phonemes, audio in pipeline("Hello world!", voice='af_heart'):
    # audio is a torch.FloatTensor at 24kHz
```

It yields segments (the model processes text in chunks). For benchmarking, we need to `torch.cat()` all segments and `torch.cuda.synchronize()` for accurate timing.

The ONNX version (`kokoro-onnx`) had a simpler API — single call, returns numpy directly. But the PyTorch version gives us the flexibility we need for weight extraction and debugging.

### Benchmark Results

After fixing the benchmark to pre-load all voices during warmup (first voice load involves downloading `.pt` files from HuggingFace and is ~1 second), we get clean numbers:

```
SYSTEM: RTX 5070 Ti, 20 CPU cores, PyTorch 2.10.0+cu128

GPU (PyTorch CUDA) — 10 utterances, 5 voices:
  [ 1/10] af_heart       8.4s audio in 0.063s  (134.8x realtime)
  [ 2/10] af_bella       9.7s audio in 0.062s  (154.5x realtime)
  [ 3/10] af_sarah       8.4s audio in 0.057s  (148.0x realtime)
  [ 4/10] bf_emma        8.6s audio in 0.063s  (137.4x realtime)
  [ 5/10] af_sky         8.7s audio in 0.066s  (131.1x realtime)
  [ 6/10] af_heart       9.5s audio in 0.063s  (151.4x realtime)
  [ 7/10] af_bella      10.0s audio in 0.060s  (166.1x realtime)
  [ 8/10] af_sarah       9.7s audio in 0.055s  (176.7x realtime)
  [ 9/10] bf_emma        8.9s audio in 0.060s  (148.9x realtime)
  [10/10] af_sky         8.7s audio in 0.057s  (150.9x realtime)

  Realtime factor:       149.5x
  Per-utterance:         mean=0.060s  median=0.061s

CPU (PyTorch CPU) — 10 utterances, 5 voices:
  Realtime factor:       4.6x
  Per-utterance:         mean=1.958s  median=1.947s
```

### Performance Summary

| Backend | RTFx | Per-utterance | Notes |
|---------|------|---------------|-------|
| ONNX GPU | ~52x | ~0.17s | onnxruntime-gpu, CUDAExecutionProvider |
| **PyTorch GPU** | **149.5x** | **0.060s** | torch 2.10.0+cu128 |
| PyTorch CPU | 4.6x | 1.96s | 20 cores |

PyTorch GPU is **2.9x faster** than ONNX Runtime GPU. This was a surprise — I expected them to be closer. PyTorch's CUDA graphs and kernel fusion are clearly doing work.

The 149.5x realtime baseline means each ~9 second utterance generates in ~60ms. That's already incredibly fast. The C++/CUDA reimplementation will need to be meaningfully faster to justify the effort — but we'll gain:
- No Python overhead (import time, GIL, etc.)
- No 110-package dependency chain
- Single static binary
- Full control over memory layout and kernel fusion
- Potential for FP16 throughout (model is FP32 currently)

### Files Changed

| File | Action |
|------|--------|
| `.python-version` | 3.13 → 3.12 |
| `pyproject.toml` | Swapped to kokoro + torch, ONNX as optional extra |
| `kokoro_speak.py` → `onnx_speak.py` | Renamed |
| `benchmark.py` → `onnx_benchmark.py` | Renamed |
| `kokoro_samples.py` → `onnx_samples.py` | Renamed |
| `kokoro_speak.py` | New — PyTorch CUDA inference |
| `benchmark.py` | New — PyTorch GPU vs CPU benchmark |
| `README.md` | New — setup instructions |

### Next
Phase 2 — weight export.
