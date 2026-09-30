<p align="center">
  <img src="rokoko.png" width="256" alt="rokoko">
</p>

# Rokoko

Fast text-to-speech on GPU. Neural G2P + Kokoro TTS in a single CUDA binary.

This is the consolidated home for inference, G2P training, model export and
regression tests. The executable uses no Python at runtime. Model assets remain
separate files.

## Build

Requires CUDA 13+ and a C++17 compiler. CUTLASS (headers only) isn't in the repo:

```bash
git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

```bash
make rokoko          # FP32 inference
make rokoko.fp16     # FP16 inference (half the download, same quality)
```

Set `CUDA_HOME` if CUDA isn't at `/usr/local/cuda-13.1`:

```bash
make rokoko CUDA_HOME=/usr/local/cuda-12.6
```

Checks and setup are documented in [tests/README.md](tests/README.md):

```bash
make test             # fast offline CPU helper/frontend checks
make test-frontend    # normalizer + native GPU G2P
```

## Usage

On first run, model files are auto-downloaded from GitHub releases to `~/.cache/rokoko/` (~200 MB for FP16, ~400 MB for FP32).

```bash
# Text to speech (plays through speakers)
./rokoko.fp16 "Hello world." --say

# Save to WAV file
./rokoko.fp16 "Hello world." -o hello.wav

# Pipe to audio player
./rokoko.fp16 "Hello world." --stdout | aplay

# Web UI
./rokoko.fp16 --serve 8080
```

The supported voice is **`af_heart`**, the highest-graded English voice in [Kokoro’s official ratings](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md#american-english).

## Acknowledgments

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by hexgrad — the original TTS model that this project reimplements in C++/CUDA (Apache 2.0 License)
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA GEMM templates (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server for web UI (MIT License)

## Options

```
--voice <name>      Voice (only: af_heart)
-o <file>           Output WAV (default: output.wav)
--say               Play audio through speakers
--stdout            Write WAV to stdout
--serve [port]      HTTP server with web UI (default: 8080)
--host <addr>       Server bind address (default: 0.0.0.0)
--weights <file>    TTS weight file
--g2p <file>        G2P model file
--voices <dir>      Directory containing af_heart.bin
-v                  Verbose output (timings, IPA, GPU info)
```

## Development and model export

- [G2P training, export and provenance](training/g2p/README.md)
- [Regression checks and reference environment](tests/README.md)
- [Retained history and repository layout](docs/HISTORY.md)

Export TTS assets from the pinned official files prepared by `tests/prepare.py`:

```sh
.venv-tests/bin/python scripts/export_weights.py \
  --official tests/models/Kokoro-82M -o /path/to/models/weights.bin \
  --voice-output /path/to/models/voices/af_heart.bin
.venv-tests/bin/python scripts/convert_v2.py \
  --weights /path/to/models/weights.bin -o /path/to/models/weights.fp16.bin
```

Use explicit `--weights`, `--g2p` and `--voices` paths to test exported assets.
The Python tools are optional development dependencies. Keep datasets,
checkpoints, model weights and generated audio outside Git.
