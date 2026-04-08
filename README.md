<p align="center">
  <img src="rokoko.png" width="256" alt="rokoko">
</p>

# Rokoko

Fast text-to-speech on GPU. Neural G2P + Kokoro TTS in a single CUDA binary.

## Build

Requires CUDA 13+ and a C++17 compiler.

```bash
make rokoko          # FP32 inference
make rokoko.fp16     # FP16 inference (half the download, same quality)
```

Set `CUDA_HOME` if CUDA isn't at `/usr/local/cuda-13.1`:

```bash
make rokoko CUDA_HOME=/usr/local/cuda-12.6
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

# Different voice
./rokoko.fp16 "Hello world." --say --voice af_bella

# Web UI
./rokoko.fp16 --serve 8080
```

Available voices: `af_heart` (default), `af_bella`, `af_sky`, `af_nicole`.

## Acknowledgments

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by hexgrad — the original TTS model that this project reimplements in C++/CUDA (Apache 2.0 License)
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA GEMM templates (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server for web UI (MIT License)

## Options

```
--voice <name>      Voice (default: af_heart)
-o <file>           Output WAV (default: output.wav)
--say               Play audio through speakers
--stdout            Write WAV to stdout
--serve [port]      HTTP server with web UI (default: 8080)
--host <addr>       Server bind address (default: 0.0.0.0)
--weights <file>    TTS weight file
--g2p <file>        G2P model file
--voices <dir>      Voice directory
-v                  Verbose output (timings, IPA, GPU info)
```
