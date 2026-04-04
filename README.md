# Rokoko

Fast text-to-speech on GPU. Neural G2P + Kokoro TTS in a single CUDA binary.

## Build

Requires CUDA 13+ and a C++17 compiler.

```bash
make rokoko
```

Set `CUDA_HOME` if CUDA isn't at `/usr/local/cuda-13.1`:

```bash
make rokoko CUDA_HOME=/usr/local/cuda-12.6
```

## Usage

On first run, model files (~364 MB weights + ~33 MB G2P + ~2 MB voices) are auto-downloaded from GitHub releases to `~/.cache/rokoko/`.

```bash
# Text to WAV file
./rokoko "Hello world." -o hello.wav

# Play directly (Linux)
./rokoko "Hello world." --stdout | aplay

# Play directly (with FFmpeg)
./rokoko "Hello world." --stdout | ffplay -nodisp -autoexit -

# Different voice
./rokoko "Hello world." --voice af_bella

# Web UI
./rokoko --serve 8080
```

Available voices: `af_heart` (default), `af_bella`, `af_sky`, `af_nicole`.

## Options

```
--voice <name>      Voice (default: af_heart)
-o <file>           Output WAV (default: output.wav)
--stdout            Write WAV to stdout
--serve [port]      HTTP server with web UI (default: 8080)
--host <addr>       Server bind address (default: 0.0.0.0)
--weights <file>    TTS weight file (default: ~/.cache/rokoko/weights.bin)
--g2p <file>        G2P model file (default: ~/.cache/rokoko/g2p.bin)
--voices <dir>      Voice directory (default: ~/.cache/rokoko/voices)
```
