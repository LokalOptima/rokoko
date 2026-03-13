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

On first run, the model bundle (~364 MB) is auto-downloaded from GitHub releases to `~/.cache/rokoko/`.

```bash
# Text to WAV
./rokoko --text "Hello world." -o output.wav

# Different voice
./rokoko --text "Hello world." --voice af_bella

# Web UI
./rokoko --serve 8080
```

Available voices: `af_heart` (default), `af_bella`, `af_sky`, `af_nicole`.

## Options

```
--text <text>       Text to synthesize
--bundle <file>     Model bundle (default: ~/.cache/rokoko/rokoko.bundle)
--voice <name>      Voice pack (default: af_heart)
-o <file>           Output WAV (default: output.wav, - for stdout)
--stdout            Write WAV to stdout
--serve [port]      Start HTTP server with web UI (default port: 8080)
--host <addr>       Server bind address (default: 0.0.0.0)
```
