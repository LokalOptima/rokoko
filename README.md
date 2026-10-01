<p align="center">
  <img src="rokoko.png" width="256" alt="rokoko">
</p>

# Rokoko

Fast text-to-speech on GPU. Neural G2P + Kokoro TTS in a single CUDA binary.

This is the consolidated home for inference, G2P training, model export and
regression tests. The executable embeds its weights, G2P model and `af_heart`
voice. It uses no Python, model downloads, external model files or Rokoko cache
at runtime.

## Build

Requires an NVIDIA GPU, its driver, the CUDA toolkit, a C++17 compiler, GNU binutils
and Python 3 (standard library only, for build-time asset preparation).
The verified setup is Linux x86-64, CUDA 13.1, CUTLASS 4.4.1, and an RTX 5070 Ti.
The default build targets the local CPU and GPU. There is no CPU inference backend.
There is one inference implementation: FP16 weights and matrix/convolution kernels,
with FP32 intermediate values and operations where required. The legacy runtime
has been removed; no precision flag or separate `rokoko.fp16` executable is needed.
CUTLASS (headers only) isn't in the repo:

```bash
git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

```bash
make rokoko          # bundled FP16 inference
```

Set `CUDA_HOME` if CUDA isn't at `/usr/local/cuda-13.1`:

```bash
make rokoko CUDA_HOME=/opt/cuda-13.1
```

CMake also provides the `rokoko_lib` target, including the embedded assets. Use
`-DROKOKO_BUILD_LIB_ONLY=ON` for a library-only build.
Asset downloads happen during the build, not CMake configuration.

```bash
cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build/cmake -j2
```

Library callers now use `context.init()` and
`pipeline.synthesize(text, audio)`. HTTP requests contain `text` and optionally
`input: "phonemes"`; the obsolete `voice` field returns HTTP 400.

Checks and setup are documented in [tests/README.md](tests/README.md):

```bash
make test             # fast offline CPU helper/frontend checks
make test-frontend    # native GPU G2P + Python reference evaluator
```

## Bundled assets

The build downloads FP16 weights, G2P V11 and
`af_heart` into `build/assets/` (CMake: `<build-directory>/assets/`). Versions,
URLs, sizes and SHA-256 checksums are pinned in [assets/manifest.json](assets/manifest.json).
Verified files are reused on subsequent builds. Corrupt files fail the build;
the build never silently substitutes another model. Downloads are atomic and
shared safely between parallel builds. `make clean` preserves downloaded assets.

The canonical G2P release is staged for publication. Until it is published,
fresh builds need the approved G2P file preseeded in the asset directory, or the
local-source option below. The weights and voice download URLs are already verified.

To build from an existing directory of approved files, without downloads:

```bash
make rokoko ASSET_SOURCE=/path/to/models OFFLINE=1
# The source contains weights.fp16.bin, g2p.bin and voices/af_heart.bin.
```

The files are copied and verified inside the build directory. Once that directory
is populated, `make rokoko OFFLINE=1` needs no source directory. Make's
`ASSET_DIR` changes the build asset location; CMake provides `ROKOKO_ASSET_DIR`,
`ROKOKO_ASSET_SOURCE` and `ROKOKO_OFFLINE` for the same purposes.

The assembler embeds the raw bytes in read-only sections. The loaders read those
bytes directly; there is no extraction into a cache or temporary directory.
After building, copy just `rokoko` (about 202 MB with the tested Make toolchain).
You can delete the build directory. Moving the executable still requires a
compatible CPU/GPU, NVIDIA driver, CUDA runtime and Linux system libraries.
Model updates require rebuilding the executable.

`./rokoko --build-info` prints the identities of the embedded assets without
initializing CUDA. File-based export and evaluation tools remain available for
development; model paths and voice selection are no longer production CLI options.

## Usage

```bash
# Text to speech (plays through speakers)
./rokoko "Hello world." --say

# Save to WAV file
./rokoko "Hello world." -o hello.wav

# Pipe to audio player
./rokoko "Hello world." --stdout | aplay

# Web UI
./rokoko --serve 8080
```

`--say` uses an installed audio player: `aplay`, `paplay`, `pw-play`, or `ffplay`.
WAV output is mono, 24 kHz, PCM16.

The supported voice is **`af_heart`**, the highest-graded English voice in [Kokoro’s official ratings](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md#american-english).

## Acknowledgments

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by hexgrad — the original TTS model that this project reimplements in C++/CUDA (Apache 2.0 License)
- **[CUTLASS](https://github.com/NVIDIA/cutlass)** by NVIDIA — CUDA GEMM templates (BSD-3-Clause License, Copyright 2017-2026 NVIDIA Corporation & Affiliates)
- **[cpp-httplib](https://github.com/yhirose/cpp-httplib)** by yhirose — HTTP server for web UI (MIT License)

## Options

```
-o <file>           Output WAV (default: output.wav)
--phonemes          Treat input as IPA (bypass normalization/G2P)
--say               Play audio through speakers
--stdout            Write WAV to stdout
--serve [port]      HTTP server with web UI (default: 8080)
--host <addr>       Server bind address (default: 0.0.0.0)
-v                  Verbose output (timings, IPA, GPU info)
--build-info        Print embedded asset identities (JSON)
--help              Show help
```

## Development and model export

- [G2P training, export and provenance](training/g2p/README.md)
- [Regression checks and reference environment](tests/README.md)
- [Retained history and repository layout](docs/HISTORY.md)

After setting up the optional [reference Python environment](tests/README.md#preparation),
export TTS assets from a local official Kokoro-82M directory containing
`config.json`, `kokoro-v1_0.pth`, and `voices/af_heart.pt`. The approved source
revision and file hashes are recorded in [the artifact manifest](tests/fixtures/artifacts.json).
The G2P checkpoint is separate; follow the [G2P export instructions](training/g2p/README.md#export-and-evaluate).

```sh
.venv-tests/bin/python scripts/export_weights.py \
  --official /path/to/Kokoro-82M -o /path/to/models/weights.bin \
  --voice-output /path/to/models/voices/af_heart.bin
.venv-tests/bin/python scripts/convert_v2.py \
  --weights /path/to/models/weights.bin -o /path/to/models/weights.fp16.bin
```

Once all runtime assets exist, `python3 tests/prepare.py --models /path/to/models` verifies
their checksums and downloads pinned official references for the tests. It requires
the runtime files first; it does not create them. See [test setup](tests/README.md)
for the independent artifact and GPU checks.
FP32 exports are optional intermediate files used only by the offline converter,
not runtime assets. To audit such an export independently against official Kokoro:

```sh
.venv-tests/bin/python tests/artifacts.py --official /path/to/Kokoro-82M \
  --models build/assets --exported-fp32 /path/to/models/weights.bin
```

The Python tools are optional development dependencies. Keep datasets,
checkpoints, model weights and generated audio outside Git.
