## 2026-03-08: Embedded Voice Pack & README Overhaul

### Embedded voice via objcopy

The af_heart voice pack (510×256 float32 matrix, ~510KB) is now linked directly into the `rokoko` binary. The Makefile uses `objcopy` to convert `voices/af_heart.bin` into a `.o` with the data in `.rodata`, and the C++ code references it via linker symbols (`_binary_voices_af_heart_bin_start/end`). No external voice file needed at runtime.

Moved voice packs from `data/voices/` to a top-level `voices/` directory so they're tracked in git (the `/data/` directory is gitignored for large generated files).

### README rewrite

The README was still describing the Python-first workflow (`kokoro_speak.py`, spacy setup, ONNX). Rewrote it to lead with the C++/CUDA pipeline:

```
make rokoko
./rokoko --text "Hello world." -o output.wav
```

Python scripts are documented as reference/comparison tools, not the primary workflow. Added sections for data export and the standalone phonemizer.
