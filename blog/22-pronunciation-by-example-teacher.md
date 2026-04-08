## 2026-03-08: Pronunciation-by-Example Teacher

### Design

Implemented a `./teacher` binary that lets users teach Kokoro custom pronunciations. The workflow: user records a word → Wav2Vec2 extracts seed phonemes → evolutionary search finds the phoneme sequence that makes Kokoro reproduce the original pronunciation → saves to user dictionary.

### Synthesis API Refactoring (Phase 4)

Moved `GpuArena` from `rokoko_cuda.cpp` to `rokoko.h` so it can be shared between binaries. Removed `static` from `rokoko_infer()`, `write_wav()`, and `precompute_weight_norms()`, adding declarations to `rokoko.h`. Guarded `main()` in `rokoko_cuda.cpp` with `#ifndef ROKOKO_NO_MAIN` so the teacher binary can link against it without symbol conflicts. Both `rokoko` and `teacher` compile cleanly from the same source files.

### User Dictionary (Phase 6)

Added JSON user dictionary support — highest priority in the lookup chain, checked before gold dict. Minimal JSON parser (~60 lines) handles flat `{"word": "phonemes"}` objects with UTF-8 escapes. Case variants (lowercase, capitalized) are auto-generated. The `load_user_dict()` call was added to `Phonemizer::load()` and also inserted into `lookup_with_pos()` (which has its own dictionary cascade independent of `lookup()`).

Key learning: the phonemize pipeline has *two* independent lookup paths — `lookup()` (basic) and `lookup_with_pos()` (POS-aware, used when POS tagger is loaded). Both needed the user dict check.

### Wav2Vec2 Custom CUDA (Phase 3)

Wrote a complete CUDA implementation of `facebook/wav2vec2-lv-60-espeak-cv-ft`:

- **CNN feature extractor**: 7 strided Conv1d layers. Reused `conv1d_general_f32` from existing `kernels.cu`.
- **Group norm**: New CUDA kernel for the first CNN layer (512 groups). Shared-memory warp reduction for mean/variance.
- **Positional conv**: Group Conv1d (groups=16, K=128) with weight norm. New kernel for group convolution.
- **Transformer encoder**: 24 layers, 16 heads, d=1024. Reused `layer_norm_f32`, `softmax_f32`, `residual_layer_norm_f32` from existing kernels. Used `cublasSgemmStridedBatched` for multi-head attention.
- **Exact GELU**: Wav2Vec2 uses exact GELU (`erf`-based), not the tanh approximation Kokoro uses. New kernel.
- **CTC decode**: Simple CPU argmax → collapse repeats → remove blanks.

The weight loading uses the same pattern as Kokoro: binary file with config header → contiguous GPU allocation → pointer arithmetic for tensor assignment.

### Evolutionary Search (Phase 5)

Population-based search with four mutation operators:
1. **Substitute**: Replace phoneme with articulatorily nearby one (weighted by PanPhon distance matrix)
2. **Delete**: Remove one phoneme (handles over-segmentation)
3. **Insert**: Add phoneme near a neighbor (handles under-segmentation)
4. **Swap**: Transpose adjacent phonemes

Each candidate is: phonemize → tokenize → Kokoro synthesize → resample 24k→16k → Wav2Vec2 extract → weighted edit distance against original recording's phonemes.

Crossover uses uniform selection from two parent sequences. Elitism preserves top-k and always keeps the original seed.

### Build System (Phase 7)

New Makefile targets:
- `teacher`: Links teacher.cpp + rokoko_cuda.cpp (with `-DROKOKO_NO_MAIN`) + wav2vec2.o
- `src/wav2vec2.o`: NVCC compile of wav2vec2.cu
- `teacher-data`: Generates `data/phoneme_distances.bin` and `data/wav2vec2.bin`

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/teacher.cpp` | ~430 | CLI + evolutionary search |
| `src/teacher.h` | ~25 | Config/candidate structs |
| `src/wav2vec2.h` | ~140 | Wav2Vec2 weight struct + API |
| `src/wav2vec2.cu` | ~430 | CUDA kernels + forward pass |
| `src/wav_io.h` | ~220 | WAV reader + sinc resampler |
| `src/phoneme_distance.h` | ~120 | Distance matrix + weighted Levenshtein |
| `scripts/export/phoneme_distances.py` | ~170 | PanPhon → binary |
| `scripts/export/wav2vec2_weights.py` | ~260 | HuggingFace → binary |
