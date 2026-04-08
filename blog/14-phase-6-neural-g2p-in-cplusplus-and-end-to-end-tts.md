## 2026-03-05: Phase 6 — Neural G2P in C++ & End-to-End TTS

### C++ Neural G2P Inference

We wrote a single-header C++ CTC Transformer inference engine (`src/g2p_model.h`, ~400 lines) that loads the exported 4M-param model and runs inference on unknown words. No ggml, no ONNX — just hand-rolled scalar C++ with optional AVX2.

The implementation:
- **Weight loading**: Reads the G2P2 binary format (magic + 9-field header + counted vocab + float32 weights in PyTorch `named_parameters()` order)
- **Pre-norm Transformer**: LayerNorm before attention and FFN (`x' = x + attn(LN(x))`, `x' = x' + ffn(LN2(x'))`)
- **CTC decode**: argmax → collapse repeats → remove blank(0) → lookup phone chars → UTF-8 encode
- **AVX2 GEMV**: 16-wide FMA unrolling with `_mm256_fmadd_ps`, ~5x faster than scalar

**Performance**: ~2ms per unknown word on CPU (AVX2). This is a fallback path — 95%+ of words hit the dictionary in <1μs.

### Integration Strategy: Dict First, Neural Second

The lookup chain in `get_word()`:
```
gold dict → silver dict → morphological stemming → supplementary dicts → espeak fallback → neural G2P → NNP letter spelling
```

Neural G2P fires only for truly unknown words (not in any dictionary, can't be stemmed, no espeak entry). This preserves the 99.8% match rate on known text while giving sensible pronunciations for novel words like "Blorfinator" or "Zylork".

Verified: Python and C++ produce identical G2P output for the same input words. Full corpus test: **99.7% match, 0 new regressions** from G2P integration.

### CMU Pronouncing Dictionary

We imported 125,889 entries from CMU (the standard English pronunciation dictionary, ARPAbet format) by mapping ARPAbet phonemes to our IPA/misaki notation:

**Key mappings**: `AA` → `ɑ`, `AE` → `æ`, `AH0` → `ə`, `AY` → `I` (misaki diphthong shorthand for aɪ), `ER` → `ɜɹ`, `R` → `ɹ`, `G` → `ɡ`, etc. 39 phoneme symbols mapped, diphthongs use misaki's uppercase shorthands (I=aɪ, A=eɪ, O=oʊ, W=aʊ, Y=ɔɪ).

After filtering entries that overlap with gold/silver and excluding abbreviations/contractions/possessives that conflict with our morphological rules: **59,630 new entries** in `data/cmu_extra.bin` (1.3 MB). Loaded automatically as a supplementary dictionary — checked after stemming, before espeak.

**Result: 0 regressions** on the full corpus. CMU only fires for words not already handled by gold/silver/stemming.

### The Stress Mark Bug

First end-to-end test: `"Hello, I'm Jarvis"` — the model clearly said "ajarvis" instead of "Jarvis". The phonemes looked fine at first glance: `ˈʤɑɹvəs`. But comparing with Python misaki's output: `ʤˈɑɹvɪs`.

The difference: **stress mark position**. Our CMU converter was placing the IPA stress mark before the syllable onset (`ˈʤɑ` — stress, then consonant, then vowel), but Kokoro's model was trained with misaki's convention where stress goes immediately before the vowel nucleus (`ʤˈɑ` — consonant, then stress, then vowel).

In ARPAbet, stress is marked on the vowel: `JH AA1 R V AH0 S` — the `1` on `AA` means primary stress. Standard IPA places the stress mark at the beginning of the syllable (before onset consonants), but misaki doesn't follow standard IPA — it places stress marks right before the vowel. Since Kokoro was trained on misaki phonemes, it interprets `ˈʤ` as something like a stressed consonant and inserts a spurious schwa-like sound, producing the "a" in "ajarvis".

The fix was simple: in the ARPAbet→IPA converter, emit the stress mark (`ˈ`/`ˌ`) as a prefix on the vowel phoneme only, not on the preceding consonant cluster. Written as a proper script (`scripts/export_cmu.py`) to make the conversion reproducible.

### End-to-End Text-to-Speech

The `./kokoro` binary now supports a `--text` flag that does everything inline — no Python, no pre-phonemization step:

```bash
./kokoro --text "Hello, I'm Jarvis and I'm your new home assistant" --voice af_heart
```

This:
1. Loads the C++ phonemizer (gold + silver + POS tagger + espeak + neural G2P + CMU) in ~114ms
2. Phonemizes the text to IPA: `həlˈO, ˌIm ʤˈɑɹvəs ænd ˌIm jʊɹ nˈu hˈOm əsˈɪstᵊnt`
3. Converts phonemes to token IDs (178-symbol vocab)
4. Loads the voice pack and extracts the style vector
5. Runs the full Kokoro-82M model on GPU
6. Writes a WAV file

Multi-chunk text is handled automatically — long inputs get split at 510 phoneme characters and each chunk is synthesized separately, with audio concatenated.

Piping directly to speakers:
```bash
./kokoro --text "Hello world" --voice af_heart --stdout | aplay -
```

The `--stdout` flag writes WAV to stdout (all status goes to stderr), so you can pipe it straight to `aplay`, `paplay`, or any audio player.

**Performance**: 3.4 seconds of audio generated in 134ms (25x realtime) on RTX 5070 Ti, including phonemizer load time. The entire pipeline from English text to WAV is a single binary with zero Python dependencies.

### The Full Stack

What started as "port PyTorch inference to CUDA" became a complete, self-contained TTS system:

| Component | Lines | Description |
|-----------|-------|-------------|
| `src/kokoro_cuda.cpp` | ~1700 | CUDA inference + end-to-end TTS |
| `src/kokoro.cpp` | ~520 | Weight loading |
| `src/kernels.cu` | ~1470 | 24 custom CUDA kernels |
| `src/phonemize.cpp` | ~2940 | Full G2P engine |
| `src/pos_tagger.h` | ~390 | Neural POS tagger |
| `src/g2p_model.h` | ~420 | Neural G2P (CTC Transformer) |

Total: ~7,440 lines of C++/CUDA. No Python. No cuDNN. No espeak-ng process. No spacy. No misaki. Just `g++`, `nvcc`, and the CUDA runtime.

```
Text in → [phonemizer (CPU)] → [Kokoro-82M (GPU)] → WAV out
  114ms                            134ms
```
