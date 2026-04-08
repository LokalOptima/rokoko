## 2026-03-03: Phase 4 — C++ Phonemizer

### The Problem

The full Kokoro pipeline requires converting English text to IPA phoneme strings before feeding the model. The Python reference uses `misaki` (a G2P library) + `spacy` (NLP for POS tagging) + `espeak-ng` (fallback pronunciation). That's ~50 extra packages and a Python runtime just for text preprocessing.

We ported the entire pipeline to standalone C++.

### What We Built

A ~2850-line C++ phonemizer that replicates misaki's behavior:

- **Dictionary lookup**: Gold (90K entries) and silver (105K entries) pronunciation dictionaries exported to binary format
- **POS-aware pronunciation**: Words like "read" (present=ɹˈid, past=ɹˈɛd), "convicts" (noun=kˈɑnvˌɪkts, verb=kənvˈɪkts) get different pronunciations based on part-of-speech
- **Neural POS tagger**: Ported spacy's `en_core_web_sm` model to C++ (~300 lines in `src/pos_tagger.h`). 6 hash embeddings → maxout mixer → 4 CNN residual layers → softmax over 50 POS tags. ~6MB weights, pure CPU with AVX2.
- **Morphological stemming**: Handles -s, -ed, -ing suffixes with phonological rules (e.g., "voiced" → voiced fricative suffix)
- **Number-to-words**: Converts "42" → "forty two", "$3.50" → "three dollars and fifty cents", ordinals, years, phone numbers
- **Compound stress resolution**: Hyphenated words get primary/secondary stress distributed across parts
- **Espeak fallback**: ~500 entries for words not in gold/silver dictionaries
- **Chunking**: Splits output at 510 phoneme characters (Kokoro's context limit)

### Match Rate

Against the Python oracle on a 17,909-sentence expanded corpus:

| Corpus | Match |
|--------|-------|
| Original (6,997 sentences) | **6,997/6,997 = 100.0%** |
| Expanded (14,554 testable) | **14,518/14,554 = 99.8%** |

The 36 remaining mismatches are POS tagger disagreements (our C++ POS tagger vs spacy disagree on a few edge cases), tokenization differences, and ground truth inconsistencies.

### The PhonoGlyphe Detour

We tried integrating PhonoGlyphe, a small neural G2P model, as a fallback for unknown words. The model was an encoder-decoder transformer (256-dim, 8 encoder + 3 decoder blocks). After porting it to a ~1000-line C++ header and testing:

**15% word accuracy.** Completely unusable. Reverted immediately.

### The Special Cases Problem

Getting from 99.4% to 99.8% required an enormous pile of special cases:
- "that" stress depends on whether it's a determiner (ðˈæt) or conjunction (ðæt), with ~20 context-dependent overrides
- "non-" prefix needs supplementary dictionary entry to avoid NNP letter-spelling
- "re-" prefix compounds need special handling to avoid merged dictionary lookup
- "US" (the abbreviation) vs "us" (the pronoun)
- "Harland's" needs an explicit possessive entry because suffix_s produces "z" after "d" but the expected form uses literal "s"

Each fix risks breaking something else. At 99.8%, the diminishing returns are severe — every remaining mismatch is a corner case that requires understanding the interaction between POS tagging, dictionary lookup, stress assignment, and morphological rules.

### Files

| File | Lines | Description |
|------|-------|-------------|
| `src/phonemize.h` | 126 | Public API |
| `src/phonemize.cpp` | ~2850 | Full G2P engine |
| `src/phonemize_main.cpp` | 86 | Standalone CLI |
| `src/pos_tagger.h` | ~300 | Neural POS tagger |
| `scripts/export_phonemizer_data.py` | — | Dict export |
| `scripts/export_pos_tagger.py` | — | POS model export |

Build: `make phonemize` (no CUDA needed, just g++ with `-mavx2 -mfma`)
