## 2026-03-08: G2P — Input Charset Normalization & Data Cleanup

### The problem: 589-character input vocabulary

The G2P model's input vocabulary had grown to 589 Unicode characters — built at runtime from whatever appeared in the training data. This included Armenian, Georgian, Greek letters, Bengali script, control characters, zero-width joiners, and other noise from LibriTTS/WikiText extraction. Only 0.08% of input data was non-ASCII, but the model was allocating embedding parameters for all 589 chars.

### Research: how do established TTS systems handle this?

Surveyed espeak-ng, Piper TTS, Coqui TTS, Amazon Polly, Google Cloud TTS, and OpenAI's voice pipeline. The consensus is clear:

- **Every English TTS system normalizes input to ASCII** (or near-ASCII). Foreign words are handled by a lexicon lookup layer *above* the G2P model, not by teaching the G2P model Unicode.
- **SSML `<phoneme>` tags** provide IPA overrides for names/loanwords — highest priority, bypasses G2P entirely.
- **Custom lexicons** (Amazon Polly PLS, Azure) map word→IPA before G2P runs.
- The **G2P model is the last resort**, handling only the target language's native charset.

### The fix: fixed ASCII vocabulary + `normalize_text()`

Replaced the runtime-built vocabulary with a fixed 96-entry charset: all 95 printable ASCII characters (space through `~`) plus PAD (ID 0). Deterministic IDs across runs — the vocab is a constant in the code, not derived from data.

Added `normalize_text()` to the data loading pipeline:
1. **NFC compose** — handles badly-formed combining marks
2. **`unidecode` transliteration** — `café→cafe`, `Dvořák→Dvorak`, `æ→ae`, `ß→ss`
3. **Drop non-printable-ASCII** — anything that survives steps 1-2 and isn't in the fixed vocab

Applied automatically in `load_tsv()`. Result: only 3 pairs dropped out of 575K (0.0005%). The normalization saves almost everything — most non-ASCII was smart punctuation (en/em dashes, curly quotes) that `unidecode` maps cleanly.

### Data cleanup

Deleted ~750 MB of intermediate/legacy data files. Renamed `g2p_train_v3_plus.tsv` → `g2p_train_v3.tsv` as the single canonical training file. Documented data versions (v1/v2/v3) and model versions separately in `docs/g2p_versions.md`.

### Final data stats

| Metric | Value |
|--------|-------|
| Training pairs | 713,497 |
| Input chars | 83.5M |
| Char vocab | 96 (fixed ASCII) |
| Phone vocab | 77 (IPA) |
| Avg sentence length | 121 chars / 23 words |
| File size | 203 MB |
