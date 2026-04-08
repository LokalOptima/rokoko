## 2026-03-07: G2P V3 — Data Expansion & Training Infrastructure

### Data Pipeline

Built a reproducible data pipeline for adding new text sources. All data is phonemized through Misaki only — no external G2P dictionaries.

**Pipeline structure:**
```
data/sources/{source}/00_raw/        → untouched downloads
data/sources/{source}/01_clean/      → cleaned text, 1 sentence/line
data/sources/{source}/02_phonemized/ → Misaki TSV (text\tphonemes)
```

**New data sources added:**

| Source | Raw utterances | Unique after dedup | Notes |
|--------|---------------|-------------------|-------|
| WikiText-103 | 576K | 575,719 | existing, cleaned |
| LibriTTS train-clean-100 | 30,078 | 30,077 | audiobook prose, CC BY 4.0 |
| LJ Speech | 13,005 | 3,242 | most overlapped with WikiText |
| **Total** | | **609,038** | +5.8% over previous 576K |

LibriTTS train-clean-360 (~200K+ more sentences) is available but not yet downloaded. Would bring total to ~800K+.

**Quality verification on merged data (609K pairs):**
- `@-@` artifacts: 0
- `ætæt` phonemes: 0
- Letter-spelling fallback: 0
- Non-Latin script: 0
- `❓` unknown markers: 0
- NUL bytes: 0
- Duplicate texts: 0

Scripts: `scripts/data/download_libritts.py`, `scripts/data/download_ljspeech.py`, `scripts/data/merge_all.py`.

### Length-Sorted Batching

Added `LengthSortedSampler` to `train.py`. Instead of random batching (where a 10-char and 200-char sentence in the same batch wastes 95% on padding), it:

1. Sorts all training examples by text length
2. Batches adjacent (similar-length) sequences together
3. Shuffles batch order each epoch (not within batches)

Standard technique from fairseq/ESPnet. Should reduce wasted compute from padding significantly.

### OOV Analysis

Ran Misaki without eSpeak fallback on a 2K sample of the 576K WikiText data to measure how many words are truly out-of-vocabulary:

- 55.8% of sentences had "OOV" — but almost all were contractions (`'ve`, `'re`) and abbreviations (`St.`, `Jr.`)
- **Only 0.3% of sentences had truly unknown words** (apostrophe-prefixed names like `'Malley`)
- Foreign names (Tchaikovsky, etc.) are in Misaki's lexicon, not OOV
- Conclusion: OOV is a non-issue. No filtering needed.
