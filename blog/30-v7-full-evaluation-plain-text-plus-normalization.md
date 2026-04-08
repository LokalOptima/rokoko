## 2026-03-10: V7 — Full Evaluation (Plain Text + Normalization)

### What changed in V7

Added targeted training data for the zero-example classes identified in the V5+ error analysis:
- **DECIMAL**: 821 pairs ("3.14" → "three point one four")
- **CONNECTOR**: 137 pairs ("w/", "w/o", "24/7", "4x4")
- **SYMBOL**: 389 pairs (feet/inches, #, @, +, degree)
- **Dollar singular**: 175 pairs ($1.xx → "one dollar and...")
- **Currencies**: Expanded euro coverage to match dollar density, added ¥, CHF, ₹, and named currencies (yen, yuan, won, peso, ruble, krona, etc.)

Total augmentation: 41,312 pairs (up from 35,306), oversampled 5x. Training: 50 epochs on 967K total pairs.

Also added `best_exact.pt` checkpoint saving — the best val_loss checkpoint is dominated by plain text and doesn't optimize for normalization accuracy. V6 showed this clearly: best val_loss (epoch 26) = 74.4% on gold tests, while epoch 42 had 82.2% exact match on validation.

### Full evaluation: plain text + all normalization classes

Built `eval_full.py` to test both plain text pronunciation (5,000 samples from the val split of g2p_train_v3.tsv) and normalization (1,484 gold test cases) in one run. This is the first time we measured plain text accuracy alongside normalization.

| Category | N | Exact% | PER |
|----------|---|--------|-----|
| **PLAIN** | **5,000** | **83.3%** | **0.49%** |
| | | | |
| TIME | 234 | 100.0% | 0.00% |
| DATE | 148 | 99.3% | 0.06% |
| FRACTION | 63 | 93.7% | 0.41% |
| MEASURE | 216 | 91.2% | 0.41% |
| ROMAN | 141 | 90.1% | 0.84% |
| CARDINAL | 131 | 86.3% | 1.38% |
| ORDINAL | 56 | 85.7% | 0.73% |
| DECIMAL | 6 | 83.3% | 1.60% |
| ABBREVIATION | 60 | 73.3% | 2.86% |
| ADDRESS | 100 | 70.0% | 2.96% |
| MONEY | 144 | 68.8% | 3.30% |
| TELEPHONE | 100 | 66.0% | 2.27% |
| RANGE | 17 | 64.7% | 3.15% |
| SYMBOL | 17 | 64.7% | 1.35% |
| CONNECTOR | 8 | 62.5% | 6.66% |
| SCORE | 16 | 50.0% | 8.96% |
| LETTERS | 20 | 35.0% | 9.83% |
| MIXED | 7 | 14.3% | 14.18% |
| **NORM TOTAL** | **1,484** | **84.4%** | **1.49%** |
| **GRAND TOTAL** | **6,484** | **83.6%** | **0.72%** |

### Normalization: targeted augmentation worked but caused regressions

| Class | V5+ | V7 | Delta |
|-------|------|------|-------|
| **DECIMAL** | 16.7% | **83.3%** | **+66.7** |
| **CONNECTOR** | 12.5% | **62.5%** | **+50.0** |
| **SYMBOL** | 35.3% | **64.7%** | **+29.4** |
| **RANGE** | 47.1% | **64.7%** | **+17.6** |
| SCORE | 75.0% | 50.0% | **-25.0** |
| ADDRESS | 84.0% | 70.0% | **-14.0** |
| ROMAN | 100.0% | 90.1% | **-9.9** |

Gains in small classes (31 cases gained) offset by losses in larger ones (28 cases lost). Classic distribution shift / catastrophic forgetting from adding new augmentation data.

### Plain text: 83.3% exact, 0.49% PER

The 0.49% PER means the model gets >99.5% of phoneme characters right on plain English text. The 83.3% exact match is lower than V3 baseline's 87.8% — that's the cost of the normalization augmentation (the model traded some plain text precision for normalization ability).

Typical plain text errors:
- **Long sequences with multiple numbers**: ISBNs, "617 Dam Busters Squadron", dates + numbers in same sentence
- **CTC stuttering on 4-digit cardinals**: "1234" sometimes garbles ("wˈθn θˈnd" instead of "wˈʌn θˈWzᵊnd")
- **Wikipedia formatting artifacts**: Georgian characters, technical identifiers like "kfreebsd-i386"

### Checkpoint selection confirmed critical

The val_loss checkpoint (best.pt, epoch 33) scored 79.0% on normalization — 5.4 points worse than best_exact (epoch 42, 84.4%). The `best_exact.pt` addition was essential.

### Where this leaves us

The model does raw text → phonemes end-to-end at 83.6% exact / 0.72% PER across all categories. For context, the current C++ pipeline (`phonemize.cpp`, 3,051 lines) uses a 7-stage fallback chain: dictionary lookups → POS disambiguation → morphological stemming → number expansion → CMU/eSpeak fallback → neural G2P → letter spelling. It loads 6 data files and handles dozens of edge cases.

The neural model replaces all of that with a single forward pass, but at 83.6% exact match it's not yet reliable enough to be a drop-in replacement — dictionary lookup is ~100% precise for known words. The realistic deployment path is hybrid: neural text normalization (expanding numbers/dates/money to words) feeding into the existing dictionary pipeline for word pronunciation.

### The journey so far

| Version | Plain | Norm | What changed |
|---------|-------|------|-------------|
| V3 | 87.8% | 10.3% | Wikipedia/LibriTTS only |
| V4 | — | 42.0% | +8.7K augmentation |
| V4b | — | 56.9% | 5× oversampling |
| V5 | — | 76.1% | +LLM data, label smoothing |
| V5+ | — | 84.6% | +rule-based date preprocessing |
| **V7** | **83.3%** | **84.4%** | +DECIMAL/CONNECTOR/SYMBOL/currencies |
