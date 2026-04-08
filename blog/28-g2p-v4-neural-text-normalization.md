## 2026-03-09: G2P V4 — Neural Text Normalization

### The Problem

The G2P V3 model (4.45M param Conformer CTC) was trained only on Wikipedia/LibriTTS text. It handles plain English words well (87.6% exact match on general text) but completely fails on semiotic classes: times ("3:45 PM"), dates ("March 9, 2026"), money ("$12.50"), telephone numbers, measurements, Roman numerals, etc.

Evaluated on 1,484 gold test cases across 18 normalization classes:
- **V3 model: 10.3% exact match**
- **Misaki baseline: 23.7% exact match**
- TIME, TELEPHONE, ROMAN, RANGE, SCORE, ADDRESS: 0% for both

### Solution: Augmentation + Label Smoothing

**Scaled augmentation data from 8.7K to 31.8K pairs** covering all 13 semiotic classes programmatically. Each critical class now has 3K-8K training examples with diverse sentence frames and value ranges. Key expansions:
- DATE (1.5K → 7.8K): Added abbreviated months, dot-separated dates, month-day only
- MONEY (1.5K → 5.1K): Dense dollar/cents, comma-formatted, pounds, euros
- MEASURE (1K → 4.3K): Added 10+ new units (dB, cal, psi, rpm), decimals, percentages
- TIME (1.6K → 3.6K): Multiple templates per time, am/pm variants, bare times
- TELEPHONE (0.6K → 3.4K): Dot-separated format, 1-800 toll-free numbers

**Fixed a critical bug in eval_normalization.py**: The `NeuralG2P` class used wrong checkpoint key names (`model_state_dict` instead of `model`, wrong constructor parameter names). It would have failed to load any model.

**Added CTC label smoothing** to train.py: `loss = (1-α)*CTC + α*(-mean(log_probs))` with α=0.1. Prevents overconfident predictions and should help with normalization patterns where multiple outputs are plausible.

### Training Config
- Data: 745K pairs (713K V3 + 32K augmentation)
- Dropout: 0.15 (up from 0.1), label smoothing: 0.1
- Same architecture: 4.45M Conformer, RoPE + RMSNorm, no conv/QK-norm
- Muon + AdamW, 50 epochs, max_tokens=115,632

### V3 Baseline (for comparison)

| Class | V3 | Misaki | N |
|-------|-----|--------|---|
| ORDINAL | 87.5% | 96.4% | 56 |
| CARDINAL | 56.5% | 85.5% | 131 |
| ABBREVIATION | 15.0% | 18.3% | 60 |
| MEASURE | 3.2% | 0.0% | 216 |
| DATE | 2.7% | 2.7% | 148 |
| MONEY | 0.0% | 100.0% | 144 |
| TIME | 0.0% | 0.0% | 234 |
| TELEPHONE | 0.0% | 0.0% | 100 |
| **OVERALL** | **10.3%** | **23.7%** | **1484** |

### V4 Results (1x augmentation)

First attempt with 32K augmentation mixed 1:1 into 713K training data.

| Class | V3 | V4 | Misaki | N |
|-------|-----|-----|--------|---|
| TIME | 0.0% | **97.0%** | 0.0% | 234 |
| ORDINAL | 87.5% | **92.9%** | 96.4% | 56 |
| MEASURE | 3.2% | **72.7%** | 0.0% | 216 |
| FRACTION | 1.6% | **76.2%** | 0.0% | 63 |
| CARDINAL | 56.5% | 59.5% | 85.5% | 131 |
| ABBREVIATION | 15.0% | **48.3%** | 18.3% | 60 |
| TELEPHONE | 0.0% | **17.0%** | 0.0% | 100 |
| DATE | 2.7% | 2.0% | 2.7% | 148 |
| MONEY | 0.0% | 2.1% | **100.0%** | 144 |
| ROMAN | 0.0% | 0.7% | 0.0% | 141 |
| **OVERALL** | **10.3%** | **42.0%** | **23.7%** | **1484** |

**Problem**: MONEY/DATE/ROMAN still terrible. CTC stuttering on $ sign, model ignoring Roman numerals (only 463 examples in 745K = 0.06%).

**Diagnosis**: 32K augmentation was only 4.3% of total data — not enough signal for the model to learn rare patterns. Also missing bare formats (e.g. "1/1/2000" without sentence wrapper).

### V4b Results (5x oversampled augmentation + bare formats)

Added bare patterns (no sentence wrapper) for DATE, MONEY, TIME, TELEPHONE, ROMAN.
Oversampled augmentation 5x: 713K + 5×35K = 890K total, augmentation at ~20%.

| Class | V3 | V4 | V4b | Misaki | N |
|-------|-----|-----|------|--------|---|
| TIME | 0.0% | 97.0% | **98.3%** | 0.0% | 234 |
| ORDINAL | 87.5% | 92.9% | **94.6%** | 96.4% | 56 |
| MEASURE | 3.2% | 72.7% | **88.9%** | 0.0% | 216 |
| FRACTION | 1.6% | 76.2% | **79.4%** | 0.0% | 63 |
| CARDINAL | 56.5% | 59.5% | **77.9%** | 85.5% | 131 |
| ABBREVIATION | 15.0% | 48.3% | **63.3%** | 18.3% | 60 |
| ROMAN | 0.0% | 0.7% | **44.7%** | 0.0% | 141 |
| MONEY | 0.0% | 2.1% | **34.7%** | **100.0%** | 144 |
| TELEPHONE | 0.0% | 17.0% | **24.0%** | 0.0% | 100 |
| RANGE | 0.0% | 11.8% | **23.5%** | 0.0% | 17 |
| DATE | 2.7% | 2.0% | **12.8%** | 2.7% | 148 |
| ADDRESS | 0.0% | 0.0% | **9.0%** | 0.0% | 100 |
| SCORE | 0.0% | 6.2% | 6.2% | 0.0% | 16 |
| SYMBOL | 5.9% | 0.0% | 5.9% | 76.5% | 17 |
| LETTERS | 35.0% | 30.0% | 35.0% | 40.0% | 20 |
| CONNECTOR | 12.5% | 0.0% | 12.5% | 62.5% | 8 |
| DECIMAL | 0.0% | 0.0% | 0.0% | 0.0% | 6 |
| MIXED | 0.0% | 0.0% | 0.0% | 0.0% | 7 |
| **OVERALL** | **10.3%** | **42.0%** | **56.9%** | **23.7%** | **1484** |

### V5+: Rule-Based Date Preprocessing (84.6% overall)

After V5 training revealed that the CTC model fundamentally can't learn month-number→month-name lookup tables, added `preprocess_text()` to the inference pipeline. This function expands date patterns to fully spoken form *before* the neural model sees them:

- `1/1/2000` → `January first, two thousand`
- `2000-01-01` → `January first, two thousand`
- `January 1, 2000` → `January first, two thousand` (day ordinal + year expansion)

Implementation: ~60 lines in `train.py` with pure-Python ordinal and year-to-words functions (no num2words dependency). Handles years 1000-2099, days 1-31.

Also fixed ADDRESS test data: spoken forms now include trailing period (raw text "She lives at 8080 Elm St." ends with period, so spoken form should too).

| Class | V5 (before) | V5+ (after) | Delta |
|-------|-------------|-------------|-------|
| DATE | 14.9% | **100.0%** | +85.1 |
| ADDRESS | 0.0% (strict) | **84.0%** | +84.0 |
| **OVERALL** | **70.4%** | **84.6%** | **+14.2** |

All other classes unchanged. The 84.6% overall exceeds the plan.md target of 80%.

**Key insight**: A hybrid approach (rule-based preprocessing for structured patterns + neural model for everything else) is far more effective than trying to make the neural model learn lookup tables. The CTC model excels at sequence-to-sequence mappings with natural alignment (words → phonemes), but fails at arbitrary code lookups (month number → month name).

### V5 Results (LLM augmentation + quality filtering)

#### LLM data pipeline

Built `augment_with_llm.py`: uses Qwen3.5-9B via llama.cpp `/completion` endpoint (NOT `/v1/chat/completions` — Qwen3.5's thinking mode consumes all tokens on reasoning otherwise).

Pipeline: LLM generates diverse (written, spoken) pairs → quality filter (remove pairs with digits in spoken form, hallucinations) → Misaki phonemizes spoken forms.

- Generated 16,404 raw pairs across 14 semiotic classes in ~55 minutes
- After dedup: 9,662 unique pairs
- After quality filtering (digits in spoken form, length mismatches): 9,502 clean pairs
- 0 Misaki phonemization failures

Also fixed a `to_roman()` bug in `augment_data.py` (early `return` inside `for` loop).

#### Training data

| Source | Pairs | Notes |
|--------|-------|-------|
| g2p_train_v3.tsv | 713,500 | Base Wikipedia/LibriTTS |
| g2p_augment_v3.tsv × 5 | 176,530 | Programmatic augmentation (fixed to_roman) |
| g2p_augment_llm_clean.tsv × 5 | 47,510 | LLM-generated diverse data |
| **Total** | **937,540** | Augmentation at 23.9% of total |

Training: 50 epochs, dropout=0.15, label_smoothing=0.1, Muon+AdamW. Best val loss at epoch 43.

#### Results (V5+: with date preprocessing + fixed ADDRESS test data)

| Class | V4b | V5+ | Misaki | N | Delta |
|-------|------|------|--------|---|-------|
| DATE | 12.8% | **100.0%** | 2.7% | 148 | +87.2 |
| ROMAN | 44.7% | **100.0%** | 0.0% | 141 | +55.3 |
| TIME | 98.3% | **99.1%** | 0.0% | 234 | +0.8 |
| FRACTION | 79.4% | **93.7%** | 0.0% | 63 | +14.3 |
| MEASURE | 88.9% | **92.1%** | 0.0% | 216 | +3.2 |
| CARDINAL | 77.9% | **84.7%** | 85.5% | 131 | +6.8 |
| ADDRESS | 9.0% | **84.0%** | 0.0% | 100 | +75.0 |
| ORDINAL | 94.6% | 82.1% | 96.4% | 56 | -12.5 |
| SCORE | 6.2% | **75.0%** | 0.0% | 16 | +68.8 |
| ABBREVIATION | 63.3% | **71.7%** | 18.3% | 60 | +8.4 |
| TELEPHONE | 24.0% | **65.0%** | 0.0% | 100 | +41.0 |
| MONEY | 34.7% | **63.2%** | **100.0%** | 144 | +28.5 |
| RANGE | 23.5% | **47.1%** | 0.0% | 17 | +23.6 |
| SYMBOL | 5.9% | **35.3%** | 76.5% | 17 | +29.4 |
| **OVERALL** | **56.9%** | **84.6%** | **23.7%** | **1484** | **+27.7** |

Minor regression: ORDINAL dropped from 94.6% → 82.1%. May be due to competition with new augmentation data.

#### Still failing: numeric dates → SOLVED by preprocessing

The model produced **garbage** for numeric date formats: `12/25/2025` → gibberish phonemes. Textual dates (`December 25, 2025`) mostly worked. The CTC model can't learn the slash→month-name mapping — essentially a lookup table (1→January, 2→February... 12→December) — from training data alone.

**Fix**: Added `preprocess_text()` to expand dates to fully spoken English before the neural model sees them. Result: **DATE 14.9% → 100%**, overall **70.4% → 84.6%**.

#### Key insights

1. **LLM augmentation works**: Adding 9.5K diverse LLM-generated pairs (on top of 35K programmatic) pushed overall from 56.9% → 76.1%.
2. **Quality filtering matters**: 1.3% of LLM pairs had digits in spoken form. Filtering these prevented training on bad data.
3. **Qwen3.5 thinking trap**: The chat completions endpoint wastes all tokens on internal reasoning. Must use raw `/completion` endpoint to bypass thinking mode.
4. **to_roman bug impact**: The early-return bug meant Roman numerals > first matching value were wrong. Fixing this + LLM data → 100% ROMAN accuracy.
5. **Hybrid approach wins**: Rule-based preprocessing for structured patterns (dates) + neural model for everything else is far more effective than pure neural. The CTC model excels at natural sequence-to-sequence mappings but fails at arbitrary code lookups.

### Remaining weak spots

1. **MONEY (63.2%)**: Still 37% error rate. Model sometimes stutters on $ sign.
2. **ORDINAL regression (82.1%)**: Down from 94.6% in V4b. May need to oversample ordinals to compensate for augmentation competition.
3. **TELEPHONE (65.0%)**: Digit-by-digit reading has some failure modes.
4. **SYMBOL/CONNECTOR/LETTERS**: Small test sets but consistently weak. Need targeted augmentation or rule-based expansion.

### Key technical insights

1. **Oversampling is critical**: 5x augmentation improved overall from 42% → 57%. The model needs normalization patterns to be ~20% of training data, not 4%.
2. **Bare formats matter**: Adding "1/1/2000" without sentence wrappers helped DATE jump from 2% → 13%.
3. **CTC label smoothing**: At α=0.1, helped with generalization but wasn't the main driver of improvement.
4. **Docker /dev/shm limit**: 64MB shm caused multi-worker dataloading to crash. Auto-detection fix: limit to 2 workers when shm < 512MB.
5. **LLM data diversity**: Template-based augmentation creates repetitive patterns. LLM-generated sentences have natural variety in vocabulary, sentence structure, and context.
