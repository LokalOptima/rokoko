## 2026-03-10: V5+ Error Analysis — Exact Match is Misleading

### The Scores Look Awful. Are They?

After reaching 84.6% overall exact match, the remaining weak classes looked terrible:

| Class | Exact% | N | Looks like... |
|-------|--------|---|---------------|
| CONNECTOR | 12.5% | 8 | catastrophic |
| DECIMAL | 16.7% | 6 | catastrophic |
| LETTERS | 35.0% | 20 | bad |
| SYMBOL | 35.3% | 17 | bad |
| RANGE | 47.1% | 17 | mediocre |
| MONEY | 63.2% | 144 | mediocre |
| TELEPHONE | 65.0% | 100 | mediocre |

Deep dive into actual error patterns reveals **three distinct failure modes**:

### 1. "Almost correct" — functionally working (MONEY, TELEPHONE, RANGE)

PER distribution analysis showed most "errors" are 1-2 character differences:

| Class | Exact | PER<5% | Avg PER | Typical Error |
|-------|-------|--------|---------|---------------|
| MONEY | 63.2% | **84.7%** | 3.1% | "dollarˈ**z**" vs "dollar" (singular/plural) |
| TELEPHONE | 65% | **83%** | 2.1% | Missing space between digit phonemes |
| RANGE | 47.1% | **76.5%** | 3.0% | Dropped stress mark on one syllable |

These classes work. A TTS engine would produce identical audio for the model's output vs the expected. The MONEY error is systematic: the model defaults to "dollars" (plural) even for $1.xx amounts, because plural examples vastly outnumber singular in training data.

### 2. Test methodology issue (LETTERS)

Every LETTERS "error" is a spacing difference:
```
Expected (Misaki on "F B I"):  ˈɛf bˈi ˌI    ← spaces between letters
Model (on "FBI"):              ˌɛfbˌiˈI      ← same phonemes, no spaces
```

The eval pipeline phonemizes the spoken form "F B I" (with spaces between letters) → naturally gets spaces in output. But the model processes "FBI" as one token → no spaces. Both produce identical TTS audio.

### 3. Zero training data (DECIMAL, CONNECTOR, SYMBOL)

| Class | Training examples | Root cause |
|-------|-------------------|------------|
| DECIMAL | **0** "point" patterns | "3.14" → confused with dates/money |
| CONNECTOR | **0** "w/", "w/o", "24/7" | Never seen these patterns |
| SYMBOL | partial (@ works, 6'2" doesn't) | Missing feet/inches, +, degree |

You can't learn from zero examples regardless of model size. A 100M parameter model would fail just as badly — this is pure data coverage.

### Would a bigger model help?

**No.** The 4.45M param model already achieves:
- TIME: 99.1% (234 cases)
- DATE: 100% (with preprocessing)
- ROMAN: 100%
- FRACTION: 93.7%
- MEASURE: 92.1%

The model has plenty of capacity. The remaining errors come from missing training data (can't learn from nothing) and CTC alignment precision (1-2 phoneme boundary artifacts that more parameters won't fix). The ROI is in data, not architecture.

### Fix plan

1. Add DECIMAL/CONNECTOR augmentation to `augment_data.py` (zero→hundreds of examples)
2. Fix LETTERS test evaluation (strip spaces before comparison)
3. Oversample "dollar" singular for MONEY
4. Retrain V6
