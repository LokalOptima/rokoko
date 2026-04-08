## Fix G2P eval methodology + NaN guard (2026-03-11)

### Lenient eval for phoneme comparison

The eval script (`eval_full.py`) was comparing model output against Misaki reference character-by-character, penalizing differences in stress marks (ˈ vs ˌ), schwa variants (ᵊ vs ə), and punctuation. This is eval noise, not real errors — e.g., ADDRESS showed 0% exact match when the pronunciations were actually ~75% correct.

Added `normalize_phonemes()` that strips stress marks, normalizes schwa variants, and removes punctuation before comparison. The report now shows **both Strict% and Lenient%** columns side by side (standard G2P eval practice per CMU Sphinx, Reichel & Pfitzinger 2008):

```
Category              N   Strict%  Lenient%    PER
----------------------------------------------------
PLAIN              5000    96.4%     98.2%   0.08%
----------------------------------------------------
ADDRESS             100     0.0%     75.0%   5.34%
DATE                148     8.8%      8.8%  26.88%
```

The lenient metric lets us distinguish "real errors" (wrong phonemes) from "style differences" (stress placement, schwa reduction) that don't affect intelligibility.

### NaN guard in training

Added a NaN/Inf check after loss computation in `train.py`. When a bad batch produces NaN loss, the training loop now skips it (zeroes gradients, updates the AMP scaler to reduce scale) instead of corrupting all model weights. This prevents training collapse from rare degenerate batches.

### Files changed

| File | Changes |
|------|---------|
| `scripts/g2p/eval_full.py` | Added `normalize_phonemes()`, dual Strict/Lenient columns in report and JSON output |
| `scripts/g2p/train.py` | Added NaN/Inf loss guard before optimizer step |
