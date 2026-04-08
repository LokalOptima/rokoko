## 2026-03-07: G2P V3 — Data Quality Crisis

### Ablation Study

We ran a 6-config ablation (10K samples, 10 epochs) to test the V3 architecture changes:

| Config | Params | Best Val Loss | PER | Exact |
|--------|--------|--------------|-----|-------|
| baseline (v2-style) | 4.75M | 0.1646 | 4.1% | 13.8% |
| **RoPE only** | **4.49M** | **0.0873** | **1.6%** | **38.4%** |
| RMSNorm only | 4.75M | 0.1599 | 3.9% | 14.6% |
| QK-Norm only | 4.75M | 0.1686 | 4.1% | 13.6% |
| Conv only | 5.58M | 0.1168 | 1.9% | 33.2% |
| full V3 | 5.31M | 0.1116 | 1.7% | 37.2% |

**RoPE** was the clear winner — 47% lower val loss, fewer params, no overfitting. Surprisingly, the full V3 config (everything on) was *worse* than RoPE alone, meaning the features interfere. ConvModule overfits after epoch 6. QK-Norm didn't help at all. RMSNorm is neutral (simpler than LayerNorm, worth keeping).

**Decision:** Keep RoPE + RMSNorm, drop QK-Norm + Conv.

### The Data Problem

We started a full 200-epoch training run, then decided to audit the data while it trained. What we found was ugly.

Our 742K training pairs come from WikiText-103 sentences phonemized by Misaki. Three contamination sources:

**1. WikiText `@-@` artifacts (16% of data, ~120K lines)**

WikiText-103 uses `@-@`, `@.@`, `@,@` as token-level separators. These aren't real text. Misaki phonemizes them literally:

```
"five @-@ star"  →  fˈIv ætæt stˈɑɹ   (should be: "five-star" → fˈIv stˈɑɹ)
"1 @.@ 4 billion" →  wˈʌn ætæt fˈɔɹ bˈɪljən  (should be: "1.4 billion")
```

The nonsense phoneme `ætæt` was the **9th most common token** in the dataset (216K occurrences). The model was spending significant capacity learning a sound that doesn't exist.

**2. Letter-spelling fallback (~2,748 lines)**

When Misaki can't phonemize a character (CJK, Cyrillic), it spells it out:
```
"いつだって" → "ʤˈæpənizlˌɛTəɹ" repeated 5 times
(literally "Japanese letter Japanese letter Japanese letter...")
```

**3. Non-Latin script (~1,900 lines)**

Sentences with CJK/Cyrillic/Arabic/Devanagari where Misaki applies English rules to non-English text. Pure noise.

### The Fix

We killed the running training and wrote `scripts/clean_g2p_data.py`:

1. Read all existing TSV data
2. Clean text: `@-@` → `-`, `@.@` → `.`, `@,@` → `,`
3. Filter: non-Latin script, letter-spelling fallback
4. Re-phonemize only the ~109K changed sentences (18 workers, 42s)
5. Keep ~468K clean sentences unchanged

Also patched `scripts/generate_g2p_parallel.py` to clean text during extraction, so this can't happen again.

**Result:** 742K → 576K pairs. Lost 22%, all garbage.

```
ætæt remaining: 0
Letter-spelling remaining: 0
Non-Latin remaining: 0
```

### Additional Data Sources

While cleaning, we researched what other text-to-IPA data we're missing:

- **LibriTTS** — 281K clean audiobook sentences designed for TTS. Phonemize with Misaki.
- **Common Voice English** — 2,500+ hours of diverse spoken text transcripts.
- **OLaPh** (2025) — 2.5M English G2P pairs from FineWeb corpus.

The low-hanging fruit is phonemizing LibriTTS/Common Voice transcripts with our existing pipeline. Better source text than Wikipedia, no tokenization artifacts.

### Restarting Training

Restarted from scratch on clean 576K data: `--no-qk-norm --no-conv --muon --compile --epochs 200`.
