## 2026-03-09: G2P V3 — Training Run & Production Readiness

### Training results

Fresh training from scratch on the full v3 dataset (713K pairs) with ASCII-normalized input, fixed 96-char vocab, and ablation-recommended config (`--no-qk-norm --no-conv`, keeping RoPE + RMSNorm).

**Command:**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/g2p/train.py train \
  --data data/g2p_train_v3.tsv --muon --compile --max-tokens 115632 --epochs 50 \
  --no-qk-norm --no-conv
```

| Metric | V1 baseline | Previous V3 (9 ep, 575K) | **New V3 (50 ep, 713K)** |
|--------|------------|--------------------------|--------------------------|
| Val loss | 0.061 | 0.021 | **0.016** |
| PER | 1.1% | 0.6% | **0.3–0.5%** |
| Exact match | 66.4% | 75.4% | **87.8%** |
| Params | 1.2M | 5.3M | **4.5M** |
| Time/epoch | — | — | **70s** |

The model converged around epoch 38–41 (best checkpoint: epoch 41, val_loss=0.0161). Training loss continued dropping (0.011→0.009) while val loss flatlined at 0.016x — classic early overfitting. The train-val gap of 0.007 was widening, confirming no benefit from further epochs.

### Auto-batch probe pitfall

`--auto-batch` probes GPU memory by running forward+backward passes to find the max `max_tokens`. But the probe runs on the **uncompiled** model (before `torch.compile`), which uses far more memory than the compiled version. First attempt found `max_tokens=18,321` (93 samples × 197 seq_len) — only 2.1 GB of 16 GB VRAM, 13% utilization.

Fix: used `--max-tokens 115632` directly from the dataloader tuner. This gave 652 batches/epoch (vs 4,121), 11.5 GB VRAM, 97% GPU utilization, and 70s/epoch (vs 90s+ with the conservative probe).

### Does the model work as a production phonemizer?

Tested the trained model on text normalization edge cases (numbers, currency, dates, abbreviations) by feeding raw text directly:

| Input | Output | Verdict |
|-------|--------|---------|
| `Dr. Smith earned 2.5M` | "Doctor Smith earned two point five million" | Mostly correct |
| `72 degrees` | "seventy two degrees" | Perfect |
| `$42 for 3 tickets` | Garbled on "dollars" | Partial |
| `35,000 feet` | Garbled | Failed |
| `3:30 PM` | Garbled | Failed |

The model has partially learned text normalization from seeing Misaki's end-to-end mappings in the training data (Wikipedia has plenty of numbers and dates). Simple numbers and common abbreviations work, but comma-separated numbers, time formats, and ordinal dates fail — not enough training examples.

### The realization: we already have a text normalizer

The C++ phonemizer (`src/phonemize.cpp`) already has:
- **Number-to-words**: `num_to_words()`, ordinals, currency ($, £, €)
- **Dictionary lookup**: gold/silver/user dictionaries with priority chain
- **Morphological stemming**: `-s`, `-ed`, `-ing` suffix rules
- **Abbreviation handling**: merges tokens like "U.S.A."

The neural G2P model sits at slot 6 in the existing fallback chain:
```
User dict → Gold dict → Silver dict → POS lookup → Espeak → Neural G2P → Letter spelling
```

So the model doesn't need to do text normalization — the C++ pipeline handles that before the model ever sees the text. The garbled outputs from testing were from bypassing the normalizer entirely, which isn't how the model is used in production.

The production architecture is already correct:
```
Raw text → C++ normalizer (numbers, dates, currency, abbreviations)
  → Dictionary lookup (known words, heteronyms, proper nouns)
  → Neural G2P model (everything else)
  → TTS synthesis
```

The remaining work is ensuring the neural G2P model is good enough to **replace espeak** in the fallback chain (slot 5), eliminating the last Python/system dependency.
