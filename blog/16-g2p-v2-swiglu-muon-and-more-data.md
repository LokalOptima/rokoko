## 2026-03-05: G2P v2 — SwiGLU, Muon, and More Data

Improving the CTC Transformer G2P model with architectural upgrades, a better optimizer, and 2x more training data.

### Data: 742K sentences (was 365K)

Extracted WikiText-103 via HuggingFace `datasets`, filtered to 20–300 char sentences, and deduplicated against the existing phonemized corpus. That yielded 164K new sentences.

Phonemization used 18 parallel workers running `misaki`/`kokoro`, finishing in 61 seconds (vs ~45 min serial). The final merged dataset: **742,470** unique text→phoneme pairs.

Pipeline script: `scripts/generate_g2p_parallel.py`

```
$ uv run python scripts/generate_g2p_parallel.py --full
Step 1: Extracting WikiText-103 sentences...
  Extracted 190,016 sentences
Step 2: Deduplicating against existing data...
  Existing sentences: 580,942
  New sentences: 164,906
Step 3: Parallel phonemization with 18 workers...
  All workers done in 60.9s
  Combined: 164,893 pairs
Step 4: Combining and cleaning all data...
  Total: 742,470 unique pairs → data/g2p_train_v2.tsv
```

### Model: SwiGLU FFN (replaces ReLU)

v1 used PyTorch's `TransformerEncoderLayer` with ReLU FFN. v2 uses custom pre-norm blocks with SwiGLU:

```
out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

Three projections instead of two — the gate and up projections are the same size (d→ff), then element-wise SiLU-gated multiplication, then down projection (ff→d). This adds ~30% more parameters for the same `d_ff` but converges faster and to lower loss.

| | v1 | v2 |
|---|---|---|
| FFN | ReLU | SwiGLU |
| d_model | 256 | 256 |
| d_ff | 1024 | 1024 |
| Params | ~4.1M | ~5.4M |
| Binary size | ~16 MB | ~22 MB |

### Optimizer: Muon for transformer weights

Instead of AdamW everywhere, we split parameters into two groups:

- **Muon** (lr=0.04): all 2D weight matrices in transformer blocks (20 tensors). Uses Newton-Schulz orthogonalization (5 steps) with Nesterov momentum (0.95).
- **AdamW** (lr=2e-3, weight_decay=0.1): embeddings, output head, biases, LayerNorm parameters (42 tensors).

Muon applies an approximate orthogonal projection to the gradient momentum buffer before each update step, which acts as a natural preconditioner for matrix-valued parameters.

### C++ inference: backward-compatible SwiGLU

The binary format already had an `activation` field in the header (was always 0 for ReLU). Now `activation=1` signals SwiGLU, and `g2p_model.h` reads three FFN weight matrices per layer (gate, up, down) instead of two (linear1, linear2).

The weight ordering follows PyTorch's `named_parameters()`:
- v1: qkv, out, ff1, ff2, norm1, norm2
- v2: norm1, qkv, out, norm2, gate, up, down

Existing v1 binaries load unchanged. We verified Python↔C++ output agreement on the same exported model:
```
Python: 'ɛlˈɛəɹˈɛɹləˈɛɹ'
C++:    'ɛlˈɛəɹˈɛɹləˈɛɹ'
```

### Early training results

| Epoch | Train Loss | Val Loss | PER | Exact Match |
|-------|-----------|----------|-----|-------------|
| 1 | 1.94 | 0.141 | 3.6% | 20.8% |
| 2 | 0.129 | 0.089 | 1.9% | 45.4% |
| 3 | 0.102 | 0.078 | 1.7% | 53.8% |
| 4 | 0.092 | 0.071 | 1.4% | 53.4% |

For comparison, v1 took ~200 epochs to reach 0.4% PER / 80% exact on its smaller dataset. v2 is at 1.4% PER after just 4 epochs — convergence is much faster. Training continues (300 epochs, ~6 min/epoch).
