## 2026-03-09: G2P V3 — C++ Port and Verification

### Goal

Port the V3 Conformer model (RMSNorm + RoPE + SwiGLU) from Python to C++ so it can run in the production phonemizer pipeline. The existing `g2p_model.h` only supported the old G2P2 format (LayerNorm + learned positional embeddings).

### Binary format: G2P3 with feature flags

The V3 model has several optional features (RoPE, QK-Norm, ConvModule, RMSNorm) that can be toggled for ablation. The best checkpoint was trained with `--no-qk-norm --no-conv`, so only RoPE and RMSNorm are active. The binary loader needs to know which features are present to read weights correctly.

Solution: encode feature flags as a bitfield in the header's reserved field:

```
bit 0: use_rope       (RoPE vs learned pos embeddings)
bit 1: use_qk_norm    (RMSNorm on Q,K before attention)
bit 2: use_conv       (Conformer ConvModule between attention and FFN)
bit 3: use_rmsnorm    (RMSNorm vs LayerNorm)
```

Current model: flags = `0b1001` = 9 (rope + rmsnorm, no qk_norm, no conv). This means each layer is just: RMSNorm → MHSA(RoPE) → residual → RMSNorm → SwiGLU → residual. No ConvModule weights, no QK-Norm weights — the loader skips those reads.

### C++ implementation challenges

**RMSNorm vs LayerNorm**: LayerNorm subtracts the mean then divides by stddev, with separate weight and bias vectors. RMSNorm has no bias and no mean subtraction — just `x * weight / sqrt(mean(x²) + eps)`. The existing `layer_norm()` was ~4 lines; added `rms_norm()` alongside it and a `norm()` dispatcher that checks the `use_rmsnorm_` flag.

**RoPE (Rotary Position Embeddings)**: The old V2 model added a learned `pos_emb[t]` vector to each input embedding. RoPE is completely different — it rotates pairs of Q,K dimensions by position-dependent angles at every attention layer, encoding relative position through the rotation.

Implementation: precompute sin/cos tables at model load time (`freq[t,i] = t / 10000^(2i/head_dim)`), then apply rotation to Q and K after projection but before attention scores. Each head's dimensions are split in half: `(q1, q2) → (q1·cos - q2·sin, q2·cos + q1·sin)`. Had to be careful about the dimension ordering — the Python `apply_rope` splits along the last dim (`x[..., :d2]` and `x[..., d2:]`), which maps to interleaving across heads in the flat C++ layout.

**Weight reading order**: G2P2 and G2P3 have completely different `named_parameters()` orderings. G2P2 reads: norm1_w, norm1_b, qkv_w, qkv_b, out_w, out_b, norm2_w, norm2_b, gate_w, ... G2P3 reads: norm1_w (no bias), qkv_w, qkv_b, out_proj_w, out_proj_b, norm_ffn_w (no bias), gate_w, ... Getting this wrong causes silent corruption (model loads but produces garbage). The manifest JSON from the exporter was essential for debugging this.

### Verification: 100% match

Built a match verifier (`scripts/g2p/verify_match.py`) that runs the same inputs through both Python and C++, comparing phoneme outputs character-by-character.

```
Testing 1118 inputs...
Results: 1118/1118 match (100.0%)
PERFECT MATCH — C++ output identical to Python for all inputs.
```

The 1118 inputs include the 117 hand-crafted test cases (proper nouns, silent letters, contractions, morphology) plus 1001 sentences sampled from the training data.

### Benchmark: ~210x real-time

```
Single words:  ~2ms/word   (485 words/s)   — test_cases.txt, avg ~7 chars
Sentences:     ~32ms/sent  (32 sentences/s) — training data, avg ~101 chars
RTFx:          ~210x real-time (6.7s audio / 32ms phonemize)
```

This is scalar C++ with AVX2 only on the gemv inner loop. Attention is still fully scalar O(T²). For a TTS system running at 5-20x real-time, the phonemizer at 210x is not the bottleneck.

### Files changed

- `src/g2p_model.h` — G2P3 loader: RMSNorm, RoPE, feature flags, backward-compatible with G2P2
- `src/test_g2p.cpp` — standalone test binary with benchmark mode
- `scripts/g2p/train.py` — export writes feature flags in header
- `scripts/g2p/verify_match.py` — Python↔C++ match verifier
- `Makefile` — `test_g2p`, `verify-g2p`, `data/g2p_v3_model.bin` targets
