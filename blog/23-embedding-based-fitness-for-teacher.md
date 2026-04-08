## 2026-03-08: Embedding-Based Fitness for Teacher

### The Problem

The teacher's evolutionary search scored candidates by: synthesize audio → Wav2Vec2 → CTC argmax → phoneme strings → weighted edit distance. But CTC argmax collapses the rich 1024-dimensional transformer embeddings down to a single phoneme ID per timestep, then collapses repeats further. Two sequences that sound nearly identical can differ by a phoneme or two due to argmax boundary effects.

### Solution: DTW on Wav2Vec2 Embeddings

The Wav2Vec2 transformer already produces `[T, 1024]` embedding vectors (`d_final_ln`) that capture vowel quality, formant structure, and prosody — far richer than any discrete phoneme representation. We just needed to *stop throwing them away*.

**Refactored `wav2vec2.cu`**: Extracted the forward pass (audio normalization → CNN → feature projection → positional conv → 24-layer transformer → final LayerNorm) into a shared `wav2vec2_forward()` helper returning `{d_final_ln, T, D}` as GPU pointers. Then:
- `wav2vec2_extract_phonemes()` = forward + CTC head + argmax decode (unchanged behavior)
- `wav2vec2_extract_embeddings()` = forward + D2H copy of `d_final_ln` (new)

No code duplication — both functions call the same ~200 line forward pass.

**DTW with cosine distance** (`embedding_dtw_distance()` in teacher.cpp): Standard Dynamic Time Warping handles the different sequence lengths between original recording and Kokoro synthesis (e.g., 50 vs 70 timesteps for the same word). Local cost is cosine distance: `1 - (a·b)/(|a|·|b|)`. Uses two-row DP (O(T) memory instead of O(T^2)). Normalized by path length for comparability across words.

**Combined fitness**: `-(0.7 × dtw_dist + 0.3 × edit_dist)`. Embeddings dominate (capture fine acoustic detail) while edit distance acts as a regularizer (more robust across different speakers/voices).

### Test Results

Ran the same tests as before (`vieques`, `kubernetes`) with `--generations 15 --population 30`:

| Word | Seed | Best candidate | Fitness | Early stop |
|------|------|----------------|---------|------------|
| vieques | biaki | biakæ | -0.367 | Gen 5 |
| kubernetes | kjuːbɚniːdz | kjuːbɚnniːds | -0.201 | Gen 5 |

Key observation: convergence is dramatically faster. Both tests early-stop at generation 5 (out of 15). The embedding DTW provides a much smoother fitness landscape than discrete phoneme edit distance — small phoneme changes produce proportional fitness changes instead of all-or-nothing phoneme mismatches. The top-3 candidates are extremely close in fitness (within 0.001), suggesting the search is finding a genuine optimum rather than random exploration.

Tradeoff: the smoother landscape means faster convergence but potentially getting stuck in local optima sooner. The old edit-distance approach explored more because its fitness was noisier. Could increase patience if more exploration is needed.

### Challenges

The main design choice was *not* doing something more complex. We considered:
- Averaging embeddings and comparing means (loses temporal structure)
- Euclidean distance instead of cosine (sensitive to magnitude, not just direction)
- Full O(T^2) DTW matrix (unnecessary — two-row DP works fine since we only need the final cost)
- A learned metric (overkill for this use case)

Memory is trivial: ~400 KB for 1-second embeddings, ~160 KB for the DTW working set, vs 1.2 GB model weights.

### Files Changed

| File | Change |
|------|--------|
| `src/wav2vec2.h` | Added `Wav2Vec2Embeddings` struct + `wav2vec2_extract_embeddings()` |
| `src/wav2vec2.cu` | Extracted `wav2vec2_forward()` helper; new `wav2vec2_extract_embeddings()` |
| `src/teacher.cpp` | Added `embedding_dtw_distance()`; combined fitness in eval loop |
