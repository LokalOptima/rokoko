## 2026-03-08: G2P V3 — Token Batching & Training Infrastructure

### Token Batching

Replaced fixed-size batching (`LengthSortedSampler`) with `TokenBatchSampler` — a batch sampler that keeps `batch_size * max_seq_len_in_batch ≈ max_tokens`, so the total number of tokens per batch is roughly constant regardless of sequence length.

**The problem with fixed batch size:** The length-sorted sampler groups similar-length sequences together, but every batch has the same number of samples. This means short-sequence batches (e.g., 30-char sentences × 584 samples) underutilize the GPU, while long-sequence batches (300-char sentences × 584 samples) risk OOM. The VRAM probe finds the max batch size at p90 sequence length, so the longest 10% of batches can still OOM.

**The fix:** `TokenBatchSampler` takes a `max_tokens` budget. For each batch of length-sorted sequences, it accumulates samples until `n_samples * max_len_in_batch > max_tokens`, then starts a new batch. Short sequences get huge batches (up to ~6,400 samples); long sequences get small batches (~190 samples). VRAM usage is consistent across all batches.

The `--auto-batch` flag now probes GPU memory at p90 sequence length, finds the max batch size, and computes `max_tokens = batch_size × seq_len`. This can also be set directly with `--max-tokens`.

### Benchmark: 1.69x speedup

Apples-to-apples on RTX 5070 Ti with the same model and data:

| | Fixed batch (old) | Token batch (new) |
|---|---|---|
| Batch size | 467 fixed | 190–6,424 variable |
| Throughput | 999 samples/s | 1,684 samples/s |
| VRAM peak | 81.5% | 86.8% |
| Batches/epoch | 1,233 | 607 |

The speedup comes from short-sequence batches being much larger — better GPU utilization since short sequences are cheap. The old approach was limited to 467 (80% of the probed 584) as a safety margin for long batches; token batching eliminates this entirely.

### AMP-friendly RMSNorm

PyTorch's `nn.RMSNorm` keeps its weight in float32, but under `torch.amp.autocast` the input arrives as float16. The fused kernel requires matching dtypes, so PyTorch silently falls back to the non-fused path with a warning. Replaced with a custom `RMSNorm` that casts `weight.to(x.dtype)` in the forward pass — enables the fused kernel and eliminates the warning.

### Dataloader pre-encoding

`G2PDataset.__init__` now pre-encodes all text/phoneme pairs into integer tensors at startup (was re-encoding every `__getitem__` call). With 575K pairs, this makes `__getitem__` a simple index lookup. Combined with token batching, even 1 dataloader worker shows 0.0% GPU data-wait time.

### Bug fix: checkpoint config keys

The `tune_dataloader.py` and `bench_dataloader.py` scripts were reading `ckpt.get("args", {})` but the checkpoint saves under `"config"`. They also used wrong key names (`"d_model"` instead of `"d"`, `"nhead"` instead of `"heads"`, etc.). This only worked by accident because the defaults matched the trained model's hyperparameters. Fixed to read the correct keys.

### Files changed

| File | Change |
|------|--------|
| `scripts/g2p/train.py` | `TokenBatchSampler`, `find_max_tokens`, `--max-tokens`, `--auto-batch`, pre-encoded dataset, data% logging |
| `scripts/g2p/model.py` | AMP-friendly `RMSNorm` replacing `nn.RMSNorm` |
| `scripts/g2p/tune_dataloader.py` | New — finds optimal `max_tokens` and `num_workers` |
| `scripts/g2p/bench_dataloader.py` | New — benchmarks dataloader throughput with token batching |
