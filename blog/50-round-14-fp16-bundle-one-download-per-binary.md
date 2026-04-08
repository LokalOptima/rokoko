## Round 14: FP16 Bundle (One Download per Binary)

**Problem**: The FP32 binary downloads a single self-contained `rokoko.bundle` (364 MB) with G2P + voices + weights. The FP16 binary downloads *two* files: the same `rokoko.bundle` for G2P/voices, plus a separate `weights.koko` (163 MB) for v2 weights. Two downloads, two files to manage, and the FP16 binary carries 327 MB of FP32 weights it never uses.

**Goal**: Each binary downloads exactly one bundle. FP32 gets `rokoko.bundle`, FP16 gets `rokoko.fp16.bundle`. Same UX, no wasted bandwidth.

### Changes

**`scripts/convert_v2.py`** — now produces a full ROKO bundle instead of a standalone `.koko` file:

1. `read_bundle_entries()`: extracts *all* entries from the source bundle (G2P, voices, weights) — not just the weights
2. Converts weights from v1 to v2 (same as before: weight norm, NHWC, padding, FP16, LSTM bias)
3. `write_koko_v2_bytes()`: builds the KOKO v2 weight blob in memory (was writing to disk)
4. `write_roko_bundle()`: packs v2 weights + G2P + voices into a ROKO bundle (16-byte header + 72-byte TOC entries + 256-byte-aligned data)

```
$ uv run scripts/convert_v2.py -o rokoko.fp16.bundle
Reading bundle: ~/.cache/rokoko/rokoko.bundle
  6 entries: g2p, voice/af_bella, voice/af_heart, voice/af_nicole, voice/af_sky, weights
  585 tensors, 163.4 MB KOKO v2 blob
  Done: 200.1 MB on disk
```

The FP16 bundle is 200 MB vs the FP32 bundle's 364 MB — 45% smaller because v2 weights are half precision.

**`src/main.cu`** — unified download logic:

The old code had two download blocks: a hardcoded `rokoko.bundle` download, then a conditional v2 weight download for FP16. Now there's one download block that calls `default_bundle_url()` / `default_bundle_filename()` — each backend provides its own URL and filename. The separate v2 weight download block is gone.

**`src/rokoko.cpp`** / **`src/rokoko_f16.cpp`** — renamed `default_weights_url/filename` → `default_bundle_url/filename`:

| Binary | `default_bundle_filename()` | `default_bundle_url()` |
|--------|---------------------------|----------------------|
| `rokoko` (FP32) | `rokoko.bundle` | `…/rokoko.bundle` |
| `rokoko.fp16` (FP16) | `rokoko.fp16.bundle` | `…/rokoko.fp16.bundle` |

### Bundle format (ROKO)

```
[4B "ROKO"] [4B version=1] [4B count] [4B padding]
[count × 72B TOC entries: 56B name + 8B offset + 8B size]
[padding to 4096]
[data: 256-byte aligned entries]
```

Both bundles use the same format. The only difference is the weights entry: v1 KOKO (FP32) vs v2 KOKO (FP16). G2P and voice entries are identical byte-for-byte copies.

### bench.sh Results

| Text | RTFx (median) | RTFx (p95) |
|------|--------------|------------|
| Short (1.6s) | 223x | 178x |
| Medium (5.7s) | 301x | 273x |
| Long (18.8s) | 354x | 348x |

STT 3/3 PASS. Performance unchanged from Round 13 — this was a packaging change, not a compute change.
