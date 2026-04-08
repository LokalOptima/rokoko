## 2026-04-04: Split Bundle into Separate Files

### Motivation

The ROKO bundle format packed weights, G2P model, and 4 voice packs into a single archive (~364 MB FP32, ~200 MB FP16). This was good for "one download, it just works" UX but added friction in practice:

- **Duplicated data**: FP32 and FP16 bundles each contained identical copies of the G2P model (~33 MB) and voices (~2 MB). Users who built both binaries downloaded 35 MB twice.
- **Atomic iteration**: Changing just the G2P model required rebuilding the entire bundle. Same for adding a voice.
- **Conversion complexity**: `convert_v2.py` had to parse the bundle, extract non-weight entries, convert weights, then repack everything. Half the script was bundle I/O, not weight conversion.

### The Split

Replaced the single `.bundle` file with individual files:

```
~/.cache/rokoko/
  weights.bin         (FP32, 327 MB)  — or weights.fp16.bin (FP16, 163 MB)
  g2p.bin             (33 MB, shared)
  voices/
    af_heart.bin      (510 KB)
    af_bella.bin      (510 KB)
    af_nicole.bin     (510 KB)
    af_sky.bin        (510 KB)
```

G2P and voices are shared between FP32/FP16 binaries — downloaded once, used by both.

### Download & Integrity

Each file is auto-downloaded from GitHub releases on first run. A single `RELEASE_BASE` URL constant constructs all download URLs. Each backend (FP32/FP16) only provides its weight filename; everything else is shared.

**Size verification**: Every cached file is checked against its expected byte size on startup. If a file is truncated or corrupt (wrong size), it's automatically deleted and re-downloaded. This catches the main failure mode — partial downloads from interrupted connections or disk-full conditions. `curl -f` also catches HTTP 4xx/5xx errors that would otherwise produce a valid-looking HTML error page.

### G2P Load Overlap

Previously, G2P loading was sequential with weight norm precomputation:

```
upload(stream) → precompute_weight_norms(stream) → g2p.load(stream)
                 [GPU busy]                         [CPU fread + GPU memcpy]
```

Now G2P loads on a separate CUDA stream in a background thread, overlapping with GPU work:

```
upload(stream) → precompute_weight_norms(stream)   [GPU: norm kernels]
               → g2p.load(g2p_stream)              [CPU: fread 33MB, GPU: memcpy]
               → join + destroy g2p_stream
```

The fread of 33 MB overlaps with the GPU kernel launches for weight normalization. Measured 25ms init improvement on warm page cache (228ms vs 253ms). On cold page cache the savings are larger since the disk I/O dominates.

### Code Cleanup

During review, several issues were caught and fixed:

- **Dead code removed**: `Weights::prefetch(const void*, size_t)` and `G2PModelCuda::load(const void*, size_t)` from-memory overloads were only used by the bundle path. Deleted.
- **dtype string mismatch**: The KOKO v2 writer emitted `"float16"` but the convention was `"fp16"`. Fixed for consistency — the C++ reader treats anything non-`"fp32"` as FP16 so it worked by accident, but would break if the reader ever did an exact string match.
- **`NUM_VOICES` constant**: Replaced with `std::size(VOICE_NAMES)` so adding a voice to the array can't desync from the loop bound.
- **Zero-length file guard**: `load_voices()` now skips empty files (which would cause `mmap(nullptr, 0, ...)` — POSIX undefined behavior on Linux).
- **Silent failures**: `load_voices()` now prints warnings when open/mmap fails, instead of silently skipping.
- **Voice mmap leak**: Added cleanup in both server and CLI exit paths.
- **`convert_v2.py` simplified**: Reads standalone `.koko` file directly (was: parse bundle → extract weights entry → convert → repack bundle). ~100 lines of bundle I/O deleted.

### Files Changed

| File | Change |
|------|--------|
| `src/main.cu` | Replace bundle loading with direct file mmap; download helper with size verification; G2P load overlap on separate stream; voice directory scanning |
| `src/rokoko.cpp` | `default_weights_filename()` + `default_weights_size()` (was `default_bundle_url/filename`) |
| `src/rokoko_f16.cpp` | Same |
| `src/weights.cpp` | Removed from-memory `prefetch()` overload |
| `src/weights.h` | Removed declaration |
| `src/g2p.h` | Removed from-memory `load()` overload |
| `src/bundle.h` | **Deleted** |
| `scripts/convert_v2.py` | Reads/writes standalone weight files, no bundle I/O |
| `scripts/extract_bundle.py` | **New**: one-time migration tool to extract existing bundles into split files |
| `Makefile` | Removed bundle.h from deps |
| `README.md` | Updated CLI docs (`--weights`, `--g2p`, `--voices` replace `--bundle`) |

### Release

Created GitHub release v2.0.0 with 7 individual assets: `weights.bin`, `weights.fp16.bin`, `g2p.bin`, and 4 voice `.bin` files. Verified end-to-end: clean cache → auto-download all files → inference plays audio.
