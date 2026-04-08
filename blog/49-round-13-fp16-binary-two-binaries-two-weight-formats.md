## Round 13: FP16 Binary (Two Binaries, Two Weight Formats)

**Goal**: Eliminate all runtime weight preparation. Pre-bake weight norm, NHWC reshape, channel padding, FP16 casting, and LSTM bias precomputation into a v2 weight file. The FP16 binary loads v2 weights and calls Cutlass FP16 directly — no maps, no fallback, zero startup compute.

### Architecture

Two binaries, two weight files:
```
make rokoko       # FP32 binary, loads v1 weights (runtime conversion)
make rokoko.fp16  # FP16 binary, loads v2 weights (pre-baked FP16)
```

Both link the same `main.o`. The linker resolves `rokoko_infer()` and `precompute_weight_norms()` from whichever `.cpp` is linked.

### File layout

| File | Purpose |
|------|---------|
| `src/rokoko.cpp` | FP32 inference: GEMM wrappers with map-based FP16 dispatch + fallback chain |
| `src/rokoko_f16.cpp` | FP16 inference: direct Cutlass FP16 calls, no maps, no fallback |
| `src/rokoko_common.h` | Shared: WAV I/O, `compute_decode_bytes`, buffer structs |
| `src/weights.h` | `__half*` companion fields alongside `float*` for all weight matrices |
| `src/weights.cpp` | `assign_v2_fp16_pointers()`: suffix-matching over v2 tensor names |
| `scripts/convert_v2.py` | Standalone Python converter: v1 KOKO → v2 KOKO (numpy, no GPU) |

### KOKO v2 Weight Format

Same structure as v1 but version=2 and mixed dtypes per tensor:
```
[4B "KOKO"] [4B version=2] [8B header_len]
[text header: "name offset size_bytes dtype shape..."]
[padding to 4096]
[data blob: 256-byte aligned tensors]
```

Tensor suffixes: `.f16` (FP16 GEMM), `.nhwc_f16` (NHWC FP16 conv), `.nhwc_f16_pad{N}` (padded NHWC FP16), `.bias_combined_fwd/rev` (precomputed LSTM bias).

The Python converter applies all transforms on CPU: weight norm, NHWC reshape, channel padding (514→520, 1090→1096, 22→24), FP16 cast, LSTM bias = bih + bhh. 966 tensors total (688 base + 192 FP16 + 74 NHWC + 12 bias), 589 MB on disk.

### Key differences in rokoko_f16.cpp

- **GEMM wrappers take `const __half*`**: always call Cutlass FP16, no `s_fp16_weights` map lookup
- **`gemm_conv1d` takes `const __half*` NHWC weight**: K=1 → GEMM, K>1 → Cutlass FP16 conv. Optional `C_in_pad` for padded channels
- **`precompute_weight_norms`**: just `w.assign_v2_fp16_pointers()` + staging buffer alloc. Zero GPU compute
- **conv_post (C_out=22)**: im2col + FP16 GEMM directly (workspace-based staging, not s_fp16_buf)
- **bilstm_gpu**: takes `__half*` wih/whh directly from struct, always FP16 GEMV

### bench.sh Results

| Text | FP32 (v1 weights) | FP16 (v2 weights) | Delta |
|------|-------|-------|-------|
| Short (1.6s) | 6.44ms | 6.45ms | ~same |
| Medium (5.7s) | 16.59ms | 16.66ms | ~same |
| Long (18.8s) | 51.49ms | 48.02ms | **−3.47ms (−6.7%)** |

RTFx: 347x (long, FP16 binary). STT 3/3 PASS on both binaries.

The long text improvement comes from slightly more efficient CUDA graph capture (no fallback dispatch logic baked in). Init time for precompute_weight_norms drops from ~80ms to ~0ms (pointer assignment only), but this is offset by the larger v2 file upload (589 MB vs 327 MB). Net init is similar.

**Next step**: strip redundant FP32 weight matrices from v2 file (keep only biases/norms/embeddings in FP32). This would cut v2 to ~260 MB and make init genuinely faster.
