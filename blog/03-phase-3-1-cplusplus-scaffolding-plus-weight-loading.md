## 2026-03-02: Phase 3.1 — C++ Scaffolding + Weight Loading

### Build System

Created a simple `Makefile` following the parakeet pattern:
- `g++` with `-std=c++17 -O3 -march=native -flto=auto`
- Links `libcudart` + `libcublas` from CUDA 13.1
- NVCC for `.cu` files (not needed yet — no custom kernels yet)
- `make kokoro` builds the binary

### Weight Structs

`src/kokoro.h` defines:
- `TensorDesc` — parsed from the text header (name, offset, size, dtype, shape)
- `Weights` — contiguous GPU allocation with named struct fields for key tensors
- Constants for model dimensions (ALBERT, text encoder, etc.)

The weight struct has explicit fields for the most important tensors (ALBERT layers, text encoder conv/LSTM, predictor shared LSTM) plus a generic `get(name)` lookup for the 600+ decoder/predictor tensors that will get their own fields as each component is implemented.

### Weight Loading (mmap + GPU upload)

Following exactly the parakeet pattern:
1. `Weights::prefetch(path)` — `mmap(MAP_PRIVATE | MAP_POPULATE)` + parse header
2. `Weights::upload(stream)` — single `cudaMalloc` + `cudaMemcpy` + pointer assignment
3. Prefetch runs in a background thread, overlapping with CUDA context init

### Naming Gotchas

Hit some naming mismatches between what I assumed and what PyTorch actually exports:
- Text encoder convs: `text_encoder.cnn.0.0.weight_v` (not `text_encoder.convs.0.conv.weight`)
- Layer norm uses `gamma`/`beta` (not `weight`/`bias`) in the text encoder
- Decoder upsampling: `decoder.generator.ups.0.weight_g` (not `decoder.ups.0.weight`)
- All convolutions use **weight normalization** (`weight_g` + `weight_v` pairs)

### Results

```
GPU:          NVIDIA GeForce RTX 5070 Ti (16189 MB free / 16612 MB total)
CUDA init:    188.5 ms
Prefetch:     188.5 ms (overlapped with CUDA init)
GPU upload:   15.8 ms
Total init:   204.3 ms
weights: 688 tensors, 156.0 MB GPU
  all key weights found
  Verified 688 / 688 tensors are accessible on GPU
```

156 MB on GPU for all model weights. Init time is dominated by CUDA context creation (~189ms) — the actual weight upload is only 16ms thanks to mmap prefetching.

### Next
Step 3.2 — ALBERT encoder implementation. Need: embedding lookup, layer norm, multi-head attention (cuBLAS GEMM), GELU activation, and the 6-iteration shared layer loop.
