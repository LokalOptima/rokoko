## 2026-03-15: Optimization Round 8 — Dual-Tile Cutlass + Depthwise Fix

### Tile Selection

The 128x128 TF32 tile from Round 7 left 94% of SMs idle on short text — only 4 CTAs for 70 SMs. Added a 64x64x16 TF32 tile that fires when `ceil(C_out/128)*ceil(T_out/128) < SM_COUNT`. This 4x increase in CTA count recovers SM occupancy on small problems while keeping the high-throughput 128x128 tile for large ones.

Also fixed `conv_transpose1d_depthwise` — was launching 1 thread per block instead of 256 threads. This was a silent correctness bug (output was correct but slow).

### Results

```
=== Dual-Tile (bench.sh, 30 runs) ===
  Short:   9.76ms / 146x RTFx  (was 11.8ms — 1.21x faster)
  Medium: 26.50ms / 201x RTFx  (was 27.5ms — 1.04x faster)
  Long:   71.37ms / 248x RTFx  (was 71.9ms — holds)
```
