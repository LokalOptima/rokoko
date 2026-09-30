# Consolidated Rokoko history

The local history was reconstructed on 2026-09-30 from retained development
milestones. These are source snapshots, not a claim that discarded experiments
never happened. Commit messages record their original source hashes. Only the
final consolidated tree has been validated with the current suite.

| Retained milestone | Original Rokoko commit |
|---|---|
| CUDA inference, neural G2P and CUTLASS kernels | `e09a349` |
| FP16 inference and the versioned weight format | `888558b` |
| Shared library API, CMake and separate runtime assets | `b0b5eb0` |
| Corrected normalizer and genuine G2P V11 | `bef3887` |
| Runtime hardening and independent regression coverage | `12c606c` |

The final consolidation also imports:

- G2P model, trainer and data-preparation tools from `LokalOptima/rokoko-g2p`
  at `936e25b`, adapted to use the production normalizer and explicit data paths.
- The FP32 TTS weight exporter from `LokalOptima/rokoko-dev` at `b7788f6`,
  adapted to load local official files. FP16 conversion remains in
  `scripts/convert_v2.py`.
- V11 run metadata and checkpoint identity. See
  [the provenance limitation](../training/g2p/README.md#shipped-v11-provenance).

Excluded from the maintained repository: the old CPU backend, dictionary/POS
frontend, pronunciation teacher, superseded runtime/normalizer copies, generated
"gold" tests, completed test plans, obsolete training plans, agent worktrees, one-off probes, and the
duplicate historical blog. Recovery bundles and source snapshots are kept
outside this repository; these excluded paths are not supported features.

The canonical source tree is organized as follows:

- `src/`: native inference, text normalization, CLI, HTTP and library API.
- `training/g2p/`: training, preparation, export and provenance.
- `scripts/`: TTS weight export and FP16 conversion.
- `tests/`: independent frontend, CPU, GPU, artifact and quality checks.

There is one maintained branch, `main`. Upstream GitHub history has not been
rewritten by this local consolidation; publishing the replacement history and
retiring the old remote repositories are separate operations.
