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

The runtime shares the forward pass in `src/rokoko.cpp` between the CUDA
executable, `rokoko`, and the new FP32 CPU executable, `rokoko.cpu`. Both embed
the same FP16 weight assets, G2P model and af_heart voice. The older on-load
conversion backend and its separate build/test paths were removed. FP32 export
remains an offline input to the converter; official Kokoro remains the
independent reference.

The canonical source tree is organized as follows:

- `src/`: native inference, text normalization, CLI, HTTP and library API.
- `training/g2p/`: training, preparation, export and provenance.
- `assets/`: pinned build-download identities and locations.
- `scripts/`: asset bundling, TTS weight export and FP16 conversion.
- `tests/`: independent frontend, CPU, GPU, artifact and quality checks.

There is one maintained branch, `main`. The consolidated history was published
on GitHub on 2026-10-01. The previous remote `main` tip,
`b0b5eb078a915d353ab3a22929b59c9c2b30322a`, is preserved on
[`archive/main-before-consolidation-2026-10-01`](https://github.com/LokalOptima/rokoko/tree/archive/main-before-consolidation-2026-10-01).
The backup and replacement are published atomically, with a lease requiring that
exact previous tip. Existing release tags/assets and the legacy `cutlass` branch
are retained. Retiring the old development/G2P repositories is a separate operation.

Existing clones of the old history need a fresh clone or deliberate migration;
merging or pulling the unrelated histories is not the consolidation workflow.
The refreshed checkout tracks `origin/main`, so subsequent commits use ordinary
`git push` and `git pull`.

