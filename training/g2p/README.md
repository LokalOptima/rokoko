# Neural G2P training

This directory owns G2P data preparation, training and checkpoint export. The
production implementation is `../../src/g2p.h`; its only text normalizer is
`../../src/normalize.h`. Python is used for development and training, never by
the TTS executable.

Use the optional Python environment described in [the test setup](../../tests/README.md).
It pins the training and reference dependencies, including Torch 2.11.0 and
Misaki 0.9.4 with American English and eSpeak fallback. Commands below run from
the repository root. Store datasets and runs outside Git, for example under
`/path/to/rokoko-data`; `data/` and `runs/` are ignored if used locally.

## Prepare and train

```sh
make tests/frontend/normalize_cli
.venv-tests/bin/python training/g2p/prepare_data.py \
  --input /path/to/raw-sentences.txt \
  --output /path/to/rokoko-data/new-base.tsv --workers 4

.venv-tests/bin/python training/g2p/train.py train \
  --data /path/to/rokoko-data/new-base.tsv \
  --out-dir /path/to/rokoko-data/runs/new-run \
  --nlayers 8 --no-conv --no-qk-norm \
  --epochs 50 --muon --compile --auto-batch
```

Preparation writes normalized text and phonemes to a TSV, plus a manifest of
input, output and normalizer hashes. It refuses to overwrite an existing dataset.
The trainer reads normalized text as-is, deduplicates before splitting, and
oversamples only the training partition. `path.tsv:5` gives a dataset fivefold
training weight. Each run records input hashes, architecture, code hashes,
optimizer configuration and environment in `train_setup.json` and checkpoints.
Use a distinct output directory for every new run; `--resume` continues a run.

`augment_data.py` can produce synthetic training pairs with `--classes` and
`--seed`. Use `--stats` to inspect the categories. These generated labels are
training material, never independent test expectations. Keep the sentences in
`tests/frontend/` out of new training datasets, including normalized variants.

## Export and evaluate

```sh
.venv-tests/bin/python training/g2p/train.py export \
  --checkpoint /path/to/rokoko-data/runs/new-run/best_exact.pt \
  --output /path/to/rokoko-data/new-g2p.bin
make test-frontend FRONTEND_PYTHON=.venv-tests/bin/python \
  G2P=/path/to/rokoko-data/new-g2p.bin
```

Export produces the native G2P3 format and a tensor manifest. The independent
frontend suite evaluates the exported model through the actual C++ runtime.
`train.py test` and `train.py eval` expect already-normalized input; they are
checkpoint diagnostics, not substitutes for those acceptance tests.

To verify the shipped V11 tooling against its checkpoint:

```sh
make test-training G2P_CHECKPOINT=/path/to/rokoko-data/training/v11/best_exact.pt
```

This checks exact export identity, Python/native phoneme agreement, shared
normalization, data preparation and its manifest. It requires CUDA for the
native G2P comparison. `make test` remains CPU-only and requires no Torch.

## Shipped V11 provenance

V11 uses 8 layers, width 256, 4 attention heads, FFN width 1024, 3x CTC upsampling,
RoPE and RMSNorm; convolution and QK normalization are disabled. The G2P3 format
and Python class name `G2PModelV3` describe its architecture family, not its
training-run version.

The original run configuration, data manifest and approved checkpoint/export
hashes are recorded in `provenance/`. The export must have MD5
`98dbb7bb697565d131a565ac644ae5da`; the regression suite also checks SHA-256.

**The original training dataset is not fully recoverable.** Two TSVs were
regenerated after V11 training; the historical manifest describes the original
files, not those regenerated replacements. The saved checkpoint can reproduce
the shipped binary, but rerunning training on the surviving TSVs cannot be
claimed to reproduce V11. Its former generated "gold" tests were invalid and
are excluded. New runs must retain their exact data and manifests.
