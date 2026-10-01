# Regression suite

`make test` is offline and CPU-only. It checks the 115 handwritten normalizer
cases, 450 reviewed snapshots, official vocabulary/token IDs, ordered chunk
coverage, UTF-8 rejection, style indexing, PCM16 encoding and scoring controls.
It also checks bounded memory reads and build asset verification without downloading models.
The token/chunk code lives in `src/phonemes.h`; including it needs no CUDA SDK.

## Preparation

Build requirements are the project's C++17/CUDA/CUTLASS requirements. The
recorded GPU run used CUDA 13.1, an RTX 5070 Ti, and Python 3.13.12. GPU and
reference scripts fail when prerequisites are absent. Build prerequisites can
download pinned assets; use `OFFLINE=1` with prepared assets to prohibit this. Keep the GPU free of other compute processes.

```sh
uv venv --python 3.13 .venv-tests
uv pip install --python .venv-tests/bin/python -r tests/requirements-reference.txt
make -j2 rokoko
python3 tests/prepare.py --models build/assets
make test-gpu OFFLINE=1
```

`prepare.py` verifies build assets against the independent approval fixture
`fixtures/artifacts.json` and fetches official Kokoro files at the pinned revision
with full SHA-256 checks. It refuses to replace existing mismatched files.
Only `weights.fp16.bin`, `g2p.bin` and `voices/af_heart.bin` are runtime inputs.
The optional `artifacts.py --exported-fp32 /path/to/weights.bin` check validates an
offline exporter intermediate; ordinary GPU tests do not require it.
Existing official files can be used with `--official /path/...` and
`make test-gpu OFFICIAL=/path/...`. The optional `MODELS` test argument identifies
files for independent comparisons; it does not override the executable's embedded assets.
See the root README for local/offline build input options.

`requirements-reference.txt` records the full reference environment, including
CUDA 13 PyTorch dependencies. The installed Torch reports `2.11.0+cu130`.
These measurements are specific to that environment, not a portability claim.

## Commands and evidence

```sh
make test
make test-frontend G2P=/path/to/runtime-models/g2p.bin
make test-gpu MODELS=/path/to/runtime-models OFFICIAL=/path/to/Kokoro-82M
make bench MODELS=/path/to/runtime-models
.venv-tests/bin/python tests/bench_reference.py --official /path/to/Kokoro-82M --models /path/to/runtime-models
.venv-tests/bin/python tests/quality.py --official /path/to/Kokoro-82M --models /path/to/runtime-models --asr /path/to/paraketto.cuda --asr-weights /path/to/paraketto-fp16.bin
.venv-tests/bin/python tests/long_tail.py --asr /path/to/paraketto.cuda --asr-weights /path/to/paraketto-fp16.bin
```

Before running the standalone quality/reference-benchmark scripts, build their frontend helpers with `make test tests/frontend/g2p_check`. Override `REFERENCE_PYTHON` to reuse a prepared environment. To run the existing
frontend evaluator with the pinned environment rather than its uv script
resolution, first build `tests/frontend/g2p_check`, then run
`.venv-tests/bin/python tests/frontend/eval_frontend.py --g2p ...`.

`test-gpu` runs the independent artifact audit, CLI/library/HTTP regressions,
reference instrumentation checks, and a real padding mutation. CUDA tests
compile a small direct-library runner (`runtime.cu`). Input and style dumps are
checked against the official config and independently verified af_heart voice pack.
The FP16 audit derives storage/layouts from official module types; it never
imports `scripts/convert_v2.py`. Folded arithmetic uses stated forward-error
bounds; copied values, padding and layouts are exact. The manifest is never
updated by ordinary tests. An available `g2p_v8.bin` is tested as a rejected
substitution; byte-corruption controls always run.

The workload covers G2P lengths 2047/2048/2049, expansion before G2P, 5,000 and
20,000-character speech, phoneme chunk boundaries, changed content/style on the
same graph key, A→B→A, five repeats, arena growth, context recreation, invalid
artifacts, cancellation and successful recovery. The current G2P format does
not carry positional-capacity metadata: the runtime exposes its 2048-codepoint
capacity. CPU tests additionally cover hard cuts through very long words.
`long_tail.py` checks the independently specified final words in all three saved
long-input tail clips using ASR. It requires the preceding GPU run.

Generated JSON, logs, trace arrays and WAVs live in ignored `tests/results/`.
Reports include source/binary hashes, artifact identities, inputs, settings and
commands. Archive that directory to retain a run; ordinary reruns overwrite
reports, never expected fixtures. `REPORT.md` records this implementation run.

## Interface and lifetime contracts

- Normalize text, then split into ordered UTF-8 spans before G2P. Trim boundary
  whitespace only. Empty, malformed or nonspeech input returns
  an error, never successful empty audio. The neural G2P is not guaranteed to
  pronounce arbitrary repeated-character strings; such failures remain visible.
- Select `pack[len(phonemes)-1]`, counting codepoints before filtering unknown
  vocabulary symbols. Token IDs independently include BOS/EOS.
- Decode at the actual duration length, including prosody, convolutions, source
  generation and STFT. Cached STFT scratch belongs to the decode arena.
- WAV is mono, 24 kHz, little-endian PCM16: clamp to [-1,1], multiply by 32767,
  truncate toward zero. Streaming emits finite little-endian float32 samples.
  These describe this supported little-endian CUDA platform.
- HTTP accepts a JSON object of string fields (`text` and optional
  `input: "phonemes"`). An obsolete `voice` field is rejected, including `af_heart`. Invalid requests return 400 before streaming
  starts. Later inference/write failures terminate the stream without a
  successful chunk terminator. A client must treat incomplete transfers as errors.
- One live `TtsContext` and serialized calls are supported. A second live
  context is rejected explicitly. Sequential destruction/recreation is tested.
  Each encode/decode/G2P graph cache holds at most 64 entries and clears when
  full; decode-arena growth also invalidates its captured graphs. The arena
  retains its largest allocation until context destruction.

## Limits of the results

No waveform-tolerance gate is invented. Repeated audio differs slightly;
reports expose this separately from exact tokens, styles, durations, shapes
and stale-output checks. Atomic reductions remain a plausible source of
variation, not a proven complete explanation. Reference captures are tested
against an untouched call and full prosody inputs re-injected; differences in
pre-rounding durations and F0/N are recorded, not excused by an arbitrary bound.

The fixed quality set has 12 independently written/reviewed references for
`af_heart`. Scoring removes case/punctuation only; digits and spelled numbers can
produce ASR-formatting errors, and wrong numbers, negation or missing words
remain errors. Paired intervals resample matched utterances and divide summed
errors by summed reference words. Reviewed cases can overlap G2P training.
This small report establishes neither generalization nor perceptual/quality
equivalence. No acceptance margin was selected. Human listening remains a separate quality assessment.

Benchmark counters identify actual graph hits and misses; sentence identity is
not used to guess cache reuse. The fixed mixed workload and memory samples are
descriptive. No speed regression gate or practical latency budget was chosen.
Reference/ASR work is outside Rokoko timing. The separate reference benchmark
reports ordinary KPipeline and model-only timings on identical phonemes; Python
model initialization is labeled separately from CLI startup.

Run `make test-bundle OFFLINE=1` for the standalone distribution check (requires
`strace`). It copies the executable alone, makes HOME/cache paths unavailable,
uses an empty PATH, denies outbound `connect` calls, and traces CLI, HTTP and
streaming synthesis. It verifies that no model files are opened or extracted.
CUDA's own disk cache is disabled for this check.

For CMake integration, run `python3 tests/cmake_smoke.py --models build/assets`.
This builds Rokoko, repeats the standalone check, and builds a separate
library-only consumer that synthesizes audio without supplying asset paths. Use fresh build directories
when checking the default architecture: old CMake caches can retain `52` from
the former initialization-order bug. Explicit `CMAKE_CUDA_ARCHITECTURES` or
`CUDAARCHS` values remain respected for cross compilation.

Only `af_heart` is supported. Voice selection and production asset-path flags are
rejected; malformed asset coverage uses a development adapter over the memory loaders. Style replay coverage uses
different rows of the same approved pack. Historical multi-voice results in REPORT.md
are retained as evidence of the earlier run.

`make test-training G2P_CHECKPOINT=/path/to/best_exact.pt` checks the consolidated
G2P training/export boundary. See [training/g2p](../training/g2p/README.md).
