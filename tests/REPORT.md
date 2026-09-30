# Implementation report — 2026-09-30

Implemented on `test/regression-suite`, based on `fix/frontend`. The generated evidence is in `tests/results/`; commands and preparation are in [README.md](README.md).

## Voice scope update

`af_heart` is now the sole supported voice, following the user's request and
[Kokoro's official overall grade A](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md#american-english).
Downloads, loading, the web UI, required artifacts and current evaluation scripts
use only this voice. The multi-voice measurements below describe the earlier run.
Changed-style cache tests now use two rows of the same pack; retired voice names
return errors even if old files remain cached. The CPU and GPU targets pass
with the reduced voice set; see `tests/results/single-voice.log`.

## Fixes and validation

- Shared CLI/library/server orchestration selects style row `len(phonemes)-1`, before unknown-symbol filtering, and splits normalized text before the 2048-codepoint G2P capacity.
- Both variants decode at true duration length. Forced 31/32/33, 63/64/65 and 127/128/129-frame cases pass. Isolated production mutations restore 32-frame padding while retaining output trimming; both fail the structural length assertion.
- A new context-recreation test exposed nonfinite audio. STFT/iSTFT scratch pointers could outlive their allocations inside cached CUDA graphs. Scratch now belongs to the decode arena; graph/operator caches and precomputed GPU allocations are released with the context.
- Encode, decode and G2P graph caches are capped at 64 entries each. One live context with serialized calls is explicitly supported; another live context is rejected. Sequential recreation, same-key content/voice changes, A→B→A and arena growth pass.
- Malformed/truncated model headers are checked before GPU use. Explicit artifact paths remain untouched. Invalid text/voice requests fail visibly, including streaming requests; cancellation is followed by a successful request.
- `make test`: 14 checks, covering 115 independent normalizer examples, 450 reviewed snapshots, official tokens, chunk coverage, UTF-8, style mutation, PCM16 and scorer controls.
- Independent artifact audit: all 688 FP32 and 585 converted tensors, all four voices, 114 vocabulary entries and G2P V11 identity pass. Byte/value corruption and local V8 substitution are rejected.
- `make test-gpu`: artifact audit, CLI/library/HTTP checks, exact styles/lengths, long-input coverage, reference instrumentation and both padding mutants pass.
- Long synthesis covers 5,000 and 20,000 characters with and without sentence punctuation. ASR recognizes the expected final four words in all six tail clips.
- Both Make variants build and run. CMake initially compiled for its compiler-default architecture, producing unusable CUTLASS kernels despite a successful build. The architecture default is now set before `project()` enables CUDA. Fresh native-architecture builds and synthesis smoke checks pass for both configurations; `tests/cmake_smoke.py` reproduces this check.

## Quality and numerical evidence

The fixed paired quality set contains 12 sentences and 103 reference words per voice. ASR is local Paraketto V2 FP16; no text corrector is enabled. Literal scoring can count numeral/spelled-number formatting differences. Reviewed sentences may overlap G2P training.

| Voice | Rokoko errors | Rokoko FP16 errors | Official Kokoro errors | Paired Rokoko − Kokoro 95% interval |
|---|---:|---:|---:|---|
| af_heart | 8/103 | 8/103 | 8/103 | [0.00, 0.00] percentage points |
| af_bella | 8/103 | 8/103 | 8/103 | [0.00, 0.00] percentage points |
| af_nicole | 5/103 | 5/103 | 4/103 | [0.00, 3.23] percentage points |
| af_sky | 8/103 | 8/103 | 8/103 | [0.00, 0.00] percentage points |

Official Kokoro on the same Rokoko phonemes has the same error counts as its ordinary pipeline in this set. Intervals resample matched utterances and use summed errors/words. An all-zero interval on 12 observed pairs does not establish population equivalence. No acceptance margin or listening claim is made.

Capture hooks and full prosody-input reinjection preserve the untouched official output exactly in the targeted cases. All per-token rounded durations match for those 12 variant/case pairs. Pre-rounding durations differ by up to 0.004107. F0 and noise differences are recorded per case; no unsupported numerical bound is used as a pass criterion. Both binaries use mixed precision.

Repeated outputs are not bit-identical. Exact token/style/length assertions and changed-input checks pass; waveform RMS/max differences for capture, replay, A→B→A, isolated graphs, arena growth and recreated contexts are retained. The scratch lifetime defect is fixed. Atomic reductions remain a possible contributor to residual variation; there is no claim that all variation has been explained or bounded.

## Timing and memory

RTX 5070 Ti, CUDA 13.1; precise driver, clocks, binary hashes and source fingerprints are saved in JSON. Numbers below are median wall milliseconds for five warm repeats of the fixed short/medium/long texts. They are descriptive, not a speed regression gate.

| Run | Short | Medium | Long |
|---|---:|---:|---:|
| Before rokoko | 7.90 | 18.92 | 57.14 |
| Before rokoko.fp16 | 7.67 | 18.38 | 52.97 |
| After rokoko | 7.88 | 18.37 | 57.09 |
| After rokoko.fp16 | 7.96 | 18.13 | 53.01 |
| Official pipeline | 22.09 | 33.01 | 98.83 |
| Official same_phonemes_model_only | 19.96 | 30.59 | 95.44 |

The before mixed workload had 15 sentences; the after workload had 250. Their aggregate latencies must not be compared as a regression estimate. Request-level graph counters, first captures, warm hits, p95, output lengths, startup and memory samples are retained. The final HTTP-escaping fix, diagnostic-only dtype fix, equivalent true-length expression and CMake configuration fix are outside the saved benchmark timing run; each run identifies its actual binary.

- rokoko: 56 encode, 50 decode, 61 G2P entries after 250 distinct sentences; reported total GPU usage 2.68 GiB. Decode arena 773 MiB. All caches remain within the 64-entry caps.
- rokoko.fp16: 56 encode, 50 decode, 61 G2P entries after 250 distinct sentences; reported total GPU usage 2.09 GiB. Decode arena 773 MiB. All caches remain within the 64-entry caps.

## Remaining limits

- Frontend guards: 991/1000 plain sentences under the historical comparison, 980/1000 exact with stress, 113/115 end-to-end handwritten cases, and 450/450 unchanged snapshots. They retain known misses and six reviewed issues; they are not claims of perfect pronunciation. Exact stress-retaining agreement is reported separately without changing the historical thresholds.
- No general audio-fidelity, deterministic-computation, perceptual-equivalence or WER noninferiority gate has been established. Human listening has not been performed by this implementation.
- Large-word CPU splitting is checked; arbitrary repeated-character gibberish can be rejected by the neural G2P rather than producing speech.
- Clean-cache release downloads were not validated: the required release assets have not been published by this task. Runtime models must be provided at the approved hashes. No release was published.
- These runs cover the recorded machine and single-context serialized use. Concurrency, other devices and long memory soaks remain outside this increment.


## Consolidation validation (2026-09-30)

The consolidated checkout was built from source with CUDA 13.1 and the existing
CUTLASS headers on the same RTX 5070 Ti. Production inference source is unchanged
from original commit `12c606c`; training and export now live alongside it.

- `make test`: all 14 CPU checks passed.
- `make test-frontend`: 115/115 handwritten normalizer cases, 991/1000 lenient
  and 980/1000 exact G2P cases, 113/115 end-to-end cases and 450/450 snapshots.
  The previously documented pronunciation misses remain.
- `make test-gpu`: both precision variants passed the runtime, reference and
  source-mutation checks using freshly exported artifacts.
- `make test-training`: V11 checkpoint export is byte-identical to the approved
  G2P artifact; Python/native predictions agree on 12 sentences. Shared
  normalization, preparation, provenance and refusal to overwrite data passed.
- The recovered FP32 exporter, FP16 converter and voice export reproduce all
  approved asset SHA-256 hashes. The independent artifact audit passed all 688
  FP32 and 585 FP16 tensors plus corruption controls. The converter's FP16
  header spelling was aligned with the shipped format; tensor bytes did not change.
- A bounded one-epoch training run on 128 existing short examples completed and
  wrote a checkpoint and verified input/source hashes. This exercises training
  plumbing only; it is not a quality result or a replacement for the shipped V11.

Consolidation logs are saved under `tests/results/consolidation/`. Model files,
checkpoints and original data remain outside source history. No new quality or
performance equivalence claim is made, and no GitHub history was rewritten.
