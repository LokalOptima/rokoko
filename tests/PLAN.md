# rokoko test plan

**Goal:** build a small, trustworthy suite that catches the known regressions and gives
repeatable evidence about speech and speed. Expand it when a concrete failure or product
claim requires more evidence.

Revised 2026-09-30 to give the work a bounded first deliverable. The executable suite and
production fixes are now implemented; see [README.md](README.md) for commands and
[REPORT.md](REPORT.md) for results and remaining limits. Historical measurements in §6
remain historical evidence, not current passing tests.

## 1. First deliverable and stopping rule

Deliver a reference-backed regression suite for both CUDA binaries and the shipped
`af_heart` voice, plus reproducible quality/performance reports on the tested machine. The suite covers
artifact identity, frontend behavior, token/style selection, long-input preservation, the
known padding error, and basic serving/cache behavior. It does not establish perceptual
equivalence or portability to untested hardware.

**This increment is complete when:**

- The checks in §2 run from a clean checkout after documented dependency/model preparation.
- Known defects addressed by this increment have reproductions that fail before the fix and
  pass afterwards, in both binaries where applicable.
- Each important assertion detects its assigned broken control for the intended reason.
- Reports identify binaries, inputs, artifacts, environment, failures and unresolved limits.
- Quality and speed results are reproducible and their claims stay within the evidence.

A missing prerequisite, missing dump, unsupported hook or inconclusive experiment is not a
pass. An unrelated finding goes into the backlog with a reproduction; it does not automatically
expand this increment. A newly found crash, content loss, or invalidating test-harness error
in these exercised paths is a blocker and must be resolved or reported as unfinished.

For each experiment, write down **the decision it will change, the observation needed, and
when to stop**. Reuse valid evidence. Do not repeat the general review or add metrics merely
because another agent suggests them. Revise this plan again only when new evidence changes
scope or the next implementation step. Use the implementation report to distinguish executed checks from remaining evidence gaps.

## 2. Implement in this order

Each item should produce an executable check. Add only the helpers and diagnostic hooks
needed for that item; no complete testing framework or model refactor is required first.

| Order | Work | Evidence needed to finish |
|---|---|---|
| 1 | Preserve `tests/frontend`; add a fast CPU target for its normalizer fixtures and direct token/chunk tests | Independent expected data; identity-normalizer and boundary faults fail assigned cases |
| 2 | Record approved artifact identities and reproduce official tensor/voice/vocab checks | Full hashes and pinned sources; corrupt files and V8 substitution detected; explicit model paths never overwritten |
| 3 | Fix style-row selection in CLI/library and both variants | Exact token IDs and 256-value style vectors match official fixtures; unsupported voice and shifted row fail |
| 4 | Split normalized text before G2P; verify long-input and output/error contracts | Ordered content coverage, boundary cases, valid audio, and visible errors through shipped interfaces |
| 5 | Establish decode at true frame length and remove the known padding error | Real lengths checked at temporal operations; padding mutation fails; targeted official-reference comparison records arithmetic differences |
| 6 | Check stateful execution and characterize repeatability | First call/replay, changed content/style, A→B→A, arena growth and context recreation tested; variation separated from stale-state faults |
| 7 | Restore reproducible intelligibility and speed reports | Independent spoken references, paired Kokoro comparison, actual graph hit/miss counters, saved results and commands |

**Keep entry points simple:** `make test` for fast offline CPU checks; keep the existing
`make test-frontend`; add `make test-gpu` for the bounded artifact/inference/runtime checks;
restore a tracked `make bench`. Use a script for the initial quality report. Introduce additional
targets only when there is real work to group. Report each target's scope and prerequisites;
missing prerequisites fail that target explicitly. Set `TEST_CMD` when the fast target exists.

Preparation may download checksum-pinned resources; ordinary tests use local files and isolated
temporary caches. Full determinism and listening studies do not block items 1–4. Consolidate
shared chunk/style orchestration where practical, while exercising CLI, library and server
integration and covering both `rokoko.cpp` and `rokoko_f16.cpp`.

## 3. Acceptance rules that matter now

### Independent artifacts and exact inputs

Use pinned official Kokoro-82M, its config and the official `af_heart` voice pack. Expected spoken
forms remain independently written or reviewed. A previous rokoko output is only a labeled
regression baseline, never proof of its own correctness.

Keep a manifest with full SHA-256 hashes, sizes, G2P version and official source revisions.
Verify identity and semantic conversion separately: a hash can preserve the wrong artifact.
The FP16 derivation must use an independent reader and official operations, not the converter
as its own oracle. `scripts/convert_v2.py` is already tracked.

Copied tensors, layouts, padding, vocab and voices are exact. Arithmetic transforms, including
folded FP32 tensors, need justified per-transform bounds. The review found FP32 folding errors
up to 5.96e-8 relative to PyTorch in local FP16 artifacts; FP32 storage alone does not justify
bitwise comparison. A blanket one-FP16-step allowance is also unjustified. Byte corruption still
fails the identity check regardless of numerical tolerance.

Style selection is `pack[len(ps) - 1]` for supported nonempty strings, using phoneme codepoints
before unknown-symbol filtering. Check tokens including BOS/EOS separately. Cover unknown and
all-unknown symbols and define empty-input behavior before indexing. Preserve the existing
ambiguous-date policy and the choice of G2P V11.

### Boundary and runtime fixtures

Use a small explicit case matrix, rather than a generic fuzzing project:

- Empty/whitespace input, malformed UTF-8, multibyte text, unknown voice and missing/truncated
  artifacts: useful errors, no crash or successful empty speech.
- Normalized G2P lengths around its runtime-defined limit (not stored in G2P3 metadata): for V11, 2047/2048/2049; include raw
  input that expands past the limit. Phoneme lengths 509/510/511, plus short and multi-chunk cases.
- Long text of about 5,000 and 20,000 characters, with and without convenient delimiters.
  Source spans cover content in order; trim only declared boundary whitespace. Check expected
  words through the last audio chunk. Plausible duration alone is insufficient.
- Independently decode WAV headers/sample counts and check finite float/PCM output. Compare
  buffered and streaming content with explicit handling of quantization and random variation.
- Streaming validation errors, cancellation/disconnect and a subsequent successful request.
  The current handler's ignored synthesis error needs a direct regression case.
- Same-key changed input/style, A→B→A, arena growth then shorter input, and sequential context
  destruction/recreation. Compare against isolated runs; cached pointers must remain valid.

### Padding, numerical diagnosis and repeatability

Use actual frame lengths first. Assert the effective lengths passed to temporal operations,
including upsampling and STFT boundaries, and test 31/32/33, 63/64/65 and a longer boundary.
Trimming padded audio or correcting only instance norm/reverse LSTM is insufficient: later
convolutions can carry padded values back into valid frames. Reintroducing padding must fail
its assigned assertion. These checks establish that the known padding error is removed;
they do not certify every arithmetic operation in the model.

For a numerical discrepancy, capture the smallest useful boundary and compare with untouched,
pinned official Kokoro. Validate reference instrumentation with overrides disabled and captured
inputs re-injected. Both binaries use mixed precision; neither is a strict-FP32 compute path.
If isolation is necessary, inject every input crossing that boundary:

| Boundary | Inputs that must come from the reference |
|---|---|
| Duration head | Duration-encoder features and speed |
| Prosody | Expanded encoder features, style and true length, not just durations |
| Decoder/generator | Aligned acoustic text features, F0, N, acoustic style, harmonic source and true lengths |

Test encoders/source generation separately only when diagnosis requires it. Initially bypass
graphs for injected diagnostics and separately test normal graph execution. A complete all-stage
injection framework is deferred.

Do five controlled repeats on the small regression set and compare first capture/replay and
A→B→A. Investigate uninitialized memory and stale state before calling variation numerical noise.
Record default-build variation. Add deterministic kernels/mode where needed for a specific
comparison; do not require cross-device bit identity or block unrelated tests on that work.

Numerical gates need independent positive controls that reflect actual weight **and activation**
casts, relevant broken controls, and validation cases separate from calibration. Check every
pre-rounding duration error; permit only an adjacent integer when its bounded interval crosses
a rounding boundary. If a sound bound is unavailable, report the discrepancy and the limited
claim supported by exact assertions. Do not invent a tolerance to get green results.

### Validate the tests and preserve evidence

Use isolated current-code mutations or the pre-fix implementation. The historical `bef3887`
canary lacks new hooks; unsupported options cannot count as fault detection. The fixed control
must pass and the broken control must reach the intended assertion. No generic mutation platform
is needed. Ordinary crashes are functional failures, but do not validate an audio-quality metric.

Save commands, source revision/dirty changes, binary and artifact hashes, dependency versions,
input/voice/seed IDs, GPU/driver and precision settings with results. Keep small fixtures and
scripts in git and preserve failing dumps/audio externally. Never automatically overwrite gold
fixtures or widen thresholds. Preserve existing frontend thresholds as labeled historical guards;
report stress-aware agreement separately.

## 4. Quality, speed and release reporting

**Intelligibility:** restore a reproducible paraketto evaluator using independent spoken forms
for existing reviewed cases. Compare both binaries with official Kokoro, saving actual phonemes
and chunks. Use rokoko phonemes through Kokoro when isolating the model from G2P. If using
LibriSpeech, use cased/punctuated transcripts and record training overlap. Never use the product
normalizer as the scoring oracle; wrong numbers, negations and omissions must remain errors.

Report corpus WER, sentence errors and paired 95% intervals per voice/binary. Bootstrap matched
utterances/groups and recompute summed errors divided by summed reference words. The proposed
0.5-percentage-point margin is a product choice to freeze before acceptance. An interval crossing
it is inconclusive. Start with a fixed dataset; expand only for a stated decision, not by sampling
until a result passes. Initial reports are evidence, not an automatic equivalence claim.

**Listening:** review representative corrected clips blindly and loudness-matched, retaining
originals to inspect gain/clipping. Record defects and preferences. A developer's failure to hear
a difference is not a threshold for automated acceptance. A powered perceptual-equivalence study
and automatic MOS predictors are deferred until a specific claim needs them.

**Speed:** commit the benchmark. Instrument actual graph hits, misses and invalidations; new text
can reuse graphs, and arena growth can clear them. Separately report cold start, warm misses,
warm hits with changed content, and a fixed mixed workload. Pin order, warmup and settings;
exclude GPU contention and keep reference/ASR work outside timing. Report end-to-end/model
latency, sample counts, median/p95 and memory for both binaries. Compare official Kokoro on the
same GPU with pinned ordinary-user settings and a separate model-only comparison on identical
phonemes. Associate outputs with correctness results. Quality limitations accompany speed claims.
A performance gate requires a preselected practical budget and repeat-trial uncertainty; raw
spread across differently sized sentences is not a regression threshold.

**Release:** validate candidate artifacts and actual binaries locally first. Once assets exist,
verify compiled download URLs using an empty temporary cache and smoke synthesis. Check both
Make/CMake configurations for a release. Remote verification is separate from implementation;
a nonexistent URL cannot pass. Record the frontend branch/release state when preparing publication.
This plan neither publishes a release nor declares existing unresolved defects acceptable.

## 5. Deferred work and its trigger

Keep these visible without making them hidden completion requirements:

| Work | Start when |
|---|---|
| Complete stage-by-stage capture/injection suite | A discrepancy cannot be localized with the small targeted checks |
| General acoustic metric calibration and fresh holdouts | A recurring audible regression escapes current checks or an automatic fidelity claim is needed |
| Powered multi-listener equivalence study | A release/product claim explicitly requires perceptual equivalence |
| Automatic naturalness predictors | Labeled defects demonstrate a useful signal beyond existing checks |
| Large corpus/power expansion | A fixed quality decision cannot be resolved by the initial paired report |
| Broad fuzzing, concurrency, long memory soaks and hardware matrix | Supported-use requirements or failures justify those extensions |
| Bucketed decode optimization | Correct exact-length execution exists and measured performance warrants the work |
| General mutation infrastructure | Maintaining the small explicit fault cases becomes a demonstrated problem |

A reproducible failure in supported use is still investigated. Deferral limits the first testing
project; it does not turn a known product defect into acceptable behavior.

## 6. Historical evidence — reference only

### Why the earlier tests were insufficient

Earlier normalizer gold data copied the normalizer in 15/18 classes; supposedly held-out
Harvard/LJSpeech/Tatoeba samples overlapped G2P training by more than 99%. The CLI formerly
replaced explicit differently sized model files, silently restoring V8; that behavior was fixed
on `fix/frontend`. Earlier FP16 work called same-binary variation a noise floor without finding
its cause. Fixed seeds alone do not guarantee deterministic floating-point reductions.

`tests/frontend` supplies existing regression protection. `tests/probe` is exploratory, not an
acceptance suite. The previous WER-generating script is absent; future numbers need an evaluator,
input manifest and command. The local benchmark has three repeatedly warmed sentences and an
optional exact-transcript check; it does not cover representative graph capture or GPU contention.

### Verified facts and open differences at the review snapshot

Prior probes reported all 688 loaded FP32 tensors matching official Kokoro: 548 checkpoint tensors
plus 140 default instance-norm affine tensors. Four voice packs and 114 vocabulary entries matched.
C++ G2P matched its Python checkpoint on 1680/1680 lines. Turn these measurements into reproducible
tests; the full FP16 derivation is still outstanding. The small FP32-folding probe described in §3 is additional
review evidence, not a replacement for that audit.

| Difference | Status / follow-up |
|---|---|
| Style row | CLI and `TtsPipeline` use `T - 2`; fix length-before-filtering semantics |
| Padded decode | Both model files round frames to 32; trimming cannot undo altered computation |
| Nondeterminism | Same-binary output varies; atomic reductions are suspects, not proven causes |
| Mixed precision | Both binaries cast weights/activations to FP16 and use TF32 elsewhere |
| Random source | Fixed/hash-based draws differ from official generation; measure effective differences separately |
| Fast sine | `__sinf` on large float32 phases remains a source-stage numerical hypothesis |
| Splitting | Phoneme punctuation/space rules differ from Kokoro text-token splitting; later chunks retain leading spaces |
| Long input | G2P rejects normalized input beyond positional capacity; split before inference and check preservation |
| WAV | PCM16 truncation needs its own declared transport contract |
| Streaming errors | Current handler ignores synthesis error return; add observable-error regression |
| Graph lifetime | Static caches capture pointers; context recreation and arena changes need coverage |

CLI/library chunk/style logic is duplicated, as is model execution in `rokoko.cpp` and
`rokoko_f16.cpp`. Every relevant fix/hook needs both variants covered. Consolidate shared
orchestration without making a wholesale model-code refactor a testing prerequisite.

### Historical audio probes, 2026-09-30

Four sentences (13, 92, 209 and 84 phonemes), one voice (`af_heart`), one GPU. Kokoro controls
changed one feature at a time with identical random draws unless stated otherwise. "Spec SNR"
compares log-magnitude spectrograms (window 1024, hop 256) and is unvalidated.

| Comparison | Durations equal | F0, largest relative change | Waveform SNR | Spec SNR |
|---|---|---|---|---|
| Kokoro, same run twice | 4/4 | 0 | identical | identical |
| Kokoro CPU vs GPU, strict FP32 | 4/4 | 0.0004–0.006% | 31–41 dB | 33–41 dB |
| Kokoro GPU, PyTorch defaults (cuDNN TF32) | 4/4 | 0.05–0.3% | 20–31 dB | 24–34 dB |
| Kokoro GPU, TF32 everywhere | 4/4 | 0.5–2.5% | 5–29 dB | 18–32 dB |
| Kokoro, weights rounded to FP16 | 3/4 | 0.08–0.6% | −3 to 25 dB | 11–28 dB |
| Kokoro, different seed | 4/4 | 0 | 19.4–19.9 dB | 21.6–22.4 dB |
| Kokoro, style row +1 | 2/4 | 8–9% | −4 to −3 dB | 6–15 dB |
| Kokoro, padded decode | 4/4 | median 0.5–1% | −4 to −3 dB | 11–12 dB |
| Kokoro, wrong voice | 0/4 | — | −4 to −2 dB | 0–4 dB |
| rokoko run vs run, `weights.bin` binary | 4/4 | — | 21–30 dB | 25–33 dB |
| rokoko run vs run, FP16 binary | 4/4 | — | 30–38 dB | 31–38 dB |
| `rokoko.fp16` vs official Kokoro | 3/4 (total frames only) | — | — | 6–11 dB |
| `rokoko.fp16` vs Kokoro with shifted style and padding | 3/4 (total frames only) | — | — | 17–19 dB (11 where the frame count differs) |


These probes explain why free-running waveform SNR is unsuitable as a general fidelity gate:
small F0 changes accumulate phase drift, and FP16 can move a duration across a rounding tie.
They suggest substantial effects from style/padding, not a measured share of perceptual quality
loss. Two seeds do not establish the allowed distribution of variation.

Historic provenance: rokoko `bef3887`, `fix/frontend`; binary MD5 prefixes `51c9eb719b79…`,
`f0cda1c3788b…`; `weights.bin` `468e919d…`, `weights.fp16.bin` `606862af…`; G2P V11
`98dbb7bb697565d131a565ac644ae5da`; kokoro 0.9.4, torch 2.11.0+cu130, RTX 5070 Ti,
driver 610.57.04, GPU reported idle. Capture full hashes in new runs.

Padding and FP16 controls were PyTorch emulations rather than isolated rokoko changes; the FP16
probe did not reproduce activation casts. The reference was installed at `~/data/rokoko_g2p/.venv`
and model files at `~/models/Kokoro-82M`; cuDNN TF32 was enabled by default there. Pin flags and
replace home-directory assumptions with documented setup before claiming clean-checkout
reproducibility. The full GPU/audio probe suite was not rerun during this revision.
