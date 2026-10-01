# Implementation report — 2026-09-30

## Experimental INT8 vocoder — evaluated 2026-10-01

An isolated build quantizes the generator residual convolutions using INT8
weights per output channel and dynamic INT8 activations per tensor. oneDNN
verbose output confirms `brg_conv_fwd:avx2_vnni` with unsigned-byte activations
and signed-byte weights. The rest of the pipeline retains FP32 arithmetic.
Production inference sources, bundled assets and `rokoko.cpu` are unchanged;
the experiment lives under `tests/experiments/` and builds into `build/int8/`.
The executable still embeds FP16 assets and quantizes weights during preparation,
so this trial measures compute changes, not a smaller download.

Five accepted warm requests per build/text were alternated on the same i7-12700,
eight workers, one request in flight and no affinity pinning. The existing guard
waits for/rejects more than 0.5 competing CPU core. These are paired results from
this run; compare the two columns rather than baselines from an earlier session.

| Speech length | FP32 CPU | Experimental INT8 | INT8 RTFx | Less time |
|---|---:|---:|---:|---:|
| 1.575 s | 176.0 ms | 139.8 ms | 11.27× | 20.6% |
| 5.725 s | 548.9 ms | 423.2 ms | 13.53× | 22.9% |
| 18.825 s | 1972.1 ms | 1533.3 ms | 12.28× | 22.3% |

The first request at the medium sentence length took 736.3 ms with INT8 versus
830.5 ms with FP32. These are single observations of new-shape overhead, excluded
from warm medians; INT8 still does not guarantee a half-second first response.

Quality and correctness:

- Pre-vocoder normalization, tokens, styles, raw/rounded durations, F0 and noise
  match the FP32 CPU traces byte for byte on all 15 cases. Audio is finite and
  sample counts are unchanged.
- **Two of 15 cases fail the existing INT8/GPU spectral-error limit of 0.1:**
  `reviewed_4` (0.11392) and the long benchmark (0.16294). Against FP32 CPU audio,
  the long benchmark also exceeds 0.1 (0.14809). No thresholds were relaxed.
- Local ASR produces identical FP32/INT8 transcripts on all 15 cases: each has
  8 word errors out of 181 reference words. On the original 12 quality fixtures,
  both have 8/103. This small diagnostic set does not establish perceptual
  equivalence; ASR does not measure timbre or audio artifacts. No human listening
  judgment has been made by this implementation.
- Independent integer-oracle tests cover zero tensors/channels, tails, worker
  boundaries, padding, dilation, stride, output guards, residual aliasing,
  changed weights after cleanup and nonfinite-input/exception recovery. Release
  tests pass with eight workers; ASan/UBSan pass with three. External static
  libraries themselves were not sanitizer-instrumented.
- The experimental binary passes the existing library lifecycle, frame-boundary,
  deterministic repeat, context-recreation, invalid-asset, CLI/HTTP and streaming
  recovery checks. Its per-weight plan cache retains only the current shape and
  is cleared with the context.

Conclusion: this selective INT8 trial saves about 21–23% of warm request time,
but has **not passed the existing audio regression limits**. Keep FP32 as the
default. Listening and further calibration or narrower layer selection are
needed before deciding whether the speed/quality tradeoff is acceptable.

Reproduction: `tests/experiments/README.md`. Evidence:
`tests/results/int8-quality/report.json` and `index.html` (paired listening),
`int8-bench/report.json` (all timing attempts and WAVs), `int8-inference/`, and
`int8-validation/`. `build/int8/build.json` records the experimental sources,
linked libraries and binary hashes. The build is checked to reproduce the
measured binaries exactly.

## Direct CPU convolutions and shared workers — verified 2026-10-01

The CPU forward pass now uses statically linked oneDNN 3.10.2 direct AVX2
convolutions alongside OpenBLAS 0.3.30. Both libraries share the existing BLAS
workers; there is still only one request in flight. Normalization, source
signal generation, conversions and large pointwise loops use those same workers.
GELU uses vector tanh. Dense and convolution input staging preserves the former
FP16 rounding while eliminating the separate half-buffer pass. Arithmetic and
cached weights remain FP32; bundled model assets and CUDA arithmetic are unchanged.
No INT8 quantization is included.

Each convolution weight retains one packed weight buffer and its most recent
primitive shape. A different shape replaces the primitive; oneDNN's independent
primitive-history cache is disabled. Staging grows to the largest input seen.
Context destruction releases this state. The shared-worker adapter uses an
isolated internal entry point in pinned OpenBLAS 0.3.30, which must be revalidated
when that dependency is upgraded.

Validation passed:

- Operator tests with one and eight workers; ASan/UBSan with three workers.
  New checks cover worker partitioning, nested jobs and exception recovery,
  exact fused half rounding, vector GELU, padded convolutions, output guards,
  residual aliasing, 160 changing lengths, bounded primitive count and cleanup.
  External library builds themselves were not sanitizer-instrumented.
- All 15 CPU/CUDA synthesis cases and 100 G2P comparisons pass the existing
  limits without changes. Text, tokens, styles, rounded durations and output
  lengths agree exactly. Maximum duration difference is 0.01348 frames;
  F0 RMSE is at most 0.08460 Hz; noise RMSE is at most 0.001298. Maximum relative
  magnitude-spectrogram error is 0.08367; RMS ratios are 1.0165–1.0232.
- Against the saved previous SIMD CPU output, maximum relative spectral error
  is 0.06897 and RMS ratios are 0.99968–1.00140. These numerical comparisons
  do not establish perceptual equivalence; paired WAVs are retained for listening.
- Library lifecycle checks, exact repeated outputs, A→B→A, arena growth,
  context recreation, malformed assets, CLI/HTTP and streaming recovery pass.
- The copied executable passes the isolated CLI/HTTP/streaming check with no
  external assets, blocked connections, unavailable HOME/cache and empty PATH.
  A separate CMake library consumer also builds and synthesizes speech.
- Embedded ELF bytes match the approved assets, and direct payload scanning
  detects duplicates. This replaces a 10 MiB non-asset-size heuristic that
  incorrectly rejected the larger static library. An appended duplicate voice
  is rejected. The executable is 227,802,848 bytes (217.25 MiB), up 26.94 MiB;
  only ordinary C/C++ system libraries are loaded dynamically.
- The CUDA build and the fast offline tests pass.

Performance on the i7-12700 uses eight workers, one request in flight and no
affinity pinning. Five accepted warm requests per binary/text are alternated
against preserved commit `b1bc302`; requests above 0.5 competing CPU core are
rejected and retained in the JSON. Startup and each first request are excluded.

| Speech length | Previous CPU | Optimized CPU | RTFx | Less time |
|---|---:|---:|---:|---:|
| 1.575 s | 286.9 ms | 157.1 ms | 10.03× | 45.2% |
| 5.725 s | 961.9 ms | 514.6 ms | 11.12× | 46.5% |
| 18.825 s | 3226.2 ms | 1878.4 ms | 10.02× | 41.8% |

The medium sentence's five accepted optimized requests range from 502.9 to
531.8 ms. Short and long medians are only just above 10×; individual requests
can be slower. This is a warm-request result, not a guarantee for every sentence.
The first medium request at its new length took 801.2 ms (7.15×), versus
1069.9 ms previously. New shapes incur convolution primitive preparation;
startup and weight packing are additional work. Startup samples were collected
before the quiet guard and are not used as clean performance comparisons.
The first long request was rejected for contention and is not a reported timing.

Measured binaries, every accepted/rejected request and WAV are retained in
`tests/results/cpu-rtfx-final/report.json`. The final optimized binary SHA-256 is
`270e520762ba2086d733524b3291ea49b505f0994a5596b1a304a6e281a4f5b6`.

Evidence: `tests/results/cpu-rtfx-parity/`, `cpu-rtfx-inference/`,
`cpu-rtfx-audio.json`, `cpu-rtfx-validation/`, and `cmake-cpu.json`. The parity directory contains
an `index.html` listening page. Timings are recorded separately from correctness.

## AVX2 activation optimization — verified 2026-10-01

The CPU style-affine/Snake loop now processes eight channels at once using AVX2
and glibc's vector sine. Double-precision normalization statistics remain
unchanged, scalar tails are retained, and fast-math remains disabled. The build
checks for the vector-math ABI; the executable uses the system glibc/libmvec.
The original OpenBLAS wait policy is retained. An experimental shorter wait
was rejected after it caused substantial latency regressions under CPU
contention; it is not part of the delivered executable.

Validation passed: independent scalar/two-pass operator oracles, SIMD tails,
unaligned and in-place output, nonfinite lanes, ASan/UBSan, all 15 CPU/CUDA
synthesis cases, 100 G2P cases, deterministic lifecycle/frame-boundary tests,
CLI/HTTP/streaming recovery, isolated standalone execution, and a separate CMake
library consumer. The final default-policy build produced bit-identical
intermediates and audio to the numerically validated wait-28 run.

Direct comparison against the preserved original CPU build's 15 audio cases
found a maximum relative magnitude-spectrogram error of 0.000378 (0.0378%).
RMS ratios were 0.999983–1.000009; durations and sample counts were unchanged.
The existing CPU/CUDA numerical limits also passed without alteration.

Performance is measured on the i7-12700 with eight threads and one request in
flight. The paired driver alternates builds, excludes initial requests, inserts
an equal idle gap, and waits for/rejects competing CPU work above 0.5 core.
Rejected attempts remain in the report; the earlier contended timing runs are
not used for the reported speedup. Each result is the median of five accepted
warm requests per build; startup and first-request latency are excluded.

| Speech length | Before | SIMD | Less time |
|---|---:|---:|---:|
| 1.575 s | 417.2 ms | 269.9 ms | 35.3% |
| 5.725 s | 1504.3 ms | 970.1 ms | 35.5% |
| 18.825 s | 4941.1 ms | 3233.7 ms | 34.6% |

Evidence: `tests/results/cpu-optimization-final/report.json` and its WAVs,
`cpu-optimization-parity/`, `cpu-optimization-audio.json`,
`cpu-optimization-inference/`, `cpu-optimization-default/`, and `cmake-cpu.json`.
The original binary identity matches the saved pre-optimization audio report.

## SIMD CPU backend — verified 2026-10-01

Added a CPU-only build alongside CUDA, sharing the forward pass, model parsing,
normalization, chunking, af_heart style selection and all three bundled assets.
CPU matrix operations use statically linked OpenBLAS 0.3.30 AVX2/FMA kernels;
F16C converts the same FP16 weights to FP32 arithmetic. At this stage, convolution im2col scratch
was tiled to 128 frames (superseded by direct convolutions above). No runtime Python, external model files, model downloads,
CUDA libraries or dynamic OpenBLAS library are needed by `rokoko.cpu`.
The supported CPU target is Linux x86-64 with AVX2, FMA and F16C.

Validation completed:

- Independent operator tests cover every half bit pattern, SIMD tails and ties,
  scalar/double GEMM oracles, leading dimensions, interleaved attention heads,
  convolution stride/dilation/padding and aliased residuals, normalization,
  duration rounding and spectral reconstruction. AddressSanitizer and
  UndefinedBehaviorSanitizer passed on this operator suite.
- The final eight-thread CPU build passed library, invalid-asset, CLI, HTTP and
  streaming checks, including frame boundaries, arena growth, changed text/style,
  cancellation/recovery and context recreation. Repeated CPU outputs in the
  lifecycle set were bit-identical. Invalid thread settings fail explicitly.
- The 12 existing quality sentences and three benchmark texts produced identical
  normalized text, phonemes, token IDs, style rows, rounded durations and audio
  lengths on CPU and CUDA. Maximum pre-rounding duration difference was 0.01605
  frames; largest per-case F0 RMSE was 0.123 Hz; noise RMSE was below 0.00187.
  Relative magnitude-spectrogram error was at most 0.0826; RMS ratios were
  1.0167–1.0235. All predefined engineering regression guards passed, as did
  their silence/gain negative controls. This is not a perceptual equivalence claim.
- A further 100 evenly sampled frontend corpus sentences produced identical
  CPU/CUDA G2P output. The reusable parity target includes this check.
- Copied CPU and CUDA executables passed CLI, HTTP and streaming checks with
  all outbound connections blocked, inaccessible HOME/cache, empty PATH and no
  external model reads or temporary model extraction. Linked ELF asset bytes
  independently match the approved manifest. The CPU binary is approximately
  200 MB and loads only ordinary C/C++ system libraries.
- Both CMake backends and separate library-only consumers built and synthesized
  valid audio. CPU configuration and compilation require no CUDA compiler.
- The existing fast offline suite and complete GPU/artifact/reference/mutation
  suite passed. A real padding mutation still reaches the intended failing assertion.

Timings on the i7-12700 (CPU, eight OpenBLAS threads) and RTX 5070 Ti (CUDA),
median wall time for five warm HTTP repeats of the same fixed texts:

| Speech length | CPU | CUDA |
|---|---:|---:|
| 1.575 s | 410.7 ms | 7.32 ms |
| 5.725 s | 1477.2 ms | 17.87 ms |
| 18.825 s | 4844.5 ms | 52.77 ms |

The long-text CPU run is about 3.9x faster than playback. Warm GPU requests
benefit from CUDA graph replay; CPU runs the forward pass on every request.
First-request and startup measurements are retained separately. A bounded
thread sweep (1, 2, 4, 8) supported the eight-thread default on this machine;
`ROKOKO_CPU_THREADS` permits 1–64. Results are descriptive, not speed guarantees.

Evidence: `tests/results/cpu-parity/` (JSON, intermediate arrays, paired WAVs and
`index.html` listening page), `cpu-g2p/report.json`, `cpu-inference/report.json`,
`cmake-cpu.json`, `cpu-gpu-bench-8threads.json` and `cpu-thread-tuning/`.
Test commands and the explicit numerical tolerances are in `tests/README.md`.


## Public build assets verified (2026-10-01)

The approved G2P V11 bytes are now published as
[`g2p.bin` in the `g2p-v11` release](https://github.com/LokalOptima/rokoko/releases/tag/g2p-v11)
in the canonical public repository. The 34,630,156-byte asset has SHA-256
`dfea20100c01c33d2ad3fa32e8ae09289b9ff3b3abf66ffe50686b4e44d7fd48`, matching
the existing approval fixture and build manifest. The versioned weights and
voice assets remain at their pinned `v2.0.1` URLs.

A new CMake build directory, with no local asset source and no GitHub token
environment variables, downloaded all three assets and built successfully.
Downloaded sizes and SHA-256 hashes matched the manifest. The resulting binary
passed the copied-executable CLI, HTTP and streaming checks with external
connections denied, no model file reads and no extraction. The embedded ELF
bytes also matched the approved hashes and occupied read-only aligned sections.
Evidence is saved in `tests/results/release-assets.json`.

The prior publication blocker is resolved. This was an asset-only release;
its tag points to the existing public `v2.0.1` commit, and the latest software
release remains unchanged. It does not publish the consolidated local source
history or an executable.

## Single FP16 runtime (2026-10-01)

The only maintained executable is now `rokoko`, using the former FP16 backend.
The old runtime converted many FP32 weights to FP16 during initialization; it was
not an independent full-precision numerical reference. Official Kokoro remains
the independent reference. FP32 intermediate values/operations used by the
retained model are unchanged.

- Removed the legacy inference backend, its unused TF32 convolution source,
  v1 runtime schema, precision switches, duplicate executable/test targets and
  their generated build outputs. `src/rokoko.cpp` is the one implementation.
- Builds and ordinary GPU tests require exactly `weights.fp16.bin`, `g2p.bin`
  and `voices/af_heart.bin`. The FP32 export is now optional offline tooling;
  its approved identity is retained separately under `export_inputs`.
- CPU checks passed (14 existing checks, bounded-reader check, 3 build-input
  checks). The complete single-runtime GPU/reference/mutation target passed,
  including all 585 independently derived converted tensors and long-input,
  graph replay, style, lifecycle and HTTP coverage.
- Make and CMake standalone checks passed with outbound connections denied and
  no external model reads or extraction. A freshly built library-only consumer
  initialized and synthesized without asset paths. The Make binary is
  202,092,448 bytes on the recorded toolchain.
- The native loader rejected the old valid v1 FP32 artifact. The optional offline
  exporter audit still passed against official Kokoro, and independent schema
  regeneration matched the checked-in FP16 schema byte for byte.
- Orkestrator and the saved FP16/FP32 listening samples were left intact. The
  G2P publication was pending at that stage; it is resolved by the public
  build-asset verification above.

The sections below record earlier stages and historical two-runtime results.

## Embedded-asset update (2026-10-01)

Rokoko now embeds one selected precision's weights, G2P V11 and af_heart, with
no runtime model paths, downloads or voice selection. Orkestrator was left
untouched at the user's request; its integration needs adapting separately.

Validation for this update:

- Make FP16/FP32 and CMake FP16/FP32 builds passed. A separate CMake project
  linking only `rokoko_lib` initialized and synthesized without asset paths.
- The actual embedded ELF bytes match the approved SHA-256 identities, are
  4096-byte aligned and reside in read-only load segments. Binary sizes with
  this toolchain are about 203 MB FP16 and 367 MB FP32.
- Copied executables passed CLI, HTTP and streaming synthesis with an empty
  PATH, unavailable HOME/cache directories and every outbound `connect` syscall
  denied. File traces showed no external model reads or temporary extraction.
- All 14 existing CPU checks, bounded byte-reader checks and 3 build-preparation
  tests passed. Concurrent input preparation, checksum failures, atomic download
  promotion and offline reuse are covered. A real Make invocation rejected a
  corrupt asset despite an already-built executable. Unchanged Make and CMake
  builds reverified inputs without recompiling or relinking.
- Both GPU suites passed, including malformed memory inputs, styles, exact
  lengths, graph replay, context recreation, cancellation and long inputs.
- Independent audit passed all 688 FP32 and 585 converted tensors, af_heart,
  vocabulary and G2P V11 identity. Reference instrumentation and both deliberate
  production padding mutations passed their intended checks.
- G2P evaluator results stayed at 980/1000 stress-sensitive matches and
  991/1000 plain-sentence matches; normalizer 115/115 and reviewed snapshots
  450/450. The same two handwritten end-to-end pronunciation misses remain.
- Anonymous fresh downloads of FP16 weights and af_heart passed checksum
  verification. The exact V11 G2P asset and release notes are staged locally;
  public G2P download verification awaits permission to publish that asset.

The historical measurements below are retained; no new perceptual equivalence
claim or performance gate is introduced by bundling.

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
