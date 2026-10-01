# INT8 vocoder experiment

This is an isolated evaluation, not a production backend or a replacement for
`rokoko.cpu`. It quantizes only convolutions inside `adain_resblock1_forward`
(the generator/noise residual blocks). Pronunciation, duration, pitch, noise
prediction, normalization and other operators retain the current CPU path.

Weights use symmetric signed INT8 with a scale per output channel. Activations
use a dynamic max-absolute scale per tensor, represented as unsigned bytes with
zero point 128. oneDNN accumulates integer products and returns FP32 output.
Unlike the FP32 path, these convolutions replace input FP16 rounding with INT8
quantization. Bundled assets remain FP16 and are quantized during preparation;
this experiment does **not** reduce the executable's asset size. Each weight
retains one current primitive shape, replaced as sentence length changes, and
all experimental caches are released with the context.

Build the normal CPU target with diagnostics first (`make cpu`). Then:

```sh
python3 tests/experiments/build_int8.py
ROKOKO_CPU_THREADS=8 build/int8/operators
build/int8/rokoko.int8 "Hello from the INT8 experiment." -o build/int8/sample.wav
```

The builder creates modified translation units only under `build/int8/`, links
against the existing pinned static dependencies and saves source/library/binary
hashes in `build/int8/build.json`. It inserts an exception-safe INT8 scope around
the generator residual function. Production sources and build flags are unchanged.
The operator test compares against an independent quantized integer oracle,
including zero tensors/channels, tails, worker boundaries, padding, dilation,
stride, distinct/aliased residuals, cleanup and rejection of nonfinite inputs.

Compare all 15 saved CPU/GPU cases and optionally score the quality fixtures and benchmark passages
with the same local ASR model:

```sh
.venv-tests/bin/python tests/experiments/check_int8.py \
  --reference tests/results/cpu-rtfx-parity \
  --asr ../paraketto/paraketto.cuda \
  --asr-weights /path/to/paraketto-fp16.bin
```

The reference report's CPU binary hash must match the current baseline. To
regenerate traces after changing production code, run `tests/cpu_parity.py`
with that reference output directory first. The experiment saves each numerical
limit failure instead of aborting at the first one. It requires byte-identical
pre-vocoder intermediates and unchanged audio lengths. Numerical thresholds and
ASR results are diagnostic; neither establishes perceptual equivalence.
Open `tests/results/int8-quality/index.html` for paired listening at equal gain.

Benchmark separately, after correctness tests finish:

```sh
python3 tests/bench_cpu.py --before build/cpu/rokoko \
  --after build/int8/rokoko.int8 --repeats 5 --output tests/results/int8-bench
```

This alternates single requests with eight workers, retains every WAV, and
waits for/rejects competing CPU work above 0.5 core. Warm timings exclude startup
and first requests. New shapes still require primitive preparation; record that
cost separately. Do not silently relax the existing audio thresholds or replace
the production binary merely because INT8 is faster.

Latest observed quality: ASR transcripts match FP32 on all 15 cases (8 errors
out of 181 words for each; 8/103 on the original 12 quality fixtures). Two of 15 INT8/GPU cases exceed the existing 0.1
spectral-error limit, with a maximum of 0.16294 on the long passage. Against the
FP32 CPU output, one case exceeds that limit (maximum 0.14809). This trial has
not passed the current audio regression limits. See the implementation report
and the saved JSON for final timing results and limitations.
