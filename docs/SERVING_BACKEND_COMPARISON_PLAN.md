# SGLang versus vLLM serving comparison plan

## Status and authorization boundary

This is a separately scoped study. It does not reopen or modify the accepted
vLLM W4A16/W4AFP8 results. The read-only audit and four-case result-gated smoke
are complete. The smoke now provides reviewed negative compatibility evidence
for the exact W4AFP8 checkpoint, but no SGLang performance result has been
accepted.

The first implementation step must be a read-only compatibility audit. Any GPU
smoke is result-gated: submit the smoke only, inspect its artifacts, and obtain
acceptance before a formal comparison is submitted.

The frozen smoke runtimes are vLLM `0.25.1+cu129` and SGLang `0.5.16`, each in
an isolated environment with PyTorch `2.11.0+cu129`. SGLang follows its official
CUDA 12 installation route, including the cu129 SGLang kernel and DeepGEMM
wheels. The prepared files are:

- `configs/deployment/sglang_vllm_llama2_13b_smoke_isambard.json`;
- `scripts/setup_isambard_sglang_env.sh`;
- `scripts/run_sglang_vllm_llama2_13b_smoke.py`;
- `scripts/run_isambard_sglang_vllm_llama2_13b_smoke.sbatch`.

The Slurm script intentionally prints a `SMOKE_RECORDED` marker rather than a
pass marker. Slurm success proves only that the four-case compatibility record
was written; it does not accept SGLang W4AFP8 support.

### Execution status (2026-08-03)

- Isambard preflight: `sbatch --test-only` accepted the script and resource
  request.
- Result-gated smoke: job `5895081`, submitted from clean revision
  `3578427052c8239049635a99c5bad70ed9225d9c`.
- Current state at submission closeout: `PENDING`, no dependency, start time
  unknown. The scheduler's test-only estimate was 2026-08-09, not a guarantee.
- SGLang environment manifest SHA-256:
  `508ab49381b655d79f1148bd49912717890b56115e6980834673fab848afe8f1`.
- No formal performance job is submitted. Compatibility remains unverified
  until job `5895081` completes and its JSON, logs, marker, revision, and hashes
  are reviewed.

### First smoke review

Job `5895081` completed `0:0` in 4 minutes 16 seconds and recorded result JSON
SHA-256 `3f4f19b550f33194851be7ceccc7acb9a07efbdd842c5a49f325aae5052f0e6c`.
Both vLLM tracks passed, generated the same eight tokens, selected the accepted
CUTLASS W4AFP8 kernel for the quantized track, and returned GPU memory from
34,099/16,803 MiB ready usage to the 3 MiB baseline.

The two SGLang tracks are **invalid as compatibility evidence**. The runner
resolved the virtual-environment Python symlink to the Cray base interpreter,
which exited before model loading with `ModuleNotFoundError: No module named
'sglang'`. This is a harness-path failure shared by BF16 and W4AFP8, not a
backend or checkpoint failure. The path handling is corrected in the next
revision and requires one replacement result-gated smoke. Job `5895081` must
not be cited as evidence that SGLang rejects either checkpoint.

The corrected replacement was submitted as job `5895358` from clean revision
`5355df8cbd1ffab62829a7fab900c9c666e708c3` after `sbatch --test-only`
accepted the script. It has no formal-job dependency. No compatibility claim
is made until its retained artifacts are reviewed.

Job `5895358` completed `0:0` in 4 minutes 20 seconds and recorded result JSON
SHA-256 `82eb8ab6c14f05946102a9a533afad73e2e0c3305ac77617920db94e7b5a28cb`.
It confirmed the virtual-environment path fix, but both SGLang cases still
failed before model loading because the optional `sgl-deep-gemm` import asserted
that `CUDA_HOME` was unset. The vLLM BF16/W4AFP8 controls again passed, including
CUTLASS W4AFP8 selection and memory recovery. Therefore job `5895358` also
contains no SGLang model-compatibility evidence.

A read-only login-node import check then passed with `CUDA_HOME` bound to the
toolkit shipped inside the isolated SGLang environment and
`SGLANG_ENABLE_JIT_DEEPGEMM=0`. JIT DeepGEMM is not used by this dense Llama
BF16/compressed-W4AFP8 smoke. These environment values are now frozen in the
config, recorded in result provenance, and validated before the four cases run.

The final corrected replacement was accepted by `sbatch --test-only` and
submitted as job `5896201` from clean revision
`f33d0029cf6f164aa85ccf715a7b9eb1d752af80`. It has no formal-job dependency;
artifact review is recorded below.

### Final smoke review

Job `5896201` completed `0:0` in 5 minutes 53 seconds. The retained result JSON
SHA-256 is `47b141a93f2c9a47109e189972531fb3fa6f8a25412efb3f120d0e0d174ed720`;
the source-manifest SHA-256 is
`456239cead1abe8664f78a64f0f273d87eba04679d8eff84ee418e5df99202b6`.
The result records vLLM `0.25.1+cu129` and SGLang `0.5.16`, both with PyTorch
`2.11.0+cu129`, plus the frozen SGLang environment values.

Both vLLM controls passed. BF16 reached the OpenAI-compatible endpoint and
returned eight tokens; the exact FP8-targeted W4AFP8 checkpoint also reached
the endpoint and selected `CutlassW4A8LinearKernel for
CompressedTensorsW4A8Fp8`. Ready GPU-memory observations were 34,323 MiB for
BF16 and 16,934 MiB for W4AFP8. These are smoke observations only, not formal
performance measurements.

SGLang reached the exact W4AFP8 model loader and failed with
`NotImplementedError: No compressed-tensors compatible scheme was found.` This
confirms the source-audit boundary for SGLang `0.5.16`: Track B cannot compare
the unchanged checkpoint across backends and stops without conversion or
re-export.

The SGLang BF16 case loaded all weights, then failed because the default FA3
attention backend attempted to import `flash_ops` from the installed
`sgl_kernel`, where that symbol is absent. This is an environment/launch
selection failure, not BF16 model incompatibility. A read-only import check
finds FlashInfer installed, but this is not GPU runnability evidence. Therefore
Track A still requires one BF16-only replacement smoke with an explicitly
available attention backend before any formal comparison decision. Track B
must not be rerun as part of that correction, and no formal job is submitted.

The reviewed compatibility record and claim boundaries are summarized in
[`SERVING_BACKEND_COMPARISON_RESULTS.md`](SERVING_BACKEND_COMPARISON_RESULTS.md).

## Research questions

1. On the same GH200, how do SGLang and vLLM compare for the same BF16
   Llama-2-13B checkpoint and the same OpenAI-compatible request stream?
2. Can both backends load the exact accepted FP8-targeted SpinQuant W4AFP8
   checkpoint without repacking, re-exporting, or changing its quantization
   metadata?
3. If exact checkpoint reuse passes, how do throughput, TTFT, TPOT/ITL,
   end-to-end latency, GPU memory, startup time, and reliability differ under a
   matched workload?

This is a serving-backend comparison, not a new rotation or quantization
study. A speed difference must not be attributed to QuaRot or SpinQuant unless
the checkpoint, request stream, runtime form, and resource limits are matched.

## Official interface basis

- SGLang documents `python -m sglang.bench_serving` as a serving benchmark for
  SGLang-native and OpenAI-compatible SGLang/vLLM endpoints. It reports request
  and token throughput, TTFT, ITL, TPOT, end-to-end latency, failures, and
  optional per-request details:
  <https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md>.
- SGLang's server arguments currently list both `w4afp8` and
  `compressed-tensors` quantization modes:
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>.
  This is capability-discovery evidence only; it does not prove that the
  project's exact checkpoint loads or selects an equivalent kernel.
- vLLM documents `vllm bench serve`, saved JSON results, per-request details,
  and configurable TTFT/TPOT/ITL/E2E percentiles:
  <https://docs.vllm.ai/en/stable/cli/bench/serve/>.

The formal study must pin exact source revisions or release versions for both
backends. Moving `main`, `latest`, or an unrecorded container tag is not an
acceptable runtime identity.

## Read-only audit result before smoke

The producing server still contains the exact FP8-targeted checkpoint and both
accepted source-gate files. Their SHA-256 values recomputed on 2026-08-03 match
the recorded values `c292e5ec...a55` and `495bcc45...d62`. The checkpoint is a
dense Llama model using `compressed-tensors`, packed group-128 INT4 weights,
dynamic per-token eight-bit floating-point activations, and `lm_head` exclusion.

SGLang `0.5.16` source recognizes that metadata shape in
`_is_wint4afp8`, but the dense compressed-tensors scheme selector does not
dispatch to it and ends with `No compressed-tensors compatible scheme was
found`. Its separate `w4afp8` implementation applies INT4+FP8 to fused MoE
experts while ordinary dense linears use the FP8 path. Job `5896201` confirmed
this boundary on GH200 for the unchanged dense Llama W4AFP8 artifact. No
conversion or SGLang-native re-export is authorized.

Primary source bindings:

- SGLang `0.5.16` compressed-tensors selector:
  <https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py>;
- SGLang `0.5.16` MoE-oriented W4AFP8 implementation:
  <https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/layers/quantization/w4afp8.py>;
- official CUDA 12 installation procedure:
  <https://docs.sglang.io/docs/get-started/install>.

## Comparison tracks

| Track | Checkpoint | Status | Claim allowed after acceptance |
|---|---|---|---|
| A: BF16 control | Exact pinned BF16 Llama-2-13B snapshot | BF16-only corrected smoke required; no formal submitted | Cross-backend serving comparison for the frozen workload |
| B: W4AFP8 primary | Exact accepted FP8-targeted SpinQuant W4AFP8 artifact and tree hash | Stopped: unchanged load failed in SGLang `0.5.16` | No cross-backend performance claim; negative compatibility boundary only |
| C: W4A16 extension | Exact accepted rotated W4A16 artifact | Optional, separately authorized after A/B | Additional precision-format comparison; not required for the first study |

If SGLang cannot load Track B unchanged, Track A may proceed, but the paired
W4AFP8 comparison stops. A newly exported SGLang-native checkpoint would be a
different experiment and must not be inserted into Track B or compared as if
the checkpoint were identical.

## Gate 0: read-only compatibility audit

Before installing or running either backend, freeze:

- BF16 and quantized checkpoint paths and complete tree SHA-256 values;
- tokenizer files and revision;
- vLLM and SGLang revisions, Python, PyTorch, CUDA, driver, and GPU identity;
- the quantization metadata expected by each loader;
- whether the exact W4AFP8 dynamic per-token FP8 input scheme, group-128 W4,
  `actorder=None`, 280 decoder-linears, and absence of runtime `g_idx` are
  supported without conversion;
- the server flags needed to align KV dtype, maximum model length, scheduling,
  prefix caching, chunked prefill, CUDA graph/eager mode, and memory limits.

The output is a compatibility matrix with `supported`, `unsupported`, or
`unverified` for each requirement. Documentation or source recognition of a
format is not enough to mark runtime loading as supported.

## Gate 1: result-gated server smoke

Run each backend in a separate isolated environment and fresh process. For
BF16 and every compatible quantized checkpoint, require:

- health and model-list endpoints;
- the same deterministic eight-token completion from the same tokenized input;
- finite and structurally valid log-probability output when the backend exposes
  the required API;
- expected packed-linear coverage and quantization metadata;
- selected kernel/runtime-form evidence from logs;
- ready GPU memory and complete process-memory recovery after shutdown;
- source manifest, environment manifest, checkpoint tree hash, stdout/stderr,
  exit status, and result JSON.

Submit smoke only. Formal quality or performance work is authorized only after
the smoke artifacts are reviewed. Do not prequeue formal work with `afterok`.

## Gate 2: matched correctness and quality

The primary performance claim requires evidence that both backends execute the
same model rather than a silently converted or degraded variant.

1. Compare deterministic token outputs and shared candidate log-probabilities
   on a frozen diagnostic set.
2. For Track B, run the existing 162 x 2048 WikiText-2 deployed-PPL protocol
   through both backends if the required target-token log-probabilities are
   available with matched semantics.
3. If one backend cannot expose the required log-probabilities, report the
   quality comparison as blocked. Do not substitute generated-text equality or
   the old vLLM PPL as a matched cross-backend quality result.
4. BoolQ is not required for the first serving-backend study. Adding it would
   be a separately declared extension, not an automatic consequence of this
   plan.

## Gate 3: primary matched serving benchmark

The first formal matrix deliberately reuses the accepted project workload:

| Parameter | Frozen value |
|---|---|
| Hardware | One Isambard GH200; no concurrent GPU workload |
| Input | Same frozen random 256-token request set and tokenizer |
| Output | Forced 64 tokens with identical sampling/EOS behavior |
| Warm-up | Four requests per fresh server |
| Measurement | 64 requests per backend/checkpoint/concurrency cell |
| Concurrency | 1 and 8 |
| KV dtype | BF16 |
| Server lifecycle | Fresh process for every cell; recovery checked after shutdown |

Use one repository-owned OpenAI-compatible client and one frozen request file
for both servers. SGLang's multi-backend benchmark may be used as an independent
cross-check, but numbers from different client implementations must not be
mixed into one primary table.

Run at least three paired repetitions per cell, alternate the backend order,
and publish every repetition. Report median and spread; do not select the best
run. There is no predeclared speedup pass threshold: a completed negative or
tied result remains valid evidence.

## Required measurements

- request, input-token, output-token, and total-token throughput;
- TTFT, TPOT or ITL, and end-to-end latency at p50, p95, and p99;
- ready and peak GPU memory, KV allocation or effective capacity, and recovery;
- server startup/load time and time to first healthy response;
- request success/failure counts and actual generated-token counts;
- selected attention and quantized-linear kernel evidence;
- complete server/client arguments, revisions, environment, prompt hash,
  checkpoint hash, raw per-request records, and summary JSON hashes.

Framework-specific memory knobs such as vLLM GPU-memory utilization and SGLang
static-memory fraction are not numerically equivalent. The primary comparison
must align the observed resource budget or effective KV capacity, not merely
copy the same percentage value into both CLIs.

## Optional second-stage workload matrix

Only after the primary 256+64 comparison is accepted, a separately approved
matrix may add bounded decode-heavy, balanced, and prefill-heavy workloads.
Those results must remain separate from the primary table. Backend-specific
autotuning, prefix caching, speculative decoding, disaggregated serving, and
other framework-exclusive features are outside the first comparison; they may
form a later "best tuned backend" track but cannot be mixed with the strictly
matched track.

## Deliverables and acceptance

The eventual study should add:

- frozen backend environment manifests and a compatibility report;
- backend-neutral server adapters and request corpus/configuration;
- smoke and formal runbooks following the repository's result-gated rule;
- raw ignored JSON/log artifacts on the producing server;
- a reviewed `docs/SERVING_BACKEND_COMPARISON_RESULTS.md` with hashes and claim
  boundaries;
- an update to `EVIDENCE_LEDGER.md` only after quality and serving evidence are
  both accepted.

Until those gates pass, the only accepted deployment numbers remain the
existing vLLM results. This plan must not be cited as SGLang performance or
compatibility evidence.
