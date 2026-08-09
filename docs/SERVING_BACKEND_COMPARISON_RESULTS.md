# SGLang versus vLLM compatibility and serving results

## Status and claim boundary

This record closes the four-case compatibility smoke for job `5896201`, the
bounded QuaRot W4A16 BoolQ compatibility sequence through job `5941763`, the
result-gated serving-client smoke `5952554`, and the formal matched serving
matrix `5960180`. The formal result supports a cross-backend serving claim for
the exact accepted QuaRot-style rotated W4A16 checkpoint on one Isambard GH200
under the frozen 256+64 request workload. It does not establish formal BoolQ
equivalence, generalize to BF16 or W4AFP8, or compare different checkpoints.

The first result-gated serving-client smoke, job `5944823`, was submitted alone
from clean revision `78ed96c` on 2026-08-07 and failed closed `1:0` after
`00:04:12`. Its reviewed artifacts identify two harness failures and no SGLang
runtime failure. It does not contain a valid paired performance result, the
concurrency-8 cell, or the three paired repetitions required for the formal
matrix. No formal job was queued at that stage.

The subsequent harness-only replacement `5949509` also failed before a paired
result: vLLM completed the smoke workload, while SGLang loaded the exact W4A16
checkpoint and captured CUDA graphs but remained at Uvicorn application
startup until timeout. Repair revision `752a01d` restored the previously
healthy checkpoint-path model ID and added bounded startup diagnostics.
Result-gated smoke `5952554` then completed both backend workloads and passed
the frozen-client and recovery gates. Formal job `5960180` subsequently passed
all 12 backend/concurrency/repetition cells and is reviewed below.

The reviewed SGLang evidence establishes one negative boundary for the exact
W4AFP8 checkpoint, identifies one correctable BF16 launch-environment failure,
and confirms that SGLang `0.5.16` can load, serve, score, and formally benchmark
the exact accepted QuaRot-style W4A16 checkpoint. The bounded 32-example result
records one prediction disagreement; it has no equivalence threshold and is
not a formal quality claim.

## Provenance

| Item | Reviewed value |
|---|---|
| Isambard job | `5896201`, `COMPLETED`, exit `0:0`, elapsed `00:05:53` |
| Project revision | `f33d0029cf6f164aa85ccf715a7b9eb1d752af80` |
| Result JSON | `results/sglang-vllm-llama2-13b/compatibility-smoke-5896201.json` |
| Result SHA-256 | `47b141a93f2c9a47109e189972531fb3fa6f8a25412efb3f120d0e0d174ed720` |
| Source manifest SHA-256 | `456239cead1abe8664f78a64f0f273d87eba04679d8eff84ee418e5df99202b6` |
| vLLM runtime | vLLM `0.25.1+cu129`, PyTorch `2.11.0+cu129`, Python `3.11.7` |
| SGLang runtime | SGLang `0.5.16`, PyTorch `2.11.0+cu129`, Python `3.11.7` |

The SGLang provenance also records the environment-provided CUDA toolkit and
`SGLANG_ENABLE_JIT_DEEPGEMM=0`. Raw JSON, logs, Slurm stdout/stderr, and the
source manifest remain on the producing server; generated artifacts are not
committed to this repository.

## Compatibility matrix

| Backend | Checkpoint | Runtime outcome | Evidence status |
|---|---|---|---|
| vLLM | BF16 Llama-2-13B | Endpoint ready; eight-token greedy completion; ready GPU memory 34,323 MiB; memory returned to baseline | Passed smoke control only |
| vLLM | Exact FP8-targeted W4AFP8 | Endpoint ready; same eight-token completion; CUTLASS `CompressedTensorsW4A8Fp8` kernel selected; ready GPU memory 16,934 MiB | Passed smoke control only |
| SGLang | BF16 Llama-2-13B | Weights loaded; default FA3 backend then failed because installed `sgl_kernel` has no `flash_ops` | Inconclusive for model compatibility; launch environment must be corrected |
| SGLang | Exact FP8-targeted W4AFP8 | Loader raised `No compressed-tensors compatible scheme was found` | Unsupported unchanged in SGLang `0.5.16`; paired Track B stops |
| vLLM | Exact QuaRot-style W4A16 | Frozen 32-example BoolQ smoke completed 64 choice requests with 28/32 correct | Passed repeated smoke control only; not formal accuracy |
| SGLang | Exact QuaRot-style W4A16 | CUDA 12.6 Marlin preflight passed; unchanged checkpoint loaded; all 64 choice requests scored; 27/32 correct | Compatibility and bounded smoke score accepted; not formal accuracy or performance |

The two ready-memory values and server-start times are single smoke
observations. They must not be reported as comparative performance numbers.
The Track A/B text-generation equality remains unavailable because neither
SGLang case in the four-case smoke reached generation. Track C uses frozen
choice log-likelihood scoring rather than generated-text equality.

## Evidence interpretation

The W4AFP8 failure occurs inside SGLang's compressed-tensors scheme selection
after the corrected interpreter and CUDA environment are active. It therefore
confirms the earlier source audit rather than repeating the invalid harness
failures from jobs `5895081` and `5895358`. Repacking or exporting a
SGLang-native checkpoint would change the artifact and is outside the current
study.

The BF16 failure has a different boundary. SGLang loaded the full model before
the attention backend failed, so this result must not be described as BF16
checkpoint incompatibility. A read-only login-node check confirmed that
FlashInfer is installed and `sgl_kernel.flash_ops` is absent, but only a GH200
server smoke can establish a working alternative attention backend.

## Bounded next decision

Track A remains eligible for a BF16-only correction, but the user has selected
Track C first. Track C job `5905638` is not accepted: its source gate passed and
the vLLM control completed 32 frozen BoolQ examples and 64 requests with 28
correct (87.5%), but the runner resolved the SGLang virtual-environment Python
symlink to the Cray base interpreter. SGLang therefore failed at module import
before server or model loading. This is a harness failure, not W4A16
compatibility evidence, and the vLLM number is smoke-only rather than a formal
accuracy result.

The path handling was corrected in revision `a3a23a7`, with the exact model,
dataset, protocol, and resource request unchanged. Replacement `5913876`
completed at the Slurm level but recorded `comparison_status=incomplete`.
SGLang reached `CompressedTensorsWNA16` and failed during GPTQ-to-Marlin repack
because its installed Ninja `1.13.0` executable was not on `PATH`. This proves
the expected weight-only scheme is selected, but it is still an environment
failure rather than end-to-end compatibility evidence. vLLM again scored
28/32; that remains smoke-only.

Revision `5445e73` exposes the existing Ninja executable and strengthens the
batch acceptance gate so an incomplete comparison returns failure. Replacement
`5917675` correctly failed closed (`FAILED`, exit `1:0`, elapsed `00:02:54`)
with `comparison_status=incomplete`. Its reviewed provenance is:

| Item | Reviewed value |
|---|---|
| Project revision | `5445e73bf9bdcc220b71ecc7f6c6ced49782b712` |
| Result JSON | `results/sglang-vllm-quarot-w4a16-boolq/boolq-compatibility-smoke-5917675.json` |
| Result SHA-256 | `f1937e4d58355a855a1a995ffb8ae72de93ed519b66186cce28eb8a2a94dd7cd` |
| Source manifest SHA-256 | `e34e93fd8a239ac2b4851d1715fc1332b859297e630b29fad73af0b84c4519aa` |
| vLLM log SHA-256 | `f8ce4b577dfb434aa5cc0c8361641b68a75972430f6331937b930e7f166004de` |
| SGLang log SHA-256 | `941bc69ddeb437065fc40843e18bf3a161d111e039fd4a45f0785c229222335d` |

vLLM again completed 32 examples and 64 requests with 28 correct. SGLang
selected `CompressedTensorsWNA16`, found Ninja, and entered GPTQ-to-Marlin JIT
compilation. NVCC then failed while compiling `source_location.h` because the
default host C++ compiler was GCC `7.5.0` and could not provide the C++20
`<version>` header. A read-only check confirmed that GCC 14 is installed and
can preprocess that header. This is a host-compiler binding failure, not
end-to-end W4A16 compatibility or incompatibility evidence.

The bounded corrective option was to bind the compiler and JIT toolchain;
subsequent corrections and the first successful SGLang W4A16 runtime boundary
are recorded below. No formal job had been submitted at that point.

## Track C reviewed runtime boundary

Jobs `5927118` and `5932590` remained pre-model toolchain failures. The former
compiled the exact Marlin source but could not link `-lcudart` from the pip CUDA
13 layout. The latter fixed that layout and compiled/loaded the module, but its
synthetic CUDA call exposed a CUDA 13 runtime/driver mismatch. Revision
`bfe2e3a` moved only SGLang's source JIT to system CUDA `12.6` and isolated the
TVM-FFI cache, without changing the model, dataset, or evaluation protocol.

Job `5940088` is the first accepted positive SGLang runtime boundary for the
exact checkpoint:

| Item | Reviewed value |
|---|---|
| Slurm outcome | `FAILED`, exit `1:0`, elapsed `00:08:15`; fail-closed comparison gate |
| Project revision | `bfe2e3a344ce2cb96701a6aaa9f82ef4e027c8b0`, clean checkout |
| Result JSON SHA-256 | `9e2afb4076051a5674c33e0250e51d487cdd60305672194300e88a4f748f2406` |
| SGLang preflight SHA-256 | `f44d01eca2e68b38a238eeaa2013783dce81b51fe08a9c8f2e7912de61087fdd` |
| Source manifest SHA-256 | `a3ba4eca84a6f23870c2600cfe5c6a31f3768c8c9bee1dafb9fef309ab697122` |
| SGLang log SHA-256 | `04aacb015e1caa41c74fad6633a07f84759a9230d3a5e1c9ff5cc5084bc24c55` |
| vLLM log SHA-256 | `db57b88d4608a80a8f3d210808836f524758600be383a7a4d5a9594ea19f402f` |
| Checkpoint tree SHA-256 | `2f22f56a5edb32e037416c78be49e617bcee796abca26822704a6ef825ff8e99` |
| SGLang toolchain gate | CUDA `12.6`, NVCC `12.6.77`, CUDART runtime `12060`, GCC/G++ `13.3.1`; exact Marlin synthetic GPU execution passed |
| SGLang model runtime | `compressed-tensors` W4A16 loaded; 6.82 GB weight memory; healthy server; first eight-request batch returned HTTP 200 |
| Comparison | `incomplete`; SGLang score rejected by harness parser before metrics were recorded |

The parser failure is exact and bounded. SGLang `0.5.16` returns an unscored
sentinel at `logprob_start_len`, followed by scores for later tokens. The
request offset `continuation_start - 1` must remain so a single-token BoolQ
choice is scored; revision `b348fc3` corrects only the parser to validate and
remove that sentinel. It also binds the responsible SGLang source hash.
Replacement smoke `5941763` was submitted alone from that clean revision and
completed successfully. Its reviewed result is:

| Item | Reviewed value |
|---|---|
| Slurm outcome | `COMPLETED`, exit `0:0`, elapsed `00:03:33`; stderr empty |
| Project revision | `b348fc32313ef9456ea755558163d2942fe0b5a1`, clean checkout |
| Result path | `results/sglang-vllm-quarot-w4a16-boolq/boolq-compatibility-smoke-5941763.json` |
| Result SHA-256 | `62346ab360a025efc2636a9ab7a0d493ea7b2ae711c979b6c84139721d7aa5b0` |
| SGLang preflight SHA-256 | `ed9c5ee7ff1b9b4f8c7578fd407b704b62720f617099247ffd33a6eb59302f1a` |
| Source manifest SHA-256 | `cf2a51f3af1a98a86b38927d10edbb0e62ff3ec01e129ec59f50be8c4f59db6b` |
| SGLang log SHA-256 | `f937b0cad1dccc64f99fa62f4fae4de0936c9c09edece13c62d4053d96f2a916` |
| vLLM log SHA-256 | `9f9b9bc155b55a474483b3d72572f72c5ec2cf958439554f1bcc4f025e7331b9` |
| vLLM smoke score | 28/32, 87.5%, 64/64 finite choice scores |
| SGLang smoke score | 27/32, 84.375%, 64/64 finite choice scores |
| Cross-backend difference | One prediction disagreement at example 6; accuracy delta SGLang-minus-vLLM `-0.03125` |
| Choice-loglikelihood difference | Mean absolute `0.0351942591`; maximum absolute `0.0953803062` |
| SGLang memory observation | Baseline 4 MiB; ready 16,735 MiB; released 4 MiB; single smoke only |

All ten SHA-256 entries in the source manifest were recomputed successfully,
including the configuration, runners, immutable source result, dataset
manifest, and preflight. SGLang's log records eight scoring batches returning
HTTP 200, with no backend exception; vLLM's log also contains no error. This
accepts unchanged-checkpoint compatibility and the recorded bounded score
differences. It does not establish formal BoolQ accuracy, statistical
equivalence, throughput, latency, concurrency, or reliability.

Track B W4AFP8 must not be rerun. Track C does not add BF16, unrotated W4A16,
SpinQuant, checkpoint conversion, or a new quantization run.

## First serving-client smoke review

Revision `78ed96c` adds the backend-neutral streaming client, frozen request
generator, predeclared corpus hashes, explicit matched cache/scheduler flags,
per-request raw timings, token-usage assertions, GPU-memory sampling, and
version/source gates. Local full-suite validation passed 179 tests with one
skip; 34 focused tests, the 280-packed-linear checkpoint gate, external source
hashes, and `sbatch --test-only` also passed on Isambard before submission.

Job `5944823` recorded:

| Item | Reviewed value |
|---|---|
| Slurm outcome | `FAILED`, exit `1:0`, elapsed `00:04:12` |
| Project revision | `78ed96c25fcf217d7256fbf1afa67e93ab645661`, clean checkout |
| Result SHA-256 | `c8b2c278917420d583302f67aed19706d336e09c7a854ff17c8e59c26e8c4aaf` |
| Request-corpus SHA-256 | `1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1` |
| SGLang preflight SHA-256 | `7b1ec75917262bfc8dcdbfc4d7b99bebc487eb63604dad7f158c5ae252885171` |
| vLLM raw SHA-256 | `8fe11fc7f0611d20cefda4ef2ab7f9b12273997cf54932b72cf93e1f68c59972` |
| vLLM log SHA-256 | `32f84b013ccc97e6fef0c1aace8172e3e1dd1e18c0a049b36c8cc6dbc0701740` |
| SGLang log SHA-256 | `360810e1466b32ffe5607b87dac96b512d9707500a68b1b6ab5cd5f6588340f9` |

The SGLang preflight passed exact Marlin synthetic GPU execution. SGLang then
loaded the unchanged checkpoint with 6.82 GB weight memory, allocated 10,485
BF16 KV tokens (4 GiB K plus 4 GiB V), captured decode graphs, and returned
HTTP 200 from `/health`. The shared launcher had not passed
`--served-model-name`, so `/v1/models` correctly returned the checkpoint path
and the harness stopped before any serving request. This is a model-name
adapter failure, not SGLang incompatibility.

vLLM completed all 13 client calls: one validation, four warm-ups, and eight
measured requests. Every call recorded exactly 256 prompt and 64 completion
tokens. The measured phase completed, but its first-eight generated-text hash
differed from job `5780631`. Since that historical artifact did not retain
prompts, output equality cannot establish request identity and must not be a
pass/fail gate. The new corpus remains independently frozen by its file,
prompt-list, and token-ID-list hashes for use identically across both backends.

The bounded repair adds SGLang's explicit served model name and retains the old
output comparison as a diagnostic only. Model, environments, request corpus,
workload, resources, and experiment matrix remain unchanged. A single
replacement result-gated smoke, job `5949509`, was submitted alone from clean
repair revision `ac82347cb4bbe467c5242ebc35faf23d7be160af`, with no formal
dependency. It failed `1:0` after `00:24:34`; result SHA-256 is
`8e17f4939dc5ffcf7d048c61a11c3a34a97c8764d26924755100bbc7ae772908`.
vLLM passed all eight measured requests, with 2.71819 requests/s, 367.67 ms p50
E2E, 22.50 ms p50 TTFT, 5.477 ms p50 TPOT, and 16,165 MiB ready GPU memory.
These are retained as one-backend smoke observations, not a cross-backend
comparison.

SGLang passed the exact Marlin preflight, loaded 6.82 GB of W4A16 weights,
allocated 10,485 BF16 KV tokens (4 GiB K plus 4 GiB V), and completed decode
CUDA-graph capture. It emitted no OOM, Python traceback, or CUDA exception, but
stopped at Uvicorn `Waiting for application startup` until the 900-second
readiness timeout. Its log SHA-256 is
`bf3f00dd5b78cc3c652d78df01302cd02581d545019c3939ca236eb5bac7cec3`.
Because no process stack was captured, this run does not distinguish a slow
application initialization from a hidden deadlock.

The final harness-only repair restored the previously healthy checkpoint path
as the common served model ID instead of adding an alias, validates the single
ID returned by `/v1/models`, uses a 1,200-second SGLang-only readiness budget,
and captures process, port, GPU, and SGLang stack diagnostics before cleanup on
timeout. Repair
revision `752a01d950e3886989ad4541f901c9a31d6ba195` passed 180 local tests with
one skip, 15 focused Isambard tests, external-source hashes, the immutable
280-linear checkpoint gate, and `sbatch --test-only`. Replacement smoke
`5952554` was then submitted alone with no formal dependency and completed
`0:0` in 3 minutes 41 seconds. Its accepted smoke result is:

| Metric | vLLM | SGLang | SGLang / vLLM |
|---|---:|---:|---:|
| Measured requests | 8 / 8 passed | 8 / 8 passed | -- |
| Request throughput | 2.7046 req/s | 3.1646 req/s | 1.1701x |
| p50 E2E | 369.80 ms | 304.38 ms | 0.8231x |
| p50 TTFT | 22.52 ms | 28.76 ms | 1.2769x |
| p50 TPOT | 5.511 ms | 4.374 ms | 0.7937x |
| Ready GPU memory | 16,166 MiB | 16,735 MiB | +569 MiB |
| Startup | 96.02 s | 35.01 s | -- |

Each backend used the same request IDs, eight 256-token prompts, and forced 64
output tokens. All requests passed; memory recovered from 16,180 to 4 MiB for
vLLM and from 16,775 to 5 MiB for SGLang. The SGLang log's final `SIGQUIT`
followed normal post-workload `SIGTERM`: the terminated detokenizer exited
`-15`, which triggered SGLang's own cleanup diagnostic after all HTTP 200
responses. It is shutdown noise rather than a measured-request failure.

The result SHA-256 is
`5430c74685e26221a397423cddb355a637015c087566df3b4ddbf24b37f5ac04`;
vLLM and SGLang raw-result SHA-256 values are
`10cbe2b73633689e26125e2b03fc774b007be0ef0582225935262529f3d6c038`
and
`c726ec313c86dfb1d60036fae3fcfda9397180ee63857b82c79302711e6b8253`.
All 15 source-manifest bindings were rehashed successfully and stderr was
empty. This closes the result-gated smoke itself and binds the accepted input
gate for the separately authorized formal matrix reviewed below.

## Formal matched serving benchmark

Formal job `5960180` completed `0:0` in `00:16:56` on one GH200. It used the
same accepted rotated W4A16 checkpoint, frozen request corpus, tokenizer,
repository-owned OpenAI-compatible streaming client, 8 GiB BF16 KV budget,
disabled prefix cache, and disabled chunked prefill as smoke `5952554`. Every
fresh server received one unmeasured validation request, four warm-up requests,
and 64 measured requests. Each measured request contained exactly 256 input
tokens and forced exactly 64 output tokens.

The matrix contains concurrency 1 and 8, three paired repetitions per cell,
and a fresh server for every backend/case/repetition. Backend order alternated
as vLLM/SGLang, SGLang/vLLM, and vLLM/SGLang. All 12 cells passed 64/64 measured
requests with zero failures, for 768 measured requests in total. The study had
no speedup pass threshold and retains every repetition rather than selecting a
best run.

### Formal provenance

| Item | Reviewed value |
|---|---|
| Slurm outcome | `5960180`, `COMPLETED`, exit `0:0`, elapsed `00:16:56`; stderr empty |
| Project revision | `5a97ab49393236a8edb44552ef4c032acf9e8058`, clean checkout |
| Result path | `results/sglang-vllm-quarot-w4a16-serving/serving-formal-5960180.json` |
| Result SHA-256 | `fcaf3fef48dfa7d11f45fe20566789e95878938d106fdd539743425fb49d64be` |
| Source-manifest SHA-256 | `071c51aa46b2e290929ddf227521fa818aeb86dd70aee61f5307a938a45ed829` |
| Effective-config SHA-256 | `1b5123d3b1bf533a5c470797262238f208d5b098b64e0b4f4624793b22fbfc5b` |
| Formal-input-gate SHA-256 | `f79315cfc0c163faa00f621e82a146b2ddcae1c7842d46381537c0f3b6ed87d6` |
| Request-corpus SHA-256 | `1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1` |
| SGLang preflight SHA-256 | `4e49a8fd010b30bcc0d1606acb9e724ff66f6003fddcabbf692594c02df81d4a` |
| Checkpoint tree SHA-256 | `2f22f56a5edb32e037416c78be49e617bcee796abca26822704a6ef825ff8e99` |
| Runtimes | vLLM `0.25.1+cu129`; SGLang `0.5.16`; PyTorch `2.11.0+cu129` |

The formal input gate independently revalidated accepted smoke `5952554`, its
request corpus and preflight, all 15 smoke-manifest entries, the unchanged
smoke producers, and the accepted source/checkpoint ancestry. The final formal
manifest contains 45 bindings, including all 12 server logs and 12 per-cell raw
records; every binding was independently rehashed successfully. vLLM selected
`MacheteLinearKernel for CompressedTensorsWNA16`; SGLang passed the exact
GPTQ-to-Marlin synthetic execution gate and loaded the checkpoint as
`compressed-tensors` W4A16 with FlashInfer attention.

### Every paired repetition

The table below reports each retained measurement. Latencies are milliseconds;
memory is total observed GPU memory in MiB.

| Rep | Concurrency | Backend | Request/s | p50 TTFT | p50 TPOT | p50 E2E | Ready / peak MiB | Startup s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | vLLM | 2.7065 | 22.92 | 5.495 | 369.10 | 16,165 / 16,179 | 106.02 |
| 1 | 1 | SGLang | 3.2512 | 28.85 | 4.381 | 304.90 | 16,734 / 16,774 | 35.01 |
| 2 | 1 | vLLM | 2.7130 | 23.05 | 5.480 | 368.25 | 16,167 / 16,181 | 90.02 |
| 2 | 1 | SGLang | 3.2678 | 28.69 | 4.395 | 305.52 | 16,736 / 16,776 | 29.01 |
| 3 | 1 | vLLM | 2.7121 | 22.60 | 5.489 | 368.43 | 16,166 / 16,180 | 90.02 |
| 3 | 1 | SGLang | 3.2724 | 28.70 | 4.390 | 305.25 | 16,735 / 16,775 | 27.01 |
| 1 | 8 | vLLM | 15.1804 | 108.17 | 6.452 | 514.83 | 16,166 / 16,462 | 86.01 |
| 1 | 8 | SGLang | 15.6536 | 178.37 | 5.199 | 508.29 | 16,735 / 16,969 | 29.01 |
| 2 | 8 | vLLM | 15.5740 | 109.83 | 6.395 | 512.94 | 16,168 / 16,464 | 96.02 |
| 2 | 8 | SGLang | 15.7404 | 178.20 | 5.206 | 508.38 | 16,736 / 16,970 | 27.01 |
| 3 | 8 | vLLM | 15.6577 | 106.84 | 6.425 | 511.25 | 16,165 / 16,461 | 98.02 |
| 3 | 8 | SGLang | 15.7549 | 177.79 | 5.190 | 508.12 | 16,735 / 16,969 | 27.01 |

### Median throughput and spread

Values are the median followed by `[minimum, maximum]` over all three
repetitions. Token throughput follows directly from the fixed 256-input and
64-output lengths but is retained explicitly in the formal result.

| Concurrency | Backend | Request/s | Input token/s | Output token/s | Total token/s |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM | 2.712 `[2.706, 2.713]` | 694.3 `[692.9, 694.5]` | 173.6 `[173.2, 173.6]` | 867.9 `[866.1, 868.2]` |
| 1 | SGLang | 3.268 `[3.251, 3.272]` | 836.6 `[832.3, 837.7]` | 209.1 `[208.1, 209.4]` | 1,045.7 `[1,040.4, 1,047.2]` |
| 8 | vLLM | 15.574 `[15.180, 15.658]` | 3,986.9 `[3,886.2, 4,008.4]` | 996.7 `[971.5, 1,002.1]` | 4,983.7 `[4,857.7, 5,010.5]` |
| 8 | SGLang | 15.740 `[15.654, 15.755]` | 4,029.5 `[4,007.3, 4,033.3]` | 1,007.4 `[1,001.8, 1,008.3]` | 5,036.9 `[5,009.2, 5,041.6]` |

### Median latency and spread

| Concurrency | Backend | Metric | p50 ms | p95 ms | p99 ms |
|---:|---|---|---:|---:|---:|
| 1 | vLLM | TTFT | 22.92 `[22.60, 23.05]` | 23.56 `[23.42, 24.54]` | 26.20 `[24.89, 32.98]` |
| 1 | SGLang | TTFT | 28.70 `[28.69, 28.85]` | 28.92 `[28.91, 29.62]` | 33.40 `[29.43, 41.21]` |
| 1 | vLLM | TPOT | 5.489 `[5.480, 5.495]` | 5.501 `[5.499, 5.530]` | 5.513 `[5.510, 5.537]` |
| 1 | SGLang | TPOT | 4.390 `[4.381, 4.395]` | 4.402 `[4.400, 4.407]` | 4.573 `[4.487, 5.388]` |
| 1 | vLLM | E2E | 368.43 `[368.25, 369.10]` | 370.19 `[369.84, 370.66]` | 372.54 `[371.24, 379.48]` |
| 1 | SGLang | E2E | 305.25 `[304.90, 305.52]` | 306.08 `[306.00, 306.25]` | 316.65 `[316.18, 368.30]` |
| 8 | vLLM | TTFT | 108.17 `[106.84, 109.83]` | 115.01 `[111.82, 115.28]` | 115.30 `[112.06, 115.55]` |
| 8 | SGLang | TTFT | 178.20 `[177.79, 178.37]` | 184.91 `[183.67, 206.14]` | 185.28 `[184.13, 206.49]` |
| 8 | vLLM | TPOT | 6.425 `[6.395, 6.452]` | 7.422 `[7.376, 7.937]` | 7.441 `[7.415, 8.297]` |
| 8 | SGLang | TPOT | 5.199 `[5.190, 5.206]` | 6.639 `[6.609, 6.649]` | 6.942 `[6.891, 6.966]` |
| 8 | vLLM | E2E | 512.94 `[511.25, 514.83]` | 518.14 `[516.49, 608.18]` | 518.43 `[516.77, 608.33]` |
| 8 | SGLang | E2E | 508.29 `[508.12, 508.38]` | 512.42 `[510.98, 533.09]` | 512.55 `[511.10, 533.33]` |

The result JSON additionally retains p50/p95/p99 inter-token latency, raw
per-request timestamps and generated-output hashes, server commands, kernel
excerpts, and every GPU-memory sample.

### Resource observations

| Concurrency | Backend | Startup s | Ready MiB | Peak MiB | Released MiB |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM | 90.02 `[90.02, 106.02]` | 16,166 `[16,165, 16,167]` | 16,180 `[16,179, 16,181]` | 3 `[2, 5]` |
| 1 | SGLang | 29.01 `[27.01, 35.01]` | 16,735 `[16,734, 16,736]` | 16,775 `[16,774, 16,776]` | 4 `[3, 5]` |
| 8 | vLLM | 96.02 `[86.01, 98.02]` | 16,166 `[16,165, 16,168]` | 16,462 `[16,461, 16,464]` | 4 `[2, 5]` |
| 8 | SGLang | 27.01 `[27.01, 29.01]` | 16,735 `[16,735, 16,736]` | 16,969 `[16,969, 16,970]` | 5 `[3, 6]` |

SGLang used 569 MiB more ready memory at both concurrency levels. Its median
peak was 595 MiB higher at concurrency 1 and 507 MiB higher at concurrency 8.
SGLang startup was substantially shorter, but startup is reported separately
and is not included in request throughput or request-latency measurements.

### Paired interpretation

| Concurrency | Request-throughput ratio | p50 TTFT ratio | p50 TPOT ratio | p50 E2E ratio |
|---:|---:|---:|---:|---:|
| 1 | 1.2045x `[1.2013, 1.2066]` | 1.2587x `[1.2448, 1.2698]` | 0.7997x `[0.7972, 0.8019]` | 0.8285x `[0.8261, 0.8296]` |
| 8 | 1.0107x `[1.0062, 1.0312]` | 1.6489x `[1.6224, 1.6640]` | 0.8077x `[0.8058, 0.8139]` | 0.9911x `[0.9873, 0.9939]` |

At concurrency 1, SGLang's median request throughput was 20.45% higher, p50
TPOT was 20.03% lower, and p50 E2E was 17.15% lower, while p50 TTFT was 25.87%
higher. At concurrency 8, request throughput was only 1.07% higher and p50 E2E
was 0.89% lower; p50 TPOT remained 19.23% lower, but p50 TTFT was 64.89%
higher. The accepted result therefore does not support a blanket claim that
one backend is faster on every metric. SGLang has a clear concurrency-1
throughput/E2E advantage and lower per-output-token time in both cells, while
vLLM has lower TTFT and essentially tied concurrency-8 throughput/E2E under
this workload.

Several vLLM logs emit `EngineDeadError` only after the client completed all 64
measured requests and the harness sent `SIGTERM`. The same log sequence records
`request processing complete` followed by resource teardown and application
shutdown; all corresponding raw cells passed and memory returned to baseline.
This is post-measurement shutdown noise, not a request or engine failure during
the benchmark. SGLang's post-workload termination diagnostics have the same
evidence boundary. The three-repetition min/max ranges are descriptive spread,
not confidence intervals, and no statistical equivalence or superiority test
was predeclared.
