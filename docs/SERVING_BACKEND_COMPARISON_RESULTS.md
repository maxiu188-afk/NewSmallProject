# SGLang versus vLLM compatibility results

## Status and claim boundary

This record closes the four-case compatibility smoke for job `5896201` and
tracks the bounded QuaRot W4A16 BoolQ compatibility sequence through accepted
job `5941763`. It is not a serving-performance result: no throughput, latency,
concurrency, formal quality, or reliability comparison is accepted here, and
no formal benchmark has been submitted.

The reviewed SGLang evidence establishes one negative boundary for the exact
W4AFP8 checkpoint, identifies one correctable BF16 launch-environment failure,
and confirms that SGLang `0.5.16` can load, serve, and score the exact accepted
QuaRot-style W4A16 checkpoint. The bounded 32-example result records one
prediction disagreement; it has no equivalence threshold and is not a formal
quality claim.

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
are recorded below. No formal job has been submitted.

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
