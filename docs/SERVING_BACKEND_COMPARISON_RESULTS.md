# SGLang versus vLLM compatibility results

## Status and claim boundary

This record closes the four-case compatibility smoke for job `5896201`. It is
not a serving-performance result: no throughput, latency, concurrency, quality,
or reliability comparison is accepted here, and no formal benchmark has been
submitted.

The only positive serving observations are vLLM smoke controls. The reviewed
SGLang evidence establishes one negative boundary for the exact W4AFP8
checkpoint and identifies one correctable BF16 launch-environment failure.

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

The two ready-memory values and server-start times are single smoke
observations. They must not be reported as comparative performance numbers.
Text equality is also not evaluated across backends because neither SGLang case
reached generation.

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
Track C first. Result-gated job `5905638` evaluates only the exact accepted
QuaRot-style rotated W4A16 checkpoint on 32 frozen BoolQ validation examples,
using matched `no`/`yes` continuation scoring through vLLM and SGLang. It is
submitted from clean revision `cf7b49a`; no formal job is submitted and no
result is claimed before artifact review.

Track B W4AFP8 must not be rerun. Track C does not add BF16, unrotated W4A16,
SpinQuant, checkpoint conversion, or a new quantization run.
