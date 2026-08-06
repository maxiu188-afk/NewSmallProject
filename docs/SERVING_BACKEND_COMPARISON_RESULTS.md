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

The bounded corrective option is to load `gcc-native/14.2` and explicitly bind
`CC`, `CXX`, and `NVCC_CCBIN`; the module alone is insufficient because `c++`
still resolves to GCC 7. No corrected job or formal job has been submitted, and
no cross-backend result is claimed.

Track B W4AFP8 must not be rerun. Track C does not add BF16, unrotated W4A16,
SpinQuant, checkpoint conversion, or a new quantization run.
