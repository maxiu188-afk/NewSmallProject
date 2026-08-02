# QuaRot-style vLLM W4A16 results on Isambard

## Scope

This is the accepted result summary for the practical serving route:
standard-layout offline QuaRot-style rotations followed by group-128 GPTQ
W4A16 in compressed-tensors format and vLLM 0.25.1. It compares the original
BF16 Llama-2-13B checkpoint with unrotated and offline-rotated W4A16
checkpoints on one NVIDIA GH200.

This route is not original QuaRot W4A4KV4. It omits online MLP/QK transforms
and the custom KV4 cache. It is also separate from the repository's owned W4A8
kernel.

## Accepted evidence chain

| Gate | Job | Accepted evidence |
|---|---:|---|
| Tiny vLLM GPU smoke | `5751780` | BF16, unrotated W4A16, and rotated W4A16 tiny checkpoints loaded and generated through the frozen GH200 runtime |
| Llama-2-13B offline inference | `5769503` | All three complete models loaded in fresh processes and returned the same eight greedy tokens |
| Deployed PPL smoke | `5839415` | All three checkpoints completed the same 4,094-target WikiText-2 slice |
| Deployed PPL formal | `5839419` | All three checkpoints completed 331,614 held-out targets through vLLM with matched runtime and provenance |
| OpenAI-compatible service smoke | `5780629` | All three servers passed health, model-listing, and deterministic completion checks |
| Matched full-model serving | `5780631` | Six model/concurrency groups each completed 64/64 measured requests with zero failures |
| Real layer-0 hook smoke | `5784966` | The packed W4A16 `LlamaDecoderLayer` executed with valid CUDA-event prefill and decode records |
| Matched layer-0 diagnostic | `5784967` | BF16 and both W4A16 variants completed all eight prefill/decode cases |

The Llama-2-13B offline inference result used vLLM `0.25.1+cu129`, PyTorch
`2.11.0+cu129`, CUDA 12.9, and GH200 SM90. The W4 checkpoints each contain 280
packed decoder linears. Across 122 shared first-token log-probability entries,
the maximum absolute differences from BF16 were `0.65765` for unrotated W4A16
and `0.37806` for rotated W4A16. This establishes bounded inference
correctness, not downstream quality or perplexity.

## Deployed-checkpoint quality result

The formal quality job evaluated 162 non-overlapping 2,048-token WikiText-2
sequences, or 331,614 scored next-token targets, through the deployed vLLM
checkpoints. Each model ran in a fresh child process with vLLM
`0.25.1+cu129`, PyTorch `2.11.0+cu129`, CUDA 12.9, and one GH200. The two W4A16
checkpoints each exposed 280 packed decoder linears and selected
`MacheteLinearKernel`.

| Model | Runtime form | PPL | Delta versus BF16 |
|---|---|---:|---:|
| BF16 | Original checkpoint through vLLM | 5.007820 | -- |
| Unrotated W4A16 | Group-128 GPTQ packed W4A16 | 5.289677 | +0.281856 (+5.63%) |
| Rotated W4A16 | Offline QuaRot-style rotation plus matched packed W4A16 | **5.132755** | +0.124934 (+2.49%) |

Rotation reduced deployed PPL by `0.156922` versus unrotated W4A16, a 2.97%
relative reduction, and recovered 55.67% of the unrotated W4A16-to-BF16 PPL
gap. It therefore provides a measured deployment-quality benefit under this
matched protocol, although the rotated checkpoint remains 2.49% above BF16.
This quality result does not establish downstream-task accuracy or unrestricted
generation quality.

Smoke job `5839415` and its dependent formal job `5839419` both completed with
exit code `0:0`. The formal manifest records `afterok:5839415`, clean revision
`7074b3ea90a7072f1ac59f49dc2a0ec25a592f3a`, token-ID SHA-256
`0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`, and
token-manifest SHA-256
`b6ed5f122ba85a9752b3dda699d73405e5a86b7db19050c1f395cbbea450f1cf`.
The accepted formal-result SHA-256 is
`f7698afcca494279cb7d3f2d50d94fd1378e03c6d6edd4506d750869cb09829a`.

## Primary full-model serving result

The matched workload used random 256-token inputs, forced 64-token outputs,
four warm-ups, 64 measured requests, concurrency 1 and 8, and an explicit
8 GiB KV cache. Every model/case pair used a fresh server with identical
request generation and server flags.

| Model | Concurrency | Requests/s | p50 TTFT (ms) | p50 TPOT (ms) | p50 E2E (ms) | Ready GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 1.760 | 21.114 | 8.684 | 568.249 | 34,099 |
| BF16 | 8 | 11.374 | 104.698 | 9.507 | 703.522 | 34,099 |
| Unrotated W4A16 | 1 | 2.718 | 22.242 | 5.488 | 368.019 | 16,125 |
| Unrotated W4A16 | 8 | 15.543 | 108.642 | 6.438 | 513.606 | 16,125 |
| Rotated W4A16 | 1 | 2.699 | 22.860 | 5.510 | 370.313 | 16,125 |
| Rotated W4A16 | 8 | 15.612 | 107.608 | 6.456 | 514.416 | 16,125 |

Relative to BF16:

- ready GPU memory fell by `52.7%` for both W4A16 checkpoints;
- request throughput improved by `1.53--1.54x` at concurrency 1 and
  `1.37x` at concurrency 8;
- p50 E2E fell by `34.8--35.2%` at concurrency 1 and `26.9--27.0%` at
  concurrency 8;
- p50 TPOT fell by `36.5--36.8%` at concurrency 1 and `32.1--32.3%` at
  concurrency 8;
- TTFT remained close to BF16 but was slightly higher for both W4 variants.

The accepted serving-result SHA-256 is
`2a683b38be4831f24897bb3d8660d3f0277c3a0c914aaebb5b8511f2f357a39a`.
These are the primary deployment results.

## Single-block diagnostic

The diagnostic measures the real vLLM
`LlamaForCausalLM.model.layers[0]` while the full model still executes to
provide real attention metadata and KV-cache state. Python hooks installed
through `LLM.apply_model()` record CUDA events around the layer. The diagnostic
uses `enforce_eager=true`; its timings are not directly interchangeable with
the compiled serving measurements above.

| Case | BF16 median (ms) | Unrotated W4A16 | Rotated W4A16 |
|---|---:|---:|---:|
| Prefill B1 x 256 | 0.711 | 0.710 (`1.00x`) | 0.730 (`0.97x`) |
| Prefill B1 x 2048 | 2.857 | 2.729 (`1.05x`) | 2.715 (`1.05x`) |
| Prefill B8 x 256 | 2.768 | 2.643 (`1.05x`) | 2.656 (`1.04x`) |
| Prefill B8 x 2048 | 20.942 | 21.558 (`0.97x`) | 21.736 (`0.96x`) |
| Decode B1, context 256 | 0.616 | 0.596 (`1.03x`) | 0.603 (`1.02x`) |
| Decode B1, context 2048 | 0.607 | 0.586 (`1.04x`) | 0.591 (`1.03x`) |
| Decode B8, context 256 | 0.617 | 0.602 (`1.03x`) | 0.611 (`1.01x`) |
| Decode B8, context 2048 | 0.601 | 0.588 (`1.02x`) | 0.593 (`1.01x`) |

The inspected BF16 layer contains `605.0 MiB` of parameters. Each W4A16 layer
contains `156.0 MiB`, including four packed merged projection parameters, a
`74.2%` reduction. The runtime selected vLLM's `MacheteLinearKernel` for
W4A16.

Layer-level speed differences are small and shape-dependent: W4A16 is roughly
`1.01--1.05x` faster in most decode and medium-prefill cases, effectively tied
at B1 x 256, and `0.96--0.97x` as fast for B8 x 2048 prefill. Rotation does
not show a stable performance advantage over unrotated W4A16.

The accepted diagnostic-result SHA-256 is
`0a5e9658c3a2205b6e2465472f81bd004370ba7b4db1c46f07123fe3a538c5cb`.

## Conclusions and remaining boundary

The practical result is positive: stable vLLM W4A16 materially reduces memory,
TPOT, E2E, and improves request throughput for the measured Llama-2-13B serving
workload on GH200. The layer diagnostic confirms that real packed W4 kernels
execute, but it does not by itself explain the larger full-model serving gain
and is not the primary performance claim.

Offline rotation does not materially change deployment performance relative to
unrotated W4A16, but the deployed-checkpoint PPL result shows a quality benefit:
PPL improved from 5.289677 to 5.132755 and recovered 55.67% of the quantization
gap to BF16. Downstream-task and broader generation-quality evaluation remain
open. Fake-quant PPL from the algorithmic study remains a separate evidence
route and must not be substituted for this packed vLLM result.
