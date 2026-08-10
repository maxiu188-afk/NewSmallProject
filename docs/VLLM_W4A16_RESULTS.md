# QuaRot-style vLLM W4A16 results on Isambard

## Scope

This is the accepted result summary for the practical serving route:
standard-layout offline QuaRot-style rotations followed by group-128 GPTQ
W4A16 in compressed-tensors format and vLLM 0.25.1. It compares the original
BF16 Llama-2-13B checkpoint with unrotated and offline-rotated W4A16
checkpoints on one NVIDIA GH200.

This route is not original QuaRot W4A4KV4. It omits online MLP/QK transforms
and the custom KV4 cache. It is also separate from the repository's owned W4A8
kernel. A separately scoped SpinQuant-derived packed-W4A16 extension is
recorded below; it does not change the accepted QuaRot-style results.

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
| QuaRot-style formal BoolQ | `5974744` | BF16 and rotated packed W4A16 each completed all 3,270 frozen examples; W4A16 reached 80.7645% versus 80.5810% BF16 |
| SpinQuant-derived export/load gate | `5945162` | The learned R1/R2 artifact produced a complete 280-linear packed W4A16 checkpoint and passed fresh-process vLLM inference |
| SpinQuant-derived formal BoolQ | `5952594` | The packed checkpoint completed all 3,270 frozen examples and reached 79.7554% versus 80.5810% BF16 |

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

## QuaRot-style rotated W4A16 formal BoolQ

Formal job `5974744` completed `0:0` in 5 minutes 4 seconds from clean
revision `5ac3b26ecb7ca3256e84d00c4780c53207a16817`. It reused the frozen
3,270-example BoolQ validation artifact and zero-shot choice-scoring protocol
used by the accepted W4AFP8 and SpinQuant W4A16 evaluations. BF16 and the
QuaRot-style rotated packed-W4A16 checkpoint ran in fresh vLLM processes and
each completed all 6,540 choice requests:

| Model | Correct | Accuracy | Delta versus BF16 |
|---|---:|---:|---:|
| BF16 | 2,635 / 3,270 | 80.5810% | -- |
| QuaRot-style rotated packed W4A16 | 2,641 / 3,270 | 80.7645% | +0.1835 pp |

The paired audit found 102 BF16-only correct examples, 108 W4A16-only correct
examples, and 210 prediction disagreements. Exact two-sided McNemar
`p=0.730161`; the observed six-answer difference is not statistically
resolved and must not be presented as either superiority or equivalence.

The accepted result SHA-256 is
`92b241ee0cce967a39df777fde97a7a9a18a4e4879c10f7bce9cf431ca75f445`;
the source-manifest SHA-256 is
`6ebd32662a2fda325673264fb98108bffd635260e6277e0790a32a5e3a3eae79`.
All ten manifest entries were independently rehashed. The result binds config
SHA-256
`944d38b4e23b71015e4db62452c4659cf009a1cf04103f0c0d29e54ff82d730a`,
dataset-manifest SHA-256
`66b7a80e9ef1df7d3bd1f07a111824bffcccff56b1ff47c4d9e81afebd1146d5`,
and examples SHA-256
`475e56b71939a8e3db8be48bcc4d344569b36cc660f4086c9ae15858d46a297f`.
The W4A16 worker recorded all 280 packed decoder linears under vLLM
`0.25.1+cu129` and PyTorch `2.11.0+cu129`.

BoolQ is an accuracy workload and does not itself produce a serving-speedup
measurement. For a compact endpoint comparison, the vLLM medians from the
formal SGLang/vLLM job `5960180` can be descriptively normalized against the
accepted BF16 rows from matched serving job `5780631`: throughput is `1.541x`
at concurrency 1 and `1.369x` at concurrency 8; p50 E2E falls by 35.2% and
27.1%; p50 TPOT falls by 36.8% and 32.4%; and ready GPU memory falls by 52.6%
at both concurrencies. TTFT increases by 8.6% and 3.3%.

This normalization is not a same-run paired speedup: job `5960180` contains no
BF16 arm and used the repository-owned client and its own frozen request
corpus. It is supported by the close same-job rotated-W4A16 ratios in
`5780631` (`1.533x` and `1.373x` throughput), but the `5780631` result remains
the primary performance evidence. The synthetic 256-input/64-output serving
measurements must not be described as BoolQ evaluation throughput.

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

## SpinQuant-derived packed-W4A16 extension

Source gate `5945162` completed `0:0` in 22 minutes 7 seconds from clean
revision `f490f58f94f4af30d4a8f326dcab0fbaf4ff5213`. It reused the accepted
100-step SpinQuant W4A16 weight-QDQ-trained R1/R2 artifact, fused the learned
rotation offline, calibrated group-128 GPTQ on 128 x 2,048 WikiText-2 tokens,
and exported a standard compressed-tensors `pack-quantized` checkpoint. The
checkpoint has 280 packed decoder linears, no activation quantization, no
online rotation modules, and selected `MacheteLinearKernel` when loaded by
vLLM `0.25.1+cu129` on GH200.

The gate launched BF16 and `spinquant_w4a16` in separate fresh processes. Both
returned the same eight greedy token IDs; this is load/inference evidence, not
task accuracy. The reviewed bindings are:

| Artifact | SHA-256 |
|---|---|
| Export result | `b21ca19f34bf24470fdec797f90821b46edb77bfc1717c23724b989aa4ff05cf` |
| Offline inference result | `fa533a64f7b8345b547f45b6fa666ca8e6181d3369dc487dc74c5cf171f995c4` |
| Source manifest | `4d11a0186139a840f9e94e27fb810957afcb7e2af53083af62cdb11934a963f4` |
| Checkpoint tree | `41a79153d2ad7e0fb598819adc538ce65ba7c1a05629459f77cb816d226ce2da` |

Formal-only BoolQ job `5952594` completed `0:0` in 5 minutes 38 seconds from
clean evaluation revision
`138ae9f0662f6441cd958c1fe9f2ebb1d99f7a3f`. It reused the exact frozen
3,270-example validation artifact and zero-shot scoring protocol from the
accepted W4AFP8 BoolQ evaluations. Both fresh-process models completed all
6,540 choice requests:

| Model | Correct | Accuracy | Delta versus BF16 |
|---|---:|---:|---:|
| BF16 | 2,635 / 3,270 | 80.5810% | -- |
| SpinQuant-derived packed W4A16 | 2,608 / 3,270 | 79.7554% | -0.8257 pp |

The paired audit found 105 BF16-only correct examples, 78 W4A16-only correct
examples, and 183 prediction disagreements. Exact two-sided McNemar
`p=0.05431`; this is above 0.05 but close enough that the result must not be
described as equivalence. Under the shared BoolQ protocol, W4A16 is 17 correct
answers (`-0.5199` pp) below the accepted FP8-targeted SpinQuant W4AFP8 result
of 80.2752%, and five answers (`+0.1529` pp) above the old INT8-trained
SpinQuant-transfer W4AFP8 result of 79.6024%.

The formal result SHA-256 is
`bdc6ac3263d14695321569b1c7b869fa11f24492e7ddbd82fb66315812549b64`;
the BF16 and W4A16 worker SHA-256 values are
`029a0884d5e995b9a1a7b3ebaa98ee2e3f714eda9a0eb9aa3c00a0a33d4334d0`
and
`95615138efa1e635a4df2399370d49c14340988a31b1a5f2e397d1027dadbe3e`.
All 13 source-manifest bindings were rehashed successfully. The W4A16 worker
recorded 280 packed linears and `MacheteLinearKernel`; the optional DeepGEMM
import warning was non-fatal. This accepts downstream deployed-checkpoint
quality. The separately accepted serving result below supplies the performance
half of this endpoint's evidence package.

## SpinQuant-derived packed-W4A16 serving result

Formal-only serving job `5961810` completed `0:0` in 11 minutes 21 seconds on
one GH200 from clean revision
`eb6eff9b4dfb77aa43374730b222d718c995527e`. It reused source gate `5945162`,
the checkpoint-tree SHA-256 recorded above, and the same random 256-token input,
forced 64-token output, four-warm-up, 64-request, concurrency 1/8, 8 GiB BF16-KV
protocol as the accepted vLLM deployment studies. BF16 and SpinQuant W4A16 ran
in fresh servers for each case.

| Model | Concurrency | Requests/s | p50 TTFT (ms) | p50 TPOT (ms) | p50 E2E (ms) | Ready GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 1.759 | 20.883 | 8.689 | 568.345 | 34,099 |
| SpinQuant-derived packed W4A16 | 1 | 2.704 | 24.237 | 5.487 | 369.925 | 16,292 |
| BF16 | 8 | 11.371 | 103.440 | 9.514 | 703.396 | 34,100 |
| SpinQuant-derived packed W4A16 | 8 | 15.681 | 107.964 | 6.390 | 510.526 | 16,126 |

Relative to its same-run BF16 control, SpinQuant W4A16 improved request
throughput by `1.537x` at concurrency 1 and `1.379x` at concurrency 8. Ready
GPU memory fell by 52.2% and 52.7%; p50 E2E fell by 34.9% and 27.4%; and p50
TPOT fell by 36.8% and 32.8%. The trade-off is p50 TTFT increasing by 16.1%
and 4.4%, so the result is not a claim that every latency metric improves.

All four cells completed 64/64 requests with zero failures and exactly 256
input plus 64 output tokens per request. Both quantized server logs selected
`MacheteLinearKernel for CompressedTensorsWNA16`; GPU memory returned to 2--3
MiB after each server. All four raw-result, server-log, and client-log hashes
matched the summary, and all 11 source-manifest bindings were independently
rehashed. The retained bindings are:

| Artifact | SHA-256 |
|---|---|
| Formal result | `8115da350e6c3c5c82f11a811f2e7ab6b1eda8bb8e504a3b3247a0f593e456e0` |
| Source manifest | `8b171af80963d7ad470c0e3298f0042a260b211d990be6d050885793e2ac47c4` |
| Checkpoint tree | `41a79153d2ad7e0fb598819adc538ce65ba7c1a05629459f77cb816d226ce2da` |

The earlier job `5960073` is rejected performance evidence: its overlong vLLM
ZeroMQ IPC path failed before serving measurement. Revision `eb6eff9` shortened
only the per-job IPC directory to `/tmp/vs-${SLURM_JOB_ID}`; model, checkpoint,
requests, resources, and benchmark protocol remained unchanged. The replacement
stderr is empty. Optional DeepGEMM import warnings and `EngineDeadError` lines
occur only during post-measurement server shutdown and do not invalidate the
complete request records.

## BoolQ and serving-acceleration endpoint view

The accepted BoolQ and serving records make a descriptive endpoint comparison
possible. Accuracy uses the same frozen task protocol. Acceleration uses the
matched serving protocol for SpinQuant W4A16 and W4AFP8, while the QuaRot row
uses the explicitly cross-job normalization described above. These are
separate formal jobs, not interleaved paired repetitions, so small differences
must not be presented as statistically resolved method or format advantages.

| Endpoint | BoolQ accuracy | Throughput ratio, c1 / c8 | p50 E2E reduction, c1 / c8 | Ready-memory reduction, c1 / c8 |
|---|---:|---:|---:|---:|
| QuaRot-style packed W4A16 | 80.7645% | 1.541x / 1.369x | 35.2% / 27.1% | 52.6% / 52.6% |
| Packed W4A16 | 79.7554% | 1.537x / 1.379x | 34.9% / 27.4% | 52.2% / 52.7% |
| FP8-targeted W4AFP8 | 80.2752% | 1.42x / 1.385x | 29.5% / 27.9% | 50.7% / 50.7% |

QuaRot-style W4A16 is 33 correct answers and 1.0092 percentage points above
SpinQuant W4A16, and 16 answers and 0.4893 percentage points above the
FP8-targeted W4AFP8 endpoint. Those are descriptive differences across
separate formal jobs, not paired significance results. SpinQuant W4AFP8 is 17
correct answers and 0.5199 percentage points above SpinQuant W4A16. Both W4A16
routes have the larger concurrency-1 throughput gain and slightly lower ready
memory than W4AFP8; concurrency-8 throughput and p50 E2E are close. TTFT
regressions remain visible in the detailed results and should be considered
when choosing an endpoint.

## Conclusions and closed boundary

The practical result is positive: stable vLLM W4A16 materially reduces memory,
TPOT, E2E, and improves request throughput for the measured Llama-2-13B serving
workload on GH200. The layer diagnostic confirms that real packed W4 kernels
execute, but it does not by itself explain the larger full-model serving gain
and is not the primary performance claim.

Offline rotation does not materially change deployment performance relative to
unrotated W4A16, but the deployed-checkpoint PPL result shows a quality benefit:
PPL improved from 5.289677 to 5.132755 and recovered 55.67% of the quantization
gap to BF16. The SpinQuant-derived extension now has both BoolQ and matched
serving evidence for its separately scoped packed checkpoint. QuaRot-style
W4A16 now also has formal BoolQ accuracy evidence; broader downstream tasks and
generation quality remain unmeasured. Fake-quant PPL from the algorithmic
study remains a separate evidence route and must not be substituted for either
packed vLLM result.
