# SpinQuant accepted results

## Accepted scope

The first accepted result is an independent, offline-fused SpinQuant
weight-only ablation on the pinned Llama-2-13B model. It learns R1 and
per-layer R2 rotations, absorbs them into ordinary Llama weights, and then
applies one-time group-128 symmetric W4 floating QDQ to the 280 decoder linear
layers. Activations and KV remain at 16 bits, and the evaluated model contains
no online rotation module. This W4A16 result remains useful evidence, but it is
not the paper's real-deployment `SpinQuant_no_had` target, which is W4A8.

This result establishes held-out WikiText-2 fake-quant quality. It is not GPTQ,
a packed checkpoint, an integer-kernel measurement, downstream accuracy, or
deployment performance.

## Accepted Isambard evidence chain

| Gate | Job | Result | Accepted evidence |
|---|---:|---|---|
| One-step full-model smoke | `5841874` | `COMPLETED (0:0)`, 3m47s | Eight 2048-token sequences; non-zero R1/R2 gradients; accepted rotation artifact |
| Formal rotation learning | `5842047` | `COMPLETED (0:0)`, 2h17m51s | 800 sequences, 100 finite-loss updates, 100 non-zero gradient records |
| Matched PPL smoke | `5847440` | `COMPLETED (0:0)`, 2m29s | All four evaluation paths completed on the reduced token set |
| Matched PPL formal | `5847441` | `COMPLETED (0:0)`, 3m25s | 162 sequences and 331,614 scored tokens; result status `passed` |
| Paper-aligned W16A8 learning smoke | `5848060` | `COMPLETED (0:0)`, 3m05s | Eight 2048-token sequences; non-zero R1/R2 gradients; activation-QDQ runtime form accepted |
| Paper-aligned W16A8 formal learning | `5848547` | `COMPLETED (0:0)`, 2h09m15s | 800 sequences, 100 finite-loss updates, 100 non-zero gradient records; accepted rotation artifact |

The first four jobs used project revision
`5d610b152434ffb04bd405be37aa172114dd6216`. The formal PPL job reused the
same retained 162 x 2048 WikiText-2 test token artifact as the deployed QuaRot
PPL study. Its token-ID SHA256 is
`0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`.

## Matched held-out PPL

| Path | Runtime form | PPL | Delta versus BF16 |
|---|---|---:|---:|
| BF16 | Unquantized reference | 5.0087 | -- |
| Unrotated W4A16 | Group-128 floating QDQ | 5.1763 | +0.1675 (+3.34%) |
| Fixed random-Hadamard R1/R2 W4A16 | QuaRot-style mechanism baseline, floating QDQ | 5.7078 | +0.6990 (+13.96%) |
| Learned SpinQuant R1/R2 W4A16 | Offline-fused floating QDQ | **5.0937** | +0.0849 (+1.70%) |

Learned R1/R2 reduced PPL by `0.0826` versus the unrotated W4A16 path, a
1.60% relative reduction. It reduced PPL by `0.6141` versus the fixed
random-Hadamard R1/R2 path, a 10.76% relative reduction. The BF16 value matches
the retained baseline, and all three W4 paths quantized the same 280 decoder
linears.

The fixed random-Hadamard row tests a QuaRot-style R1/R2 mechanism under this
matched RTN-QDQ protocol. It is not the accepted QuaRot GPTQ/vLLM checkpoint
and must not be reported as that deployment result.

## Artifact provenance

- formal rotation tensor SHA256:
  `303c614f425ea5d37e138643dd52a2410a4747fda7e3038587c93a2db05a57fe`;
- formal rotation-manifest SHA256:
  `825701f474aa38d9b3c063775429813c9ce7673f229282df7ed6b7d54f156ad5`;
- formal PPL result SHA256:
  `e70ee79229dd3a19f8f1f7a588af3a32ebf76bdf745645d52019fd77941b1574`;
- retained test-token manifest SHA256:
  `b6ed5f122ba85a9752b3dda699d73405e5a86b7db19050c1f395cbbea450f1cf`.

The result JSON is generated as
`results/spinquant-isambard-llama2-13b/fake-quant-ppl-formal.json` in the
isolated Isambard SpinQuant checkout. Large generated artifacts remain outside
Git.

## Paper-protocol and deployment boundaries

The paper distinguishes `SpinQuant_no_had`, whose learned R1/R2 rotations can
be absorbed into weights, from `SpinQuant_had`, which adds online Hadamard R3
and R4 for aggressively quantized activations and KV cache. The paper reports
the real-deployment `no_had` route at W4A8: R1/R2 are fused offline, while
activations are dynamically quantized to A8. The completed W4A16 evaluation
obeys the same offline-rotation constraint but is only a weight-only ablation,
not the paper deployment precision.

The paper's main learning protocol is different from this accepted experiment:
it optimizes rotations while quantizing activations and retaining 16-bit
weights, then applies GPTQ to the learned rotated weights. The current runner
instead optimizes directly against group-128 W4 weight QDQ with A16/KV16 and
uses RTN-style floating QDQ for evaluation. Therefore the result is a complete
matched evaluation of the implemented W4A16 weight-only path, but it is not a
literal reproduction of the paper's full fake-quant matrix and must not be
compared directly with a paper-table PPL as if the protocols were identical.

The W16A8 activation-only learning objective has now completed both Isambard
gates. Smoke job `5848060` passed in 3 minutes 5 seconds. Formal job `5848547`
passed in 2 hours 9 minutes 15 seconds using 800 x 2048 calibration tokens and
100 updates; all losses and rotation-gradient records were finite and every
gradient record was non-zero. Its first/last ten-step mean losses were
`1.651698` and `1.561623`, with a best loss of `1.32231` at step 94. Final
R1/R2 orthogonality errors were `1.3709e-6` and `5.96e-7`. The accepted
rotation tensor SHA256 is
`62cbc73e26d8993f068331e924c19738a40bc89813e680316d1978df3d452fda`;
the formal result SHA256 is
`11a71e105b93640763886e9dbc165feb37e96adffd1a0cf9ebcdc42113d6097e`.
This remains rotation-learning evidence only.

The remaining stages are:

1. finish the paper-aligned `no_had` fake-quant protocol by applying
   post-learning GPTQ W4 to the accepted W16A8-trained rotations, then evaluate
   the no-had W4A8KV16 path on matched held-out tokens;
2. reproduce the `had` fake-quant path with R3/R4 for low-bit activation/KV
   experiments, while keeping it out of the offline-only deployment path;
3. export a compressed-tensors W4A8 checkpoint and test fresh-process load plus
   fixed-token correctness;
4. test selected-kernel runtime form and performance separately. Current stable
   vLLM documentation does not list INT4-weight/INT8-activation acceleration on
   Hopper, so loading a checkpoint is not sufficient evidence of real A8
   execution or acceleration.
