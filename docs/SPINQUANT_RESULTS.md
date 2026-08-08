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
| Corrected paper-aligned W16A8 learning smoke | `5854268` | `COMPLETED (0:0)`, 2m47s | Eight 2048-token sequences; paper-aligned A8 coverage assertions and non-zero R1/R2 gradients accepted |
| Corrected paper-aligned W16A8 formal learning | `5854269` | `COMPLETED (0:0)`, 2h12m09s | 800 sequences, 100 finite-loss updates, 100 non-zero gradient records; accepted corrected rotation artifact |
| Corrected-A8, strong-GPTQ no-had W4A8 PPL smoke | `5854890` | `COMPLETED (0:0)`, 12m33s | Two held-out sequences; BF16, unrotated, and learned-rotation paths completed with 8-sequence GPTQ calibration |
| Corrected-A8, strong-GPTQ no-had W4A8 PPL formal | `5854891` | `COMPLETED (0:0)`, 19m50s | 162 sequences, 331,614 scored tokens, and 128-sequence GPTQ calibration; all three paths accepted as a matched diagnostic |

The first four jobs used project revision
`5d610b152434ffb04bd405be37aa172114dd6216`. The formal PPL job reused the
same retained 162 x 2048 WikiText-2 test token artifact as the deployed QuaRot
PPL study. Its token-ID SHA256 is
`0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`.

## Matched held-out W4A16 ablation PPL

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

## Corrected-A8, strong-GPTQ no-had W4A8 diagnostic PPL

| Path | Runtime form | PPL | Delta versus BF16 |
|---|---|---:|---:|
| BF16 | Unquantized reference | 5.0087 | -- |
| Unrotated W4A8KV16 | GPTQ group-128 W4 plus dynamic asymmetric per-token A8 QDQ | 5.1480 | +0.1392 (+2.78%) |
| SpinQuant no-had W4A8KV16 | Offline R1/R2, then matched GPTQ W4 and A8 QDQ | **5.1627** | +0.1539 (+3.07%) |

The learned-rotation path was `0.0147` PPL worse than the matched unrotated
control, a 0.285% relative increase. Both W4A8 paths used the same 128 x 2048
GPTQ calibration artifact, quantized all 280 decoder linears, used group-128
weights with activation ordering, used group size 128 for `o_proj`, included
zero for ungrouped asymmetric A8, left `lm_head` and KV at 16 bits, and scored
the same 331,614 held-out targets. The result is retained as a completed
matched **strong-GPTQ diagnostic** with no observed rotation benefit. It is not
the paper-aligned GPTQ endpoint and must not be compared numerically with a
paper-table PPL without an explicit protocol-change label. A paper-aligned run
would require `group_size=-1`, no activation ordering, and the paper's
weight-clipping step. That endpoint is outside the closed phase; the existing
result remains immutable.

The formal PPL result SHA256 is
`0b5403d3a0423efd9b3d7673107ed9b25d58350a7caf7643432aefcc2ef4c6fd`;
the formal source-manifest SHA256 is
`713be9fd759677ed19dad3a5f3a8ffd045c0f1b420686c66259ed03b4d772727`.

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

An activation audit subsequently found that the first W16A8 chain did not use
the paper-aligned grouped-A8 treatment for `o_proj`, so jobs `5848060` and
`5848547` are superseded for the no-had quality claim. Corrected smoke job
`5854268` and formal job `5854269` use dynamic asymmetric per-token A8, include
zero for ungrouped operators, and use group size 128 for `o_proj`. The formal
job passed in 2 hours 12 minutes 9 seconds using 800 x 2048 calibration tokens
and 100 updates. Its first/last ten-step mean losses were `1.651692` and
`1.561438`, with a best loss of `1.322814` at step 94. Final R1/R2
orthogonality errors were `1.6689e-6` and `7.1526e-7`. The corrected rotation
tensor SHA256 is
`ac08587f537b37be176cdd8ed6ab2137bc2b287e0e9f1e7a1e9aa9fb3ed31c4a`;
its manifest SHA256 is
`b52466ef3eed8a97e859fed1c5593fc6fd7591e16ed1b20599235fc2a1aec695`,
and the formal result SHA256 is
`fc17656fe9d2684ccd4e96859c3e8d4f48a909be7e72ffd46b8f5c013fbe43e9`.
The rotation artifact is accepted as the source of the completed corrected
PPL result; its training loss alone is not quality evidence.

## Completed W4AFP8 deployment continuation

The formerly planned W4AFP8 continuation is complete. The corrected learned
rotation was retained as an INT8-trained transfer endpoint, and a separate
FP8-targeted rotation was trained without overwriting it. Both were exported
under the backend-constrained group-128/no-actorder/min-max W4AFP8 protocol.
Packed fresh-process loading, formal deployed PPL, matched full-model serving,
and BoolQ are accepted in [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md).

The original reproduction and W4AFP8 continuation remain closed. A later,
separately scoped deployment extension exported the accepted 100-step W4A16
weight-QDQ-trained R1/R2 artifact as a standard packed W4A16 checkpoint.
Export/load gate `5945162` is accepted; formal BoolQ job `5952594` is submitted
but still pending with no result artifact. This does not convert the fake-quant
PPL above into packed-deployment evidence and does not establish BoolQ,
serving, or acceleration. Its provenance and current state are recorded in
[`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md).

The separately labelled paper-GPTQ W4A8 endpoint, the paper's online `had`
R3/R4 extension, and FP8 KV remain unmeasured scope boundaries rather than
pending stages.
