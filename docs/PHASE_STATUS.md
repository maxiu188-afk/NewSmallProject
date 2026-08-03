# Current phase status

This file is the concise current-state entry point. The dated update history
through 2026-07-22 is preserved in
[`archive/plans/PHASE_STATUS_2026-07-22.md`](archive/plans/PHASE_STATUS_2026-07-22.md).

## Status summary

| Workstream | Current status | Evidence boundary |
|---|---|---|
| Algorithmic fake quant | Complete for the formal Llama-2-13B study | QuaRot + GPTQ W4A4 reached PPL 5.8376 versus BF16 5.0087; all low-bit rows are floating QDQ and keep K/V at 16 bits |
| SpinQuant fake quant | Corrected-A8 strong-GPTQ diagnostic complete: jobs 5854890/5854891 passed; PPL was 5.0087 BF16, 5.1480 unrotated W4A8, and 5.1627 SpinQuant W4A8 | Group-128 plus activation-order GPTQ is not the paper GPTQ protocol; retain the result, but label it separately and do not compare it directly with the planned group-size -1/no-actorder/clipped run |
| Owned packed W4A8 | Correctness complete on RTX 6000 Ada; selected-linear portability accepted on GH200 | All 280 decoder linears matched the packed oracle on Ada; the Isambard result covers one `q_proj`, not the full decoder; K/V remain BF16 |
| Official QuaRot full model | Complete on RTX 6000 Ada | Real Llama-2-13B W4A4KV4 reduced model-resident memory from 26.29 GB to 7.18 GB but was slower at batch one; no packed-checkpoint PPL result |
| Official QuaRot single block | Complete on RTX 6000 Ada | W4 completed 14/14 cases; 2048-token prefill gained 1.53--1.68x; batch-16/context-4096 layer E2E gained 1.28x; this is not full-model latency |
| vLLM W4A16 serving and quality | Complete on GH200 for matched Llama-2-13B deployed PPL, serving, and layer-0 protocols | Rotated packed W4A16 reached PPL 5.132755 versus 5.289677 unrotated and 5.007820 BF16; W4A16 cut ready GPU memory by 52.7% and improved request throughput by 1.37--1.54x; no downstream-task result |
| vLLM W4AFP8 deployment | Deployed PPL and BoolQ accepted on GH200; old INT8-trained-transfer service smoke 5882787 is accepted and benchmark formal 5883004 is running; new FP8-targeted export/load source gate 5881273 remains pending | No W4AFP8 acceleration result exists until benchmark 5883004 completes and is accepted |

## Current deployment decision

The official-backend route is complete for the current presentation scope.
Its accepted results are consolidated in
[`OFFICIAL_QUAROT_RESULTS.md`](OFFICIAL_QUAROT_RESULTS.md).

The owned W4A8 kernel remains correctness evidence, but its RTX 6000 Ada
performance smoke was much slower than BF16. Stable vLLM does not provide the
selected NVIDIA W4A8 serving path, and this phase will not add an out-of-tree
plugin or another full CUDA backend.

The completed serving route uses offline-fusible QuaRot-style rotation
followed by group-128 GPTQ W4A16 in standard compressed-tensors format. It is
reported as **QuaRot-style W4A16**, not original QuaRot W4A4KV4, because it
omits online MLP/QK transforms and the custom KV4 cache.

This W4A16 decision applies to the completed QuaRot workstream only. The
SpinQuant paper's real-deployment `no_had` route remains W4A8, with
offline-fused R1/R2 and activation quantization retained at inference. Its
current INT8 W4A8 fake-quant chain completed with paper-aligned A8 coverage but
a stronger group-128/activation-order GPTQ protocol. It is retained as a
separately labelled diagnostic; a paper-GPTQ endpoint remains to be run.

After that chain is accepted, the next real-deployment study changes to a
joint, hardware-aligned W4AFP8 route for both QuaRot and SpinQuant on GH200.
It requires deployed-checkpoint PPL plus matched full-model serving performance;
the existing fake-quant evaluator cannot establish W4AFP8 accuracy. Its local
export, backend-audit, deployed-PPL, and serving paths are prepared and tested
statically. Corrected rotation provenance is frozen. The first W4AFP8 gates
were cancelled before allocation when the GPTQ protocol changed. Jobs
`5859043`/`5859044` later failed stale-calibration validation, and
`5873446`/`5873447` passed calibration and backend audit before an over-strict
observer-name check rejected LLM Compressor's canonical `memoryless_minmax`
name. Revision `9d691e9` accepts only the equivalent
`minmax`/`memoryless_minmax` names and still rejects MSE observers. QuaRot
validation job `5873544` completed in 38m30s, and SpinQuant-transfer job
`5873545` then completed in 19m44s through `afterok:5873544`. Both returned
`0:0`; all fixed-token outputs matched BF16. Joint four-model source-gate job
`5874345` then completed `0:0` in 4m35s. Its inference and capability SHA-256
hashes are frozen in PPL revision `1f3e4cb`, whose provenance and retained-token
preflights passed. PPL smoke `5874806` completed `0:0` in 6m24s; formal job
`5874807` then completed `0:0` in 5m00s through `afterok:5874806`. Formal PPL
was 5.007820 BF16, 5.136105 unrotated W4AFP8, 5.248356 QuaRot-style W4AFP8,
and 5.230155 SpinQuant-transfer W4AFP8. Both rotations were worse than the
matched unrotated control; SpinQuant-transfer was slightly better than
QuaRot-style. See [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md). The deployed-quality
half is complete, while matched serving remains required.
See [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md) and
[`W4AFP8_ISAMBARD_RUNBOOK.md`](W4AFP8_ISAMBARD_RUNBOOK.md).

To isolate why the backend-compatible unrotated GPTQ checkpoint outperformed
the QuaRot-style checkpoint, revision `6f609b0` adds a matched MSE-clipping
diagnostic without replacing the accepted min/max artifacts. Gate `5875320`
and smoke `5875322` completed `0:0`, with smoke constrained by
`afterok:5875320`. Review accepted the source revision, observer metadata,
packed coverage, absence of runtime `g_idx`, fixed token counts, and artifact
hashes. Formal job `5875865` was then submitted through
`afterok:5875322` and completed `0:0` in 7m02s. Over 331,614 targets, MSE
PPL was 5.163774 unrotated and 5.216687 QuaRot-style. The matched min/max PPL
was 5.136105 and 5.248356, so MSE reduced the rotation gap by 52.8622% without
changing the winner. The accepted four-model deployment result above remains
unchanged.

The W4AFP8 **formal deployment result is a two-part evidence package**, not a
performance-only benchmark. Accuracy must be measured from the packed
checkpoints through vLLM on the retained 162 x 2048 WikiText-2 tokens
(331,614 scored next-token targets), reporting total NLL and PPL for BF16,
unrotated W4AFP8, QuaRot-style W4AFP8, and SpinQuant W4AFP8. Acceleration must
be measured separately with the matched serving protocol, reporting throughput,
TTFT, TPOT, end-to-end latency, GPU memory, request completion, and selected
kernel evidence. A method is not deployment-complete if either formal accuracy
or formal serving evidence is missing.

## Latest accepted result

FP8-targeted SpinQuant formal training job `5876984` completed `0:0` in
2h10m04s and is accepted as rotation-training evidence. It recorded 100 finite
losses and 100 non-zero gradient maxima over 800 x 2048 calibration tokens.
Training R1/R2 orthogonality errors were `1.7285e-6` and `5.9605e-7`; the
rotation SafeTensors SHA-256 is
`383004941a14e40f256abd4a615246a9adbfe1308ecd08896f167f5b6c2566ec`.
This does not establish W4AFP8 quality or acceleration. Isolated source gate
`5881273` was submitted from revision `60aa629` to create and load the new
packed checkpoint without replacing the accepted INT8-transfer artifacts; it
was `PENDING` at the 2026-08-03 snapshot. Its wall-time limit was reduced in
place from 24 hours to 6 hours, without cancellation or resubmission; prior
comparable export/load gates completed in well under one hour.

The missing serving experiment for the accepted INT8-trained rotation transfer
is being run independently. Result-gated service smoke `5882787` completed
`0:0` in 8m09s and is accepted: all four endpoints succeeded, all generated
texts matched, GPU memory recovered after shutdown, and all three quantized
logs selected `CutlassW4A8LinearKernel`. Its result SHA-256 is
`d3acdc424cd6796700a9ad937ceb26efca735a9a07adfed1b645a1403d09af7c`.
Benchmark formal `5883004` was then submitted from the same frozen revision
`ad971f9`; it entered `RUNNING` at 2026-08-03 09:20:13 and is not yet
acceleration evidence.

The latest accepted diagnostic is formal BoolQ job `5876591`, submitted after
manual acceptance of smoke `5876321`. It scored all 3,270 validation examples:
80.5810% BF16, 78.4098% unrotated W4AFP8, 78.5627% QuaRot-style W4AFP8, and
79.6024% SpinQuant-transfer W4AFP8. The formal JSON SHA-256 is
`b278567aae262fdd6f4379d4004817f17faba51fa304077f03dd1479b8bdb824`.
This is downstream quality evidence, not serving-acceleration evidence.

The accepted min/max-versus-MSE job `5875865` remains the latest observer
diagnostic.
It scored 331,614 retained WikiText-2 targets per model and confirmed that MSE
clipping improves QuaRot PPL from 5.248356 to 5.216687 while worsening the
unrotated control from 5.136105 to 5.163774. Its formal result SHA-256 is
`c6e0d88b06f27c6147c069db4f7b06e479ab9566348aa28e9cd7953ba97cbbdf`.
The primary four-model deployed-quality result remains job `5874807`, including
SpinQuant-transfer; neither result is serving-acceleration evidence.

The Isambard evidence chain is complete through Llama-2-13B offline inference,
OpenAI-compatible service smoke, matched full-model serving, and the dependent
real-vLLM layer-0 diagnostic. Formal serving job `5780631` completed all six
model/concurrency groups with 64/64 requests and zero failures. Against BF16,
both W4A16 variants reduced ready GPU memory from 34,099 to 16,125 MiB, raised
request throughput by 1.53--1.54x at concurrency 1 and 1.37x at concurrency 8,
and reduced p50 E2E by 26.9--35.2%.

Dependent jobs `5784966` and `5784967` then passed the packed-W4 hook smoke and
eight-case single-block comparison. W4A16 reduced layer-0 parameter bytes by
74.2%, but measured speedups were shape-dependent and small: approximately
0.96--1.05x versus BF16. This diagnostic used eager execution and is not a
replacement for the compiled serving result.

The deployed-checkpoint quality gate has also passed. Smoke job `5839415` and
formal job `5839419` completed with exit code `0:0`; the formal run scored
331,614 WikiText-2 targets through the real vLLM checkpoints. PPL was 5.007820
for BF16, 5.289677 for unrotated W4A16, and 5.132755 for rotated W4A16. Rotation
reduced PPL by 2.97% relative to unrotated W4A16 and recovered 55.67% of its PPL
gap to BF16 while leaving serving performance materially unchanged. The
remaining quality boundary is downstream-task or broader generation evaluation,
not deployed-checkpoint perplexity.

The SpinQuant work is isolated under its own implementation, configuration,
script, and test directories. It does not modify the completed QuaRot pipeline
or the remote vLLM checkout. Jobs `5841874` and `5842047` accepted one-step and
100-step full-model rotation training; dependent jobs `5847440` and `5847441`
then accepted the matched held-out fake-quant PPL gate. Learned offline R1/R2
reached PPL 5.0937, compared with 5.1763 unrotated and 5.7078 for the fixed
random-Hadamard R1/R2 mechanism baseline. This W4A16 result is retained as a
weight-only ablation, not the paper deployment target.

The first activation-only rotation chain was superseded after the A8 audit
found a paper-protocol mismatch at `o_proj`. Corrected jobs `5854268` and
`5854269` passed the one-step and 100-step W16A8 learning gates. Corrected GPTQ
W4A8 PPL smoke `5854890` and formal job `5854891` then passed on the retained
held-out tokens. SpinQuant no-had reached PPL `5.1627`, slightly worse than the
matched unrotated W4A8 control at `5.1480`; this completes the reproduction
chain but does not establish a learned-rotation benefit. See
[`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md) for protocol and claim boundaries
and [`SPINQUANT_PLAN.md`](SPINQUANT_PLAN.md) for the next stages.

See [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md) for the accepted metrics,
[`VLLM_W4A16_ISAMBARD_RUNBOOK.md`](VLLM_W4A16_ISAMBARD_RUNBOOK.md) for the
procedure, and
[`QUAROT_REAL_DEPLOYMENT_ROADMAP.md`](QUAROT_REAL_DEPLOYMENT_ROADMAP.md) for
the full decision record.
