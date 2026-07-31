# Current phase status

This file is the concise current-state entry point. The dated update history
through 2026-07-22 is preserved in
[`archive/plans/PHASE_STATUS_2026-07-22.md`](archive/plans/PHASE_STATUS_2026-07-22.md).

## Status summary

| Workstream | Current status | Evidence boundary |
|---|---|---|
| Algorithmic fake quant | Complete for the formal Llama-2-13B study | QuaRot + GPTQ W4A4 reached PPL 5.8376 versus BF16 5.0087; all low-bit rows are floating QDQ and keep K/V at 16 bits |
| SpinQuant fake quant | Historical offline-fused W4A16 training/PPL passed; paper-aligned W16A8 activation-only smoke 5848060 and formal training 5848547 also passed on GH200 | Rotation learning is complete; post-learning GPTQ, no-had W4A8 fake-quant evaluation, A4/`had` R3/R4, downstream accuracy, packing, and deployment remain separate gates |
| Owned packed W4A8 | Correctness complete on RTX 6000 Ada; selected-linear portability accepted on GH200 | All 280 decoder linears matched the packed oracle on Ada; the Isambard result covers one `q_proj`, not the full decoder; K/V remain BF16 |
| Official QuaRot full model | Complete on RTX 6000 Ada | Real Llama-2-13B W4A4KV4 reduced model-resident memory from 26.29 GB to 7.18 GB but was slower at batch one; no packed-checkpoint PPL result |
| Official QuaRot single block | Complete on RTX 6000 Ada | W4 completed 14/14 cases; 2048-token prefill gained 1.53--1.68x; batch-16/context-4096 layer E2E gained 1.28x; this is not full-model latency |
| vLLM W4A16 serving | Complete on GH200 for the matched Llama-2-13B serving and layer-0 diagnostic protocols | W4A16 cut ready GPU memory by 52.7% and improved request throughput by 1.37--1.54x; rotation had no material performance effect; no deployed-checkpoint PPL or downstream-quality result |

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
SpinQuant paper's real-deployment `no_had` route is W4A8, with offline-fused
R1/R2 and activation quantization retained at inference. The first revised
SpinQuant deployment target is therefore W4A8KV16. Fake-quant integration is a
low-to-medium change, but accelerated INT4-weight/INT8-activation serving on
GH200 is a separate high-risk backend gate: stable vLLM does not currently
list this combination as supported on Hopper.

## Latest accepted result

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

The remaining QuaRot scientific boundary is deployed-checkpoint quality. Its
fixed-token offline gate establishes bounded execution correctness, but this
phase has not measured WikiText-2 PPL or a downstream task on the packed W4A16
checkpoint. Rotation did not materially change serving or layer timing, so any
rotation benefit must be evaluated through a separate quality protocol rather
than inferred from performance.

The SpinQuant work is isolated under its own implementation, configuration,
script, and test directories. It does not modify the completed QuaRot pipeline
or the remote vLLM checkout. Jobs `5841874` and `5842047` accepted one-step and
100-step full-model rotation training; dependent jobs `5847440` and `5847441`
then accepted the matched held-out fake-quant PPL gate. Learned offline R1/R2
reached PPL 5.0937, compared with 5.1763 unrotated and 5.7078 for the fixed
random-Hadamard R1/R2 mechanism baseline. This W4A16 result is retained as a
weight-only ablation, not the paper deployment target.

Paper-aligned activation-only rotation learning also completed. Job `5848060`
passed the one-step W16A8 smoke, and job `5848547` completed all 100 formal
updates in 2 hours 9 minutes 15 seconds with finite losses and non-zero
rotation gradients. The accepted learned rotations now feed the remaining
post-learning GPTQ and no-had W4A8 fake-quant gates. See
[`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md) for protocol and claim boundaries
and [`SPINQUANT_PLAN.md`](SPINQUANT_PLAN.md) for the next stages.

See [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md) for the accepted metrics,
[`VLLM_W4A16_ISAMBARD_RUNBOOK.md`](VLLM_W4A16_ISAMBARD_RUNBOOK.md) for the
procedure, and
[`QUAROT_REAL_DEPLOYMENT_ROADMAP.md`](QUAROT_REAL_DEPLOYMENT_ROADMAP.md) for
the full decision record.
