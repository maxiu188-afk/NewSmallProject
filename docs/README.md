# Documentation index

This directory separates the current source of truth from historical evidence.
Start with the documents in **Current status and results**. Material under
[`archive/`](archive/README.md) is retained for provenance and reproduction,
but it is not the current project plan.

## Slurm smoke submission rule

Classify every smoke before submitting its formal job:

1. **Result-gated smoke:** if formal submission requires reading or accepting
   the smoke JSON, logs, numerical values, hashes, or artifacts, submit only
   the smoke. Do not prequeue formal. Wait for the user to report completion,
   then review the result and submit formal only after acceptance. If Slurm no
   longer accepts the completed smoke as a dependency, use a read-only
   acceptance job when the formal wrapper requires `afterok`; never rerun the
   same smoke merely to recreate a scheduler dependency.
2. **Runnability-only smoke:** if the smoke only proves that the fixed job can
   execute and formal does not require a separate result decision, submit
   smoke and formal together with `formal --dependency=afterok:<smoke-job>`.
   Do not continuously monitor the smoke.

The presence of an `afterok` option does not determine the category. The
controlling question is whether a smoke result must be inspected before formal
is authorized. Record the category in the relevant runbook and never convert a
result-gated smoke into an automatically queued formal job.

## Current status and results

- [`PHASE_STATUS.md`](PHASE_STATUS.md): concise current state, latest accepted
  result, and remaining evidence boundary.
- [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md): claim-to-job-to-artifact index,
  recorded hashes, scope boundaries, and the pending read-only durability
  audit.
- [`FAKE_QUANT_RESULTS.md`](FAKE_QUANT_RESULTS.md): consolidated algorithmic
  fake-quant results from local smokes through the formal Llama-2-13B study.
- [`OFFICIAL_QUAROT_RESULTS.md`](OFFICIAL_QUAROT_RESULTS.md): consolidated
  official-backend full-model and paper-aligned single-block results.
- [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md): accepted Isambard GH200
  deployed-checkpoint PPL, full-model serving result, and same-environment
  layer-0 diagnostic.
- [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md): accepted packed-checkpoint vLLM PPL
  and matched serving for the INT8-trained transfer and FP8-targeted SpinQuant
  endpoints, plus the accepted min/max-versus-MSE and old/new SpinQuant BoolQ
  diagnostics.
- [`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md): accepted Llama-2-13B
  rotation-training and matched held-out fake-quant PPL evidence, including the
  corrected no-had W4A8 deployment boundary.
- [`SPINQUANT_PLAN.md`](SPINQUANT_PLAN.md): retained SpinQuant implementation
  and execution decision record; its experimental stages are closed.
- [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md): completed joint
  QuaRot/SpinQuant GH200 deployment plan, including FP8-targeted PPL and matched
  full-model serving acceptance.
- [`W4AFP8_ISAMBARD_RUNBOOK.md`](W4AFP8_ISAMBARD_RUNBOOK.md): isolated local
  preparation and accepted GH200 export, PPL, downstream-diagnostic, and
  serving evidence.
- [`QUAROT_REAL_DEPLOYMENT_ROADMAP.md`](QUAROT_REAL_DEPLOYMENT_ROADMAP.md):
  completed QuaRot routes plus the completed joint W4AFP8 extension.
- [`QUAROT_PROGRESS_REPORT_ZH.tex`](QUAROT_PROGRESS_REPORT_ZH.tex): concise
  Chinese progress report for XeLaTeX/Overleaf.

## Active implementation and runbook

- [`VLLM_W4A16_ISAMBARD_RUNBOOK.md`](VLLM_W4A16_ISAMBARD_RUNBOOK.md):
  Isambard GH200 environment, completed gates, and reproduction procedure.
- [`W4A8_CUDA_KERNEL_RESULTS.md`](W4A8_CUDA_KERNEL_RESULTS.md): owned packed-W4
  and W4A8 correctness record, including the 280-linear decoder gate.
- [`PACKED_W4_FORMAT.md`](PACKED_W4_FORMAT.md): owned checkpoint format and
  numerical contract.

## Separately scoped serving-backend study

- [`SERVING_BACKEND_COMPARISON_PLAN.md`](SERVING_BACKEND_COMPARISON_PLAN.md):
  bounded SGLang-versus-vLLM comparison plan with exact-checkpoint,
  quality-parity, matched-resource, and result-gated acceptance rules.
- [`SERVING_BACKEND_COMPARISON_RESULTS.md`](SERVING_BACKEND_COMPARISON_RESULTS.md):
  reviewed GH200 compatibility-smoke record. It confirms that SGLang `0.5.16`
  cannot load the exact unchanged dense W4AFP8 checkpoint, and records the
  accepted exact-checkpoint QuaRot W4A16 BoolQ smoke: vLLM 28/32, SGLang
  27/32, one prediction disagreement. Serving-client smoke `5944823` failed on
  two reviewed harness gates after SGLang successfully loaded and became
  healthy. Harness-only repair revision `ac82347` was validated, and its single
  replacement result-gated smoke `5949509` failed after vLLM completed the
  workload and SGLang loaded/captured graphs but stalled before Uvicorn
  application startup completed. A bounded repair restores the known-good
  checkpoint-path model ID and adds timeout diagnostics; no formal job is
  queued. This is not yet formal quality or serving-performance evidence.

## Method, policy, and reusable reference

- [`QUAROT_METHOD_AND_EXECUTION_NOTE.md`](QUAROT_METHOD_AND_EXECUTION_NOTE.md):
  method and algorithm/deployment distinction.
- [`PORTABLE_PIPELINE.md`](PORTABLE_PIPELINE.md): configuration-driven portable
  pipeline and local commands.
- [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md): result labels and claim
  boundaries.
- [`CUDA_PYTORCH_COMPATIBILITY_POLICY.md`](CUDA_PYTORCH_COMPATIBILITY_POLICY.md):
  approved runtime-pair policy and mandatory checks.
- [`UPSTREAM_AUDIT.md`](UPSTREAM_AUDIT.md): pinned upstream reference audit.

## Historical material

Completed runbooks, superseded plans, preliminary smokes, and the detailed
records merged into the current summaries are listed in
[`archive/README.md`](archive/README.md). Archive files remain tracked so that
commands, provenance, and artifact hashes are not lost.
