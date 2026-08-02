# Documentation index

This directory separates the current source of truth from historical evidence.
Start with the documents in **Current status and results**. Material under
[`archive/`](archive/README.md) is retained for provenance and reproduction,
but it is not the current project plan.

## Current status and results

- [`PHASE_STATUS.md`](PHASE_STATUS.md): concise current state, latest accepted
  result, and remaining evidence boundary.
- [`FAKE_QUANT_RESULTS.md`](FAKE_QUANT_RESULTS.md): consolidated algorithmic
  fake-quant results from local smokes through the formal Llama-2-13B study.
- [`OFFICIAL_QUAROT_RESULTS.md`](OFFICIAL_QUAROT_RESULTS.md): consolidated
  official-backend full-model and paper-aligned single-block results.
- [`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md): accepted Isambard GH200
  deployed-checkpoint PPL, full-model serving result, and same-environment
  layer-0 diagnostic.
- [`W4AFP8_RESULTS.md`](W4AFP8_RESULTS.md): accepted packed-checkpoint vLLM PPL
  for BF16, unrotated W4AFP8, QuaRot-style W4AFP8, and SpinQuant transfer, plus
  the accepted min/max-versus-MSE observer diagnostic.
- [`SPINQUANT_RESULTS.md`](SPINQUANT_RESULTS.md): accepted Llama-2-13B
  rotation-training and matched held-out fake-quant PPL evidence, including the
  corrected no-had W4A8 deployment boundary.
- [`SPINQUANT_PLAN.md`](SPINQUANT_PLAN.md): independent SpinQuant fake-quant
  implementation status, clean-room boundary, W4A8 migration assessment, and
  staged reproduction plan.
- [`W4AFP8_DEPLOYMENT_PLAN.md`](W4AFP8_DEPLOYMENT_PLAN.md): active joint
  QuaRot/SpinQuant GH200 deployment plan; deployed-checkpoint quality is
  accepted, the MSE observer diagnostic is complete, and
  matched full-model acceleration remains.
- [`W4AFP8_ISAMBARD_RUNBOOK.md`](W4AFP8_ISAMBARD_RUNBOOK.md): isolated local
  preparation, accepted GH200 export/PPL evidence, and remaining serving
  procedure.
- [`QUAROT_REAL_DEPLOYMENT_ROADMAP.md`](QUAROT_REAL_DEPLOYMENT_ROADMAP.md):
  completed QuaRot routes plus the planned joint W4AFP8 extension.
- [`QUAROT_PROGRESS_REPORT_ZH.tex`](QUAROT_PROGRESS_REPORT_ZH.tex): concise
  Chinese progress report for XeLaTeX/Overleaf.

## Active implementation and runbook

- [`VLLM_W4A16_ISAMBARD_RUNBOOK.md`](VLLM_W4A16_ISAMBARD_RUNBOOK.md):
  Isambard GH200 environment, completed gates, and reproduction procedure.
- [`W4A8_CUDA_KERNEL_RESULTS.md`](W4A8_CUDA_KERNEL_RESULTS.md): owned packed-W4
  and W4A8 correctness record, including the 280-linear decoder gate.
- [`PACKED_W4_FORMAT.md`](PACKED_W4_FORMAT.md): owned checkpoint format and
  numerical contract.

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
