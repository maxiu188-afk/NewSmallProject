# Documentation archive

This directory contains superseded, duplicated, or completed documentation.
Files are retained for provenance and reproducibility; they should not be used
as the current project status or next-action list.

## `fake_quant/`

Detailed source records merged into
[`../FAKE_QUANT_RESULTS.md`](../FAKE_QUANT_RESULTS.md):

- the formal Llama-2-13B BF16, RTN, GPTQ, component, and calibration records;
- the local tiny-model and SmolLM2 fake-quant records;
- the preliminary RunPod A40 fake-quant record and local scope note.

## `results/`

Detailed source records merged into
[`../OFFICIAL_QUAROT_RESULTS.md`](../OFFICIAL_QUAROT_RESULTS.md):

- the official Llama-2-13B full-model W4A4KV4 result;
- the official Llama-2-7B paper-aligned single-block result.

## `smokes/`

Completed local, Windows CUDA, modern-runtime, and portable-pipeline smoke
records that were superseded by formal model and server results.

## `plans/`

Superseded or completed planning documents:

- the original Llama-2-13B evaluation plan;
- the deferred Qwen roadmap;
- the owned-kernel deployment plan superseded by the two-route roadmap;
- the full dated phase-status history through 2026-07-22.

## `runbooks/`

Completed operational procedures for RunPod, the GPTQ study, the W4A8 layer
and full-decoder gates, the W4A8 performance smoke, and the official
single-block matrix. The active server procedure is now
[`../VLLM_W4A16_ISAMBARD_RUNBOOK.md`](../VLLM_W4A16_ISAMBARD_RUNBOOK.md).

## Archive rule

When a historical document is still needed for an exact command, environment,
hash, or failure record, link to it explicitly. Do not copy its old status or
next-step wording back into a current document.
