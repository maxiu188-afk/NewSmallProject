# Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 — Reproduction contract | complete | Offline config validation, run-manifest generation, environment templates, taxonomy, and upstream audit pass local smoke checks |
| 1 — Primitive correctness | complete (reference scope) | Independent CPU tests pass for QDQ, int4 packing, Hadamard invariants, rotation equivalence, and LLaMA-2-relevant dimension planning |
| 2 — Model equivalence | complete for local smoke scope | Framework-free one-token check passes with <=2.3e-16 error; two-layer PyTorch/Transformers random LLaMA passes with logits <=1.2e-07 and hidden states <=6.0e-07, with no model download |
| 3 — Fake-quant accuracy | local mechanism smoke complete | F0–F4 random tiny-model QDQ and pinned SmolLM2-135M synthetic-input QDQ matrices ran successfully; neither establishes pretrained text accuracy or includes KV4 |
| 3b — Portable LLaMA pipeline | partial pretrained smoke complete | Generic model/data/runtime configuration, larger synthetic GQA LLaMA equivalence, and W4A4 QDQ smoke pass locally; pinned SmolLM2-135M executes offline with residual, V/O, and 12 x 128 MLP equivalence checks |
| 4 — CUDA/kernel correctness | out of current scope | Requires NVIDIA CUDA; not attempted |
| 5 — Performance | out of current scope | Requires real deployment environment; not attempted |
| 6 — Presentation package | partial | Plan, audit, execution taxonomy, local results, and method note are available; no empirical accuracy/performance table exists |

## 2026-07-16 implementation update

The table above records the completed historical runs. The portable algorithm
now additionally implements Q/K per-head Hadamard immediately after RoPE,
K-cache QDQ, and separate V-projection-output QDQ. The new
`synthetic_llama_w4a4kv4_smoke.json` configuration performs sequential cached
decoding so that the KV QDQ path is exercised. Static checks plus the isolated
Windows PyTorch/Transformers CUDA smoke pass; see
`WINDOWS_CUDA_SMOKE.md`. The result remains synthetic fake-quant evidence only.

## Local dependency boundary

Common development dependencies may be added when a concrete task requires
them. The next approved isolated smoke environment is Windows PyTorch and
Transformers using the local NVIDIA GPU. It remains uncreated while the
algorithm implementation is under review. Datasets, Accelerate, LM-Eval, and
the upstream CUDA extension remain outside that local smoke environment.

## RunPod material

`scripts/runpod_preflight.py`, `docs/RUNPOD_SMOKE_RUNBOOK.md`, and the RunPod
configuration templates are used only after the local GPU smoke gate passes.
They remain unsuitable for unreviewed large-scale execution.
