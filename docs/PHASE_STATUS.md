# Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 — Reproduction contract | complete | Offline config validation, run-manifest generation, environment templates, taxonomy, and upstream audit pass local smoke checks |
| 1 — Primitive correctness | complete (reference scope) | Independent CPU tests pass for QDQ, int4 packing, Hadamard invariants, rotation equivalence, and LLaMA-2-relevant dimension planning |
| 2 — Model equivalence | complete for local smoke scope | Framework-free one-token check passes with <=2.3e-16 error; two-layer PyTorch/Transformers random LLaMA passes with logits <=1.2e-07 and hidden states <=6.0e-07, with no model download |
| 3 — Fake-quant accuracy | CUDA synthetic-input smoke complete; text evaluation pending | Local F0–F5 checks plus a RunPod A40 matched SmolLM2-135M W4A4KV4 naive/QuaRot smoke completed; the server result uses fixed synthetic tokens and is not a WikiText-2 PPL claim |
| 3b — Portable LLaMA pipeline | partial offline pretrained smoke complete | Generic model/data/runtime configuration, larger synthetic GQA LLaMA equivalence, and W4A4KV4 QDQ smoke pass locally; pinned SmolLM2-135M executes offline with residual, V/O, Q/K-after-RoPE, and 12 x 128 MLP checks |
| 4 — CUDA/kernel correctness | reference-build failure captured; kernel not built | The A40 preflight and cu128 runtime check passed. The unmodified upstream editable build reached CMake configuration, then failed because nested legacy `pip install -e` calls used build isolation without PyTorch; no source patch or kernel claim was made |
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

## 2026-07-16 RunPod update

The server preflight recorded an NVIDIA A40 (`sm_86`), driver `570.211.01`,
CUDA compiler `12.8.93`, Python `3.11.13`, PyTorch `2.11.0+cu128`, and
Transformers `5.14.1`. A 16-bit SmolLM2-135M rotation check on CUDA produced
mean/max absolute logit errors of `1.347205e-05` / `2.908707e-04`. The matched
synthetic-token W4A4KV4 smoke produced `8.3990068` / `44.7422943` for the
naive control and `5.0678024` / `33.9174576` for QuaRot. See
`RUNPOD_FAKE_QUANT_RESULTS.md` for configurations, boundaries, and the next
WikiText-2 step.

## Local dependency boundary

Common development dependencies may be added when a concrete task requires
them. The isolated Windows PyTorch and Transformers smoke environment has been
used only for the recorded local NVIDIA-GPU F5 check. Datasets, Accelerate,
LM-Eval, and the upstream CUDA extension remain outside that local smoke
environment.

## RunPod material

`scripts/runpod_preflight.py`, `docs/RUNPOD_SMOKE_RUNBOOK.md`, and the RunPod
configuration templates have now been used for preflight and the small CUDA
fake-quant gate. A pinned WikiText-2 evaluation and official LLaMA-2 model
access remain required before a text-accuracy claim or large-scale execution.
