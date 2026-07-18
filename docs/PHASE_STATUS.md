# Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 — Reproduction contract | complete | Offline config validation, run-manifest generation, environment templates, taxonomy, and upstream audit pass local smoke checks |
| 1 — Primitive correctness | complete (reference scope) | Independent CPU tests pass for QDQ, int4 packing, Hadamard invariants, rotation equivalence, and LLaMA-2-relevant dimension planning |
| 2 — Model equivalence | complete for local smoke scope | Framework-free one-token check passes with <=2.3e-16 error; two-layer PyTorch/Transformers random LLaMA passes with logits <=1.2e-07 and hidden states <=6.0e-07, with no model download |
| 3 — Fake-quant accuracy | F4 GPTQ text result complete; naive GPTQ control pending | Local F0–F5 checks plus a RunPod A40 matched SmolLM2-135M W4A4KV4 naive/QuaRot smoke completed. On Llama-2-13B WikiText-2, RTN F3/F4 records PPL 8719.68 / 12.46; QuaRot GPTQ F4 improves to 5.84; the same-run BF16 F0 is 5.0087 |
| 3b — Portable LLaMA pipeline | partial offline pretrained smoke complete | Generic model/data/runtime configuration, larger synthetic GQA LLaMA equivalence, and W4A4KV4 QDQ smoke pass locally; pinned SmolLM2-135M executes offline with residual, V/O, Q/K-after-RoPE, and 12 x 128 MLP checks |
| 4 — CUDA/kernel correctness | reference-build failure captured; kernel not built | The A40 preflight and cu128 runtime check passed. The unmodified upstream editable build reached CMake configuration, then failed because nested legacy `pip install -e` calls used build isolation without PyTorch; no source patch or kernel claim was made |
| 5 — Performance | out of current scope | Requires real deployment environment; not attempted |
| 6 — Presentation package | partial | Plan, audit, execution taxonomy, and empirical Llama-2-13B BF16/RTN/GPTQ accuracy records are available; no deployment-performance table exists |

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

## 2026-07-17 Llama-2-13B F0 update

The formal server baseline completed on CUDA with `meta-llama/Llama-2-13b-hf`
at revision `5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1` and
`Salesforce/wikitext` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
The BF16 F0 evaluation scored mean NLL `1.6111028514668504` and PPL
`5.008331629066348` over 331,614 next-token targets (162 x 2048-token
sequences). This is an internally reproducible full-precision control, not a
GPTQ, QuaRot low-bit, KV-cache, kernel, memory, or throughput claim. The
server result, manifests, pinned config, and preflight record were verified
and copied locally before shutdown; see `LLAMA2_13B_BF16_BASELINE.md`.

## 2026-07-17 Llama-2-13B RTN W4A4 update

The formal fake-quant F3/F4 pair completed on one RTX 6000 Ada using the same
pinned model and WikiText-2 protocol as F0. The naive RTN W4A4 control scored
PPL `8719.677539`; the QuaRot W4A4 candidate scored `12.460633`. Both rows
score 331,614 next-token targets, leave K/V at 16 bits, and are QDQ simulation
rather than GPTQ or deployment evidence. The strict result-comparison gate
passed; see `LLAMA2_13B_RTN_W4A4_RESULTS.md`.

## 2026-07-17 GPTQ server result

The code, matched naive/QuaRot GPTQ W4A4 configurations, and the calibration
contract are committed. A CUDA 12.4 RTX 6000 Ada session passed a tiny LLaMA
end-to-end GPTQ smoke, then completed the formal QuaRot F4 GPTQ command. It
scored PPL `5.837575` against a same-run BF16 PPL `5.008716`, improving on the
matched QuaRot RTN PPL `12.460633`. The raw server artifacts were copied and
SHA-256 checked locally before shutdown. The naive GPTQ control remains open;
see `LLAMA2_13B_GPTQ_W4A4_RESULTS.md`.

## Local dependency boundary

Common development dependencies may be added when a concrete task requires
them. The isolated Windows PyTorch and Transformers smoke environment has been
used only for the recorded local NVIDIA-GPU F5 check. Datasets, Accelerate,
LM-Eval, and the upstream CUDA extension remain outside that local smoke
environment.

## RunPod material

`scripts/runpod_preflight.py`, `docs/RUNPOD_SMOKE_RUNBOOK.md`, and the RunPod
configuration templates have now been used for preflight, the small CUDA
fake-quant gate, the pinned Llama-2-13B BF16 text baseline, and a matched RTN
W4A4 F3/F4 text pair, and a reviewed QuaRot GPTQ F4 text result with a
separate pinned calibration policy. QuaRot remains the baseline; the upstream
reference implementation is not a required kernel dependency for the owned
deployment track.
