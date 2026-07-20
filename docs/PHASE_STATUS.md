# Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 — Reproduction contract | complete | Offline config validation, run-manifest generation, environment templates, taxonomy, and upstream audit pass local smoke checks |
| 1 — Primitive correctness | complete (reference scope) | Independent CPU tests pass for QDQ, int4 packing, Hadamard invariants, rotation equivalence, and LLaMA-2-relevant dimension planning |
| 2 — Model equivalence | complete for local smoke scope | Framework-free one-token check passes with <=2.3e-16 error; two-layer PyTorch/Transformers random LLaMA passes with logits <=1.2e-07 and hidden states <=6.0e-07, with no model download |
| 3 — Fake-quant accuracy | GPTQ F3/F4, component, and calibration study complete | Local F0–F5 checks plus a RunPod A40 matched SmolLM2-135M W4A4KV4 naive/QuaRot smoke completed. On Llama-2-13B WikiText-2, GPTQ naive F3 / complete F4 records PPL 8624.35 / 5.84; 32/64/128 calibration and cumulative-component rows are reviewed; the same-run BF16 F0 is 5.0087 |
| 3b — Portable LLaMA pipeline | partial offline pretrained smoke complete | Generic model/data/runtime configuration, larger synthetic GQA LLaMA equivalence, and W4A4KV4 QDQ smoke pass locally; pinned SmolLM2-135M executes offline with residual, V/O, Q/K-after-RoPE, and 12 x 128 MLP checks |
| 4 — CUDA/kernel correctness | W4A8 int4 GEMM, formal GPTQ-packed Llama linear export, and one selected `q_proj` layer/logits gate passed on RTX 6000 Ada | The owned W4A8 CUDA kernel exactly matches independent int32 references for `(5120,5120)`, `(13824,5120)`, and `(5120,13824)`. A 40-layer/280-linear F4 GPTQ run exported a self-describing act-order packed `q_proj`; the CUDA module agrees with its packed oracle to `5.66e-07`. The formal fixed-token decoder-layer and full-model-logits comparison then passed for that one replacement. Isambard-AI has a rebuilt ARM64 environment and an equivalent gate queued, but no Isambard GPU result yet. Full-model integration and KV4 remain separate gates. |
| 5 — Performance | not started | Requires a Phase-2 packed full-model path and matching numerical results; no timing or memory claim exists |
| 6 — Presentation package | partial | Plan, audit, execution taxonomy, empirical Llama-2-13B accuracy records, W4A8 kernel report, and Isambard command reference are available; no deployment-performance table exists |

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
SHA-256 checked locally before shutdown. The naive GPTQ control was then still
open;
see `LLAMA2_13B_GPTQ_W4A4_RESULTS.md`.

## 2026-07-18 GPTQ component and calibration update

The reviewed CUDA study completed all nine planned rows on the same pinned
Llama-2-13B/WikiText-2 protocol. The missing naive GPTQ F3 control scored PPL
`8624.350905`; complete QuaRot F4 scored `5.837575`. The 32/64/128 calibration
and cumulative component rows are fully documented in
`LLAMA2_13B_GPTQ_ABLATION_CALIBRATION_RESULTS.md`. Raw JSON/log artifacts were
recovered and SHA-256 checked before this status update.

## 2026-07-20 W4A8 layer gate and Isambard preparation

The first formal real-model Phase-2 correctness gate passed on the RTX 6000
Ada RunPod environment. It loaded the pinned Llama-2-13B revision, applied the
recorded QuaRot transformation, and replaced only the exported formal GPTQ
tensor `model.layers.0.self_attn.attention.q_proj`. The independent packed
W4/A8 oracle and the owned CUDA `W4A8Linear` produced identical captured
decoder-layer output and full-model logits for the fixed `[1, 16]` token input
after their outputs were explicitly converted back to the surrounding BF16
module boundary. The recovered report has SHA-256
`406d16c805fcf95b5854a13701b1a58e8f47bffd3f7642d9942ba24dcf207c61`.

The Isambard-AI `u6rt` project environment was then rebuilt directly in the
persistent home directory, not as a Slurm environment-installation job. The
ARM64 CUDA 12.8 wheel index does not publish `torch==2.8.0`; the recorded
Isambard environment instead uses `torch==2.9.0+cu128`, Transformers `5.14.1`,
Datasets `5.0.0`, CUDA compiler `12.6`, and `ninja==1.11.1.4`. The pinned model
snapshot and WikiText-2 train/test splits were reloaded successfully in offline
mode. Initial concurrent Hub download remnants were removed only after the
offline snapshot was verified.

The corresponding Isambard GPU gate is submitted as Slurm job `5732906` and
was pending for scheduler priority at the time of this update. A queued job is
not evidence: no Isambard CUDA result, all-linear integration, generation
smoke, KV4 result, PPL, timing, throughput, or memory claim is made here.

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
