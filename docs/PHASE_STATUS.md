# Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 — Reproduction contract | complete | Offline config validation, run-manifest generation, environment templates, taxonomy, and upstream audit pass local smoke checks |
| 1 — Primitive correctness | complete (reference scope) | Independent CPU tests pass for QDQ, int4 packing, Hadamard invariants, rotation equivalence, and LLaMA-2-relevant dimension planning |
| 2 — Model equivalence | complete for local smoke scope | Framework-free one-token check passes with <=2.3e-16 error; two-layer PyTorch/Transformers random LLaMA passes with logits <=1.2e-07 and hidden states <=6.0e-07, with no model download |
| 3 — Fake-quant accuracy | GPTQ F3/F4, component, and calibration study complete | Local F0–F5 checks plus a RunPod A40 matched SmolLM2-135M W4A4KV4 naive/QuaRot smoke completed. On Llama-2-13B WikiText-2, GPTQ naive F3 / complete F4 records PPL 8624.35 / 5.84; 32/64/128 calibration and cumulative-component rows are reviewed; the same-run BF16 F0 is 5.0087 |
| 3b — Portable LLaMA pipeline | partial offline pretrained smoke complete | Generic model/data/runtime configuration, larger synthetic GQA LLaMA equivalence, and W4A4KV4 QDQ smoke pass locally; pinned SmolLM2-135M executes offline with residual, V/O, Q/K-after-RoPE, and 12 x 128 MLP checks |
| 4 — CUDA/kernel correctness | Complete 280-linear W4A8 decoder gate passed on RTX 6000 Ada; selected-linear Isambard portability passed | The owned kernel exactly matches int32 references for all three Llama linear shapes. The full sharded GPTQ checkpoint replaced all 280 decoder linears on RTX 6000 Ada; all 40 fixed-token layer outputs and final logits matched the packed oracle with maximum error `0.0`, and BF16-K/V generation logits were finite. Isambard job `5742443` independently passed the kernel and selected-`q_proj` layer/logits gate on GH200. No Isambard full-decoder or KV4 result exists. |
| 5 — Performance | full-model extension and paper-aligned single-block run complete | The 280-linear full-model extension reduced model-resident allocated memory from 26.29 GB to 7.18 GB but was slower at batch one. In the paper-aligned Llama-2-7B single-block matrix on the same RTX 6000 Ada, W4A4KV4 completed 14/14 cases, accelerated paired 2048-token prefill by 1.53--1.68x, and reached 1.28x layer-e2e at batch 16/context 4096. FP16 completed 13/14 and OOMed at batch 64/sequence 2048; W4 completed that point at 16.86 GB peak. |
| 6 — Presentation package | Route A complete; Route B serving follows | Faithful upstream Llama-2-13B W4A4KV4 full-model execution and the paper-protocol single-block matrix are both documented. The next implementation is QuaRot-style vLLM W4A16 on Isambard, with full-model offline inference and serving as the primary practical result and single-block timing as a diagnostic. |

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

## 2026-07-21 Isambard gate repair and current state

The initial Isambard jobs `5732906` and `5735242` failed before any CUDA,
model, or numerical work. Slurm copies an `.sbatch` script to its spool
directory before execution; the script had derived `project_root` from
`BASH_SOURCE`, and therefore looked for the checked export artifact under
`/var/spool/slurmd/...` rather than in `NewSmallProject`. The logged missing
input was the formal packed `q_proj` artifact, not a missing cache or a failed
GPTQ export.

`scripts/run_isambard_w4a8_layer_gate.sbatch` now takes the project root from
`SLURM_SUBMIT_DIR` (with the source-relative location retained for direct
execution). Its Slurm `--test-only` validation passed, and job `5739260` was
submitted on 2026-07-21 at 09:26 UK time. The scheduler snapshot immediately
after submission was `PENDING`, `Reason=None`, `StartTime=Unknown`; a separate
test-only estimate indicated approximately 2026-07-25 11:28 as the earliest
currently available allocation. This is scheduling information only, not a
failure or a result.

The batch log now emits one unambiguous stage marker for each boundary:
`module_load`, `inputs`, `source_manifest`, `preflight`, `kernel_smoke`,
`layer_smoke`, and `assert_result`. Consequently, any later failure can be
attributed to a specific prerequisite, CUDA kernel build/execution, or
fixed-token layer comparison without rebuilding the environment, re-downloading
the model, or re-exporting the W4 artifact. A successful job still proves only
the selected W4A8 `q_proj` replacement; all other linears and K/V remain BF16.

## Quantization naming boundary

The historic QuaRot fake-quant accuracy study is W4A4 (and F5 is W4A4KV4): it
uses floating QDQ simulation. The Phase-2 job is instead W4A8 execution: it
uses a 4-bit GPTQ-packed `q_proj` weight artifact and quantizes that layer's
runtime input to int8 before the owned CUDA int32 accumulator. The layer gate
currently consumes the historic F4 configuration file named
`llama2_13b_wikitext2_quarot_w4a4_gptq.json` as its transformation/export
provenance; that filename does not change the W4A8 execution path or turn the
job into a W4A4 fake-quant result.

## 2026-07-21 local full-decoder gate preparation

The next Phase-2 implementation slice is ready for review locally without
starting another GPU server. GPTQ capture can now stream each completed packed
linear through a callback, and the full-checkpoint exporter writes one CPU
`.pt` shard per transformed decoder linear. It publishes the self-describing
`manifest.json` atomically only after all expected tensors exist. The loader
checks every SHA-256, dtype, shape, input permutation, bias boundary, model
linear name, source config, and model revision before replacement.

The corresponding correctness script installs all indexed linears through the
independent packed Torch oracle, captures all 40 decoder outputs plus final
logits, converts the same buffers to the owned CUDA modules without reloading
the model, repeats the fixed-token forward, and then runs a short finite-logit
greedy-generation smoke with BF16 K/V. The checked-in Isambard batch entry
keeps preflight, kernel, export, validation, full-model comparison, and final
assertion in one staged job. RunPod uses the same scripts only as a fallback.

On macOS, the focused Torch tests pass for streamed callback capture,
checkpoint creation/load, deliberate checksum corruption rejection, exact
linear-set enforcement, and reference-to-CUDA module conversion. This is code
preparation only: the full exporter and numerical comparison require NVIDIA
CUDA and have not run. Isambard remains the primary formal environment once
its service is healthy; the existing selected-linear job is not cancelled by
this preparation. See `PHASE2_GPTQ_W4A8_FULL_MODEL_RUNBOOK.md`.

## 2026-07-21 local performance preparation

The full-decoder RunPod correctness gate subsequently passed for all 280
linears: all 40 captured layer outputs and final logits matched the independent
packed oracle with maximum error `0.0`, and the BF16-K/V generation smoke had
finite logits. The recovered result JSON has SHA-256
`0a848e22ff1881ecd1ee76d829e3c39d43e2f9224892cdd1e61ada6be9b208d8`;
the linked checkpoint manifest has SHA-256
`caa12e485087e4bc5630c950e5b96041ba90e770cc40cdd7ae5defba64641e33`.

Phase-3 benchmark code is now prepared locally and refuses to run without that
passed evidence. BF16 and W4A8 use identical fixed inputs and the same QuaRot
transform, run sequentially to avoid dual-model residency, and write every raw
CUDA-event and synchronized-wall sample plus peak allocated/reserved memory.
The packed oracle, model loading, checkpoint validation, and extension build
are excluded from timing. Isambard remains the primary formal target; RunPod is
the fallback.

The subsequent RTX 6000 Ada smoke completed with result SHA-256
`29528757e7f843efb501df3c7d920cbcffe38af4d02e3916f93209f31b030989`.
W4A8 peak allocated memory was about 28.4--28.8% of BF16, but its measured
speedup ratios were below one: about `0.024x` for 16-token prefill, `0.294x`
for fixed-context one-token decode, and `0.079x` for four-token generation.
The formal grid was stopped because the correctness-oriented kernel is not a
competitive performance backend. See `W4A8_PERFORMANCE_RUNBOOK.md`.

## 2026-07-21 two-route real-deployment reset

Deployment work now follows `QUAROT_REAL_DEPLOYMENT_ROADMAP.md`. Route A is a
faithful build and execution of the pinned upstream W4A4/CUDA/e2e backend,
starting with source, primitive, KV4, and one-layer gates on x86-64 NVIDIA
CUDA. Route B keeps only offline-fusible rotations in a standard Llama
checkpoint, then uses a format supported by stable vLLM.

The current stable vLLM support matrix marks GPU W4A8 unsupported and W4A16
GPTQ/AWQ/Marlin supported on Ada and Hopper. The selected first serving format
is therefore group-128 GPTQ W4A16; no custom vLLM plugin or new CUDA kernel is
planned. A tiny GQA Llama offline-rotation smoke already passes direct and
save/reload equivalence with maximum logits error
`3.5762786865234375e-07`, while retaining standard decoder `nn.Linear`
modules. This is rotation-only evidence, not quantization or vLLM execution.

## 2026-07-21 official QuaRot W4A4KV4 result

Route A now executes the complete exported Llama-2-13B checkpoint using the
pinned upstream backend on an RTX 6000 Ada. All 280 decoder projections are
upstream packed `Linear4bit` modules, activations use the upstream W4 path, and
the paged K/V cache is 4 bit. The full-checkpoint smoke passed packed-weight,
scale, finite-logit, cache-length, and exact repeated-generation checks. Peak
allocated memory in that short smoke was 7.19 GB.

The matched repeated benchmark compares this model against the upstream QuaRot
FP16 model/cache implementation with sequential loading and identical fixed
tokens. At 2048 prefill + 32 decode tokens (3 warm-ups, 10 repeats), median
prefill was 392.51 ms FP16 versus 441.70 ms W4A4KV4, decode was 52.28 versus
79.51 ms/token, and end to end was 2076.90 versus 2945.97 ms. Model-resident
allocated memory fell from 26.29 GB to 7.18 GB. Thus the faithful backend is a
real low-bit, large memory-saving deployment on Ada, but not a speedup. See
`OFFICIAL_QUAROT_W4A4_RESULTS.md` for scope, hashes, raw samples, and remaining
accuracy boundary.

## 2026-07-22 Isambard acceptance and single-block result

Isambard job `5742443` completed with exit code `0:0` on one GH200. Its owned
W4A8 kernel produced exact int32 accumulators and a maximum scaled FP32 error of
`3.814697265625e-06`. For the fixed `[1,16]` Llama-2-13B input, the selected
layer-0 `q_proj` CUDA replacement and the independent packed oracle matched at
the decoder-layer output and final logits with maximum and mean errors `0.0`.
This closes only the selected-linear Isambard portability gate; other linears
and K/V remained BF16.

The paper-aligned official QuaRot single-block run has now completed on RTX
6000 Ada. The runner retained the upstream Llama-2-7B decoder layer, W4A4
linears, KV4 cache, and official 3-warm-up / 10-step / 10-repeat structure. W4
completed all 14 cases; FP16 completed 13 and retained one capacity OOM at
batch 64/sequence 2048. W4 prefill speedup was 1.53--1.68x on the paired points,
and batch-16/context-4096 layer-e2e speedup was 1.28x. See
`OFFICIAL_QUAROT_SINGLE_BLOCK_RESULTS.md`.

Route B is now next on Isambard: its vLLM W4A16 single-block result is
diagnostic, while matched full-model offline inference and serving remain the
primary practical evidence.

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
