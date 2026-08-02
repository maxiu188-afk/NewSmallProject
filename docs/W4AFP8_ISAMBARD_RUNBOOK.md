# W4AFP8 Isambard preparation and runbook

## Current state

The QuaRot-only job `5854439` and isolated SpinQuant-transfer job `5857916`
were cancelled before allocation on 2026-08-01 after the GPTQ protocol was
revised from static activation ordering to no activation ordering. Both have
zero runtime and provide no result. Jobs `5859043`/`5859044` then failed the
same-revision calibration check. Their replacements `5873446`/`5873447`
accepted calibration and passed the GH200 CUTLASS capability audit, but both
stopped before `oneshot()` quantization because the exporter required the
configured observer alias `minmax` while LLM Compressor 0.12.0 resolves the
same direct-min/max observer as `memoryless_minmax`. They produced no
checkpoint.

Revision `9d691e9cf828` accepts those two equivalent names while continuing to
reject `memoryless_mse`; the pinned quantizer preflight and all ten W4AFP8 unit
tests passed on Isambard. QuaRot validation job `5873544` completed `0:0` in
38m30s. SpinQuant-transfer main job `5873545` then completed `0:0` in 19m44s
through `afterok:5873544`. All three compressed checkpoints report 280 packed
decoder linears, zero runtime `g_idx` tensors, and 7,202,132,925 bytes. Both
isolated gates selected `CutlassW4A8LinearKernel`, loaded their checkpoints
through vLLM, and reproduced the same fixed eight tokens as BF16.

The four-model PPL and serving validators require one complete source-gate
result rather than a synthetic merge of the two isolated results. Joint job
`5874345` reused all three checkpoints and completed `0:0` in 4m35s from the
same revision. Its combined inference result SHA-256 is
`6a22855a720b95e7230dd15644ced73b28297a960e29d5d2823666de28254f2a`; its
capability result SHA-256 is
`048fce61ad390e0ac4dfb89db2f1801c5f47468e993a5ffbdb6b4b4e7d61633a`.
Those hashes are frozen in PPL revision `1f3e4cb`. PPL smoke job `5874806`
completed `0:0` in 6m24s, and formal job `5874807` completed `0:0` in 5m00s
through `afterok:5874806`. The formal job scored 331,614 targets per model:
PPL was 5.007820 BF16, 5.136105 unrotated W4AFP8, 5.248356 QuaRot-style W4AFP8,
and 5.230155 SpinQuant-transfer W4AFP8. The formal result SHA-256 is
`78a81dcdb17d8393247abd65d2f7e00b030e78817dc6d8b0699a8c15e48481f3`.
The scheduler dependencies and joint gate are operational; the evidence
boundaries remain distinct because:

- W4AFP8 uses a separate Isambard checkout from the fake-quant runs;
- W4AFP8 uses a separate
  `${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8/` artifact root;
- the QuaRot-only mode neither reads nor exports a SpinQuant rotation;
- the SpinQuant mode uses the checksummed corrected job `5854269` rotation;
- the SpinQuant job must pass its own packed-checkpoint and inference checks;
- fake-quant outputs remain immutable and separate from deployment artifacts.

The prepared path covers all later evidence gates:

| Gate | Prepared entry point | Runtime evidence |
|---|---|---|
| Backend capability | `scripts/audit_vllm_w4afp8_backend.py` | GH200 selects `CutlassW4A8LinearKernel` for every Llama-2-13B shape and rejects runtime `g_idx` |
| Export and load | `scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch` | Three separate packed checkpoints, 280 decoder linears each, fresh-process vLLM inference |
| Deployed accuracy | `scripts/run_isambard_vllm_w4afp8_llama2_13b_ppl.sbatch` | BF16 plus three W4AFP8 checkpoints on the retained 162 x 2048 WikiText-2 tokens |
| Service and acceleration | `scripts/run_isambard_vllm_w4afp8_llama2_13b_serving.sbatch` | Kernel log proof, endpoint smoke, concurrency-1/8 latency and throughput, memory and recovery |

The official LLM Compressor `W4AFP8` preset is group-128 symmetric INT4
weights plus symmetric dynamic per-token FP8 activations. The pinned vLLM
kernel requires Hopper SM90, FP8 E4M3 activations, no zero points, no runtime
activation-order `g_idx`, K/N divisible by 128, and BF16 output. These are
encoded as validation failures rather than documentation-only assumptions.
The revised deployment GPTQ keeps the required group size 128 but explicitly
sets `actorder=None` and uses min/max weight ranges without MSE clipping. This
is the weaker deployable protocol, not the paper's `group_size=-1` GPTQ.

Official references:

- [LLM Compressor scheme selection](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-scheme/)
- [vLLM quantization support](https://docs.vllm.ai/en/stable/features/quantization/)
- [LLM Compressor repository and W4AFP8 examples](https://github.com/vllm-project/llm-compressor)
- [vLLM repository](https://github.com/vllm-project/vllm)

## QuaRot execution sequence

1. Run a scheduler preflight and then the isolated export/load gate. The job
   materializes the same-revision calibration artifact before export:

   ```bash
   sbatch --test-only \
     scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot
   sbatch scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot
   ```

2. Isolated jobs `5873544`/`5873545` and joint job `5874345` passed terminal,
   marker, capability, export, inference, and checkpoint-provenance checks.
3. The joint result and capability hashes are frozen in the four-model PPL
   config. Smoke `5874806` and dependent formal job `5874807` passed. The
   serving config remains pending and must freeze the same joint gate before
   its own smoke/formal chain is submitted.

The main PPL and serving jobs depend on accepted smoke results. They do not use
smoke outcomes to change the formal protocol.

## Formal deployment evidence contract

Formal deployment is accepted only as the combination of two terminal,
provenance-checked results:

| Required formal result | Primary records | Acceptance boundary |
|---|---|---|
| Deployed accuracy | Total NLL, PPL, 331,614 scored targets, checkpoint and token hashes for BF16 plus all W4AFP8 variants | Must execute the packed checkpoints through vLLM; fake-quant PPL and fixed-token inference cannot substitute |
| Serving acceleration | Throughput, TTFT, TPOT, end-to-end latency, ready/peak GPU memory, completed/failed requests, kernel log proof | Must use the matched full-model concurrency-1/8 protocol; checkpoint size and layer timing cannot substitute |

Submit a PPL smoke before the PPL formal job and a serving smoke before the
serving formal job. The final report must join both accepted formal results by
checkpoint hashes and present quality and performance together. Do not label a
variant as real-deployment complete when only one side has passed.

## SpinQuant transfer execution sequence

The first W4AFP8 SpinQuant checkpoint is a transfer diagnostic from the
corrected W16A8-trained rotation, not an FP8-targeted learned endpoint. Current
main job `5873545` is scheduler-gated on validation job `5873544`. The
submission protocol is:

```bash
validation_job=$(sbatch --parsable --time=06:00:00 \
  scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot)
sbatch --time=06:00:00 \
  --dependency="afterok:${validation_job}" \
  scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch spinquant
```

Acceptance requires the general gate marker plus
`ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_SPINQUANT_TRANSFER_GATE_PASSED`, 280 packed
linears in both quantized checkpoints, no runtime `g_idx`, and selected
`CutlassW4A8LinearKernel` capability evidence. Do not submit deployed PPL or
serving from this gate until its result and checkpoint hashes are frozen.

## Result boundaries

- The current INT8 W4A8 fake-quant result is not W4AFP8 quality evidence.
- Accepted Gate 1 jobs `5873544` and `5873545` are export/load correctness only, not
  formal accuracy or acceleration evidence.
- Joint source-gate job `5874345` passed, but it is still correctness and
  provenance evidence rather than a PPL result.
- PPL smoke `5874806` and formal job `5874807` passed; the formal JSON is
  accepted deployed-quality evidence.
- Backend capability selection is not checkpoint correctness, PPL, or speed.
- Fixed-token inference is not full held-out accuracy.
- Deployed PPL is not acceleration.
- Acceleration requires the matched full-model service result plus the selected
  W4AFP8 kernel pattern in every quantized server log.
- The initial SpinQuant checkpoint remains labelled `INT8-trained rotation
  transfer to W4AFP8`. If it is not competitive with unrotated W4AFP8, the
  accepted SpinQuant endpoint requires FP8-targeted rotation learning.
