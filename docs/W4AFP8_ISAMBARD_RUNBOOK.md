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
| Downstream diagnostic | `scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch` | Zero-shot BoolQ validation accuracy for the same four checkpoints under the QuaRot-pinned LM Evaluation Harness task definition |
| Service and acceleration | `scripts/run_isambard_vllm_w4afp8_llama2_13b_serving.sbatch` | Kernel log proof, endpoint smoke, concurrency-1/8 latency and throughput, memory and recovery |

The accepted INT8-trained-transfer checkpoints use isolated serving revision
`ad971f9`, which freezes joint source gate `5874345` without changing any
gate-producing input. Result-gated service smoke `5882787` completed `0:0` in
8m09s and passed review of its JSON, four endpoint checks, kernel logs, memory
recovery, hashes, and revision. Its result SHA-256 is
`d3acdc424cd6796700a9ad937ceb26efca735a9a07adfed1b645a1403d09af7c`.
Benchmark formal `5883004` was consequently submitted from the same checkout
and completed `0:0` in 19m48s. All eight model/case groups completed 64/64
requests with zero failures, all six quantized cases selected
`CutlassW4A8LinearKernel`, and every server recovered to 1--4 MiB after
shutdown. The result SHA-256 is
`df63917675621f891280cf2cf5960e1b394a815f14fd8096125085ea02edae6f`.

The separately rooted FP8-targeted chain is also accepted. Revision `318bac2`
freezes source gate `5881273`. PPL smoke `5884993` completed `0:0` in 7m00s,
and formal `5884996` completed `0:0` in 5m18s through `afterok:5884993`.
Formal PPL was 5.007820 BF16, 5.136105 unrotated, 5.248356 QuaRot-style, and
5.219583 FP8-targeted SpinQuant over 331,614 targets; result SHA-256:
`9f17bc86ee664960400dd26f2182ceef72d76dce0a1b75f055331074a7ff7e0a`.
Service smoke `5884997` completed `0:0` in 8m31s, and benchmark `5884998`
completed `0:0` in 18m47s through `afterok:5884997`. All 512/512 requests
succeeded, every quantized server selected CUTLASS, all processes recovered to
1--3 MiB, and the FP8-targeted endpoint reached 1.42x/1.385x BF16 request
throughput with 50.7% lower ready GPU memory. Formal serving SHA-256:
`a10044d965d93ceca756e203a86ad5f72018a2b1028a65b552315a59ea1a80b7`.

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
   corresponding serving smoke `5882787` and benchmark `5883004` also passed
   from their frozen serving revision.

The main PPL and serving jobs depend on accepted smoke results. They do not use
smoke outcomes to change the formal protocol.

## BoolQ downstream diagnostic

The BoolQ diagnostic is motivated by the larger separation in SpinQuant Table
7 for LLaMA-2-13B W4A8KV16: 75.3% GPTQ versus 81.5% SpinQuant without online
Hadamard transforms. The current deployed checkpoints are W4AFP8, and the
SpinQuant checkpoint transfers a rotation trained against INT8 activations.
Consequently, this experiment can test whether the current deployed rotations
help a downstream task, but it cannot reproduce or refute the paper's W4A8
endpoint.

The public SpinQuant repository states that its reported table was produced
with an internal LLaMA codebase and its released evaluator covers WikiText-2
PPL only. The closest public frozen task protocol is the LM Evaluation Harness
commit pinned by QuaRot, `9b0b15b1ccace3534ffbd13298c569869ce8eaf3`:
zero-shot BoolQ validation, prompt
`{passage}\nQuestion: {question}?\nAnswer:`, choices `no`/`yes` with one leading
delimiter space, and raw accuracy. The local runner reproduces that
log-likelihood construction while executing the packed checkpoints through
vLLM.

Materialize the immutable 3,270-example validation artifact on the login node:

```bash
module load cray-python/3.11.7
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
"$HOME/.venvs/newsmallproject-llmcompressor-0.12.0/bin/python" \
  scripts/prepare_boolq_validation.py \
  --config configs/deployment/vllm_w4afp8_llama2_13b_boolq_isambard.json \
  --output-dir \
  "${PROJECTDIR}/${USER}/newsmallproject-vllm/boolq-lm-eval-v1-3de24cf"
```

Then submit smoke only:

```bash
mkdir -p results/vllm-w4afp8-boolq-llama2-13b
sbatch --test-only \
  scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch smoke
sbatch scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch smoke
```

Do not submit formal until the 32-example smoke JSON and logs have been
reviewed. After acceptance, submit the full validation split with the required
dependency:

```bash
sbatch --dependency=afterok:<SMOKE_JOB_ID> \
  scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch formal
```

Accepted execution: smoke `5876321` completed `0:0`; formal `5876591` then
completed `0:0` through `afterok:5876321`. Formal accuracy was 80.5810% BF16,
78.4098% unrotated W4AFP8, 78.5627% QuaRot-style W4AFP8, and 79.6024%
SpinQuant-transfer W4AFP8. Result SHA-256:
`b278567aae262fdd6f4379d4004817f17faba51fa304077f03dd1479b8bdb824`.
This accepts the downstream diagnostic only; serving acceleration is recorded
separately by accepted benchmark `5883004`.

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

Apply the repository smoke rule before queueing either formal job. If the next
step requires inspection of directional PPL, request failures, kernel logs,
hashes, or another smoke artifact, submit smoke only and wait for the user to
report completion. Queue `formal --dependency=afterok:<smoke-job>` immediately
only when the smoke is explicitly classified as runnability-only and no result
decision is required. In neither case should the smoke be continuously
monitored.

For a result-gated smoke that has already been accepted but is no longer a
valid Slurm dependency target, do not rerun training solely to obtain a fresh
job ID. A short read-only acceptance job may revalidate the existing result and
source hashes, with formal depending on that job and continuing to name the
original accepted smoke artifact.

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

The transfer checkpoint subsequently reached 79.6024% on the accepted BoolQ
formal run, versus 78.4098% unrotated, while its deployed PPL remained worse
than unrotated. The follow-up therefore trains a separately named FP8-targeted
rotation with:

```bash
sbatch --test-only scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch smoke
sbatch scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch smoke
```

Do not submit its formal mode until the smoke result is manually accepted.
Training does not alter the frozen transfer checkpoint and does not by itself
satisfy either the W4AFP8 quality or serving-acceleration gate.

Accepted FP8-targeted training smoke `5876912` completed `0:0` in 2m48s. Its
result and rotation SHA-256 values are respectively
`bf1c927f2a821c76aeb7b39d2ef1fbe291bbbf135c33e50af09322631e8a83f2` and
`a88e745beaf9601d04af0bd07a2c92798bf86062332319782e553fbca9092194`.
Read-only acceptance job `5876983` completed `0:0` in one second. Dependent
formal training job `5876984` completed `0:0` in 2h10m04s with 100 finite
losses, 100 non-zero gradient maxima, and a passed rotation artifact. Its
formal result SHA-256 is
`f34f450e03dd59ec9b26942731b470b080908399f9d53089da2fd38bb903f1d1`,
and its rotation SafeTensors SHA-256 is
`383004941a14e40f256abd4a615246a9adbfe1308ecd08896f167f5b6c2566ec`.

Revision `60aa629` adds isolated config and path overrides for the new
FP8-targeted export without changing the accepted transfer artifacts. Joint
source gate `5881273` completed `0:0` in 54m02s and passed result review. Each
W4AFP8 checkpoint has 280 packed decoder linears, group-128 min/max weights,
dynamic per-token FP8 inputs, and zero runtime `g_idx` tensors. All three
quantized fresh-process loads selected `CutlassW4A8LinearKernel`, and BF16 plus
all three W4AFP8 variants generated the same eight greedy tokens. The joint
result, capability result, and source-manifest SHA-256 values are respectively
`c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55`,
`495bcc458681f5473e4b1ad50db96b249c82c1ef88cd0f7085e505e5dbdc8d62`,
and `df05eae813350c33608acbda2e5c054661cd99d7cfc321d92ee22101b5b96f84`.
The optional DeepGEMM import and post-exit NCCL cleanup warnings were non-fatal;
CUTLASS was selected and the job reached its required pass marker. No separate
smoke was required for this result-gated source gate.

Downstream revision `318bac2` retains those hashes and separates quality from
timing. Runnability-only PPL smoke/formal jobs `5884993`/`5884996` and service
smoke/benchmark jobs `5884997`/`5884998` all completed `0:0`. Formal PPL for
FP8-targeted SpinQuant is 5.219583, versus 5.136105 unrotated and 5.248356
QuaRot-style. The benchmark completed 512/512 requests with no failures and
measured 2.486/15.701 requests/s at concurrency 1/8, versus 1.752/11.338 for
BF16, with ready memory 16,803 versus 34,099 MiB. This accepts core deployment
quality and acceleration with BF16 KV. It does not establish FP8 KV or an
FP8-targeted BoolQ result.

The FP8-targeted BoolQ follow-up uses
`configs/deployment/vllm_w4afp8_fp8_spinquant_llama2_13b_boolq_isambard.json`
and the accepted source gate `5881273`. The batch wrapper accepts isolated
config, SpinQuant-checkpoint, artifact-root, and result-directory overrides.
This smoke is **result-gated**, because its 32-example directional accuracy,
paired predictions, hashes, and logs must be reviewed before authorizing all
3,270 validation examples. Submit only the smoke, with a one-hour override:

```bash
artifact_root="${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8-fp8-targeted-5876984"
results_dir="${PWD}/results/vllm-w4afp8-fp8-targeted-5876984-boolq"
mkdir -p "${results_dir}"
sbatch --time=01:00:00 \
  --output="${results_dir}/%x-%j.out" \
  --error="${results_dir}/%x-%j.err" \
  --export="ALL,VLLM_W4AFP8_ARTIFACT_ROOT=${artifact_root},VLLM_W4AFP8_SPINQUANT_CHECKPOINT_DIR=${artifact_root}/spinquant-w4afp8-fp8-targeted-5876984,VLLM_W4AFP8_BOOLQ_CONFIG=${PWD}/configs/deployment/vllm_w4afp8_fp8_spinquant_llama2_13b_boolq_isambard.json,VLLM_W4AFP8_RESULTS_DIR=${results_dir}" \
  scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch smoke
```

Do not queue formal until the smoke is accepted. The wrapper's default limit
is six hours for a later formal run; smoke explicitly uses one hour.

Execution note: clean revision `183bd8f` passed source/data preflight, but
initial smoke `5886682` failed `1:0` in five seconds before model execution.
The materialized dataset manifest correctly retained the original BoolQ config
SHA, while the new model-source config had a different SHA. Revision `3639cc4`
adds an explicit `expected_manifest_config_sha256` equal to the recorded
manifest value; it still verifies dataset revision, fingerprint, examples SHA,
and row count and rejects any other manifest config hash. Corrected smoke
`5886913` is queued with a one-hour limit. Formal `5886914` is queued with a
six-hour limit and `afterok:5886913`.

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
- Accepted BoolQ smoke `5876321` and formal `5876591` are downstream-quality
  evidence, not PPL or acceleration.
- Acceleration requires the matched full-model service result plus the selected
  W4AFP8 kernel pattern in every quantized server log.
- The initial SpinQuant checkpoint remains labelled `INT8-trained rotation
  transfer to W4AFP8`. The separately named FP8-targeted rotation has passed
  export, PPL, and serving gates. BoolQ remains required only before making a
  downstream-task claim for that endpoint.
