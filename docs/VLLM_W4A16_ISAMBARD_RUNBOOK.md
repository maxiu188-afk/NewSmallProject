# vLLM W4A16 on Isambard-AI

## Scope and evidence boundary

This route evaluates standard-layout offline QuaRot-style rotations followed by
group-128 GPTQ W4A16 and stable vLLM. The primary endpoint is a matched
full-model serving comparison; a single transformer block is retained only as
a diagnostic. It is not original QuaRot W4A4, the owned W4A8 checkpoint format,
or a custom vLLM plugin.

Environment installation, package downloads, cache inspection, and tiny
checkpoint generation run directly on an Isambard login node. Do not submit
those operations to Slurm. Submit a GPU job only when the prepared artifacts
need CUDA validation.

## Frozen platform stack

- Isambard project: `u6rt`, account `brics.u6rt`, partition `workq`
- Host architecture: aarch64; formal accelerator: NVIDIA GH200 (SM 9.0)
- Python: `cray-python/3.11.7`
- Serving: PyTorch 2.11.0+cu129 and vLLM 0.25.1 official aarch64 wheel
- Quantization: PyTorch 2.11.0+cu129 and LLM Compressor 0.12.0
- Checkpoint format: compressed-tensors GPTQ W4A16, group size 128

Serving and quantization use separate virtual environments. vLLM 0.25.1 pins
`compressed-tensors==0.17.0`, whereas LLM Compressor 0.12.0 pins 0.17.1.
Combining them would make the environment internally inconsistent.

## Login-node preparation (no Slurm)

Keep the old `$HOME/NewSmallProject` checkout untouched because it contains the
accepted W4A8 work. Use the clean dedicated checkout:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
bash scripts/setup_isambard_vllm_env.sh
```

The setup verifies the official vLLM wheel SHA-256, installs two isolated
environments, runs `pip check`, tests CPU-side imports, and records frozen
package manifests under `$HOME/.cache/newsmallproject-vllm/manifests/`.
Success requires `ISAMBARD_VLLM_ENV_READY`.

PyTorch's official aarch64 CUDA 12.9 dependency currently installs
`nvidia-cusparselt-cu12==0.7.1` with an internal `manylinux2014_sbsa` wheel tag.
Stock `pip check` does not recognize that tag and emits one platform warning.
The preflight accepts only that exact package/version/tag combination, records
the warning, and still requires successful Torch imports plus the GH200 smoke.

Then run the serving-package preflight and prepare the deterministic tiny
checkpoints on the login node:

```bash
export VLLM_ENV="$HOME/.venvs/newsmallproject-vllm-0.25.1"
export QUANTIZER_ENV="$HOME/.venvs/newsmallproject-llmcompressor-0.12.0"

"$VLLM_ENV/bin/python" scripts/check_isambard_vllm_env.py \
  --output "$HOME/.cache/newsmallproject-vllm/preflight/login-preflight.json"
CUDA_VISIBLE_DEVICES="" "$QUANTIZER_ENV/bin/python" \
  scripts/prepare_vllm_w4a16_tiny_checkpoints.py \
  --output-dir "$HOME/.cache/newsmallproject-vllm/tiny-checkpoints"
```

The tiny gate produces BF16, unrotated W4A16, and offline-rotated W4A16
checkpoints. Its manifest proves deterministic preparation and compressed-
tensors metadata only; it does not prove that vLLM kernels execute on GH200.
The GPU job hashes this manifest and the controlling source files, and rejects
a dirty checkout or a manifest generated from a different Git revision.

## Short GPU smoke

Only after both login-node checks pass, validate the request and submit one
short job from the clean checkout:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
mkdir -p results/vllm-w4a16-isambard-smoke
sbatch --test-only scripts/run_isambard_vllm_w4a16_smoke.sbatch
sbatch scripts/run_isambard_vllm_w4a16_smoke.sbatch
```

The job loads all three tiny checkpoints through the same vLLM runtime and
runs deterministic offline inference. Success requires both an exit code of
`0:0` and `ISAMBARD_VLLM_W4A16_SMOKE_PASSED` in the output. The JSON report is
written to `results/vllm-w4a16-isambard-smoke/`.

The job sets `VLLM_USE_FLASHINFER_SAMPLER=0`. Isambard does not expose an
NVCC-equipped CUDA 12.9 toolkit in the frozen runtime, so FlashInfer's sampling
JIT cannot build there. This uses vLLM's native sampler fallback only; it does
not replace the FlashAttention or W4A16 linear execution paths under test.

Do not continuously poll a queued or running job. Check one scheduler snapshot,
continue local implementation or documentation work, and inspect `sacct` plus
the retained logs later.

## Next gates

The tiny GH200 gate passed as job `5751780`. The next gate goes directly to the
pinned `meta-llama/Llama-2-13b-hf` revision already cached on Isambard; no
SmolLM intermediate is used. BF16 uses the original snapshot, while unrotated
and offline-rotated W4A16 use the same materialized 128 x 2048-token WikiText
calibration set. The two compressed checkpoints live under the project
filesystem rather than the nearly-full home quota.

Prepare the shared calibration tokens directly on the login node:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ARTIFACT_ROOT="${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16"
mkdir -p "$ARTIFACT_ROOT"
"$HOME/.venvs/newsmallproject-llmcompressor-0.12.0/bin/python" \
  scripts/prepare_vllm_w4a16_llama2_calibration.py \
  --config configs/deployment/vllm_w4a16_llama2_13b_isambard.json \
  --output-dir "$ARTIFACT_ROOT/calibration-128x2048"
```

Only after the calibration manifest passes, submit the GPU export/inference
gate:

```bash
mkdir -p results/vllm-w4a16-llama2-13b
sbatch --test-only scripts/run_isambard_vllm_w4a16_llama2_13b_gate.sbatch
sbatch scripts/run_isambard_vllm_w4a16_llama2_13b_gate.sbatch
```

The job is stage-resumable for completed exports and refuses to overwrite a
partial checkpoint. Success requires 280 packed decoder linears in each W4A16
checkpoint, bounded pre-quantization rotation error, successful vLLM inference
for all three complete models, and
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_GATE_PASSED`.

### Job 5758738 inference-only recovery

Job `5758738` completed both expensive full-model exports before failing in
`offline_inference`. Each checkpoint contains 280 packed decoder linears and
occupies 7.20 GB; the rotated pre-quantization BF16 logit error was 0.1875,
within the frozen 0.25 tolerance. The BF16 vLLM engine also loaded and
generated successfully.

The failure was process lifetime rather than a checkpoint or W4A16 failure.
The original inference driver created all three vLLM engines sequentially in
one Python process. After the BF16 engine finished, the process still held a
large V1-engine GPU reservation, so the unrotated W4A16 engine saw only
24.58 GiB free, below its requested 71.25 GiB reservation.

The repaired driver gives each model its own child process. Process exit
releases CUDA and vLLM state before the next model starts. Reuse of the
completed exports is allowed only when their recorded revision is an ancestor
of the retry revision and all export-producing source files are byte-unchanged
in Git. The retry job also rechecks report hashes, rotation tolerance, and all
280 packed weights before inference:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
sbatch --test-only \
  scripts/run_isambard_vllm_w4a16_llama2_13b_inference_retry.sbatch
sbatch scripts/run_isambard_vllm_w4a16_llama2_13b_inference_retry.sbatch
```

This job does not repeat calibration or quantization. Success requires exit
code `0:0`, three per-model JSON files under
`offline-inference-retry-models/`, and both
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_INFERENCE_RETRY_PASSED` and the overall gate
marker in its output.

Job `5769503` completed this recovery in 2 minutes 58 seconds with exit code
`0:0`. BF16, unrotated W4A16, and rotated W4A16 all loaded through vLLM
0.25.1+cu129 on one GH200 and returned the same eight greedy tokens. Across
the 122 shared first-token logprob entries, the maximum absolute errors against
BF16 were 0.65765 for unrotated W4A16 and 0.37806 for rotated W4A16. This is
accepted full-model offline inference evidence, not quality or performance
evidence.

## Deployed-checkpoint PPL smoke and formal result

The PPL gate evaluates the original BF16 checkpoint and both packed W4A16
checkpoints through vLLM on the retained WikiText-2 test stream. It materializes
162 non-overlapping 2,048-token sequences and scores all 331,614 next-token
targets. Each model runs in a fresh child process, and the formal job requires
an `afterok` dependency on a passing same-revision smoke:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
smoke_job="$(
  sbatch --parsable \
    scripts/run_isambard_vllm_w4a16_llama2_13b_ppl.sbatch smoke
)"
sbatch --dependency="afterok:${smoke_job}" \
  scripts/run_isambard_vllm_w4a16_llama2_13b_ppl.sbatch formal
```

Smoke job `5839415` completed in 7 minutes 41 seconds with exit code `0:0` and
scored 4,094 targets. Formal job `5839419` completed in 5 minutes 51 seconds
with exit code `0:0`, recorded `afterok:5839415`, and emitted
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_PPL_FORMAL_PASSED`. Both used clean revision
`7074b3ea90a7072f1ac59f49dc2a0ec25a592f3a`.

| Model | Scored targets | PPL |
|---|---:|---:|
| BF16 | 331,614 | 5.007820 |
| Unrotated packed W4A16 | 331,614 | 5.289677 |
| Rotated packed W4A16 | 331,614 | **5.132755** |

The rotated checkpoint reduced PPL by 2.97% relative to unrotated W4A16 and
recovered 55.67% of the PPL gap to BF16. Both W4 checkpoints contained 280
packed decoder linears and selected `MacheteLinearKernel`. Optional DeepGEMM
import and NCCL process-group cleanup warnings were non-fatal; all per-model,
aggregate-result, provenance, marker, and Slurm exit-code assertions passed.

The accepted formal-result SHA-256 is
`f7698afcca494279cb7d3f2d50d94fd1378e03c6d6edd4506d750869cb09829a`.
The retained token-ID SHA-256 is
`0f49a76a5cc6f3841356f09fee93eb5a8de9cc37d6b65af54e40614d6eac0de9`.

## SpinQuant-derived W4A16 BoolQ formal job

The separately accepted SpinQuant-derived packed W4A16 endpoint is bound to
export/load gate `5945162`, clean revision `f490f58`, export-result SHA-256
`b21ca19f34bf24470fdec797f90821b46edb77bfc1717c23724b989aa4ff05cf`,
inference-result SHA-256
`fa533a64f7b8345b547f45b6fa666ca8e6181d3369dc487dc74c5cf171f995c4`,
and checkpoint-tree SHA-256
`41a79153d2ad7e0fb598819adc538ce65ba7c1a05629459f77cb816d226ce2da`.
The BoolQ input validator rechecks those records, their source manifest, the
Machete kernel log marker, Git ancestry, unchanged gate-producing files, the
checkpoint tree, 280 packed decoder linears, and exact W4A16 metadata.

This job reuses the immutable 3,270-example artifact and the exact zero-shot
prompt, choice order, token boundary, and raw-accuracy scoring used by the
accepted W4AFP8 BoolQ runs. It evaluates only matched BF16 and SpinQuant W4A16
models, each in a fresh vLLM process. It is formal-only: the complete export
and fresh-process load gate has already passed, and a smaller BoolQ subset
would not authorize a different formal configuration or result decision.
`sbatch --test-only` is still required for scheduler validation, but it is not
an experimental smoke.

From the clean dedicated checkout:

```bash
cd "$HOME/NewSmallProject-spinquant-w4a16-export"
mkdir -p results/vllm-spinquant-w4a16-boolq-llama2-13b
sbatch --test-only \
  scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_boolq.sbatch formal
sbatch \
  scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_boolq.sbatch formal
```

Success requires exit code `0:0`, 3,270 examples and 6,540 choice requests for
both models, and
`ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_BOOLQ_PASSED`. The job-specific JSON
is written below `results/vllm-spinquant-w4a16-boolq-llama2-13b/`. No BoolQ
metric is accepted until that JSON, the per-model records, hashes, logs, and
Slurm state are reviewed. This is downstream deployed-quality evidence only;
it is not the paper's W4A8KV16 endpoint and provides no PPL or acceleration
claim.

## Service smoke and dependent formal benchmark

The service stage uses the separate serving configuration at
`configs/deployment/vllm_w4a16_llama2_13b_serving_isambard.json`; the accepted
export configuration remains byte-unchanged. The smoke starts a fresh
OpenAI-compatible server for each complete model, checks `/health` and
`/v1/models`, and sends one deterministic eight-token `/v1/completions`
request. It also requires GPU memory to return to the pre-server baseline
before starting the next model.

The formal benchmark is submitted with an `afterok` dependency on the smoke.
It additionally parses the smoke JSON and requires the same Git revision, so a
missing, failed, or stale smoke cannot authorize measurement:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
mkdir -p results/vllm-w4a16-serving-llama2-13b
sbatch --test-only \
  scripts/run_isambard_vllm_w4a16_llama2_13b_service_smoke.sbatch
sbatch --test-only \
  scripts/run_isambard_vllm_w4a16_llama2_13b_serving_benchmark.sbatch

smoke_job="$(
  sbatch --parsable \
    scripts/run_isambard_vllm_w4a16_llama2_13b_service_smoke.sbatch
)"
sbatch --dependency="afterok:${smoke_job}" \
  scripts/run_isambard_vllm_w4a16_llama2_13b_serving_benchmark.sbatch
```

Both jobs reuse the accepted inference result from job `5769503` only after
checking its SHA-256, ancestry, runtime, model paths, and unchanged
inference-producing source files.

The matched benchmark uses random 256-token inputs and forced 64-token outputs:
64 measured requests plus four warmups at concurrency 1 and concurrency 8.
Every model/case pair gets a fresh server. Server flags, random seed, tokenizer,
requests, and an explicit 8 GiB KV cache are identical. The result retains
client-side TTFT, TPOT, ITL, E2E, request throughput, output/total-token
throughput, p50/p90/p99 values, detailed raw requests, and sampled GPU memory.
The fixed KV allocation prevents vLLM's automatic cache sizing from hiding
model-memory differences.

Smoke success requires
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_SERVICE_SMOKE_PASSED`. Formal success requires
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_SERVING_BENCHMARK_PASSED`. Benchmark duration
and Slurm elapsed time include server startup and teardown and are not reported
as request latency.

The first service submission, job `5777529`, completed the BF16 health, model
listing, and deterministic completion requests before the orchestration code
misclassified TCP `TIME_WAIT` state as an active listener when preparing the
second server. The server and engine had shut down normally; no W4A16 model had
started, and this was not a vLLM, checkpoint, API, or GPU-memory failure. The
port guard now checks whether a listener accepts a connection instead of
attempting to bind the recently used port. Its dependent benchmark job
`5777531` was cancelled after Slurm marked the dependency as never satisfiable.

The corrected service smoke, job `5780629`, completed with exit code `0:0`.
Its dependent formal benchmark, job `5780631`, also completed with exit code
`0:0`. All six model/case groups completed 64 of 64 requests with no failures,
using exactly 256 input and 64 output tokens. The accepted result SHA-256 is
`2a683b38be4831f24897bb3d8660d3f0277c3a0c914aaebb5b8511f2f357a39a`.

| Model | Concurrency | Requests/s | p50 TTFT (ms) | p50 TPOT (ms) | p50 E2E (ms) | Ready GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 1.760 | 21.114 | 8.684 | 568.249 | 34,099 |
| BF16 | 8 | 11.374 | 104.698 | 9.507 | 703.522 | 34,099 |
| Unrotated W4A16 | 1 | 2.718 | 22.242 | 5.488 | 368.019 | 16,125 |
| Unrotated W4A16 | 8 | 15.543 | 108.642 | 6.438 | 513.606 | 16,125 |
| Rotated W4A16 | 1 | 2.699 | 22.860 | 5.510 | 370.313 | 16,125 |
| Rotated W4A16 | 8 | 15.612 | 107.608 | 6.456 | 514.416 | 16,125 |

These are the primary deployment results. Both W4A16 checkpoints roughly halve
resident model-plus-cache memory and improve batch-one and concurrency-eight
request throughput in this matched configuration. Rotation does not produce a
material serving-performance separation from unrotated W4A16.

## Dependent single-block diagnostic

The follow-up measures the real vLLM `LlamaDecoderLayer` at
`model.model.layers[0]`, not a reimplemented PyTorch block. The full model still
executes so vLLM supplies the real attention metadata and KV cache. Forward
hooks installed through `LLM.apply_model()` record CUDA events around layer 0;
each checkpoint runs in a fresh child process so vLLM and GPU resources are
released between variants.

This diagnostic deliberately sets `enforce_eager=true`, because Python hooks
must not be bypassed by a compiled graph. Its numbers therefore explain the
layer path but are not directly interchangeable with the compiled full-model
serving benchmark above.

The smoke runs the packed unrotated W4A16 layer with one prefill and one short
decode case. The formal job requires both an `afterok` Slurm dependency and a
passing smoke JSON from the same Git revision. It compares BF16, unrotated
W4A16, and rotated W4A16 across batch sizes 1 and 8, prompt lengths 256 and
2048, and separate prefill and 16-step decode measurements:

```bash
cd "$HOME/NewSmallProject-vllm-ready"
mkdir -p results/vllm-w4a16-single-block-llama2-13b
sbatch --test-only \
  scripts/run_isambard_vllm_w4a16_llama2_13b_single_block_smoke.sbatch
sbatch --test-only \
  scripts/run_isambard_vllm_w4a16_llama2_13b_single_block_benchmark.sbatch

smoke_job="$(
  sbatch --parsable \
    scripts/run_isambard_vllm_w4a16_llama2_13b_single_block_smoke.sbatch
)"
sbatch --dependency="afterok:${smoke_job}" \
  scripts/run_isambard_vllm_w4a16_llama2_13b_single_block_benchmark.sbatch
```

Smoke success requires
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_SINGLE_BLOCK_SMOKE_PASSED`. Formal success
requires
`ISAMBARD_VLLM_W4A16_LLAMA2_13B_SINGLE_BLOCK_BENCHMARK_PASSED`.

The hook smoke, job `5784966`, completed in 1 minute 23 seconds with exit code
`0:0`. Its dependent formal job, `5784967`, completed in 4 minutes 44 seconds
with exit code `0:0`; the retained manifest records
`afterok:5784966`, clean revision
`3733d4c1daf38f8de3ebe7e1075800a9d8bd86fb`, and the accepted serving-result
hash.

The formal result passed all eight cases for all three models. The real vLLM
layer contained 605.0 MiB of parameters in BF16 and 156.0 MiB in each W4A16
variant, a 74.2% reduction. Median W4A16/BF16 layer speed ranged from 0.96x for
the largest B8 x 2048 prefill to 1.05x for B1 x 2048 prefill; decode cases
ranged from 1.01x to 1.04x. Rotation did not show a stable timing advantage.
The accepted result SHA-256 is
`0a5e9658c3a2205b6e2465472f81bd004370ba7b4db1c46f07123fe3a538c5cb`.

vLLM logged that W4A16 selected `MacheteLinearKernel`. The optional DeepGEMM
probe could not find a CUDA toolkit, but it was not the selected W4 execution
path. Child-process shutdown also emitted a non-fatal NCCL cleanup warning;
all result assertions and Slurm exit codes passed.

The consolidated measurements and evidence boundaries are in
[`VLLM_W4A16_RESULTS.md`](VLLM_W4A16_RESULTS.md).

No throughput, latency, quality, or full-model claim follows from the platform
or tiny smoke gates.
