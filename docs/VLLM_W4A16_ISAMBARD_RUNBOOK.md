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

No throughput, latency, quality, or full-model claim follows from the platform
or tiny smoke gates.
