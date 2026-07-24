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

After that:

1. Add a one-request `/health`, `/v1/models`,
   and deterministic completion service smoke.
2. Measure matched BF16, unrotated W4A16, and rotated W4A16 full-model serving;
   run the single-block W4A16 diagnostic in the same environment.

No throughput, latency, quality, or full-model claim follows from the platform
or tiny smoke gates.
