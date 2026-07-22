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
sbatch --test-only scripts/run_isambard_vllm_w4a16_smoke.sbatch
sbatch scripts/run_isambard_vllm_w4a16_smoke.sbatch
```

The job loads all three tiny checkpoints through the same vLLM runtime and
runs deterministic offline inference. Success requires both an exit code of
`0:0` and `ISAMBARD_VLLM_W4A16_SMOKE_PASSED` in the output. The JSON report is
written to `results/vllm-w4a16-isambard-smoke/`.

Do not continuously poll a queued or running job. Check one scheduler snapshot,
continue local implementation or documentation work, and inspect `sacct` plus
the retained logs later.

## Next gates

1. Repeat offline rotation and matched calibration on the smallest approved
   pretrained Llama checkpoint.
2. Add fixed-token offline comparison and a one-request `/health`, `/v1/models`,
   and deterministic completion service smoke.
3. Scale only a passed workflow to Llama-2-13B.
4. Measure matched BF16, unrotated W4A16, and rotated W4A16 full-model serving;
   run the single-block W4A16 diagnostic in the same environment.

No throughput, latency, quality, or full-model claim follows from the platform
or tiny smoke gates.
