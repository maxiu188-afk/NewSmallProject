#!/bin/bash
# Run the paper-aligned activation-only SpinQuant no-had one-step CUDA smoke.
# Weights remain BF16 during rotation learning; activations use unclipped,
# asymmetric per-token A8 floating QDQ, with head-dimension groups on o_proj.
# Post-learning GPTQ is a later gate.
# Submit from a clean, isolated SpinQuant checkout:
#   sbatch --test-only scripts/run_isambard_spinquant_llama2_13b_w16a8_1step_smoke.sbatch
#   sbatch scripts/run_isambard_spinquant_llama2_13b_w16a8_1step_smoke.sbatch
#SBATCH --job-name=spinquant-13b-w16a8-smoke
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=results/spinquant-isambard-llama2-13b/%x-%j.out
#SBATCH --error=results/spinquant-isambard-llama2-13b/%x-%j.err

set -euo pipefail

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
training_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
calibration_config="${project_root}/configs/spinquant/llama2_13b_rotation_calibration_8_smoke.json"
training_config="${project_root}/configs/spinquant/llama2_13b_w16a8_rotation_train_1step_smoke.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
artifact_root="${SPINQUANT_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-paper-nohad-w16a8}"
results_dir="${project_root}/results/spinquant-isambard-llama2-13b"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-spinquant-w16a8-${SLURM_JOB_ID}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=inputs"
for required in \
    "${training_env}/bin/python" \
    "${calibration_config}" \
    "${training_config}" \
    "${model_snapshot}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "SPINQUANT_13B_W16A8_SMOKE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SPINQUANT_13B_W16A8_SMOKE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
revision_label="${git_revision:0:12}"
calibration_dir="${artifact_root}/calibration-8x2048-${revision_label}"
rotation_dir="${artifact_root}/rotations-1step-${SLURM_JOB_ID}"
rotation_manifest="${rotation_dir}/rotation-manifest.json"
result="${results_dir}/spinquant-13b-w16a8-1step-${SLURM_JOB_ID}.json"
source_manifest="${results_dir}/spinquant-13b-w16a8-1step-${SLURM_JOB_ID}-source-manifest.txt"
mkdir -p "${artifact_root}" "${results_dir}"

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=runtime"
"${training_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import platform
import torch

assert platform.python_version() == "3.11.7", platform.python_version()
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("transformers") == "5.10.1"
assert metadata.version("datasets") == "5.0.0"
assert metadata.version("accelerate") == "1.13.0"
assert metadata.version("safetensors") == "0.8.0"
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["HF_DATASETS_OFFLINE"] == "1"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SPINQUANT_13B_W16A8_SMOKE_RUNTIME_ACCEPTED")
PY

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'training_environment=%s\n' "${training_env}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${calibration_config}" \
        "${training_config}" \
        "${project_root}/repro/spinquant/artifacts.py" \
        "${project_root}/repro/spinquant/calibration.py" \
        "${project_root}/repro/spinquant/activation_qdq.py" \
        "${project_root}/repro/spinquant/llama_adapter.py" \
        "${project_root}/repro/spinquant/rotations.py" \
        "${project_root}/repro/spinquant/stiefel.py" \
        "${project_root}/repro/spinquant/training.py" \
        "${project_root}/scripts/spinquant/prepare_wikitext2_calibration.py" \
        "${project_root}/scripts/spinquant/train_llama_rotations.py" \
        "${project_root}/scripts/run_isambard_spinquant_llama2_13b_w16a8_1step_smoke.sbatch" \
        "${model_snapshot}/config.json"
} > "${source_manifest}"

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=calibration"
if [[ ! -e "${calibration_dir}/calibration-manifest.json" && \
      ! -e "${calibration_dir}/token-ids.json" ]]; then
    "${training_env}/bin/python" \
        "${project_root}/scripts/spinquant/prepare_wikitext2_calibration.py" \
        --config "${calibration_config}" \
        --model-snapshot "${model_snapshot}" \
        --output-dir "${calibration_dir}"
elif [[ ! -f "${calibration_dir}/calibration-manifest.json" || \
        ! -f "${calibration_dir}/token-ids.json" ]]; then
    echo "SPINQUANT_13B_W16A8_SMOKE_PARTIAL_CALIBRATION=${calibration_dir}" >&2
    exit 1
fi

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=train"
if [[ -e "${rotation_dir}" ]]; then
    echo "SPINQUANT_13B_W16A8_SMOKE_REFUSES_EXISTING_ROTATION_DIR=${rotation_dir}" >&2
    exit 1
fi
"${training_env}/bin/python" \
    "${project_root}/scripts/spinquant/train_llama_rotations.py" \
    --config "${training_config}" \
    --model-snapshot "${model_snapshot}" \
    --calibration-manifest "${calibration_dir}/calibration-manifest.json" \
    --output-dir "${rotation_dir}"

echo "SPINQUANT_13B_W16A8_SMOKE_STAGE=assert_result"
"${training_env}/bin/python" - \
    "${rotation_manifest}" "${result}" "${git_revision}" "${SLURM_JOB_ID}" <<'PY'
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from repro.spinquant.artifacts import load_rotation_artifact

manifest_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
revision = sys.argv[3]
job_id = sys.argv[4]
loaded = load_rotation_artifact(
    manifest_path,
    maximum_orthogonality_error=1e-4,
)
manifest = loaded["manifest"]
provenance = manifest["provenance"]
training = provenance["training"]
assert provenance["project_revision"] == revision, provenance
assert provenance["model"]["revision"] == (
    "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
), provenance
assert training["status"] == "passed", training
assert training["rotation_objective"] == "activation_qdq", training
assert training["adapter"]["weight_bits"] == 16, training
assert training["adapter"]["activation_bits"] == 8, training
assert training["adapter"]["weight_runtime_form"] == "unquantized", training
assert training["adapter"]["activation_runtime_form"] == "floating_qdq_with_ste", training
assert training["adapter"]["activation_granularity"] == (
    "per_token_last_axis_o_proj_grouped"
), training
assert training["adapter"]["activation_o_proj_group_size"] == 128, training
assert training["adapter"]["activation_ungrouped_include_zero"] is True, training
assert training["optimizer_steps"] == 1, training
assert training["gradient_accumulation_steps"] == 8, training
assert training["calibration_sequences"] == 8, training
assert training["sequence_length"] == 2048, training
assert training["gradient_maxima"][0] > 0, training
assert max(training["orthogonality_error"].values()) <= 1e-4, training
assert manifest["rotation"]["r1_shape"] == [5120, 5120], manifest
assert manifest["rotation"]["r2_shape"] == [40, 128, 128], manifest
tensor_path = manifest_path.parent / manifest["file"]["name"]
digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
assert digest == manifest["file"]["sha256"], manifest
result = {
    "status": "passed",
    "scope": (
        "paper-aligned Llama-2-13B no-had W16A8 activation-only one-step "
        "rotation smoke; no GPTQ, PPL, packing, kernel, or deployment claim"
    ),
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "slurm_job_id": job_id,
    "project_revision": revision,
    "rotation_manifest": str(manifest_path),
    "rotation_safetensors_sha256": digest,
    "training": training,
    "runtime": provenance["runtime"],
}
result_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("ISAMBARD_SPINQUANT_LLAMA2_13B_W16A8_1STEP_SMOKE_PASSED")
PY
