#!/bin/bash
# Train Llama-2-13B SpinQuant rotations against the exact vLLM dynamic
# per-token FP8 E4M3 activation quantizer. Submit formal only after reviewing
# the smoke result:
#   sbatch scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch smoke
#   sbatch --dependency=afterok:<SMOKE_JOB> \
#     scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch formal <SMOKE_JOB>
#SBATCH --job-name=spinquant-13b-fp8
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=results/spinquant-isambard-llama2-13b-fp8/%x-%j.out
#SBATCH --error=results/spinquant-isambard-llama2-13b-fp8/%x-%j.err

set -euo pipefail

mode="${1:-}"
smoke_job="${2:-}"
if [[ "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 {smoke|formal} [smoke-job]" >&2
    exit 2
fi
if [[ "${mode}" == "formal" ]]; then
    if [[ ! "${smoke_job}" =~ ^[0-9]+$ ]]; then
        echo "formal mode requires the accepted smoke job id" >&2
        exit 2
    fi
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "SPINQUANT_13B_FP8_FORMAL_REQUIRES_AFTEROK_DEPENDENCY" >&2
        exit 2
    fi
fi

echo "SPINQUANT_13B_FP8_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
training_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
artifact_root="${SPINQUANT_FP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-fp8-targeted}"
results_dir="${project_root}/results/spinquant-isambard-llama2-13b-fp8"

if [[ "${mode}" == "smoke" ]]; then
    calibration_config="${project_root}/configs/spinquant/llama2_13b_rotation_calibration_8_smoke.json"
    training_config="${project_root}/configs/spinquant/llama2_13b_w16afp8_rotation_train_1step_smoke.json"
    calibration_label="8x2048"
    result_prefix="spinquant-13b-fp8-1step"
    expected_steps=1
    expected_sequences=8
else
    calibration_config="${project_root}/configs/spinquant/llama2_13b_rotation_calibration_800.json"
    training_config="${project_root}/configs/spinquant/llama2_13b_w16afp8_rotation_train_100.json"
    calibration_label="800x2048"
    result_prefix="spinquant-13b-fp8-100step"
    expected_steps=100
    expected_sequences=800
fi

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-spinquant-fp8-${mode}-${SLURM_JOB_ID}"
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

echo "SPINQUANT_13B_FP8_STAGE=inputs"
for required in \
    "${training_env}/bin/python" \
    "${calibration_config}" \
    "${training_config}" \
    "${model_snapshot}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "SPINQUANT_13B_FP8_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SPINQUANT_13B_FP8_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
revision_label="${git_revision:0:12}"
calibration_dir="${artifact_root}/calibration-${calibration_label}-${revision_label}"
rotation_dir="${artifact_root}/rotations-${mode}-${SLURM_JOB_ID}"
rotation_manifest="${rotation_dir}/rotation-manifest.json"
result="${results_dir}/${result_prefix}-${SLURM_JOB_ID}.json"
source_manifest="${results_dir}/${result_prefix}-${SLURM_JOB_ID}-source-manifest.txt"
mkdir -p "${artifact_root}" "${results_dir}"

if [[ "${mode}" == "formal" ]]; then
    smoke_result="${results_dir}/spinquant-13b-fp8-1step-${smoke_job}.json"
    smoke_source_manifest="${results_dir}/spinquant-13b-fp8-1step-${smoke_job}-source-manifest.txt"
    for required in "${smoke_result}" "${smoke_source_manifest}"; do
        if [[ ! -f "${required}" ]]; then
            echo "SPINQUANT_13B_FP8_FORMAL_MISSING_SMOKE_INPUT=${required}" >&2
            exit 1
        fi
    done
    echo "SPINQUANT_13B_FP8_STAGE=smoke_gate"
    grep -E '^[0-9a-f]{64}  ' "${smoke_source_manifest}" | sha256sum --check --status
    "${training_env}/bin/python" - \
        "${smoke_result}" "${git_revision}" "${smoke_job}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
revision = sys.argv[2]
job_id = sys.argv[3]
assert smoke["status"] == "passed", smoke
assert smoke["slurm_job_id"] == job_id, smoke
assert subprocess.run(
    ["git", "merge-base", "--is-ancestor", smoke["project_revision"], revision],
    check=False,
).returncode == 0, smoke
training = smoke["training"]
assert training["rotation_objective"] == "fp8_activation_qdq", training
assert training["optimizer_steps"] == 1, training
assert training["calibration_sequences"] == 8, training
assert training["adapter"]["activation_type"] == "float", training
assert training["adapter"]["activation_dtype"] == "float8_e4m3fn", training
assert training["adapter"]["activation_runtime_form"] == (
    "vllm_dynamic_per_token_fp8_e4m3fn_qdq_with_ste"
), training
assert training["adapter"]["activation_granularity"] == "per_token_last_axis", training
assert training["adapter"]["activation_o_proj_group_size"] == -1, training
assert max(training["orthogonality_error"].values()) <= 1e-4, training
print("SPINQUANT_13B_FP8_FORMAL_SMOKE_GATE_ACCEPTED")
PY
fi

echo "SPINQUANT_13B_FP8_STAGE=runtime"
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
assert hasattr(torch, "float8_e4m3fn")
assert torch.finfo(torch.float8_e4m3fn).max == 448.0
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["HF_DATASETS_OFFLINE"] == "1"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SPINQUANT_13B_FP8_RUNTIME_ACCEPTED")
PY

echo "SPINQUANT_13B_FP8_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'mode=%s\n' "${mode}"
    printf 'training_environment=%s\n' "${training_env}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    if [[ "${mode}" == "formal" ]]; then
        printf 'accepted_smoke_job=%s\n' "${smoke_job}"
    fi
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
        "${project_root}/scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch" \
        "${model_snapshot}/config.json"
    if [[ "${mode}" == "formal" ]]; then
        sha256sum "${smoke_result}" "${smoke_source_manifest}"
    fi
} > "${source_manifest}"

echo "SPINQUANT_13B_FP8_STAGE=calibration"
if [[ ! -e "${calibration_dir}/calibration-manifest.json" && \
      ! -e "${calibration_dir}/token-ids.json" ]]; then
    "${training_env}/bin/python" \
        "${project_root}/scripts/spinquant/prepare_wikitext2_calibration.py" \
        --config "${calibration_config}" \
        --model-snapshot "${model_snapshot}" \
        --output-dir "${calibration_dir}"
elif [[ ! -f "${calibration_dir}/calibration-manifest.json" || \
        ! -f "${calibration_dir}/token-ids.json" ]]; then
    echo "SPINQUANT_13B_FP8_PARTIAL_CALIBRATION=${calibration_dir}" >&2
    exit 1
fi

echo "SPINQUANT_13B_FP8_STAGE=train"
if [[ -e "${rotation_dir}" ]]; then
    echo "SPINQUANT_13B_FP8_REFUSES_EXISTING_ROTATION_DIR=${rotation_dir}" >&2
    exit 1
fi
training_started_epoch="$(date +%s)"
"${training_env}/bin/python" \
    "${project_root}/scripts/spinquant/train_llama_rotations.py" \
    --config "${training_config}" \
    --model-snapshot "${model_snapshot}" \
    --calibration-manifest "${calibration_dir}/calibration-manifest.json" \
    --output-dir "${rotation_dir}"
training_finished_epoch="$(date +%s)"

echo "SPINQUANT_13B_FP8_STAGE=assert_result"
"${training_env}/bin/python" - \
    "${rotation_manifest}" "${result}" "${git_revision}" "${SLURM_JOB_ID}" \
    "${mode}" "${expected_steps}" "${expected_sequences}" \
    "${training_started_epoch}" "${training_finished_epoch}" "${smoke_job}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

from repro.spinquant.artifacts import load_rotation_artifact

manifest_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
revision = sys.argv[3]
job_id = sys.argv[4]
mode = sys.argv[5]
expected_steps = int(sys.argv[6])
expected_sequences = int(sys.argv[7])
started = int(sys.argv[8])
finished = int(sys.argv[9])
smoke_job = sys.argv[10]
loaded = load_rotation_artifact(manifest_path, maximum_orthogonality_error=1e-4)
manifest = loaded["manifest"]
provenance = manifest["provenance"]
training = provenance["training"]
assert provenance["project_revision"] == revision, provenance
assert provenance["model"]["revision"] == (
    "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
), provenance
assert training["status"] == "passed", training
assert training["rotation_objective"] == "fp8_activation_qdq", training
assert training["adapter"]["weight_bits"] == 16, training
assert training["adapter"]["activation_bits"] == 8, training
assert training["adapter"]["activation_type"] == "float", training
assert training["adapter"]["activation_dtype"] == "float8_e4m3fn", training
assert training["adapter"]["weight_runtime_form"] == "unquantized", training
assert training["adapter"]["activation_runtime_form"] == (
    "vllm_dynamic_per_token_fp8_e4m3fn_qdq_with_ste"
), training
assert training["adapter"]["activation_granularity"] == "per_token_last_axis", training
assert training["adapter"]["activation_o_proj_group_size"] == -1, training
assert training["optimizer_steps"] == expected_steps, training
assert training["gradient_accumulation_steps"] == 8, training
assert training["calibration_sequences"] == expected_sequences, training
assert training["sequence_length"] == 2048, training
assert len(training["gradient_maxima"]) == expected_steps, training
assert len(training["update_losses"]) == expected_steps, training
assert all(math.isfinite(value) and value > 0 for value in training["gradient_maxima"])
assert all(math.isfinite(value) for value in training["update_losses"])
assert max(training["orthogonality_error"].values()) <= 1e-4, training
assert manifest["rotation"]["r1_shape"] == [5120, 5120], manifest
assert manifest["rotation"]["r2_shape"] == [40, 128, 128], manifest
tensor_path = manifest_path.parent / manifest["file"]["name"]
digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
assert digest == manifest["file"]["sha256"], manifest
assert finished >= started
result = {
    "status": "passed",
    "scope": (
        f"Llama-2-13B no-had {mode} rotation training against exact vLLM "
        "dynamic per-token FP8 E4M3 activation QDQ; no W4AFP8 export, PPL, "
        "BoolQ, serving, or acceleration claim"
    ),
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "slurm_job_id": job_id,
    "accepted_smoke_job_id": smoke_job or None,
    "project_revision": revision,
    "rotation_manifest": str(manifest_path),
    "rotation_safetensors_sha256": digest,
    "training_duration_seconds": finished - started,
    "training": training,
    "runtime": provenance["runtime"],
}
result_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"ISAMBARD_SPINQUANT_LLAMA2_13B_FP8_{mode.upper()}_PASSED")
PY
