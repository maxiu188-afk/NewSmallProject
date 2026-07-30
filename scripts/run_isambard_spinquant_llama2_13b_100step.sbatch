#!/bin/bash
# Run the formal 100-step Llama-2-13B SpinQuant W4A16 rotation training job.
# The accepted one-step smoke is a content-hash and result gate, not merely a
# scheduler dependency. No GPTQ export, PPL evaluation, or deployment occurs.
# Submit from a clean, isolated SpinQuant checkout:
#   sbatch --test-only scripts/run_isambard_spinquant_llama2_13b_100step.sbatch
#   sbatch scripts/run_isambard_spinquant_llama2_13b_100step.sbatch
#SBATCH --job-name=spinquant-13b-100step
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=results/spinquant-isambard-llama2-13b/%x-%j.out
#SBATCH --error=results/spinquant-isambard-llama2-13b/%x-%j.err

set -euo pipefail

echo "SPINQUANT_13B_FORMAL_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
training_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
calibration_config="${project_root}/configs/spinquant/llama2_13b_rotation_calibration_800.json"
training_config="${project_root}/configs/spinquant/llama2_13b_w4a16_rotation_train_100.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
artifact_root="${SPINQUANT_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-w4a16}"
results_dir="${project_root}/results/spinquant-isambard-llama2-13b"
smoke_job="5841874"
smoke_result="${results_dir}/spinquant-13b-1step-${smoke_job}.json"
smoke_source_manifest="${results_dir}/spinquant-13b-1step-${smoke_job}-source-manifest.txt"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-spinquant-formal-${SLURM_JOB_ID}"
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

echo "SPINQUANT_13B_FORMAL_STAGE=inputs"
for required in \
    "${training_env}/bin/python" \
    "${calibration_config}" \
    "${training_config}" \
    "${model_snapshot}/config.json" \
    "${smoke_result}" \
    "${smoke_source_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SPINQUANT_13B_FORMAL_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SPINQUANT_13B_FORMAL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
revision_label="${git_revision:0:12}"
calibration_dir="${artifact_root}/calibration-800x2048-${revision_label}"
rotation_dir="${artifact_root}/rotations-100step-${SLURM_JOB_ID}"
rotation_manifest="${rotation_dir}/rotation-manifest.json"
result="${results_dir}/spinquant-13b-100step-${SLURM_JOB_ID}.json"
mkdir -p "${artifact_root}" "${results_dir}"

echo "SPINQUANT_13B_FORMAL_STAGE=smoke_gate"
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
assert training["status"] == "passed", training
assert training["optimizer_steps"] == 1, training
assert training["gradient_accumulation_steps"] == 8, training
assert training["calibration_sequences"] == 8, training
assert training["sequence_length"] == 2048, training
assert len(training["gradient_maxima"]) == 1, training
assert training["gradient_maxima"][0] > 0, training
assert max(training["orthogonality_error"].values()) <= 1e-4, training
print("SPINQUANT_13B_FORMAL_SMOKE_GATE_ACCEPTED")
PY

echo "SPINQUANT_13B_FORMAL_STAGE=runtime"
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
print("SPINQUANT_13B_FORMAL_RUNTIME_ACCEPTED")
PY

echo "SPINQUANT_13B_FORMAL_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'accepted_smoke_job=%s\n' "${smoke_job}"
    printf 'accepted_smoke_result=%s\n' "${smoke_result}"
    printf 'training_environment=%s\n' "${training_env}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    printf 'scratchdir=%s\n' "${SCRATCHDIR}"
    sha256sum \
        "${calibration_config}" \
        "${training_config}" \
        "${project_root}/repro/spinquant/artifacts.py" \
        "${project_root}/repro/spinquant/calibration.py" \
        "${project_root}/repro/spinquant/llama_adapter.py" \
        "${project_root}/repro/spinquant/rotations.py" \
        "${project_root}/repro/spinquant/stiefel.py" \
        "${project_root}/repro/spinquant/training.py" \
        "${project_root}/scripts/spinquant/prepare_wikitext2_calibration.py" \
        "${project_root}/scripts/spinquant/train_llama_rotations.py" \
        "${project_root}/scripts/run_isambard_spinquant_llama2_13b_100step.sbatch" \
        "${model_snapshot}/config.json" \
        "${smoke_result}" \
        "${smoke_source_manifest}"
} > "${results_dir}/spinquant-13b-100step-${SLURM_JOB_ID}-source-manifest.txt"

echo "SPINQUANT_13B_FORMAL_STAGE=calibration"
if [[ ! -e "${calibration_dir}/calibration-manifest.json" && \
      ! -e "${calibration_dir}/token-ids.json" ]]; then
    "${training_env}/bin/python" \
        "${project_root}/scripts/spinquant/prepare_wikitext2_calibration.py" \
        --config "${calibration_config}" \
        --model-snapshot "${model_snapshot}" \
        --output-dir "${calibration_dir}"
elif [[ ! -f "${calibration_dir}/calibration-manifest.json" || \
        ! -f "${calibration_dir}/token-ids.json" ]]; then
    echo "SPINQUANT_13B_FORMAL_PARTIAL_CALIBRATION=${calibration_dir}" >&2
    exit 1
fi

echo "SPINQUANT_13B_FORMAL_STAGE=train"
if [[ -e "${rotation_dir}" ]]; then
    echo "SPINQUANT_13B_FORMAL_REFUSES_EXISTING_ROTATION_DIR=${rotation_dir}" >&2
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

echo "SPINQUANT_13B_FORMAL_STAGE=assert_result"
"${training_env}/bin/python" - \
    "${rotation_manifest}" "${result}" "${git_revision}" "${SLURM_JOB_ID}" \
    "${smoke_job}" "${training_started_epoch}" "${training_finished_epoch}" <<'PY'
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
smoke_job = sys.argv[5]
started = int(sys.argv[6])
finished = int(sys.argv[7])
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
assert training["optimizer_steps"] == 100, training
assert training["gradient_accumulation_steps"] == 8, training
assert training["calibration_sequences"] == 800, training
assert training["sequence_length"] == 2048, training
assert len(training["gradient_maxima"]) == 100, training
assert len(training["update_losses"]) == 100, training
assert all(
    math.isfinite(value) and value > 0
    for value in training["gradient_maxima"]
), training
assert all(math.isfinite(value) for value in training["update_losses"]), training
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
        "formal Llama-2-13B R1/R2 W4A16 100-step rotation training; "
        "no PPL, downstream quality, GPTQ, packing, kernel, or deployment claim"
    ),
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "slurm_job_id": job_id,
    "accepted_smoke_job_id": smoke_job,
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
print("ISAMBARD_SPINQUANT_LLAMA2_13B_100STEP_PASSED")
PY
