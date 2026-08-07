#!/bin/bash
# Export and load-gate the accepted SpinQuant-derived packed W4A16 endpoint.
# Login-node prerequisite: materialize calibration with this revision/config.
# This is a result-gated source job; it does not run BoolQ or serving timing.
#SBATCH --job-name=vllm-13b-spin-w4a16-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/vllm-spinquant-w4a16-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-spinquant-w4a16-llama2-13b/%x-%j.err

set -euo pipefail

echo "VLLM_SPINQUANT_W4A16_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
config="${project_root}/configs/deployment/vllm_spinquant_w4a16_llama2_13b_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
quantizer_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
artifact_root="${VLLM_SPINQUANT_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-spinquant-w4a16}"
calibration_dir="${artifact_root}/calibration-128x2048"
checkpoint_dir="${artifact_root}/spinquant-w4a16"
rotation_manifest="${SPINQUANT_W4A16_ROTATION_MANIFEST:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-w4a16/rotations-100step-5842047/rotation-manifest.json}"
rotation_tensors="${rotation_manifest%/*}/rotations.safetensors"
training_result="${SPINQUANT_W4A16_TRAINING_RESULT:-${HOME}/NewSmallProject-spinquant-ready/results/spinquant-isambard-llama2-13b/spinquant-13b-100step-5842047.json}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
results_dir="${VLLM_SPINQUANT_W4A16_RESULTS_DIR:-${project_root}/results/vllm-spinquant-w4a16-llama2-13b}"
export_report="${results_dir}/spinquant-export-${SLURM_JOB_ID}.json"
inference_result="${results_dir}/spinquant-offline-inference-${SLURM_JOB_ID}.json"
source_manifest="${results_dir}/spinquant-source-manifest-${SLURM_JOB_ID}.txt"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_SPINQUANT_W4A16_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${quantizer_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${calibration_dir}/manifest.json" \
    "${calibration_dir}/dataset" \
    "${training_result}" \
    "${rotation_manifest}" \
    "${rotation_tensors}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_SPINQUANT_W4A16_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done

git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_SPINQUANT_W4A16_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi

mkdir -p "${artifact_root}" "${results_dir}"
for output in \
    "${checkpoint_dir}" \
    "${checkpoint_dir}.incomplete" \
    "${export_report}" \
    "${inference_result}" \
    "${source_manifest}"; do
    if [[ -e "${output}" ]]; then
        echo "VLLM_SPINQUANT_W4A16_REFUSES_EXISTING_OUTPUT=${output}" >&2
        exit 1
    fi
done
available_kib="$(df --output=avail "${artifact_root}" | tail -1)"
if (( available_kib < 10 * 1024 * 1024 )); then
    echo "VLLM_SPINQUANT_W4A16_INSUFFICIENT_SPACE_KIB=${available_kib}" >&2
    exit 1
fi

echo "VLLM_SPINQUANT_W4A16_STAGE=provenance_gate"
"${quantizer_env}/bin/python" - \
    "${project_root}" \
    "${config}" \
    "${calibration_dir}/manifest.json" \
    "${training_result}" \
    "${rotation_manifest}" \
    "${rotation_tensors}" \
    "${git_revision}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

project_root, config_path, calibration_path, training_path, manifest_path, tensor_path = map(
    Path, sys.argv[1:7]
)
revision = sys.argv[7]
sys.path.insert(0, str(project_root))
from repro.vllm_w4a16 import validate_spinquant_export_config


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


config = json.loads(config_path.read_text(encoding="utf-8"))
validate_spinquant_export_config(config)
rotation = config["rotation"]
calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
training = json.loads(training_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

assert config["status"] == "accepted_rotation_ready_for_export_gate", config
assert calibration["status"] == "passed", calibration
assert calibration["project_revision"] == revision, calibration
assert calibration["config_sha256"] == sha256(config_path), calibration
assert training["status"] == "passed", training
assert training["slurm_job_id"] == rotation["spinquant_training_job_id"], training
assert training["project_revision"] == rotation["spinquant_training_project_revision"], training
assert sha256(training_path) == rotation["spinquant_training_result_sha256"]
assert training["rotation_safetensors_sha256"] == rotation["spinquant_rotation_safetensors_sha256"]
assert sha256(manifest_path) == rotation["spinquant_rotation_manifest_sha256"]
assert sha256(tensor_path) == rotation["spinquant_rotation_safetensors_sha256"]
assert manifest["file"]["sha256"] == rotation["spinquant_rotation_safetensors_sha256"]
assert manifest["provenance"]["project_revision"] == rotation["spinquant_training_project_revision"]
assert manifest["rotation"]["r1_shape"] == [5120, 5120], manifest
assert manifest["rotation"]["r2_shape"] == [40, 128, 128], manifest
print("VLLM_SPINQUANT_W4A16_PROVENANCE_ACCEPTED")
PY

echo "VLLM_SPINQUANT_W4A16_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'training_job=5842047\n'
    printf 'artifact_root=%s\n' "${artifact_root}"
    sha256sum \
        "${config}" \
        "${project_root}/repro/vllm_w4a16.py" \
        "${project_root}/repro/spinquant/artifacts.py" \
        "${project_root}/repro/spinquant/offline.py" \
        "${project_root}/repro/spinquant/rotations.py" \
        "${project_root}/scripts/export_vllm_w4a16_llama2_13b.py" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
        "${project_root}/scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_gate.sbatch" \
        "${calibration_dir}/manifest.json" \
        "${training_result}" \
        "${rotation_manifest}" \
        "${rotation_tensors}"
} > "${source_manifest}"

echo "VLLM_SPINQUANT_W4A16_STAGE=runtime"
"${quantizer_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("llmcompressor") == "0.12.0"
assert metadata.version("compressed-tensors") == "0.17.1"
assert metadata.version("transformers") == "5.10.1"
assert metadata.version("safetensors") == "0.8.0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("VLLM_SPINQUANT_W4A16_RUNTIME_ACCEPTED")
PY

echo "VLLM_SPINQUANT_W4A16_STAGE=export"
"${quantizer_env}/bin/python" \
    "${project_root}/scripts/export_vllm_w4a16_llama2_13b.py" \
    --config "${config}" \
    --calibration-dir "${calibration_dir}" \
    --mode spinquant \
    --rotation-manifest "${rotation_manifest}" \
    --output-dir "${checkpoint_dir}" \
    --report "${export_report}"

echo "VLLM_SPINQUANT_W4A16_STAGE=vllm_load"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
    --config "${config}" \
    --original "${model_snapshot}" \
    --spinquant "${checkpoint_dir}" \
    --output "${inference_result}"

echo "VLLM_SPINQUANT_W4A16_STAGE=assert_result"
"${serving_env}/bin/python" - \
    "${export_report}" \
    "${inference_result}" \
    "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

export, inference = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:3]
]
revision = sys.argv[3]
assert export["status"] == "passed", export
assert export["project_revision"] == revision, export
assert export["mode"] == "spinquant", export
assert export["rotation"]["kind"] == "spinquant_learned_offline", export
assert export["rotation"]["training_job_id"] == "5842047", export
assert export["rotation"]["online_rotation_modules"] == 0, export
assert export["packed_decoder_linear_count"] == 280, export
assert len(export["checkpoint_tree_sha256"]) == 64, export
assert inference["status"] == "passed", inference
assert inference["project_revision"] == revision, inference
assert set(inference["models"]) == {"bf16", "spinquant_w4a16"}, inference
comparison = inference["comparisons"]["spinquant_w4a16_vs_bf16"]
assert comparison["common_logprob_tokens"] > 0, comparison
print("ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_GATE_PASSED")
PY
