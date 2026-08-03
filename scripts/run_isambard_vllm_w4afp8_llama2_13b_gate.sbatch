#!/bin/bash
# Export/load gate for isolated QuaRot, isolated SpinQuant-transfer, or joint
# W4AFP8 studies.
# Run the isolated QuaRot gate as:
#   sbatch scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch quarot
#SBATCH --job-name=vllm-13b-w4afp8-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=results/vllm-w4afp8-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4afp8-llama2-13b/%x-%j.err

set -euo pipefail

variant_set="${1:-}"
if [[ "${variant_set}" != "quarot" && \
      "${variant_set}" != "spinquant" && \
      "${variant_set}" != "joint" ]]; then
    echo "usage: sbatch $0 {quarot|spinquant|joint}" >&2
    exit 2
fi

echo "VLLM_W4AFP8_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
config="${VLLM_W4AFP8_CONFIG:-${project_root}/configs/deployment/vllm_w4afp8_llama2_13b_isambard.json}"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
quantizer_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
artifact_root="${VLLM_W4AFP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8}"
calibration_dir="${artifact_root}/calibration-128x2048"
unrotated_dir="${artifact_root}/unrotated-w4afp8"
quarot_dir="${artifact_root}/quarot-w4afp8"
spinquant_dir="${VLLM_W4AFP8_SPINQUANT_CHECKPOINT_DIR:-${artifact_root}/spinquant-w4afp8-int8-transfer}"
results_dir="${VLLM_W4AFP8_RESULTS_DIR:-${project_root}/results/vllm-w4afp8-llama2-13b}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
spinquant_artifact_root="${SPINQUANT_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-spinquant/llama2-13b-paper-nohad-w16a8}"
rotation_manifest="${SPINQUANT_ROTATION_MANIFEST:-${spinquant_artifact_root}/rotations-100step-5854269/rotation-manifest.json}"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-w4afp8-${SLURM_JOB_ID}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "VLLM_W4AFP8_STAGE=inputs"
required_inputs=(
    "${serving_env}/bin/python"
    "${quantizer_env}/bin/python"
    "${config}"
    "${model_snapshot}/config.json"
)
if [[ "${variant_set}" != "quarot" ]]; then
    required_inputs+=("${rotation_manifest}")
fi
for required in "${required_inputs[@]}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_W4AFP8_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_W4AFP8_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${artifact_root}" "${results_dir}"

echo "VLLM_W4AFP8_STAGE=calibration"
if [[ ! -e "${calibration_dir}" ]]; then
    "${quantizer_env}/bin/python" \
        "${project_root}/scripts/prepare_vllm_w4a16_llama2_calibration.py" \
        --config "${config}" \
        --output-dir "${calibration_dir}"
elif [[ ! -f "${calibration_dir}/manifest.json" || \
        ! -d "${calibration_dir}/dataset" ]]; then
    echo "VLLM_W4AFP8_PARTIAL_CALIBRATION=${calibration_dir}" >&2
    exit 1
fi
"${quantizer_env}/bin/python" - \
    "${calibration_dir}/manifest.json" "${config}" "${git_revision}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
assert manifest["status"] == "passed", manifest
assert manifest["project_revision"] == sys.argv[3], manifest
assert manifest["config_sha256"] == digest, manifest
assert manifest["dataset"]["materialized_samples"] == 128, manifest
assert manifest["dataset"]["sequence_length"] == 2048, manifest
print("VLLM_W4AFP8_CALIBRATION_ACCEPTED")
PY

echo "VLLM_W4AFP8_STAGE=source_manifest"
{
    printf 'variant_set=%s\n' "${variant_set}"
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'config=%s\n' "${config}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    printf 'results_dir=%s\n' "${results_dir}"
    sha256sum \
        "${config}" \
        "${project_root}/repro/vllm_w4afp8.py" \
        "${project_root}/repro/offline_llama_rotation.py" \
        "${project_root}/scripts/audit_vllm_w4afp8_backend.py" \
        "${project_root}/scripts/export_vllm_w4afp8_llama2_13b.py" \
        "${project_root}/scripts/prepare_vllm_w4a16_llama2_calibration.py" \
        "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_gate.py" \
        "${project_root}/scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch" \
        "${calibration_dir}/manifest.json"
} > "${results_dir}/source-manifest-${SLURM_JOB_ID}.txt"
if [[ "${variant_set}" != "quarot" ]]; then
    sha256sum \
        "${project_root}/repro/spinquant/artifacts.py" \
        "${project_root}/repro/spinquant/offline.py" \
        "${rotation_manifest}" \
        >> "${results_dir}/source-manifest-${SLURM_JOB_ID}.txt"
fi

echo "VLLM_W4AFP8_STAGE=capability_audit"
capability_report="${results_dir}/backend-capability-${SLURM_JOB_ID}.json"
"${serving_env}/bin/python" \
    "${project_root}/scripts/audit_vllm_w4afp8_backend.py" \
    --config "${config}" \
    --output "${capability_report}"

echo "VLLM_W4AFP8_STAGE=quantizer_runtime"
"${quantizer_env}/bin/python" - "${config}" <<'PY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import torch

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = config["runtime"]
assert torch.__version__ == runtime["torch_version"], torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("llmcompressor") == runtime["llmcompressor_version"]
assert metadata.version("compressed-tensors") == runtime["quantizer_compressed_tensors_version"]
assert torch.cuda.is_available()
assert list(torch.cuda.get_device_capability(0)) == runtime["compute_capability"]
print(torch.cuda.get_device_name(0))
PY

echo "VLLM_W4AFP8_STAGE=export"
validate_or_export() {
    local variant="$1"
    local checkpoint="$2"
    local report="$3"
    shift 3
    if [[ -f "${report}" && -f "${checkpoint}/config.json" ]]; then
        "${quantizer_env}/bin/python" - "${report}" "${git_revision}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "passed", report
assert report["project_revision"] == sys.argv[2], report
assert report["variant"] == sys.argv[3], report
assert report["packed_decoder_linear_count"] == 280, report
assert report["runtime_g_idx_tensors"] == 0, report
print(f"VLLM_W4AFP8_REUSING_EXPORT={sys.argv[3]}")
PY
        return
    fi
    if [[ -e "${checkpoint}" || -e "${checkpoint}.incomplete" || -e "${report}" ]]; then
        echo "VLLM_W4AFP8_PARTIAL_EXPORT_REQUIRES_INSPECTION=${variant}" >&2
        exit 1
    fi
    "${quantizer_env}/bin/python" \
        "${project_root}/scripts/export_vllm_w4afp8_llama2_13b.py" \
        --config "${config}" \
        --calibration-dir "${calibration_dir}" \
        --variant "${variant}" \
        --output-dir "${checkpoint}" \
        --report "${report}" \
        "$@"
}

validate_or_export \
    unrotated_w4afp8 \
    "${unrotated_dir}" \
    "${results_dir}/unrotated-export.json"
if [[ "${variant_set}" != "spinquant" ]]; then
    validate_or_export \
        quarot_w4afp8 \
        "${quarot_dir}" \
        "${results_dir}/quarot-export.json"
fi
if [[ "${variant_set}" != "quarot" ]]; then
    validate_or_export \
        spinquant_w4afp8 \
        "${spinquant_dir}" \
        "${results_dir}/spinquant-transfer-export.json" \
        --rotation-manifest "${rotation_manifest}"
fi

echo "VLLM_W4AFP8_STAGE=offline_inference"
gate_result="${results_dir}/${variant_set}-offline-inference-${SLURM_JOB_ID}.json"
gate_command=(
    "${serving_env}/bin/python"
    "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_gate.py"
    --variant-set "${variant_set}"
    --config "${config}"
    --model "bf16=${model_snapshot}"
    --model "unrotated_w4afp8=${unrotated_dir}"
    --output "${gate_result}"
)
if [[ "${variant_set}" != "spinquant" ]]; then
    gate_command+=(--model "quarot_w4afp8=${quarot_dir}")
fi
if [[ "${variant_set}" != "quarot" ]]; then
    gate_command+=(--model "spinquant_w4afp8=${spinquant_dir}")
fi
"${gate_command[@]}"

echo "VLLM_W4AFP8_STAGE=assert_result"
"${serving_env}/bin/python" - \
    "${capability_report}" "${gate_result}" "${git_revision}" "${variant_set}" <<'PY'
import json
import sys
from pathlib import Path

capability, gate = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:3]
]
variant_set = sys.argv[4]
expected = {"bf16", "unrotated_w4afp8", "quarot_w4afp8"}
if variant_set == "spinquant":
    expected = {"bf16", "unrotated_w4afp8", "spinquant_w4afp8"}
elif variant_set == "joint":
    expected.add("spinquant_w4afp8")
assert capability["status"] == "passed", capability
assert {row["kernel"] for row in capability["selections"]} == {
    "CutlassW4A8LinearKernel"
}, capability
assert gate["status"] == "passed", gate
assert gate["project_revision"] == sys.argv[3], gate
assert gate["variant_set"] == variant_set, gate
assert set(gate["models"]) == expected, gate
for name in expected - {"bf16"}:
    checkpoint = gate["models"][name]["checkpoint"]
    assert checkpoint["packed_decoder_linear_count"] == 280, checkpoint
    assert checkpoint["runtime_g_idx_tensors"] == 0, checkpoint
print("ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_GATE_PASSED")
PY

if [[ "${variant_set}" == "quarot" ]]; then
    echo "ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_QUAROT_GATE_PASSED"
elif [[ "${variant_set}" == "spinquant" ]]; then
    echo "ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_SPINQUANT_TRANSFER_GATE_PASSED"
fi
