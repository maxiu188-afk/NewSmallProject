#!/bin/bash
# Export matched Llama-2-13B W4A16 checkpoints and validate all three in vLLM.
# Calibration materialization is a login-node prerequisite.
#SBATCH --job-name=vllm-llama2-13b-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=results/vllm-w4a16-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4a16-llama2-13b/%x-%j.err

set -euo pipefail

echo "VLLM_13B_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
config="${project_root}/configs/deployment/vllm_w4a16_llama2_13b_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
quantizer_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
calibration_dir="${artifact_root}/calibration-128x2048"
unrotated_dir="${artifact_root}/unrotated-w4a16"
rotated_dir="${artifact_root}/rotated-w4a16"
results_dir="${project_root}/results/vllm-w4a16-llama2-13b"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_13B_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${quantizer_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${calibration_dir}/manifest.json" \
    "${calibration_dir}/dataset"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_13B_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_13B_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
calibration_revision="$("${quantizer_env}/bin/python" - "${calibration_dir}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["project_revision"])
PY
)"
if [[ "${calibration_revision}" != "${git_revision}" ]]; then
    echo "VLLM_13B_CALIBRATION_REVISION_MISMATCH" >&2
    exit 1
fi
mkdir -p "${artifact_root}" "${results_dir}"
available_kib="$(df --output=avail "${artifact_root}" | tail -1)"
if (( available_kib < 20 * 1024 * 1024 )); then
    echo "VLLM_13B_INSUFFICIENT_ARTIFACT_SPACE_KIB=${available_kib}" >&2
    exit 1
fi

echo "VLLM_13B_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'artifact_root=%s\n' "${artifact_root}"
    sha256sum \
        "${config}" \
        "${project_root}/repro/offline_llama_rotation.py" \
        "${project_root}/repro/structured_hadamard.py" \
        "${project_root}/scripts/export_vllm_w4a16_llama2_13b.py" \
        "${project_root}/scripts/run_isambard_vllm_w4a16_llama2_13b_gate.sbatch" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
        "${calibration_dir}/manifest.json"
} > "${results_dir}/source-manifest.txt"

echo "VLLM_13B_STAGE=runtime"
"${quantizer_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import torch

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("llmcompressor") == "0.12.0"
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (9, 0)
print(torch.cuda.get_device_name(0))
PY

echo "VLLM_13B_STAGE=unrotated_export"
validate_or_export() {
    local mode="$1"
    local checkpoint="$2"
    local report="$3"
    if [[ -f "${report}" && -f "${checkpoint}/config.json" ]]; then
        "${quantizer_env}/bin/python" - "${report}" "${git_revision}" "${mode}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "passed", report
assert report["project_revision"] == sys.argv[2], report["project_revision"]
assert report["mode"] == sys.argv[3], report["mode"]
assert report["packed_decoder_linear_count"] == 280, report
print(f"VLLM_13B_REUSING_EXPORT={sys.argv[3]}")
PY
        return
    fi
    if [[ -e "${checkpoint}" || -e "${checkpoint}.incomplete" || -e "${report}" ]]; then
        echo "VLLM_13B_PARTIAL_EXPORT_REQUIRES_INSPECTION=${mode}" >&2
        exit 1
    fi
    "${quantizer_env}/bin/python" \
        "${project_root}/scripts/export_vllm_w4a16_llama2_13b.py" \
        --config "${config}" \
        --calibration-dir "${calibration_dir}" \
        --mode "${mode}" \
        --output-dir "${checkpoint}" \
        --report "${report}"
}

validate_or_export \
    unrotated \
    "${unrotated_dir}" \
    "${results_dir}/unrotated-export.json"

echo "VLLM_13B_STAGE=rotated_export"
validate_or_export \
    rotated \
    "${rotated_dir}" \
    "${results_dir}/rotated-export.json"

echo "VLLM_13B_STAGE=offline_inference"
"${serving_env}/bin/python" "${project_root}/scripts/run_vllm_w4a16_llama2_13b_gate.py" \
    --config "${config}" \
    --original "${model_snapshot}" \
    --unrotated "${unrotated_dir}" \
    --rotated "${rotated_dir}" \
    --output "${results_dir}/offline-inference.json"

echo "VLLM_13B_STAGE=assert_result"
"${serving_env}/bin/python" - \
    "${results_dir}/unrotated-export.json" \
    "${results_dir}/rotated-export.json" \
    "${results_dir}/offline-inference.json" <<'PY'
import json
import sys
from pathlib import Path

unrotated, rotated, inference = [
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
]
assert unrotated["status"] == "passed", unrotated
assert rotated["status"] == "passed", rotated
assert unrotated["packed_decoder_linear_count"] == 280, unrotated
assert rotated["packed_decoder_linear_count"] == 280, rotated
assert rotated["rotation"]["residual_hadamard_kind"] == "structured", rotated
assert inference["status"] == "passed", inference
assert set(inference["models"]) == {"bf16", "unrotated_w4a16", "rotated_w4a16"}
print("ISAMBARD_VLLM_W4A16_LLAMA2_13B_GATE_PASSED")
PY
