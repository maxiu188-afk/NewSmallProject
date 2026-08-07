#!/bin/bash
# Result-gated smoke only. Do not submit a formal job until this artifact is reviewed.
# Submit from a clean project checkout with:
#   sbatch scripts/run_isambard_sglang_vllm_quarot_w4a16_boolq_smoke.sbatch
#SBATCH --job-name=sgl-vllm-w4a16-boolq
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/sglang-vllm-quarot-w4a16-boolq/%x-%j.out
#SBATCH --error=results/sglang-vllm-quarot-w4a16-boolq/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7
module load cuda/12.6
module load gcc-native/13.2
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

config="${project_root}/configs/deployment/sglang_vllm_quarot_w4a16_boolq_smoke_isambard.json"
vllm_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
sglang_env="${SGLANG_ENV_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-sglang/sglang-0.5.16-cu129}"
export PATH="${sglang_env}/bin:${PATH}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export NVCC_CCBIN="${CXX}"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
rotated_dir="${artifact_root}/rotated-w4a16"
source_result="${W4A16_SOURCE_RESULT:-${HOME}/NewSmallProject-vllm-ready/results/vllm-w4a16-llama2-13b/offline-inference-retry.json}"
dataset_root="${BOOLQ_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/boolq-lm-eval-v1-3de24cf}"
dataset_manifest="${dataset_root}/manifest.json"
results_dir="${project_root}/results/sglang-vllm-quarot-w4a16-boolq"
output="${results_dir}/boolq-compatibility-smoke-${SLURM_JOB_ID}.json"
log_dir="${results_dir}/boolq-compatibility-smoke-${SLURM_JOB_ID}-logs"
source_manifest="${results_dir}/boolq-compatibility-smoke-${SLURM_JOB_ID}-source-manifest.txt"
sglang_preflight="${results_dir}/boolq-compatibility-smoke-${SLURM_JOB_ID}-sglang-preflight.json"

for required in \
    "${vllm_env}/bin/python" \
    "${sglang_env}/bin/python" \
    "${sglang_env}/bin/ninja" \
    "${CUDA_HOME}/bin/nvcc" \
    "${CUDA_HOME}/lib64/libcudart.so" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    "${project_root}/scripts/setup_isambard_sglang_env.sh" \
    "${config}" \
    "${model_snapshot}/tokenizer.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${dataset_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SGLANG_VLLM_W4A16_BOOLQ_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" ]]; then
    echo "SGLANG_VLLM_W4A16_BOOLQ_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SGLANG_VLLM_W4A16_BOOLQ_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}" "${log_dir}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TVM_FFI_CACHE_DIR="${PROJECTDIR}/${USER}/newsmallproject-sglang/tvm-ffi-cache/cuda-12.6-gcc-13.2-tvmffi-0.1.11"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
mkdir -p "${TVM_FFI_CACHE_DIR}"

echo "SGLANG_VLLM_W4A16_BOOLQ_NINJA=$(ninja --version)"
echo "SGLANG_VLLM_W4A16_BOOLQ_STAGE=runtime_preflight"
"${sglang_env}/bin/python" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    --config "${config}" \
    --output "${sglang_preflight}"

echo "SGLANG_VLLM_W4A16_BOOLQ_STAGE=source_gate"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --model "${rotated_dir}"

{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5769503\n'
    printf 'dataset_manifest=%s\n' "${dataset_manifest}"
    printf 'model=%s\n' "${rotated_dir}"
    printf 'formal_job=not_submitted\n'
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_boolq.py" \
        "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
        "${project_root}/scripts/setup_isambard_sglang_env.sh" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
        "${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py" \
        "${project_root}/scripts/run_isambard_sglang_vllm_quarot_w4a16_boolq_smoke.sbatch" \
        "${source_result}" \
        "${dataset_manifest}" \
        "${sglang_preflight}"
} > "${source_manifest}"

echo "SGLANG_VLLM_W4A16_BOOLQ_STAGE=evaluate"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_boolq.py" \
    --config "${config}" \
    --dataset-manifest "${dataset_manifest}" \
    --tokenizer-model "${model_snapshot}" \
    --model-path "${rotated_dir}" \
    --sglang-python "${sglang_env}/bin/python" \
    --sglang-preflight "${sglang_preflight}" \
    --max-examples 32 \
    --log-dir "${log_dir}" \
    --output "${output}"

"${vllm_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "recorded", result
assert result["project_revision"] == sys.argv[2], result
assert result["model"] == "quarot_w4a16", result
assert result["evaluated_examples"] == 32, result
assert result["expected_requests_per_backend"] == 64, result
assert set(result["cases"]) == {"vllm", "sglang"}, result
assert result["runtimes"]["sglang_preflight"]["record"]["status"] == "passed", result
assert result["comparison_status"] == "both_backends_scored", result
assert all(case["status"] == "passed" for case in result["cases"].values()), result
print("ISAMBARD_SGLANG_VLLM_QUAROT_W4A16_BOOLQ_SMOKE_RECORDED")
PY
