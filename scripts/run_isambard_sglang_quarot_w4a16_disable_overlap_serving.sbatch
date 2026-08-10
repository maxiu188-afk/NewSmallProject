#!/bin/bash
# Protocol-matched SGLang third arm; does not rerun vLLM or default SGLang.
# Submit from a clean project checkout with:
#   sbatch scripts/run_isambard_sglang_quarot_w4a16_disable_overlap_serving.sbatch
# This file only defines the job. It never submits another job itself.
#SBATCH --job-name=sgl-w4a16-no-overlap
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=results/sglang-vllm-quarot-w4a16-serving/%x-%j.out
#SBATCH --error=results/sglang-vllm-quarot-w4a16-serving/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7
module load cuda/12.6
module load gcc-native/13.2
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

spec="${project_root}/configs/deployment/sglang_quarot_w4a16_disable_overlap_serving_formal_isambard.json"
client_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
sglang_env="${SGLANG_ENV_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-sglang/sglang-0.5.16-cu129}"
export PATH="${sglang_env}/bin:${PATH}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export NVCC_CCBIN="${CXX}"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
rotated_dir="${artifact_root}/rotated-w4a16"
source_result="${W4A16_SOURCE_RESULT:-${HOME}/NewSmallProject-vllm-ready/results/vllm-w4a16-llama2-13b/offline-inference-retry.json}"
sglang_source_root="${sglang_env}/lib/python3.11/site-packages/sglang"
results_dir="${project_root}/results/sglang-vllm-quarot-w4a16-serving"
reference_result="${results_dir}/serving-formal-5960180.json"
reference_manifest="${results_dir}/serving-formal-5960180-source-manifest.txt"
prefix="${results_dir}/sglang-disable-overlap-formal-${SLURM_JOB_ID}"
output="${prefix}.json"
request_corpus="${prefix}-requests.json"
log_dir="${prefix}-logs"
raw_dir="${prefix}-raw"
source_manifest="${prefix}-source-manifest.txt"
sglang_preflight="${prefix}-sglang-preflight.json"
effective_config="${prefix}-effective-config.json"
input_record="${prefix}-input-gate.json"

for required in \
    "${client_env}/bin/python" \
    "${sglang_env}/bin/python" \
    "${sglang_env}/bin/ninja" \
    "${CUDA_HOME}/bin/nvcc" \
    "${CUDA_HOME}/lib64/libcudart.so" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
    "${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py" \
    "${project_root}/scripts/run_sglang_quarot_w4a16_disable_overlap_serving.py" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    "${project_root}/scripts/validate_sglang_quarot_w4a16_disable_overlap_inputs.py" \
    "${spec}" \
    "${model_snapshot}/tokenizer.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${reference_result}" \
    "${reference_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SGLANG_W4A16_DISABLE_OVERLAP_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
for refused in \
    "${output}" \
    "${request_corpus}" \
    "${log_dir}" \
    "${raw_dir}" \
    "${source_manifest}" \
    "${sglang_preflight}" \
    "${effective_config}" \
    "${input_record}"; do
    if [[ -e "${refused}" ]]; then
        echo "SGLANG_W4A16_DISABLE_OVERLAP_REFUSES_EXISTING_OUTPUT=${refused}" >&2
        exit 1
    fi
done

git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SGLANG_W4A16_DISABLE_OVERLAP_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TVM_FFI_CACHE_DIR="${PROJECTDIR}/${USER}/newsmallproject-sglang/tvm-ffi-cache/cuda-12.6-gcc-13.2-tvmffi-0.1.11"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
mkdir -p "${TVM_FFI_CACHE_DIR}"

echo "SGLANG_W4A16_DISABLE_OVERLAP_STAGE=reference_formal_gate"
"${client_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_quarot_w4a16_disable_overlap_inputs.py" \
    --project-root "${project_root}" \
    --config "${spec}" \
    --reference-result "${reference_result}" \
    --reference-source-manifest "${reference_manifest}" \
    --output-config "${effective_config}" \
    --output-record "${input_record}"

echo "SGLANG_W4A16_DISABLE_OVERLAP_STAGE=openai_source_gate"
"${client_env}/bin/python" - "${effective_config}" "${sglang_source_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
expected = config["sglang"]["expected_openai_source_sha256"]
observed = {name: sha256(root / name) for name in expected}
if observed != expected:
    raise RuntimeError(f"SGLang OpenAI source hashes drifted: {observed}")
print("SGLANG_W4A16_DISABLE_OVERLAP_OPENAI_SOURCES_ACCEPTED")
PY

echo "SGLANG_W4A16_DISABLE_OVERLAP_NINJA=$(ninja --version)"
echo "SGLANG_W4A16_DISABLE_OVERLAP_STAGE=runtime_preflight"
"${sglang_env}/bin/python" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    --config "${effective_config}" \
    --output "${sglang_preflight}"

echo "SGLANG_W4A16_DISABLE_OVERLAP_STAGE=immutable_model_gate"
"${client_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    --project-root "${project_root}" \
    --config "${effective_config}" \
    --source-result "${source_result}" \
    --model "${rotated_dir}"

echo "SGLANG_W4A16_DISABLE_OVERLAP_STAGE=third_arm_matrix"
"${client_env}/bin/python" \
    "${project_root}/scripts/run_sglang_quarot_w4a16_disable_overlap_serving.py" \
    --config "${effective_config}" \
    --input-record "${input_record}" \
    --tokenizer-model "${model_snapshot}" \
    --model-path "${rotated_dir}" \
    --sglang-python "${sglang_env}/bin/python" \
    --sglang-preflight "${sglang_preflight}" \
    --request-corpus "${request_corpus}" \
    --log-dir "${log_dir}" \
    --raw-dir "${raw_dir}" \
    --output "${output}"

"${client_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["comparison_status"] == "protocol_matched_historical_third_arm_complete", result
assert result["project_revision"] == sys.argv[2], result
assert result["model"] == "quarot_w4a16", result
assert result["reference_formal_job"]["job_id"] == "5960180", result
assert result["experiment"]["server_argument"] == "--disable-overlap-schedule", result
assert result["request_corpus"]["sha256"] == "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1", result
assert len(result["repetitions"]) == 3, result
for repetition in result["repetitions"]:
    assert set(repetition["cases"]) == {"latency_c1", "throughput_c8"}, repetition
    for case in repetition["cases"].values():
        assert case["status"] == "passed", case
        assert case["metrics"]["completed"] == 64, case
        assert case["metrics"]["failed"] == 0, case
        assert case["metrics"]["input_tokens"] == 64 * 256, case
        assert case["metrics"]["output_tokens"] == 64 * 64, case
        assert case["server_command"].count("--disable-overlap-schedule") == 1, case
assert result["aggregate"]["status"] == "third_arm_complete", result
assert result["aggregate"]["pairing_boundary"] == "not_paired_with_reference_job_5960180", result
print("ISAMBARD_SGLANG_QUAROT_W4A16_DISABLE_OVERLAP_FORMAL_PASSED")
PY

{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'reference_formal_job=5960180\n'
    printf 'backend=sglang\n'
    printf 'server_argument=--disable-overlap-schedule\n'
    printf 'formal_repetitions=3\n'
    printf 'formal_concurrency=1,8\n'
    sha256sum \
        "${spec}" \
        "${project_root}/configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json" \
        "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
        "${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py" \
        "${project_root}/scripts/run_sglang_quarot_w4a16_disable_overlap_serving.py" \
        "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
        "${project_root}/scripts/validate_sglang_quarot_w4a16_disable_overlap_inputs.py" \
        "${project_root}/scripts/run_isambard_sglang_quarot_w4a16_disable_overlap_serving.sbatch" \
        "${source_result}" \
        "${reference_result}" \
        "${reference_manifest}" \
        "${effective_config}" \
        "${input_record}" \
        "${sglang_preflight}" \
        "${request_corpus}" \
        "${output}"
    find "${log_dir}" "${raw_dir}" -type f -print0 | sort -z | xargs -0 sha256sum
} > "${source_manifest}"

echo "SGLANG_W4A16_DISABLE_OVERLAP_RESULT=${output}"
echo "SGLANG_W4A16_DISABLE_OVERLAP_SOURCE_MANIFEST=${source_manifest}"
