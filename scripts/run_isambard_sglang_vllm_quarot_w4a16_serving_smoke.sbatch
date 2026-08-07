#!/bin/bash
# Result-gated serving-client smoke only. Do not submit the formal matrix yet.
# Submit from a clean project checkout with:
#   sbatch scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_smoke.sbatch
#SBATCH --job-name=sgl-vllm-w4a16-serve
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/sglang-vllm-quarot-w4a16-serving/%x-%j.out
#SBATCH --error=results/sglang-vllm-quarot-w4a16-serving/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7
module load cuda/12.6
module load gcc-native/13.2
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

config="${project_root}/configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json"
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
legacy_raw="${HOME}/NewSmallProject-vllm-ready/results/vllm-w4a16-serving-llama2-13b/serving-benchmark-raw/rotated_w4a16-latency_c1.json"
vllm_source_root="${vllm_env}/lib/python3.11/site-packages/vllm"
sglang_source_root="${sglang_env}/lib/python3.11/site-packages/sglang"
results_dir="${project_root}/results/sglang-vllm-quarot-w4a16-serving"
output="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}.json"
request_corpus="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}-requests.json"
log_dir="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}-logs"
raw_dir="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}-raw"
source_manifest="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}-source-manifest.txt"
sglang_preflight="${results_dir}/serving-client-smoke-${SLURM_JOB_ID}-sglang-preflight.json"

for required in \
    "${vllm_env}/bin/python" \
    "${vllm_env}/bin/vllm" \
    "${sglang_env}/bin/python" \
    "${sglang_env}/bin/ninja" \
    "${CUDA_HOME}/bin/nvcc" \
    "${CUDA_HOME}/lib64/libcudart.so" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    "${config}" \
    "${model_snapshot}/tokenizer.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${legacy_raw}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SGLANG_VLLM_W4A16_SERVING_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
for refused in "${output}" "${request_corpus}" "${source_manifest}" "${sglang_preflight}"; do
    if [[ -e "${refused}" ]]; then
        echo "SGLANG_VLLM_W4A16_SERVING_REFUSES_EXISTING_OUTPUT=${refused}" >&2
        exit 1
    fi
done

git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SGLANG_VLLM_W4A16_SERVING_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}" "${log_dir}" "${raw_dir}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TVM_FFI_CACHE_DIR="${PROJECTDIR}/${USER}/newsmallproject-sglang/tvm-ffi-cache/cuda-12.6-gcc-13.2-tvmffi-0.1.11"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
mkdir -p "${TVM_FFI_CACHE_DIR}"

echo "SGLANG_VLLM_W4A16_SERVING_STAGE=external_source_gate"
"${vllm_env}/bin/python" - \
    "${config}" "${vllm_source_root}" "${sglang_source_root}" "${legacy_raw}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for root, expected in (
    (Path(sys.argv[2]), config["vllm"]["expected_source_sha256"]),
    (Path(sys.argv[3]), config["sglang"]["expected_openai_source_sha256"]),
):
    observed = {name: sha256(root / name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"external source hashes drifted: {observed}")
legacy = config["request_corpus"]["legacy_vllm_evidence"]
if sha256(sys.argv[4]) != legacy["raw_result_sha256"]:
    raise RuntimeError("accepted vLLM serving raw result hash drifted")
print("SGLANG_VLLM_W4A16_SERVING_EXTERNAL_SOURCES_ACCEPTED")
PY

echo "SGLANG_VLLM_W4A16_SERVING_NINJA=$(ninja --version)"
echo "SGLANG_VLLM_W4A16_SERVING_STAGE=runtime_preflight"
"${sglang_env}/bin/python" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    --config "${config}" \
    --output "${sglang_preflight}"

echo "SGLANG_VLLM_W4A16_SERVING_STAGE=source_gate"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --model "${rotated_dir}"

echo "SGLANG_VLLM_W4A16_SERVING_STAGE=client_smoke"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
    --config "${config}" \
    --tokenizer-model "${model_snapshot}" \
    --model-path "${rotated_dir}" \
    --vllm-executable "${vllm_env}/bin/vllm" \
    --sglang-python "${sglang_env}/bin/python" \
    --sglang-preflight "${sglang_preflight}" \
    --request-corpus "${request_corpus}" \
    --log-dir "${log_dir}" \
    --raw-dir "${raw_dir}" \
    --output "${output}"

"${vllm_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "recorded", result
assert result["comparison_status"] == "both_backends_benchmark_client_passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["model"] == "quarot_w4a16", result
assert result["request_corpus"]["sha256"] == "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1", result
assert set(result["cases"]) == {"vllm:smoke_c1", "sglang:smoke_c1"}, result
assert all(case["status"] == "passed" for case in result["cases"].values()), result
for case in result["cases"].values():
    assert case["metrics"]["completed"] == 8, case
    assert case["metrics"]["failed"] == 0, case
    assert case["metrics"]["input_tokens"] == 8 * 256, case
    assert case["metrics"]["output_tokens"] == 8 * 64, case
assert result["cases"]["vllm:smoke_c1"]["legacy_vllm_output_diagnostic"]["policy"] == "diagnostic_only_not_corpus_identity", result
assert result["comparison"]["status"] == "smoke_observation_not_formal_performance_evidence", result
print("ISAMBARD_SGLANG_VLLM_QUAROT_W4A16_SERVING_SMOKE_RECORDED")
PY

{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5769503\n'
    printf 'compatibility_gate_job=5941763\n'
    printf 'legacy_serving_job=5780631\n'
    printf 'model=%s\n' "${rotated_dir}"
    printf 'formal_job=not_submitted\n'
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
        "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
        "${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py" \
        "${project_root}/scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_smoke.sbatch" \
        "${source_result}" \
        "${legacy_raw}" \
        "${sglang_preflight}" \
        "${request_corpus}" \
        "${output}"
    find "${log_dir}" "${raw_dir}" -type f -print0 | sort -z | xargs -0 sha256sum
} > "${source_manifest}"

echo "SGLANG_VLLM_W4A16_SERVING_RESULT=${output}"
echo "SGLANG_VLLM_W4A16_SERVING_SOURCE_MANIFEST=${source_manifest}"
