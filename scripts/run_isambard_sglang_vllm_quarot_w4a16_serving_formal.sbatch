#!/bin/bash
# Formal serving matrix. Requires the retained, manually accepted job-5952554 smoke.
# Submit from a clean project checkout with:
#   sbatch scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_formal.sbatch
#SBATCH --job-name=sgl-vllm-w4a16-formal
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=results/sglang-vllm-quarot-w4a16-serving/%x-%j.out
#SBATCH --error=results/sglang-vllm-quarot-w4a16-serving/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7
module load cuda/12.6
module load gcc-native/13.2
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

formal_spec="${project_root}/configs/deployment/sglang_vllm_quarot_w4a16_serving_formal_isambard.json"
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
smoke_result="${results_dir}/serving-client-smoke-5952554.json"
smoke_request_corpus="${results_dir}/serving-client-smoke-5952554-requests.json"
smoke_preflight="${results_dir}/serving-client-smoke-5952554-sglang-preflight.json"
smoke_manifest="${results_dir}/serving-client-smoke-5952554-source-manifest.txt"
output="${results_dir}/serving-formal-${SLURM_JOB_ID}.json"
request_corpus="${results_dir}/serving-formal-${SLURM_JOB_ID}-requests.json"
log_dir="${results_dir}/serving-formal-${SLURM_JOB_ID}-logs"
raw_dir="${results_dir}/serving-formal-${SLURM_JOB_ID}-raw"
source_manifest="${results_dir}/serving-formal-${SLURM_JOB_ID}-source-manifest.txt"
sglang_preflight="${results_dir}/serving-formal-${SLURM_JOB_ID}-sglang-preflight.json"
effective_config="${results_dir}/serving-formal-${SLURM_JOB_ID}-effective-config.json"
formal_input_record="${results_dir}/serving-formal-${SLURM_JOB_ID}-input-gate.json"

for required in \
    "${vllm_env}/bin/python" \
    "${vllm_env}/bin/vllm" \
    "${sglang_env}/bin/python" \
    "${sglang_env}/bin/ninja" \
    "${CUDA_HOME}/bin/nvcc" \
    "${CUDA_HOME}/lib64/libcudart.so" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving_formal.py" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_serving_formal_inputs.py" \
    "${formal_spec}" \
    "${model_snapshot}/tokenizer.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${legacy_raw}" \
    "${smoke_result}" \
    "${smoke_request_corpus}" \
    "${smoke_preflight}" \
    "${smoke_manifest}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SGLANG_VLLM_W4A16_FORMAL_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
for refused in \
    "${output}" \
    "${request_corpus}" \
    "${source_manifest}" \
    "${sglang_preflight}" \
    "${effective_config}" \
    "${formal_input_record}"; do
    if [[ -e "${refused}" ]]; then
        echo "SGLANG_VLLM_W4A16_FORMAL_REFUSES_EXISTING_OUTPUT=${refused}" >&2
        exit 1
    fi
done

git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SGLANG_VLLM_W4A16_FORMAL_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

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

echo "SGLANG_VLLM_W4A16_FORMAL_STAGE=accepted_smoke_gate"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_serving_formal_inputs.py" \
    --project-root "${project_root}" \
    --formal-config "${formal_spec}" \
    --smoke-result "${smoke_result}" \
    --smoke-request-corpus "${smoke_request_corpus}" \
    --smoke-preflight "${smoke_preflight}" \
    --smoke-source-manifest "${smoke_manifest}" \
    --output-config "${effective_config}" \
    --output-record "${formal_input_record}"

echo "SGLANG_VLLM_W4A16_FORMAL_STAGE=external_source_gate"
"${vllm_env}/bin/python" - \
    "${effective_config}" "${vllm_source_root}" "${sglang_source_root}" "${legacy_raw}" <<'PY'
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
print("SGLANG_VLLM_W4A16_FORMAL_EXTERNAL_SOURCES_ACCEPTED")
PY

echo "SGLANG_VLLM_W4A16_FORMAL_NINJA=$(ninja --version)"
echo "SGLANG_VLLM_W4A16_FORMAL_STAGE=runtime_preflight"
"${sglang_env}/bin/python" \
    "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
    --config "${effective_config}" \
    --output "${sglang_preflight}"

echo "SGLANG_VLLM_W4A16_FORMAL_STAGE=immutable_model_gate"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    --project-root "${project_root}" \
    --config "${effective_config}" \
    --source-result "${source_result}" \
    --model "${rotated_dir}"

echo "SGLANG_VLLM_W4A16_FORMAL_STAGE=paired_matrix"
"${vllm_env}/bin/python" \
    "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving_formal.py" \
    --config "${effective_config}" \
    --formal-input-record "${formal_input_record}" \
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
assert result["status"] == "passed", result
assert result["comparison_status"] == "formal_matrix_complete", result
assert result["project_revision"] == sys.argv[2], result
assert result["model"] == "quarot_w4a16", result
assert result["serving_smoke_gate"]["job_id"] == "5952554", result
assert result["request_corpus"]["sha256"] == "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1", result
assert len(result["repetitions"]) == 3, result
assert [item["backend_order"] for item in result["repetitions"]] == [
    ["vllm", "sglang"],
    ["sglang", "vllm"],
    ["vllm", "sglang"],
], result
for repetition in result["repetitions"]:
    assert set(repetition["cases"]) == {
        "vllm:latency_c1",
        "sglang:latency_c1",
        "vllm:throughput_c8",
        "sglang:throughput_c8",
    }, repetition
    assert repetition["comparison"]["status"] == "paired", repetition
    for case in repetition["cases"].values():
        assert case["status"] == "passed", case
        assert case["metrics"]["completed"] == 64, case
        assert case["metrics"]["failed"] == 0, case
        assert case["metrics"]["input_tokens"] == 64 * 256, case
        assert case["metrics"]["output_tokens"] == 64 * 64, case
assert result["aggregate"]["status"] == "formal_matrix_complete", result
print("ISAMBARD_SGLANG_VLLM_QUAROT_W4A16_SERVING_FORMAL_PASSED")
PY

{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'accepted_serving_smoke_job=5952554\n'
    printf 'source_gate_job=5769503\n'
    printf 'compatibility_gate_job=5941763\n'
    printf 'model=%s\n' "${rotated_dir}"
    printf 'formal_repetitions=3\n'
    printf 'formal_concurrency=1,8\n'
    sha256sum \
        "${formal_spec}" \
        "${project_root}/configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json" \
        "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving.py" \
        "${project_root}/scripts/run_sglang_vllm_quarot_w4a16_serving_formal.py" \
        "${project_root}/scripts/preflight_sglang_wna16_runtime.py" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_serving_formal_inputs.py" \
        "${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py" \
        "${project_root}/scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_smoke.sbatch" \
        "${project_root}/scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_formal.sbatch" \
        "${source_result}" \
        "${legacy_raw}" \
        "${smoke_result}" \
        "${smoke_request_corpus}" \
        "${smoke_preflight}" \
        "${smoke_manifest}" \
        "${effective_config}" \
        "${formal_input_record}" \
        "${sglang_preflight}" \
        "${request_corpus}" \
        "${output}"
    find "${log_dir}" "${raw_dir}" -type f -print0 | sort -z | xargs -0 sha256sum
} > "${source_manifest}"

echo "SGLANG_VLLM_W4A16_FORMAL_RESULT=${output}"
echo "SGLANG_VLLM_W4A16_FORMAL_SOURCE_MANIFEST=${source_manifest}"
