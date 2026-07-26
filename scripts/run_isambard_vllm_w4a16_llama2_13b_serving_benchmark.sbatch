#!/bin/bash
# Run the formal matched serving benchmark only after the service smoke passes.
# Submit with: sbatch --dependency=afterok:<smoke-job-id> <this-script>
#SBATCH --job-name=vllm-13b-serving-bench
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/vllm-w4a16-serving-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4a16-serving-llama2-13b/%x-%j.err

set -euo pipefail

echo "VLLM_13B_BENCHMARK_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
config="${project_root}/configs/deployment/vllm_w4a16_llama2_13b_serving_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
unrotated_dir="${artifact_root}/unrotated-w4a16"
rotated_dir="${artifact_root}/rotated-w4a16"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
source_result="${project_root}/results/vllm-w4a16-llama2-13b/offline-inference-retry.json"
results_dir="${project_root}/results/vllm-w4a16-serving-llama2-13b"
smoke_result="${results_dir}/service-smoke.json"
output="${results_dir}/serving-benchmark.json"
log_dir="${results_dir}/serving-benchmark-logs"
raw_result_dir="${results_dir}/serving-benchmark-raw"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_13B_BENCHMARK_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${serving_env}/bin/vllm" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${model_snapshot}/tokenizer.json" \
    "${unrotated_dir}/config.json" \
    "${rotated_dir}/config.json" \
    "${source_result}" \
    "${smoke_result}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_13B_BENCHMARK_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_13B_BENCHMARK_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}" "${log_dir}" "${raw_result_dir}"

echo "VLLM_13B_BENCHMARK_STAGE=smoke_dependency"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_serving_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --smoke-result "${smoke_result}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --model "rotated_w4a16=${rotated_dir}"

echo "VLLM_13B_BENCHMARK_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5769503\n'
    printf 'slurm_dependency=%s\n' "${SLURM_JOB_DEPENDENCY:-unset}"
    printf 'artifact_root=%s\n' "${artifact_root}"
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_isambard_vllm_w4a16_llama2_13b_serving_benchmark.sbatch" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_serving.py" \
        "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_serving_inputs.py" \
        "${source_result}" \
        "${smoke_result}"
} > "${results_dir}/serving-benchmark-source-manifest.txt"

echo "VLLM_13B_BENCHMARK_STAGE=runtime"
"${serving_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import torch

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert metadata.version("vllm") == "0.25.1+cu129"
assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("VLLM_13B_BENCHMARK_RUNTIME_ACCEPTED")
PY

echo "VLLM_13B_BENCHMARK_STAGE=matched_serving"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4a16_llama2_13b_serving.py" \
    --mode benchmark \
    --config "${config}" \
    --vllm-executable "${serving_env}/bin/vllm" \
    --tokenizer "${model_snapshot}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --model "rotated_w4a16=${rotated_dir}" \
    --log-dir "${log_dir}" \
    --raw-result-dir "${raw_result_dir}" \
    --output "${output}"

echo "VLLM_13B_BENCHMARK_STAGE=assert_result"
"${serving_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
expected_models = {"bf16", "unrotated_w4a16", "rotated_w4a16"}
expected_cases = {"latency_c1", "throughput_c8"}
assert set(result["models"]) == expected_models
for model in result["models"].values():
    assert set(model) == expected_cases, model
    for case in model.values():
        assert case["metrics"]["request_throughput"] > 0, case
        assert case["metrics"]["output_throughput"] > 0, case
        assert case["peak_gpu_memory_used_mib"] > 0, case
        assert all(math.isfinite(value) for value in case["metrics"].values())
print("ISAMBARD_VLLM_W4A16_LLAMA2_13B_SERVING_BENCHMARK_PASSED")
PY
