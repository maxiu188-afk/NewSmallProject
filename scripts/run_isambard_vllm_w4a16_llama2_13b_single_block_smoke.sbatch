#!/bin/bash
# Validate the real vLLM layer-0 hook with one packed W4A16 checkpoint.
# Submit only after the accepted full-model serving benchmark is present.
#SBATCH --job-name=vllm-13b-block-smoke
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm-w4a16-single-block-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4a16-single-block-llama2-13b/%x-%j.err

set -euo pipefail

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
config="${project_root}/configs/deployment/vllm_w4a16_llama2_13b_single_block_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
unrotated_dir="${artifact_root}/unrotated-w4a16"
rotated_dir="${artifact_root}/rotated-w4a16"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
source_result="${project_root}/results/vllm-w4a16-serving-llama2-13b/serving-benchmark.json"
results_dir="${project_root}/results/vllm-w4a16-single-block-llama2-13b"
worker_dir="${results_dir}/smoke-workers"
output="${results_dir}/single-block-smoke.json"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${model_snapshot}/config.json" \
    "${unrotated_dir}/config.json" \
    "${rotated_dir}/config.json" \
    "${source_result}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_13B_SINGLE_BLOCK_SMOKE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_13B_SINGLE_BLOCK_SMOKE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}" "${worker_dir}"

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=source_gate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_single_block_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --model "rotated_w4a16=${rotated_dir}"

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'serving_smoke_job=5780629\n'
    printf 'serving_benchmark_job=5780631\n'
    printf 'artifact_root=%s\n' "${artifact_root}"
    sha256sum \
        "${config}" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_single_block.py" \
        "${project_root}/scripts/validate_vllm_w4a16_llama2_13b_single_block_inputs.py" \
        "${project_root}/scripts/run_isambard_vllm_w4a16_llama2_13b_single_block_smoke.sbatch" \
        "${source_result}"
} > "${results_dir}/single-block-smoke-source-manifest.txt"

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=layer_hook"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4a16_llama2_13b_single_block.py" \
    --config "${config}" \
    --protocol smoke \
    --model "unrotated_w4a16=${unrotated_dir}" \
    --worker-dir "${worker_dir}" \
    --output "${output}"

echo "VLLM_13B_SINGLE_BLOCK_SMOKE_STAGE=assert_result"
"${serving_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["protocol"] == "smoke", result
assert set(result["models"]) == {"unrotated_w4a16"}, result
model = result["models"]["unrotated_w4a16"]
assert model["layer_inspection"]["packed_parameter_count"] > 0, model
assert set(model["cases"]) == {
    "smoke-prefill-b1-s128",
    "smoke-decode-b1-s128-d2",
}
assert model["cases"]["smoke-prefill-b1-s128"]["layer_call_elapsed_ms"]["count"] == 2
assert model["cases"]["smoke-decode-b1-s128-d2"]["layer_call_elapsed_ms"]["count"] == 4
print("ISAMBARD_VLLM_W4A16_LLAMA2_13B_SINGLE_BLOCK_SMOKE_PASSED")
PY
