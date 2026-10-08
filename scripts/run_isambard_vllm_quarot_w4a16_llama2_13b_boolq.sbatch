#!/bin/bash
# Run BF16 and QuaRot-style rotated W4A16 on the full frozen BoolQ split.
# Submit only in explicit formal mode:
#   sbatch scripts/run_isambard_vllm_quarot_w4a16_llama2_13b_boolq.sbatch formal
#SBATCH --job-name=vllm-13b-quarot-w4a16-boolq
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm-quarot-w4a16-boolq-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-quarot-w4a16-boolq-llama2-13b/%x-%j.err

set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 formal" >&2
    exit 2
fi

module load cray-python/3.11.7
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
config="${project_root}/configs/deployment/vllm_quarot_w4a16_llama2_13b_boolq_isambard.json"
artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
dataset_root="${BOOLQ_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/boolq-lm-eval-v1-3de24cf}"
dataset_manifest="${dataset_root}/manifest.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
rotated_dir="${artifact_root}/rotated-w4a16"
source_result="${W4A16_SOURCE_RESULT:-${HOME}/NewSmallProject-vllm-ready/results/vllm-w4a16-llama2-13b/offline-inference-retry.json}"
results_dir="${VLLM_QUAROT_W4A16_BOOLQ_RESULTS_DIR:-${project_root}/results/vllm-quarot-w4a16-boolq-llama2-13b}"
output="${results_dir}/boolq-formal-${SLURM_JOB_ID}.json"
evaluation_manifest="${results_dir}/boolq-source-manifest-${SLURM_JOB_ID}.txt"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-quarot-w4a16-boolq-${SLURM_JOB_ID}"
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

echo "VLLM_QUAROT_W4A16_BOOLQ_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${dataset_manifest}" \
    "${source_result}" \
    "${model_snapshot}/config.json" \
    "${model_snapshot}/tokenizer.json" \
    "${rotated_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_QUAROT_W4A16_BOOLQ_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" || -e "${evaluation_manifest}" ]]; then
    echo "VLLM_QUAROT_W4A16_BOOLQ_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_QUAROT_W4A16_BOOLQ_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_QUAROT_W4A16_BOOLQ_STAGE=source_gate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --model "${rotated_dir}"

echo "VLLM_QUAROT_W4A16_BOOLQ_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5769503\n'
    printf 'serving_reference_job=5960180\n'
    sha256sum \
        "${config}" \
        "${project_root}/repro/vllm_w4a16.py" \
        "${project_root}/repro/vllm_w4afp8.py" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_ppl.py" \
        "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_boolq.py" \
        "${project_root}/scripts/run_vllm_quarot_w4a16_llama2_13b_boolq.py" \
        "${project_root}/scripts/validate_sglang_vllm_quarot_w4a16_inputs.py" \
        "${project_root}/scripts/run_isambard_vllm_quarot_w4a16_llama2_13b_boolq.sbatch" \
        "${dataset_manifest}" \
        "${source_result}"
} > "${evaluation_manifest}"

echo "VLLM_QUAROT_W4A16_BOOLQ_STAGE=evaluate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_quarot_w4a16_llama2_13b_boolq.py" \
    --config "${config}" \
    --dataset-manifest "${dataset_manifest}" \
    --tokenizer-model "${model_snapshot}" \
    --model "bf16=${model_snapshot}" \
    --model "quarot_w4a16=${rotated_dir}" \
    --max-examples 3270 \
    --output "${output}"

"${serving_env}/bin/python" - "${output}" "$(git rev-parse HEAD)" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["evaluated_examples"] == 3270, result
assert result["evaluated_requests"] == 6540, result
assert tuple(result["models"]) == ("bf16", "quarot_w4a16"), result
for model in result["models"].values():
    assert model["evaluated_examples"] == 3270, model
    assert model["evaluated_requests"] == 6540, model
paired = result["comparisons"]["paired_correctness"]
assert paired["prediction_disagreements"] == (
    paired["bf16_only_correct"] + paired["quarot_w4a16_only_correct"]
), paired
assert math.isfinite(paired["mcnemar_exact_two_sided_p"]), paired
print("ISAMBARD_VLLM_QUAROT_W4A16_LLAMA2_13B_BOOLQ_PASSED")
PY

echo "VLLM_QUAROT_W4A16_BOOLQ_RESULT=${output}"
echo "VLLM_QUAROT_W4A16_BOOLQ_SOURCE_MANIFEST=${evaluation_manifest}"
