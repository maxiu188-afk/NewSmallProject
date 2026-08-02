#!/bin/bash
# Run smoke first. Submit formal only after reviewing the smoke artifact, with:
# sbatch --dependency=afterok:<SMOKE_JOB_ID> scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch formal
#SBATCH --job-name=vllm-13b-w4afp8-boolq
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/vllm-w4afp8-boolq-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4afp8-boolq-llama2-13b/%x-%j.err

set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 {smoke|formal}" >&2
    exit 2
fi

module load cray-python/3.11.7
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
config="${project_root}/configs/deployment/vllm_w4afp8_llama2_13b_boolq_isambard.json"
artifact_root="${VLLM_W4AFP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8}"
dataset_root="${BOOLQ_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/boolq-lm-eval-v1-3de24cf}"
dataset_manifest="${dataset_root}/manifest.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
unrotated_dir="${artifact_root}/unrotated-w4afp8"
quarot_dir="${artifact_root}/quarot-w4afp8"
spinquant_dir="${artifact_root}/spinquant-w4afp8-int8-transfer"
results_dir="${project_root}/results/vllm-w4afp8-boolq-llama2-13b"
smoke_result="${results_dir}/boolq-smoke.json"
formal_result="${results_dir}/boolq-formal.json"

mapfile -t source_paths < <("${serving_env}/bin/python" - "${config}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["source_gate"]
if gate.get("status") != "accepted":
    raise SystemExit("W4AFP8 source gate is still pending")
print(gate["result"])
print(gate["capability_result"])
PY
)
source_result="${project_root}/${source_paths[0]}"
capability_result="${project_root}/${source_paths[1]}"

if [[ "${mode}" == "smoke" ]]; then
    output="${smoke_result}"
    max_examples=32
    expected_requests=64
else
    output="${formal_result}"
    max_examples=3270
    expected_requests=6540
fi

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-w4afp8-boolq-${SLURM_JOB_ID}"
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

echo "VLLM_W4AFP8_BOOLQ_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${dataset_manifest}" \
    "${source_result}" \
    "${capability_result}" \
    "${model_snapshot}/config.json" \
    "${model_snapshot}/tokenizer.json" \
    "${unrotated_dir}/config.json" \
    "${quarot_dir}/config.json" \
    "${spinquant_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_W4AFP8_BOOLQ_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" ]]; then
    echo "VLLM_W4AFP8_BOOLQ_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_W4AFP8_BOOLQ_DIRTY_CHECKOUT" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_W4AFP8_BOOLQ_STAGE=source_gate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_vllm_w4afp8_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --source-result "${source_result}" \
    --capability-result "${capability_result}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4afp8=${unrotated_dir}" \
    --model "quarot_w4afp8=${quarot_dir}" \
    --model "spinquant_w4afp8=${spinquant_dir}"

if [[ "${mode}" == "formal" ]]; then
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "VLLM_W4AFP8_BOOLQ_FORMAL_REQUIRES_AFTEROK" >&2
        exit 1
    fi
    "${serving_env}/bin/python" - "${smoke_result}" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert smoke["status"] == "passed", smoke
assert smoke["evaluated_examples"] == 32, smoke
assert smoke["evaluated_requests"] == 64, smoke
assert smoke["project_revision"] == sys.argv[2], smoke
assert set(smoke["models"]) == {
    "bf16", "unrotated_w4afp8", "quarot_w4afp8", "spinquant_w4afp8"
}, smoke
print("VLLM_W4AFP8_BOOLQ_SMOKE_ACCEPTED")
PY
fi

echo "VLLM_W4AFP8_BOOLQ_STAGE=evaluate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_boolq.py" \
    --config "${config}" \
    --dataset-manifest "${dataset_manifest}" \
    --tokenizer-model "${model_snapshot}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4afp8=${unrotated_dir}" \
    --model "quarot_w4afp8=${quarot_dir}" \
    --model "spinquant_w4afp8=${spinquant_dir}" \
    --max-examples "${max_examples}" \
    --output "${output}"

"${serving_env}/bin/python" - "${output}" "${max_examples}" "${expected_requests}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["evaluated_examples"] == int(sys.argv[2]), result
assert result["evaluated_requests"] == int(sys.argv[3]), result
assert set(result["models"]) == {
    "bf16", "unrotated_w4afp8", "quarot_w4afp8", "spinquant_w4afp8"
}, result
for model in result["models"].values():
    assert model["evaluated_examples"] == int(sys.argv[2]), model
    assert model["evaluated_requests"] == int(sys.argv[3]), model
print("ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_BOOLQ_PASSED")
PY
