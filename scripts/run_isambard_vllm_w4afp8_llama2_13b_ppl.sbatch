#!/bin/bash
# Prepared only. The config source_gate must be frozen after the deployment gate.
# Run later as smoke, then formal with afterok dependency.
#SBATCH --job-name=vllm-13b-w4afp8-ppl
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/vllm-w4afp8-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4afp8-llama2-13b/%x-%j.err

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
config="${VLLM_W4AFP8_PPL_CONFIG:-${project_root}/configs/deployment/vllm_w4afp8_llama2_13b_ppl_isambard.json}"
artifact_root="${VLLM_W4AFP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8}"
w4a16_artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
tokens_manifest="${w4a16_artifact_root}/ppl-wikitext2-test-162x2048/manifest.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
unrotated_dir="${artifact_root}/unrotated-w4afp8"
quarot_dir="${artifact_root}/quarot-w4afp8"
spinquant_dir="${VLLM_W4AFP8_SPINQUANT_CHECKPOINT_DIR:-${artifact_root}/spinquant-w4afp8-int8-transfer}"
results_dir="${VLLM_W4AFP8_RESULTS_DIR:-${project_root}/results/vllm-w4afp8-llama2-13b}"
smoke_result="${results_dir}/deployed-ppl-smoke.json"
formal_result="${results_dir}/deployed-ppl-formal.json"

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
    max_sequences=2
    expected_tokens=4094
else
    output="${formal_result}"
    max_sequences=162
    expected_tokens=331614
fi

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-w4afp8-ppl-${SLURM_JOB_ID}"
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

echo "VLLM_W4AFP8_PPL_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${config}" \
    "${tokens_manifest}" \
    "${source_result}" \
    "${capability_result}" \
    "${model_snapshot}/config.json" \
    "${unrotated_dir}/config.json" \
    "${quarot_dir}/config.json" \
    "${spinquant_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_W4AFP8_PPL_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" ]]; then
    echo "VLLM_W4AFP8_PPL_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_W4AFP8_PPL_DIRTY_CHECKOUT" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_W4AFP8_PPL_STAGE=source_gate"
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
        echo "VLLM_W4AFP8_PPL_FORMAL_REQUIRES_AFTEROK" >&2
        exit 1
    fi
    "${serving_env}/bin/python" - "${smoke_result}" <<'PY'
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert smoke["status"] == "passed", smoke
assert smoke["evaluated_sequences"] == 2, smoke
assert smoke["scored_tokens"] == 4094, smoke
assert set(smoke["models"]) == {
    "bf16", "unrotated_w4afp8", "quarot_w4afp8", "spinquant_w4afp8"
}, smoke
print("VLLM_W4AFP8_PPL_SMOKE_ACCEPTED")
PY
fi

echo "VLLM_W4AFP8_PPL_STAGE=evaluate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_ppl.py" \
    --config "${config}" \
    --tokens-manifest "${tokens_manifest}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_w4afp8=${unrotated_dir}" \
    --model "quarot_w4afp8=${quarot_dir}" \
    --model "spinquant_w4afp8=${spinquant_dir}" \
    --max-sequences "${max_sequences}" \
    --output "${output}"

"${serving_env}/bin/python" - "${output}" "${expected_tokens}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["scored_tokens"] == int(sys.argv[2]), result
assert set(result["models"]) == {
    "bf16", "unrotated_w4afp8", "quarot_w4afp8", "spinquant_w4afp8"
}, result
print("ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_PPL_PASSED")
PY
