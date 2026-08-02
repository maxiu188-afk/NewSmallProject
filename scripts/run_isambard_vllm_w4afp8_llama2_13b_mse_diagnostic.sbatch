#!/bin/bash
# Isolated W4AFP8 min/max versus MSE observer diagnostic.
# Submit gate first, then submit smoke with afterok on the gate. Do not submit
# formal until the smoke result has been reviewed and accepted.
#SBATCH --job-name=vllm-13b-w4afp8-mse
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/vllm-w4afp8-mse-diagnostic/%x-%j.out
#SBATCH --error=results/vllm-w4afp8-mse-diagnostic/%x-%j.err

set -euo pipefail

mode="${1:-}"
gate_job_id="${2:-}"
if [[ "${mode}" != "gate" && "${mode}" != "smoke" && "${mode}" != "formal" ]]; then
    echo "usage: sbatch $0 {gate|smoke|formal} [gate-job-id]" >&2
    exit 2
fi
if [[ "${mode}" != "gate" && ! "${gate_job_id}" =~ ^[0-9]+$ ]]; then
    echo "${mode} requires the accepted gate job id" >&2
    exit 2
fi

module load cray-python/3.11.7
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
config="${project_root}/configs/deployment/vllm_w4afp8_llama2_13b_mse_diagnostic_isambard.json"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
quantizer_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
base_artifact_root="${VLLM_W4AFP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8}"
diagnostic_root="${VLLM_W4AFP8_MSE_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8-mse-diagnostic}"
calibration_dir="${base_artifact_root}/calibration-128x2048"
unrotated_minmax_dir="${base_artifact_root}/unrotated-w4afp8"
quarot_minmax_dir="${base_artifact_root}/quarot-w4afp8"
unrotated_mse_dir="${diagnostic_root}/unrotated-mse-w4afp8"
quarot_mse_dir="${diagnostic_root}/quarot-mse-w4afp8"
w4a16_artifact_root="${VLLM_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4a16}"
tokens_manifest="${w4a16_artifact_root}/ppl-wikitext2-test-162x2048/manifest.json"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
results_dir="${project_root}/results/vllm-w4afp8-mse-diagnostic"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-w4afp8-mse-${SLURM_JOB_ID}"
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_W4AFP8_MSE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${diagnostic_root}" "${results_dir}"

required=(
    "${config}"
    "${serving_env}/bin/python"
    "${quantizer_env}/bin/python"
    "${calibration_dir}/manifest.json"
    "${calibration_dir}/dataset"
    "${model_snapshot}/config.json"
    "${unrotated_minmax_dir}/config.json"
    "${quarot_minmax_dir}/config.json"
)
if [[ "${mode}" != "gate" ]]; then
    required+=("${tokens_manifest}" "${unrotated_mse_dir}/config.json" "${quarot_mse_dir}/config.json")
fi
for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "VLLM_W4AFP8_MSE_MISSING_INPUT=${path}" >&2
        exit 1
    fi
done

if [[ "${mode}" == "gate" ]]; then
    echo "VLLM_W4AFP8_MSE_STAGE=export"
    validate_or_export() {
        local variant="$1"
        local checkpoint="$2"
        local report="$3"
        if [[ -f "${report}" && -f "${checkpoint}/config.json" ]]; then
            "${quantizer_env}/bin/python" - "${report}" "${git_revision}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "passed", report
assert report["project_revision"] == sys.argv[2], report
assert report["variant"] == sys.argv[3], report
assert report["checkpoint_weight_observer"] in {"mse", "memoryless_mse"}, report
assert report["packed_decoder_linear_count"] == 280, report
assert report["runtime_g_idx_tensors"] == 0, report
print(f"VLLM_W4AFP8_MSE_REUSING_EXPORT={sys.argv[3]}")
PY
            return
        fi
        if [[ -e "${checkpoint}" || -e "${checkpoint}.incomplete" || -e "${report}" ]]; then
            echo "VLLM_W4AFP8_MSE_PARTIAL_EXPORT=${variant}" >&2
            exit 1
        fi
        "${quantizer_env}/bin/python" \
            "${project_root}/scripts/export_vllm_w4afp8_llama2_13b_mse_diagnostic.py" \
            --config "${config}" \
            --calibration-dir "${calibration_dir}" \
            --variant "${variant}" \
            --output-dir "${checkpoint}" \
            --report "${report}"
    }

    validate_or_export \
        unrotated_mse_w4afp8 \
        "${unrotated_mse_dir}" \
        "${results_dir}/unrotated-mse-export.json"
    validate_or_export \
        quarot_mse_w4afp8 \
        "${quarot_mse_dir}" \
        "${results_dir}/quarot-mse-export.json"

    echo "VLLM_W4AFP8_MSE_STAGE=load_gate"
    gate_result="${results_dir}/mse-offline-inference-${SLURM_JOB_ID}.json"
    "${serving_env}/bin/python" \
        "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_gate.py" \
        --variant-set quarot \
        --config "${config}" \
        --model "bf16=${model_snapshot}" \
        --model "unrotated_w4afp8=${unrotated_mse_dir}" \
        --model "quarot_w4afp8=${quarot_mse_dir}" \
        --output "${gate_result}"
    "${serving_env}/bin/python" - "${gate_result}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert set(result["models"]) == {"bf16", "unrotated_w4afp8", "quarot_w4afp8"}, result
for name in ("unrotated_w4afp8", "quarot_w4afp8"):
    checkpoint = result["models"][name]["checkpoint"]
    weights = next(iter(checkpoint["quantization_config"]["config_groups"].values()))["weights"]
    assert weights["observer"] in {"mse", "memoryless_mse"}, weights
    assert checkpoint["packed_decoder_linear_count"] == 280, checkpoint
    assert checkpoint["runtime_g_idx_tensors"] == 0, checkpoint
print("ISAMBARD_VLLM_W4AFP8_MSE_GATE_PASSED")
PY
    exit 0
fi

gate_result="${results_dir}/mse-offline-inference-${gate_job_id}.json"
for path in \
    "${gate_result}" \
    "${results_dir}/unrotated-mse-export.json" \
    "${results_dir}/quarot-mse-export.json"; do
    if [[ ! -f "${path}" ]]; then
        echo "VLLM_W4AFP8_MSE_MISSING_GATE_RESULT=${path}" >&2
        exit 1
    fi
done
"${serving_env}/bin/python" - "${gate_result}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
print("VLLM_W4AFP8_MSE_GATE_ACCEPTED")
PY

if [[ "${mode}" == "smoke" ]]; then
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "VLLM_W4AFP8_MSE_SMOKE_REQUIRES_AFTEROK" >&2
        exit 1
    fi
    output="${results_dir}/deployed-ppl-smoke-${gate_job_id}.json"
    max_sequences=2
    expected_tokens=4094
else
    output="${results_dir}/deployed-ppl-formal-${gate_job_id}.json"
    max_sequences=162
    expected_tokens=331614
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "VLLM_W4AFP8_MSE_FORMAL_REQUIRES_AFTEROK" >&2
        exit 1
    fi
    smoke_result="${results_dir}/deployed-ppl-smoke-${gate_job_id}.json"
    "${serving_env}/bin/python" - "${smoke_result}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert result["evaluated_sequences"] == 2, result
assert result["scored_tokens"] == 4094, result
print("VLLM_W4AFP8_MSE_PPL_SMOKE_ACCEPTED")
PY
fi
if [[ -e "${output}" ]]; then
    echo "VLLM_W4AFP8_MSE_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi

echo "VLLM_W4AFP8_MSE_STAGE=${mode}"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_mse_ppl.py" \
    --config "${config}" \
    --tokens-manifest "${tokens_manifest}" \
    --model "bf16=${model_snapshot}" \
    --model "unrotated_minmax_w4afp8=${unrotated_minmax_dir}" \
    --model "quarot_minmax_w4afp8=${quarot_minmax_dir}" \
    --model "unrotated_mse_w4afp8=${unrotated_mse_dir}" \
    --model "quarot_mse_w4afp8=${quarot_mse_dir}" \
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
    "bf16",
    "unrotated_minmax_w4afp8",
    "quarot_minmax_w4afp8",
    "unrotated_mse_w4afp8",
    "quarot_mse_w4afp8",
}, result
print("ISAMBARD_VLLM_W4AFP8_MSE_PPL_PASSED")
PY
