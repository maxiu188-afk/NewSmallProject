#!/bin/bash
# Prepared only. Freeze the accepted source gate before submitting smoke/benchmark.
#SBATCH --job-name=vllm-13b-w4afp8-serve
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=results/vllm-w4afp8-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-w4afp8-llama2-13b/%x-%j.err

set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "smoke" && "${mode}" != "benchmark" ]]; then
    echo "usage: sbatch $0 {smoke|benchmark}" >&2
    exit 2
fi

module load cray-python/3.11.7
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
vllm_executable="${serving_env}/bin/vllm"
config="${VLLM_W4AFP8_SERVING_CONFIG:-${project_root}/configs/deployment/vllm_w4afp8_llama2_13b_serving_isambard.json}"
artifact_root="${VLLM_W4AFP8_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
unrotated_dir="${artifact_root}/unrotated-w4afp8"
quarot_dir="${artifact_root}/quarot-w4afp8"
spinquant_dir="${VLLM_W4AFP8_SPINQUANT_CHECKPOINT_DIR:-${artifact_root}/spinquant-w4afp8-int8-transfer}"
results_dir="${VLLM_W4AFP8_RESULTS_DIR:-${project_root}/results/vllm-w4afp8-llama2-13b}"
smoke_result="${results_dir}/service-smoke.json"
benchmark_result="${results_dir}/serving-benchmark.json"

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
    log_dir="${results_dir}/service-smoke-logs"
else
    output="${benchmark_result}"
    log_dir="${results_dir}/serving-benchmark-logs"
fi
raw_result_dir="${results_dir}/serving-benchmark-raw"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-w4afp8-serving-${SLURM_JOB_ID}"
mkdir -p "${SCRATCHDIR}"
export TMPDIR="${SCRATCHDIR}"
export TMP="${SCRATCHDIR}"
export TEMP="${SCRATCHDIR}"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_W4AFP8_SERVING_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${vllm_executable}" \
    "${config}" \
    "${source_result}" \
    "${capability_result}" \
    "${model_snapshot}/config.json" \
    "${unrotated_dir}/config.json" \
    "${quarot_dir}/config.json" \
    "${spinquant_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_W4AFP8_SERVING_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" || -e "${log_dir}" ]]; then
    echo "VLLM_W4AFP8_SERVING_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_W4AFP8_SERVING_DIRTY_CHECKOUT" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_W4AFP8_SERVING_STAGE=source_gate"
validator=(
    "${serving_env}/bin/python"
    "${project_root}/scripts/validate_vllm_w4afp8_inputs.py"
    --project-root "${project_root}"
    --config "${config}"
    --source-result "${source_result}"
    --capability-result "${capability_result}"
    --model "bf16=${model_snapshot}"
    --model "unrotated_w4afp8=${unrotated_dir}"
    --model "quarot_w4afp8=${quarot_dir}"
    --model "spinquant_w4afp8=${spinquant_dir}"
)
if [[ "${mode}" == "benchmark" ]]; then
    if [[ "${SLURM_JOB_DEPENDENCY:-}" != afterok:* ]]; then
        echo "VLLM_W4AFP8_BENCHMARK_REQUIRES_AFTEROK" >&2
        exit 1
    fi
    validator+=(--smoke-result "${smoke_result}")
fi
"${validator[@]}"

echo "VLLM_W4AFP8_SERVING_STAGE=${mode}"
runner=(
    "${serving_env}/bin/python"
    "${project_root}/scripts/run_vllm_w4afp8_llama2_13b_serving.py"
    --mode "${mode}"
    --config "${config}"
    --vllm-executable "${vllm_executable}"
    --tokenizer "${model_snapshot}"
    --model "bf16=${model_snapshot}"
    --model "unrotated_w4afp8=${unrotated_dir}"
    --model "quarot_w4afp8=${quarot_dir}"
    --model "spinquant_w4afp8=${spinquant_dir}"
    --log-dir "${log_dir}"
    --output "${output}"
)
if [[ "${mode}" == "benchmark" ]]; then
    runner+=(--raw-result-dir "${raw_result_dir}")
fi
"${runner[@]}"

"${serving_env}/bin/python" - "${output}" "${mode}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
expected = {
    "bf16", "unrotated_w4afp8", "quarot_w4afp8", "spinquant_w4afp8"
}
assert result["status"] == "passed", result
assert set(result["models"]) == expected, result
if mode == "smoke":
    for name in expected - {"bf16"}:
        assert result["models"][name]["kernel_evidence"]["matched"], result
else:
    for name in expected:
        for case in ("latency_c1", "throughput_c8"):
            metrics = result["models"][name][case]["metrics"]
            assert metrics["request_throughput"] > 0, metrics
    for name in expected - {"bf16"}:
        for case in ("latency_c1", "throughput_c8"):
            assert result["models"][name][case]["kernel_evidence"]["matched"], result
print(f"ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_{mode.upper()}_PASSED")
PY
