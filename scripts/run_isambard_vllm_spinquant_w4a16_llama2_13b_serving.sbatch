#!/bin/bash
# Run the formal matched BF16 versus SpinQuant W4A16 serving benchmark.
# The accepted export/load gate already exercised the exact packed checkpoint,
# so this fail-closed measurement intentionally has no smoke dependency.
#SBATCH --job-name=vllm-13b-spin-w4a16-serve
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm-spinquant-w4a16-serving-llama2-13b/%x-%j.out
#SBATCH --error=results/vllm-spinquant-w4a16-serving-llama2-13b/%x-%j.err

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
vllm_executable="${serving_env}/bin/vllm"
config="${project_root}/configs/deployment/vllm_spinquant_w4a16_llama2_13b_serving_isambard.json"
artifact_root="${VLLM_SPINQUANT_W4A16_ARTIFACT_ROOT:-${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-spinquant-w4a16}"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
spinquant_dir="${artifact_root}/spinquant-w4a16"
results_dir="${VLLM_SPINQUANT_W4A16_SERVING_RESULTS_DIR:-${project_root}/results/vllm-spinquant-w4a16-serving-llama2-13b}"
output="${results_dir}/serving-formal-${SLURM_JOB_ID}.json"
log_dir="${results_dir}/serving-formal-${SLURM_JOB_ID}-logs"
raw_result_dir="${results_dir}/serving-formal-${SLURM_JOB_ID}-raw"
evaluation_manifest="${results_dir}/serving-source-manifest-${SLURM_JOB_ID}.txt"

mapfile -t source_paths < <("${serving_env}/bin/python" - "${config}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["source_gate"]
if gate.get("status") != "accepted":
    raise SystemExit("SpinQuant W4A16 source gate is not accepted")
print(gate["export_result"])
print(gate["inference_result"])
print(gate["source_manifest"])
print(gate["stdout"])
PY
)
export_result="${project_root}/${source_paths[0]}"
inference_result="${project_root}/${source_paths[1]}"
source_manifest="${project_root}/${source_paths[2]}"
source_stdout="${project_root}/${source_paths[3]}"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-vllm-spinquant-w4a16-serving-${SLURM_JOB_ID}"
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

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${vllm_executable}" \
    "${config}" \
    "${export_result}" \
    "${inference_result}" \
    "${source_manifest}" \
    "${source_stdout}" \
    "${model_snapshot}/config.json" \
    "${model_snapshot}/tokenizer.json" \
    "${spinquant_dir}/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_SPINQUANT_W4A16_SERVING_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
for refused in "${output}" "${log_dir}" "${raw_result_dir}" "${evaluation_manifest}"; do
    if [[ -e "${refused}" ]]; then
        echo "VLLM_SPINQUANT_W4A16_SERVING_REFUSES_EXISTING_OUTPUT=${refused}" >&2
        exit 1
    fi
done
git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_SPINQUANT_W4A16_SERVING_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=source_gate"
"${serving_env}/bin/python" \
    "${project_root}/scripts/validate_vllm_spinquant_w4a16_serving_inputs.py" \
    --project-root "${project_root}" \
    --config "${config}" \
    --export-result "${export_result}" \
    --inference-result "${inference_result}" \
    --source-manifest "${source_manifest}" \
    --stdout "${source_stdout}" \
    --model "bf16=${model_snapshot}" \
    --model "spinquant_w4a16=${spinquant_dir}"

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_gate_job=5945162\n'
    printf 'execution_kind=formal_only\n'
    sha256sum \
        "${config}" \
        "${project_root}/repro/vllm_w4a16.py" \
        "${project_root}/scripts/run_vllm_w4a16_llama2_13b_serving.py" \
        "${project_root}/scripts/run_vllm_spinquant_w4a16_llama2_13b_serving.py" \
        "${project_root}/scripts/validate_vllm_spinquant_w4a16_boolq_inputs.py" \
        "${project_root}/scripts/validate_vllm_spinquant_w4a16_serving_inputs.py" \
        "${project_root}/scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_serving.sbatch" \
        "${export_result}" \
        "${inference_result}" \
        "${source_manifest}" \
        "${source_stdout}"
} > "${evaluation_manifest}"

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=runtime"
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
print("VLLM_SPINQUANT_W4A16_SERVING_RUNTIME_ACCEPTED")
PY

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=benchmark"
"${serving_env}/bin/python" \
    "${project_root}/scripts/run_vllm_spinquant_w4a16_llama2_13b_serving.py" \
    --mode benchmark \
    --config "${config}" \
    --vllm-executable "${vllm_executable}" \
    --tokenizer "${model_snapshot}" \
    --model "bf16=${model_snapshot}" \
    --model "spinquant_w4a16=${spinquant_dir}" \
    --log-dir "${log_dir}" \
    --raw-result-dir "${raw_result_dir}" \
    --output "${output}"

echo "VLLM_SPINQUANT_W4A16_SERVING_STAGE=assert_result"
"${serving_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import math
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert result["project_revision"] == sys.argv[2], result
assert tuple(result["models"]) == ("bf16", "spinquant_w4a16"), result
expected_cases = {"latency_c1", "throughput_c8"}
for name, cases in result["models"].items():
    assert set(cases) == expected_cases, cases
    for case in cases.values():
        assert case["metrics"]["request_throughput"] > 0, case
        assert case["metrics"]["output_throughput"] > 0, case
        assert case["peak_gpu_memory_used_mib"] > 0, case
        assert all(math.isfinite(value) for value in case["metrics"].values())
        if name == "spinquant_w4a16":
            assert case["kernel_evidence"]["matched"] is True, case
for case in expected_cases:
    comparison = result["comparisons"][case]["spinquant_w4a16_vs_bf16"]
    assert all(math.isfinite(value) and value > 0 for value in comparison.values())
print("ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_SERVING_FORMAL_PASSED")
PY
