#!/bin/bash
# Matched performance measurements for the already-correct QuaRot BF16 and
# full-decoder W4A8 paths. Embedding, lm_head, and K/V remain BF16.
#SBATCH --job-name=w4a8-benchmark
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "W4A8_BENCHMARK_STAGE=module_load"
module load gcc-native/13.2 cray-python/3.11.7 cuda/12.6
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="${CXX}"

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv_path="${HOME}/.venvs/newsmallproject-w4a8"
results_dir="${project_root}/results/phase3-w4a8-benchmark"
checkpoint_manifest="results/phase2-w4a8-full-checkpoint/manifest.json"
correctness_result="results/phase2-w4a8-full-model/isambard-full-model-smoke.json"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST="9.0"

echo "W4A8_BENCHMARK_STAGE=inputs"
for required in \
    "${venv_path}/bin/python" \
    "${HF_HUB_CACHE}/models--meta-llama--Llama-2-13b-hf" \
    "${project_root}/${checkpoint_manifest}" \
    "${project_root}/${correctness_result}" \
    "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark_smoke.json" \
    "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "W4A8_BENCHMARK_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -n "$(git -C "${project_root}" status --porcelain)" ]]; then
    echo "W4A8_BENCHMARK_DIRTY_CHECKOUT=${project_root}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "W4A8_BENCHMARK_STAGE=source_manifest"
{
    printf 'git_revision='; git -C "${project_root}" rev-parse HEAD
    printf 'git_status\n'; git -C "${project_root}" status --short
    sha256sum \
        "${project_root}/repro/w4a8_benchmark.py" \
        "${project_root}/scripts/run_llama_w4a8_benchmark.py" \
        "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark_smoke.json" \
        "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark.json" \
        "${project_root}/${checkpoint_manifest}" \
        "${project_root}/${correctness_result}"
} > "${results_dir}/isambard-source-manifest.txt"

echo "W4A8_BENCHMARK_STAGE=preflight"
"${venv_path}/bin/python" "${project_root}/scripts/runpod_preflight.py" \
    --output "${results_dir}/isambard-preflight.json"

echo "W4A8_BENCHMARK_STAGE=smoke"
"${venv_path}/bin/python" "${project_root}/scripts/run_llama_w4a8_benchmark.py" \
    --config "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark_smoke.json" \
    --correctness-result "${correctness_result}" \
    --output "${results_dir}/isambard-benchmark-smoke.json"

echo "W4A8_BENCHMARK_STAGE=formal"
"${venv_path}/bin/python" "${project_root}/scripts/run_llama_w4a8_benchmark.py" \
    --config "${project_root}/configs/deployment/llama2_13b_w4a8_benchmark.json" \
    --correctness-result "${correctness_result}" \
    --output "${results_dir}/isambard-benchmark.json"

echo "W4A8_BENCHMARK_STAGE=assert_result"
"${venv_path}/bin/python" - "${results_dir}/isambard-benchmark.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "complete", result["status"]
assert set(result["modes"]) == {"bf16", "w4a8"}, result["modes"].keys()
assert result["modes"]["w4a8"]["installed_decoder_linears"] == 280
assert result["comparison"], result
print("ISAMBARD_PHASE3_W4A8_BENCHMARK_PASSED")
PY
