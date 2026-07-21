#!/bin/bash
# Formal Isambard-AI correctness gate for one GPTQ-exported Llama q_proj.
#
# Scope: one packed W4/A8 q_proj is compared with its independent packed oracle
# inside Llama-2-13B. All other linears and K/V remain BF16. This does not
# evaluate a packed full model, PPL, KV4, or performance.
#SBATCH --job-name=w4a8-layer-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "W4A8_LAYER_GATE_STAGE=module_load"
module load cray-python/3.11.7 cuda/12.6

# Slurm copies the batch script to its spool directory before executing it, so
# BASH_SOURCE no longer identifies the checkout.  SLURM_SUBMIT_DIR retains the
# directory from which `sbatch` was invoked; keep the source-relative form only
# for direct local execution.
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv_path="${HOME}/.venvs/newsmallproject-w4a8"
results_dir="${project_root}/results/phase2-w4a8-layer"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST="9.0"

echo "W4A8_LAYER_GATE_STAGE=inputs"
for required in \
    "${venv_path}/bin/python" \
    "${HF_HUB_CACHE}/models--meta-llama--Llama-2-13b-hf" \
    "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt" \
    "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "W4A8_LAYER_GATE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
mkdir -p "${results_dir}"

echo "W4A8_LAYER_GATE_STAGE=source_manifest"
{
    printf 'git_revision='; git -C "${project_root}" rev-parse HEAD
    printf 'git_status\n'; git -C "${project_root}" status --short
    sha256sum \
        "${project_root}/repro/w4a8_linear.py" \
        "${project_root}/scripts/run_llama_gptq_w4a8_layer_smoke.py" \
        "${project_root}/scripts/run_w4a8_cuda_smoke.py" \
        "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt" \
        "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json"
} > "${results_dir}/isambard-source-manifest.txt"

echo "W4A8_LAYER_GATE_STAGE=preflight"
"${venv_path}/bin/python" "${project_root}/scripts/runpod_preflight.py" \
    --output "${results_dir}/isambard-preflight.json"

echo "W4A8_LAYER_GATE_STAGE=kernel_smoke"
"${venv_path}/bin/python" "${project_root}/scripts/run_w4a8_cuda_smoke.py" \
    --output "${results_dir}/isambard-w4a8-kernel-smoke.json"

echo "W4A8_LAYER_GATE_STAGE=layer_smoke"
"${venv_path}/bin/python" "${project_root}/scripts/run_llama_gptq_w4a8_layer_smoke.py" \
    --config "${project_root}/configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json" \
    --artifact "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.pt" \
    --artifact-report "${project_root}/results/phase2-w4a8-gptq-export/llama2-13b-f4-layer0-qproj.json" \
    --output "${results_dir}/isambard-llama2-13b-f4-qproj-layer-smoke.json" \
    --sequence-length 16

echo "W4A8_LAYER_GATE_STAGE=assert_result"
"${venv_path}/bin/python" - "${results_dir}/isambard-llama2-13b-f4-qproj-layer-smoke.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["passed"], result
assert result["layer_output"]["allclose"], result["layer_output"]
assert result["logits"]["allclose"], result["logits"]
print("ISAMBARD_PHASE2_LAYER_GATE_PASSED")
PY
