#!/bin/bash
# Export and validate all GPTQ-packed Llama decoder linears, then compare the
# independent packed oracle with the owned CUDA W4A8 path at every layer.
# K/V, embeddings, and lm_head remain BF16. This is not a PPL or timing job.
#SBATCH --job-name=w4a8-full-gate
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "W4A8_FULL_GATE_STAGE=module_load"
module load cray-python/3.11.7 cuda/12.6

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv_path="${HOME}/.venvs/newsmallproject-w4a8"
checkpoint_dir="${project_root}/results/phase2-w4a8-full-checkpoint"
results_dir="${project_root}/results/phase2-w4a8-full-model"
config_path="${project_root}/configs/pipeline/llama2_13b_wikitext2_quarot_w4a4_gptq.json"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export TORCH_CUDA_ARCH_LIST="9.0"

echo "W4A8_FULL_GATE_STAGE=inputs"
for required in \
    "${venv_path}/bin/python" \
    "${HF_HUB_CACHE}/models--meta-llama--Llama-2-13b-hf" \
    "${config_path}"; do
    if [[ ! -e "${required}" ]]; then
        echo "W4A8_FULL_GATE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -n "$(git -C "${project_root}" status --porcelain)" ]]; then
    echo "W4A8_FULL_GATE_DIRTY_CHECKOUT=${project_root}" >&2
    exit 1
fi
mkdir -p "${checkpoint_dir}" "${results_dir}"

echo "W4A8_FULL_GATE_STAGE=source_manifest"
{
    printf 'git_revision='; git -C "${project_root}" rev-parse HEAD
    printf 'git_status\n'; git -C "${project_root}" status --short
    sha256sum \
        "${config_path}" \
        "${project_root}/repro/gptq.py" \
        "${project_root}/repro/w4a8_checkpoint.py" \
        "${project_root}/repro/w4a8_linear.py" \
        "${project_root}/scripts/export_llama_gptq_w4a8_checkpoint.py" \
        "${project_root}/scripts/run_llama_gptq_w4a8_full_model_smoke.py"
} > "${results_dir}/isambard-source-manifest.txt"

echo "W4A8_FULL_GATE_STAGE=preflight"
"${venv_path}/bin/python" "${project_root}/scripts/runpod_preflight.py" \
    --output "${results_dir}/isambard-preflight.json"

echo "W4A8_FULL_GATE_STAGE=kernel_smoke"
"${venv_path}/bin/python" "${project_root}/scripts/run_w4a8_cuda_smoke.py" \
    --output "${results_dir}/isambard-w4a8-kernel-smoke.json"

echo "W4A8_FULL_GATE_STAGE=checkpoint"
if [[ -f "${checkpoint_dir}/manifest.json" ]]; then
    "${venv_path}/bin/python" "${project_root}/scripts/validate_w4a8_checkpoint.py" \
        --manifest "${checkpoint_dir}/manifest.json"
else
    "${venv_path}/bin/python" "${project_root}/scripts/export_llama_gptq_w4a8_checkpoint.py" \
        --config "${config_path}" \
        --output-dir "${checkpoint_dir}" \
        --report "${results_dir}/isambard-checkpoint-export.json"
fi

echo "W4A8_FULL_GATE_STAGE=full_model_smoke"
"${venv_path}/bin/python" "${project_root}/scripts/run_llama_gptq_w4a8_full_model_smoke.py" \
    --config "${config_path}" \
    --manifest "${checkpoint_dir}/manifest.json" \
    --output "${results_dir}/isambard-full-model-smoke.json" \
    --sequence-length 16 \
    --generation-tokens 8

echo "W4A8_FULL_GATE_STAGE=assert_result"
"${venv_path}/bin/python" - \
    "${checkpoint_dir}/manifest.json" \
    "${results_dir}/isambard-full-model-smoke.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert manifest["tensor_count"] == 280, manifest["tensor_count"]
assert result["precision"]["decoder_linear_count"] == 280, result["precision"]
assert result["passed"], result
assert result["logits"]["allclose"], result["logits"]
assert all(item["allclose"] for item in result["layer_outputs"]), result["layer_outputs"]
assert result["generation"]["finite_logits"], result["generation"]
print("ISAMBARD_PHASE2_FULL_MODEL_GATE_PASSED")
PY
