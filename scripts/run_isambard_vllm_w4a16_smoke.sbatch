#!/bin/bash
# Short GH200 gate for the already-prepared vLLM runtime and tiny checkpoints.
# Environment installation and checkpoint generation must happen on the login
# node before this is submitted.
#SBATCH --job-name=vllm-w4a16-smoke
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=results/vllm-w4a16-isambard-smoke/%x-%j.out
#SBATCH --error=results/vllm-w4a16-isambard-smoke/%x-%j.err

set -euo pipefail

echo "VLLM_SMOKE_STAGE=module_load"
module load cray-python/3.11.7

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
checkpoint_root="${HOME}/.cache/newsmallproject-vllm/tiny-checkpoints"
results_dir="${project_root}/results/vllm-w4a16-isambard-smoke"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "VLLM_SMOKE_STAGE=inputs"
for required in \
    "${serving_env}/bin/python" \
    "${checkpoint_root}/manifest.json" \
    "${checkpoint_root}/bf16/config.json" \
    "${checkpoint_root}/unrotated-w4a16/config.json" \
    "${checkpoint_root}/rotated-w4a16/config.json"; do
    if [[ ! -e "${required}" ]]; then
        echo "VLLM_SMOKE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done

git_revision="$(git -C "${project_root}" rev-parse HEAD)"
git_status="$(git -C "${project_root}" status --short)"
if [[ -n "${git_status}" ]]; then
    echo "VLLM_SMOKE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
manifest_revision="$("${serving_env}/bin/python" - "${checkpoint_root}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["project_revision"])
PY
)"
if [[ "${manifest_revision}" != "${git_revision}" ]]; then
    echo "VLLM_SMOKE_REVISION_MISMATCH=source:${git_revision},checkpoint:${manifest_revision}" >&2
    exit 1
fi
mkdir -p "${results_dir}"

echo "VLLM_SMOKE_STAGE=source_manifest"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    sha256sum \
        "${project_root}/configs/deployment/vllm_w4a16_isambard.json" \
        "${project_root}/scripts/run_vllm_w4a16_tiny_smoke.py" \
        "${checkpoint_root}/manifest.json"
} > "${results_dir}/source-manifest.txt"

echo "VLLM_SMOKE_STAGE=runtime"
"${serving_env}/bin/python" - <<'PY'
import os
import torch
import vllm

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert vllm.__version__.partition("+")[0] == "0.25.1", vllm.__version__
assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (9, 0)
print(torch.cuda.get_device_name(0))
PY

echo "VLLM_SMOKE_STAGE=offline_inference"
"${serving_env}/bin/python" "${project_root}/scripts/run_vllm_w4a16_tiny_smoke.py" \
    --config "${project_root}/configs/deployment/vllm_w4a16_isambard.json" \
    --checkpoint-root "${checkpoint_root}" \
    --output "${results_dir}/tiny-offline-inference.json"

echo "VLLM_SMOKE_STAGE=assert_result"
"${serving_env}/bin/python" - "${results_dir}/tiny-offline-inference.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "passed", result
assert set(result["models"]) == {"bf16", "unrotated_w4a16", "rotated_w4a16"}
print("ISAMBARD_VLLM_W4A16_SMOKE_PASSED")
PY
