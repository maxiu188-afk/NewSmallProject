#!/bin/bash
# Rebuild the W4A8 evaluation environment on one Isambard-AI GH200.
# Submit from the project root with: sbatch scripts/setup_isambard_w4a8_env.sbatch
#
# This script intentionally installs and validates only the software stack.
# Model weights are prefetched separately after Hugging Face authentication.
#
# The project and cache paths resolve from $HOME so no machine-specific path is
# committed.  Do not use node-local storage for either of them.
#SBATCH --job-name=w4a8-env
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load cray-python/3.11.7 cuda/12.6

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${HOME}/.venvs/newsmallproject-w4a8"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export TORCH_CUDA_ARCH_LIST="9.0"

mkdir -p "${HF_HOME}" "${venv_path%/*}"

if [[ ! -x "${venv_path}/bin/python" ]]; then
    python3 -m venv "${venv_path}"
fi

"${venv_path}/bin/python" -m pip install --upgrade pip
# Isambard-AI's ARM64 cu128 index does not provide torch 2.8.0.  Keep this
# distinct from the earlier x86_64 RunPod runtime, which did provide it.
"${venv_path}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0"
"${venv_path}/bin/python" -m pip install -r "${project_root}/requirements-runpod-eval.txt"

"${venv_path}/bin/python" - <<'PY'
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated Isambard GPU job")
if shutil.which("nvcc") is None:
    raise RuntimeError("nvcc is unavailable; the W4A8 CUDA extension cannot build")

print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("device=", torch.cuda.get_device_name(0))
print("capability=", torch.cuda.get_device_capability(0))
print("nvcc=", shutil.which("nvcc"))
PY

echo "W4A8_ISAMBARD_ENV_READY"
