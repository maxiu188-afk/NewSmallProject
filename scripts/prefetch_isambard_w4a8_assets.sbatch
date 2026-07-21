#!/bin/bash
# Prefetch the pinned Llama-2-13B revision and Wikitext-2 into the persistent
# Isambard home cache.  Submit after `hf auth login` has completed there.
#SBATCH --job-name=w4a8-prefetch
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
# Login nodes can kill the default eight-way Hub/Xet transfer.  A one-worker
# HTTP transfer is slower but stays within the login-node resource limit and
# resumes from complete blobs on a subsequent run.
export HF_HUB_DISABLE_XET=1

if [[ ! -x "${venv_path}/bin/hf" ]]; then
    echo "W4A8 environment is absent; submit setup_isambard_w4a8_env.sbatch first." >&2
    exit 1
fi

"${venv_path}/bin/hf" auth whoami
"${venv_path}/bin/hf" download meta-llama/Llama-2-13b-hf \
    --revision 5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1 \
    --cache-dir "${HF_HUB_CACHE}" \
    --max-workers 1

"${venv_path}/bin/python" - <<'PY'
from datasets import load_dataset
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "meta-llama/Llama-2-13b-hf",
    revision="5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1",
    local_files_only=True,
)
for split in ("train", "test"):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    )
    print(f"wikitext split={split} rows={len(dataset)}")
print(f"model_snapshot={model_path}")
PY

echo "W4A8_ISAMBARD_ASSETS_READY"
