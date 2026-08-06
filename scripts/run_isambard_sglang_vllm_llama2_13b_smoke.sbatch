#!/bin/bash
# Record the result-gated exact-checkpoint compatibility matrix only.
# A successful Slurm exit means that evidence was recorded, not that all tracks passed.
#SBATCH --job-name=sglang-vllm-13b-smoke
#SBATCH --account=brics.u6rt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=results/sglang-vllm-llama2-13b/%x-%j.out
#SBATCH --error=results/sglang-vllm-llama2-13b/%x-%j.err

set -euo pipefail

module load cray-python/3.11.7
project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

config="${project_root}/configs/deployment/sglang_vllm_llama2_13b_smoke_isambard.json"
runner="${project_root}/scripts/run_sglang_vllm_llama2_13b_smoke.py"
vllm_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
sglang_env="${PROJECTDIR}/${USER}/newsmallproject-sglang/sglang-0.5.16-cu129"
source_evidence_root="${SGLANG_VLLM_SOURCE_EVIDENCE_ROOT:-${HOME}/NewSmallProject-vllm-ready}"
artifact_root="${PROJECTDIR}/${USER}/newsmallproject-vllm/llama2-13b-w4afp8-fp8-targeted-5876984"
model_snapshot="${HOME}/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
w4afp8_model="${artifact_root}/spinquant-w4afp8-fp8-targeted-5876984"
source_result="${source_evidence_root}/results/vllm-w4afp8-fp8-targeted-5876984/joint-offline-inference-5881273.json"
capability_result="${source_evidence_root}/results/vllm-w4afp8-fp8-targeted-5876984/backend-capability-5881273.json"
results_dir="${project_root}/results/sglang-vllm-llama2-13b"
output="${results_dir}/compatibility-smoke-${SLURM_JOB_ID}.json"
log_dir="${results_dir}/compatibility-smoke-${SLURM_JOB_ID}-logs"
source_manifest="${results_dir}/compatibility-smoke-${SLURM_JOB_ID}-source-manifest.txt"

task_tmp_parent="${SCRATCHDIR:-/tmp}"
export SCRATCHDIR="${task_tmp_parent%/}/newsmallproject-sglang-vllm-${SLURM_JOB_ID}"
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
export CUDA_HOME="${sglang_env}/lib/python3.11/site-packages/nvidia/cu13"
export SGLANG_ENABLE_JIT_DEEPGEMM=0

for required in \
    "${config}" \
    "${runner}" \
    "${vllm_env}/bin/vllm" \
    "${sglang_env}/bin/python" \
    "${CUDA_HOME}/bin/nvcc" \
    "${model_snapshot}/config.json" \
    "${w4afp8_model}/config.json" \
    "${source_result}" \
    "${capability_result}"; do
    if [[ ! -e "${required}" ]]; then
        echo "SGLANG_VLLM_SMOKE_MISSING_INPUT=${required}" >&2
        exit 1
    fi
done
if [[ -e "${output}" || -e "${log_dir}" ]]; then
    echo "SGLANG_VLLM_SMOKE_REFUSES_EXISTING_OUTPUT=${output}" >&2
    exit 1
fi

git_revision="$(git rev-parse HEAD)"
git_status="$(git status --short)"
if [[ -n "${git_status}" ]]; then
    echo "SGLANG_VLLM_SMOKE_DIRTY_CHECKOUT" >&2
    printf '%s\n' "${git_status}" >&2
    exit 1
fi
source_revision="$(git -C "${source_evidence_root}" rev-parse HEAD)"
if [[ "${source_revision}" != "60aa6295909f60997836b9c52dce7e5d4b2b1cc1" ]]; then
    echo "SGLANG_VLLM_SMOKE_SOURCE_REVISION_MISMATCH=${source_revision}" >&2
    exit 1
fi
printf '%s  %s\n' \
    "c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55" \
    "${source_result}" | sha256sum --check --status
printf '%s  %s\n' \
    "495bcc458681f5473e4b1ad50db96b249c82c1ef88cd0f7085e505e5dbdc8d62" \
    "${capability_result}" | sha256sum --check --status

mkdir -p "${results_dir}"
{
    printf 'git_revision=%s\n' "${git_revision}"
    printf 'git_status=clean\n'
    printf 'source_revision=%s\n' "${source_revision}"
    sha256sum \
        "${config}" \
        "${runner}" \
        "${project_root}/scripts/run_isambard_sglang_vllm_llama2_13b_smoke.sbatch" \
        "${source_result}" \
        "${capability_result}" \
        "${PROJECTDIR}/${USER}/newsmallproject-sglang/manifests/sglang-0.5.16-cu129-aarch64.txt"
} > "${source_manifest}"

"${vllm_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

assert metadata.version("vllm") == "0.25.1+cu129"
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SGLANG_VLLM_SMOKE_VLLM_RUNTIME_ACCEPTED")
PY

"${sglang_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import torch

assert metadata.version("sglang") == "0.5.16"
assert metadata.version("sglang-kernel") == "0.4.5+cu129"
assert metadata.version("sgl-deep-gemm") == "0.1.4.post1+cu129"
assert metadata.version("cuda-python") == "12.9.4"
assert metadata.version("cuda-bindings") == "12.9.7"
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert os.path.isdir(os.environ["CUDA_HOME"]), os.environ["CUDA_HOME"]
assert os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"
assert torch.cuda.get_device_capability(0) == (9, 0)
print("SGLANG_VLLM_SMOKE_SGLANG_RUNTIME_ACCEPTED")
PY

"${vllm_env}/bin/python" "${runner}" \
    --config "${config}" \
    --vllm-executable "${vllm_env}/bin/vllm" \
    --sglang-python "${sglang_env}/bin/python" \
    --model "bf16=${model_snapshot}" \
    --model "fp8_targeted_w4afp8=${w4afp8_model}" \
    --log-dir "${log_dir}" \
    --output "${output}"

"${vllm_env}/bin/python" - "${output}" "${git_revision}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "recorded", result
assert result["project_revision"] == sys.argv[2], result
assert set(result["cases"]) == {
    "vllm:bf16",
    "vllm:fp8_targeted_w4afp8",
    "sglang:bf16",
    "sglang:fp8_targeted_w4afp8",
}, result
print(f"SGLANG_VLLM_COMPATIBILITY_STATUS={result['compatibility_status']}")
print("ISAMBARD_SGLANG_VLLM_LLAMA2_13B_SMOKE_RECORDED")
PY
