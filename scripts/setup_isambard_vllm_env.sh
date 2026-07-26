#!/bin/bash
# Configure the persistent Isambard vLLM and quantizer environments.
# Run directly on a login node. This script deliberately refuses Slurm jobs.

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "VLLM_ENV_ERROR=run this setup directly on the login node, not in Slurm" >&2
    exit 2
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "VLLM_ENV_ERROR=expected aarch64" >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load cray-python/3.11.7

serving_env="${HOME}/.venvs/newsmallproject-vllm-0.25.1"
quantizer_env="${HOME}/.venvs/newsmallproject-llmcompressor-0.12.0"
wheel_cache="${HOME}/.cache/newsmallproject-vllm/wheels"
manifest_dir="${HOME}/.cache/newsmallproject-vllm/manifests"
vllm_wheel="${wheel_cache}/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_aarch64.whl"
vllm_url="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_aarch64.whl"
vllm_sha256="bdffbe35b2c1ab8f2a9dcc337b657261d9b192c92c217e5a2f98a8835fe78daa"
llmcompressor_sha256="e17675737f41861391ee833265e3b96785feace4b96cf099df61119829a00f4d"

mkdir -p "${wheel_cache}" "${manifest_dir}" "${HOME}/.venvs"

pip_check_with_sbsa_guard() {
    local python_path="$1"
    local check_output
    local check_status=0
    check_output="$("${python_path}" -m pip check 2>&1)" || check_status=$?
    if [[ ${check_status} -eq 0 ]]; then
        printf '%s\n' "${check_output}"
        return 0
    fi
    if [[ "$(uname -m)" == "aarch64" ]] && \
       [[ "${check_output}" == "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform" ]]; then
        local wheel_metadata
        wheel_metadata="$(find "$(dirname "${python_path}")/../lib" \
            -path '*nvidia_cusparselt_cu12-0.7.1.dist-info/WHEEL' -print -quit)"
        if [[ -n "${wheel_metadata}" ]] && \
           grep -Fxq 'Tag: py3-none-manylinux2014_sbsa' "${wheel_metadata}"; then
            echo "VLLM_ENV_KNOWN_SBSA_TAG_WARNING=${check_output}"
            return 0
        fi
    fi
    printf '%s\n' "${check_output}" >&2
    return "${check_status}"
}

echo "VLLM_ENV_STAGE=download_vllm_wheel"
if [[ ! -f "${vllm_wheel}" ]]; then
    curl --fail --location --retry 3 --output "${vllm_wheel}" "${vllm_url}"
fi
printf '%s  %s\n' "${vllm_sha256}" "${vllm_wheel}" | sha256sum --check --status

echo "VLLM_ENV_STAGE=serving_environment"
if [[ ! -x "${serving_env}/bin/python" ]]; then
    python3 -m venv "${serving_env}"
fi
"${serving_env}/bin/python" -m pip install --upgrade "pip==26.1.2"
"${serving_env}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
"${serving_env}/bin/python" -m pip install "${vllm_wheel}"
pip_check_with_sbsa_guard "${serving_env}/bin/python"

echo "VLLM_ENV_STAGE=quantizer_environment"
if [[ ! -x "${quantizer_env}/bin/python" ]]; then
    python3 -m venv "${quantizer_env}"
fi
"${quantizer_env}/bin/python" -m pip install --upgrade "pip==26.1.2"
"${quantizer_env}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.11.0"
"${quantizer_env}/bin/python" -m pip download \
    --dest "${wheel_cache}" --no-deps --only-binary=:all: "llmcompressor==0.12.0"
llmcompressor_wheel="${wheel_cache}/llmcompressor-0.12.0-py3-none-any.whl"
printf '%s  %s\n' "${llmcompressor_sha256}" "${llmcompressor_wheel}" | sha256sum --check --status
"${quantizer_env}/bin/python" -m pip install "${llmcompressor_wheel}"
pip_check_with_sbsa_guard "${quantizer_env}/bin/python"

echo "VLLM_ENV_STAGE=manifests"
"${serving_env}/bin/python" -m pip freeze > "${manifest_dir}/vllm-0.25.1-cu129-aarch64.txt"
"${quantizer_env}/bin/python" -m pip freeze > "${manifest_dir}/llmcompressor-0.12.0-cu129-aarch64.txt"
sha256sum "${manifest_dir}"/*.txt > "${manifest_dir}/SHA256SUMS"

echo "VLLM_ENV_STAGE=login_imports"
"${serving_env}/bin/python" - <<'PY'
import platform
import torch
import vllm

assert platform.machine() == "aarch64"
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert vllm.__version__.partition("+")[0] == "0.25.1", vllm.__version__
assert not torch.cuda.is_available(), "login-node setup unexpectedly owns a GPU"
print("VLLM_LOGIN_IMPORT_PASSED")
PY
"${quantizer_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
from llmcompressor.modifiers.gptq import GPTQModifier  # noqa: F401

assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert metadata.version("llmcompressor") == "0.12.0"
assert metadata.version("compressed-tensors") == "0.17.1"
print("LLMCOMPRESSOR_LOGIN_IMPORT_PASSED")
PY

du -sh "${serving_env}" "${quantizer_env}"
echo "ISAMBARD_VLLM_ENV_READY"
