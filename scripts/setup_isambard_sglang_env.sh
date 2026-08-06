#!/bin/bash
# Configure an isolated CUDA 12.9 SGLang environment on Isambard login nodes.

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "SGLANG_ENV_ERROR=run this setup on a login node, not in Slurm" >&2
    exit 2
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "SGLANG_ENV_ERROR=expected aarch64" >&2
    exit 2
fi
if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "SGLANG_ENV_ERROR=PROJECTDIR is not set" >&2
    exit 2
fi
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load cray-python/3.11.7
module load gcc-native/14.2

environment_root="${PROJECTDIR}/${USER}/newsmallproject-sglang"
sglang_env="${environment_root}/sglang-0.5.16-cu129"
manifest_dir="${environment_root}/manifests"
mkdir -p "${environment_root}" "${manifest_dir}"
export PATH="${sglang_env}/bin:${PATH}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export NVCC_CCBIN="${CXX}"

pip_check_with_sbsa_guard() {
    local python_path="$1"
    local check_output
    local check_status=0
    check_output="$("${python_path}" -m pip check 2>&1)" || check_status=$?
    if [[ ${check_status} -eq 0 ]]; then
        printf '%s\n' "${check_output}"
        return 0
    fi
    local cuda_override_warning=
    local sbsa_warning=
    local unexpected_warning=
    while IFS= read -r warning; do
        case "${warning}" in
            "sglang 0.5.16 has requirement cuda-python>=13.0, but you have cuda-python 12.9.4.")
                cuda_override_warning="${warning}"
                ;;
            "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform")
                sbsa_warning="${warning}"
                ;;
            *)
                unexpected_warning="${unexpected_warning}${warning}"$'\n'
                ;;
        esac
    done <<< "${check_output}"
    if [[ -n "${cuda_override_warning}" && -n "${sbsa_warning}" && \
        -z "${unexpected_warning}" ]]; then
        local wheel_metadata
        wheel_metadata="$(find "$(dirname "${python_path}")/../lib" \
            -path '*nvidia_cusparselt_cu12-0.7.1.dist-info/WHEEL' \
            -print -quit)"
        if [[ -n "${wheel_metadata}" ]] && \
            grep -Fxq 'Tag: py3-none-manylinux2014_sbsa' "${wheel_metadata}"; then
            echo "SGLANG_ENV_CUDA12_OVERRIDE=${cuda_override_warning}"
            echo "SGLANG_ENV_KNOWN_SBSA_TAG_WARNING=${sbsa_warning}"
            return 0
        fi
    fi
    printf '%s\n' "${check_output}" >&2
    return "${check_status}"
}

if [[ ! -x "${sglang_env}/bin/python" ]]; then
    python3 -m venv "${sglang_env}"
fi
"${sglang_env}/bin/python" -m pip install --upgrade \
    "pip==26.1.2" "uv==0.12.1"

uv_command=("${sglang_env}/bin/uv" pip install --python "${sglang_env}/bin/python")
"${uv_command[@]}" --prerelease=allow "sglang==0.5.16" "ninja==1.13.0"
"${sglang_env}/bin/python" -m pip uninstall --yes \
    cuda-core nvidia-cusparselt-cu13
"${uv_command[@]}" --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu129 \
    "torch==2.11.0" \
    "torchaudio==2.11.0" \
    "torchvision==0.26.0" \
    "nvidia-cusparselt-cu12==0.7.1"
"${uv_command[@]}" --force-reinstall \
    "cuda-python==12.9.4" \
    "cuda-bindings==12.9.7" \
    "cuda-pathfinder==1.6.0" \
    "numpy==2.3.5"
"${sglang_env}/bin/python" -m pip uninstall --yes cuda-core
"${uv_command[@]}" --force-reinstall \
    --index-url https://docs.sglang.ai/whl/cu129/ \
    "sglang-kernel==0.4.5"
"${uv_command[@]}" --force-reinstall --no-deps \
    --index-url https://docs.sglang.ai/whl/cu129/ \
    "sgl-deep-gemm==0.1.4.post1"

pip_check_with_sbsa_guard "${sglang_env}/bin/python"
export CUDA_HOME="${sglang_env}/lib/python3.11/site-packages/nvidia/cu13"
printf '#include <version>\n' | "${CXX}" -std=c++20 -x c++ -E - >/dev/null
"${sglang_env}/bin/python" -m pip freeze \
    > "${manifest_dir}/sglang-0.5.16-cu129-aarch64.txt"
{
    printf 'compiler_module=gcc-native/14.2\n'
    printf 'CC=%s\n' "${CC}"
    printf 'CC_version=%s\n' "$("${CC}" -dumpfullversion -dumpversion | head -1)"
    printf 'CXX=%s\n' "${CXX}"
    printf 'CXX_version=%s\n' "$("${CXX}" -dumpfullversion -dumpversion | head -1)"
    printf 'NVCC_CCBIN=%s\n' "${NVCC_CCBIN}"
    printf 'ninja=%s\n' "$(command -v ninja)"
    printf 'ninja_version=%s\n' "$(ninja --version)"
    printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version
} > "${manifest_dir}/sglang-0.5.16-cu129-toolchain.txt"
sha256sum \
    "${manifest_dir}/sglang-0.5.16-cu129-aarch64.txt" \
    "${manifest_dir}/sglang-0.5.16-cu129-toolchain.txt" \
    > "${manifest_dir}/SHA256SUMS"

"${sglang_env}/bin/python" - <<'PY'
import importlib.metadata as metadata
import platform
import torch

assert platform.machine() == "aarch64", platform.machine()
assert metadata.version("sglang") == "0.5.16"
assert metadata.version("sglang-kernel") == "0.4.5+cu129"
assert metadata.version("sgl-deep-gemm") == "0.1.4.post1+cu129"
assert metadata.version("apache-tvm-ffi") == "0.1.11"
assert metadata.version("flashinfer-python") == "0.6.14"
assert metadata.version("cuda-python") == "12.9.4"
assert metadata.version("cuda-bindings") == "12.9.7"
assert metadata.version("ninja") == "1.13.0"
assert metadata.version("transformers") == "5.12.1"
assert torch.__version__ == "2.11.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
assert not torch.cuda.is_available(), "login-node setup unexpectedly owns a GPU"
print("ISAMBARD_SGLANG_0_5_16_CU129_ENV_READY")
PY

du -sh "${sglang_env}"
