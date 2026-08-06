#!/bin/bash
# Add the development-name CUDART link expected by TVM-FFI's CUDA JIT linker.

set -euo pipefail

if [[ $# -ne 1 || "${1}" != /* ]]; then
    echo "SGLANG_CUDA_LAYOUT_ERROR=expected one absolute CUDA_HOME path" >&2
    exit 2
fi

cuda_home="$1"
cudart_soname="libcudart.so.13"
cudart_runtime="${cuda_home}/lib/${cudart_soname}"
cudart_link_dir="${cuda_home}/lib64"
cudart_link="${cudart_link_dir}/libcudart.so"

if [[ ! -f "${cudart_runtime}" ]]; then
    echo "SGLANG_CUDA_LAYOUT_ERROR=missing ${cudart_runtime}" >&2
    exit 2
fi
if [[ -e "${cudart_link}" || -L "${cudart_link}" ]]; then
    if [[ ! -L "${cudart_link}" ]] || \
        [[ ! "${cudart_link}" -ef "${cudart_runtime}" ]]; then
        echo "SGLANG_CUDA_LAYOUT_ERROR=unexpected ${cudart_link}" >&2
        exit 2
    fi
else
    mkdir -p "${cudart_link_dir}"
    ln -s "../lib/${cudart_soname}" "${cudart_link}"
fi

if [[ ! "${cudart_link}" -ef "${cudart_runtime}" ]]; then
    echo "SGLANG_CUDA_LAYOUT_ERROR=link validation failed" >&2
    exit 2
fi

printf 'SGLANG_CUDA_LAYOUT_CUDART_RUNTIME=%s\n' "${cudart_runtime}"
printf 'SGLANG_CUDA_LAYOUT_CUDART_LINK=%s\n' "${cudart_link}"
echo "SGLANG_CUDA_JIT_LAYOUT_READY"
