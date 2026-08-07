#!/usr/bin/env python3
"""Fail fast on the exact SGLang WNA16 toolchain and API contract."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SGLANG_SOURCE_RELATIVE_PATHS = {
    "sglang.jit_kernel.gptq_marlin_repack": "jit_kernel/gptq_marlin_repack.py",
    "sglang.jit_kernel.gptq_marlin_repack.cuh": (
        "jit_kernel/csrc/gemm/marlin/gptq_marlin_repack.cuh"
    ),
    "sglang.srt.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_wNa16": (
        "srt/layers/quantization/compressed_tensors/schemes/"
        "compressed_tensors_wNa16.py"
    ),
    "sglang.srt.managers.io_struct": "srt/managers/io_struct.py",
    "sglang.srt.managers.schedule_batch": "srt/managers/schedule_batch.py",
    "sglang.srt.managers.tokenizer_manager": "srt/managers/tokenizer_manager.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    )
    return completed.stdout.strip()


def _major(version: str) -> int:
    match = re.match(r"^(\d+)", version)
    if match is None:
        raise RuntimeError(f"cannot parse version major: {version!r}")
    return int(match.group(1))


def _require_env_executable(name: str) -> Path:
    raw_value = os.environ.get(name)
    if not raw_value:
        raise RuntimeError(f"{name} is not set")
    path = Path(raw_value)
    if not path.is_absolute() or not os.access(path, os.X_OK):
        raise RuntimeError(f"{name} is not an absolute executable: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sglang_config = config["sglang"]
    toolchain = sglang_config["toolchain"]

    cc = _require_env_executable("CC")
    cxx = _require_env_executable("CXX")
    nvcc_ccbin = _require_env_executable("NVCC_CCBIN")
    if not os.path.samefile(cxx, nvcc_ccbin):
        raise RuntimeError("CXX and NVCC_CCBIN select different host compilers")

    cc_version = _run([str(cc), "-dumpfullversion", "-dumpversion"]).splitlines()[0]
    cxx_version = _run([str(cxx), "-dumpfullversion", "-dumpversion"]).splitlines()[0]
    expected_compiler_major = int(toolchain["expected_compiler_major"])
    if _major(cc_version) != expected_compiler_major:
        raise RuntimeError(f"CC version drifted: {cc_version}")
    if _major(cxx_version) != expected_compiler_major:
        raise RuntimeError(f"CXX version drifted: {cxx_version}")
    _run(
        [str(cxx), "-std=c++20", "-x", "c++", "-E", "-"],
        input_text="#include <version>\n",
    )

    ninja = shutil.which("ninja")
    if ninja is None:
        raise RuntimeError("ninja is not on PATH")
    ninja_cli_version = _run([ninja, "--version"])
    ninja_package_version = metadata.version("ninja")
    expected_ninja = str(toolchain["expected_ninja_version"])
    if ninja_package_version != expected_ninja or not ninja_cli_version.startswith(
        expected_ninja
    ):
        raise RuntimeError(
            f"Ninja version drifted: package={ninja_package_version} "
            f"cli={ninja_cli_version}"
        )

    pinned_packages = {
        "apache-tvm-ffi": str(toolchain["expected_tvm_ffi_version"]),
        "compressed-tensors": str(
            toolchain["expected_compressed_tensors_version"]
        ),
        "flashinfer-python": str(toolchain["expected_flashinfer_version"]),
        "sglang-kernel": str(toolchain["expected_sglang_kernel_version"]),
        "transformers": str(toolchain["expected_transformers_version"]),
    }
    package_versions = {
        package: metadata.version(package) for package in pinned_packages
    }
    drifted_packages = {
        package: {"expected": expected, "observed": package_versions[package]}
        for package, expected in pinned_packages.items()
        if package_versions[package] != expected
    }
    if drifted_packages:
        raise RuntimeError(f"SGLang dependency versions drifted: {drifted_packages}")

    cray_cuda_version = os.environ.get("CRAY_CUDA_VERSION")
    if cray_cuda_version != str(toolchain["expected_cray_cuda_version"]):
        raise RuntimeError(f"CUDA module drifted: {cray_cuda_version}")

    cuda_home = Path(os.environ.get("CUDA_HOME", ""))
    if not cuda_home.is_absolute() or not cuda_home.is_dir():
        raise RuntimeError(
            f"CUDA_HOME is not an absolute toolkit directory: {cuda_home}"
        )
    nvcc = cuda_home / "bin/nvcc"
    if not os.access(nvcc, os.X_OK):
        raise RuntimeError(f"CUDA_HOME does not provide nvcc: {nvcc}")
    nvcc_version = _run([str(nvcc), "--version"])
    release_match = re.search(r"release\s+(\d+\.\d+)", nvcc_version)
    nvcc_release = release_match.group(1) if release_match else None
    if nvcc_release != str(toolchain["expected_nvcc_release"]):
        raise RuntimeError(f"NVCC release drifted: {nvcc_release}")
    version_match = re.search(r"\bV(\d+\.\d+\.\d+)\b", nvcc_version)
    nvcc_build_version = version_match.group(1) if version_match else None
    if nvcc_build_version != str(toolchain["expected_nvcc_version"]):
        raise RuntimeError(f"NVCC build version drifted: {nvcc_build_version}")

    cache_dir_raw = os.environ.get("TVM_FFI_CACHE_DIR", "")
    jit_cache_dir = Path(cache_dir_raw)
    if not jit_cache_dir.is_absolute() or not jit_cache_dir.is_dir():
        raise RuntimeError(
            f"TVM_FFI_CACHE_DIR is not an absolute existing directory: {jit_cache_dir}"
        )
    if jit_cache_dir.name != str(toolchain["jit_cache_namespace"]):
        raise RuntimeError(f"TVM-FFI cache namespace drifted: {jit_cache_dir}")

    cudart_soname = str(toolchain["expected_cudart_soname"])
    cudart_runtime_dir = (cuda_home / "lib64").resolve()
    cudart_runtime = cudart_runtime_dir / cudart_soname
    cudart_link = cuda_home / "lib64/libcudart.so"
    if not cudart_runtime.is_file():
        raise RuntimeError(f"CUDART runtime is missing: {cudart_runtime}")
    if not cudart_link.is_symlink() or not os.path.samefile(
        cudart_link, cudart_runtime
    ):
        raise RuntimeError(
            f"TVM-FFI CUDART link is missing or drifted: {cudart_link}"
        )
    ld_library_paths = [
        Path(value).resolve()
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if value
    ]
    if cudart_runtime_dir not in ld_library_paths:
        raise RuntimeError(
            f"LD_LIBRARY_PATH does not contain CUDART runtime: {cudart_runtime_dir}"
        )

    with tempfile.TemporaryDirectory(prefix="sglang-cudart-link-") as temporary:
        probe_dir = Path(temporary)
        probe_source = probe_dir / "probe.cpp"
        probe_library = probe_dir / "cudart-link-probe.so"
        probe_source.write_text(
            'extern "C" int cudaRuntimeGetVersion(int*);\n'
            'extern "C" int probe(int* version) { '
            "return cudaRuntimeGetVersion(version); }\n",
            encoding="utf-8",
        )
        try:
            _run(
                [
                    str(cxx),
                    "-std=c++20",
                    "-fPIC",
                    "-shared",
                    "-Wl,--no-undefined",
                    str(probe_source),
                    f"-L{cuda_home / 'lib64'}",
                    "-lcudart",
                    "-o",
                    str(probe_library),
                ]
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "CUDART host-link probe failed: "
                f"stdout={error.stdout!r} stderr={error.stderr!r}"
            ) from error
        probe_module = ctypes.CDLL(str(probe_library), mode=ctypes.RTLD_LOCAL)
        probe_function = probe_module.probe
        probe_function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        probe_function.restype = ctypes.c_int
        cudart_runtime_version = ctypes.c_int()
        cudart_status = probe_function(ctypes.byref(cudart_runtime_version))
        if cudart_status != 0 or cudart_runtime_version.value <= 0:
            raise RuntimeError(
                "CUDART dynamic-load probe failed: "
                f"status={cudart_status} version={cudart_runtime_version.value}"
            )

    import torch

    if metadata.version("sglang") != str(sglang_config["expected_version"]):
        raise RuntimeError("SGLang version drifted")
    if torch.__version__ != str(sglang_config["expected_torch_version"]):
        raise RuntimeError(f"PyTorch version drifted: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("SGLang WNA16 preflight requires the allocated CUDA GPU")
    capability = list(torch.cuda.get_device_capability())
    if capability != list(config["runtime"]["compute_capability"]):
        raise RuntimeError(f"compute capability drifted: {capability}")

    sglang_root = Path(
        metadata.distribution("sglang").locate_file("sglang")
    ).resolve()
    source_records = {}
    for name, relative_path in SGLANG_SOURCE_RELATIVE_PATHS.items():
        source = sglang_root / relative_path
        if not source.is_file():
            raise RuntimeError(f"SGLang source is missing: {source}")
        source_records[name] = {"path": str(source), "sha256": _sha256(source)}
    expected_source_sha256 = sglang_config["expected_source_sha256"]
    observed_source_sha256 = {
        name: record["sha256"] for name, record in source_records.items()
    }
    if observed_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "SGLang source hashes drifted: "
            f"expected={expected_source_sha256} observed={observed_source_sha256}"
        )

    from sglang.jit_kernel.gptq_marlin_repack import (
        _jit_gptq_marlin_repack_module,
        gptq_marlin_repack,
    )
    from sglang.srt.managers.io_struct import GenerateReqInput

    request = GenerateReqInput(
        input_ids=[[1, 2, 3], [1, 2, 3, 4]],
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=[1, 2],
        top_logprobs_num=0,
        return_text_in_logprobs=False,
    )
    request.normalize_batch_and_arguments()
    if request.return_logprob != [True, True]:
        raise RuntimeError("SGLang batch return_logprob normalization drifted")
    if request.logprob_start_len != [1, 2]:
        raise RuntimeError("SGLang batch logprob_start_len normalization drifted")
    if request.top_logprobs_num != [0, 0]:
        raise RuntimeError("SGLang batch top_logprobs_num normalization drifted")

    started = time.monotonic()
    jit_module = _jit_gptq_marlin_repack_module()
    jit_compile_seconds = time.monotonic() - started
    if not callable(getattr(jit_module, "gptq_marlin_repack", None)):
        raise RuntimeError("compiled GPTQ-Marlin module is missing its entrypoint")

    size_k = 128
    size_n = 128
    num_bits = 4
    packed = torch.zeros(
        (size_k // (32 // num_bits), size_n),
        dtype=torch.int32,
        device="cuda",
    )
    permutation = torch.empty(0, dtype=torch.int32, device="cuda")
    repacked = gptq_marlin_repack(
        packed,
        permutation,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
    )
    torch.cuda.synchronize()
    expected_shape = (size_k // 16, size_n * 16 // (32 // num_bits))
    if tuple(repacked.shape) != expected_shape or repacked.dtype != torch.int32:
        raise RuntimeError(
            f"GPTQ-Marlin repack output drifted: {repacked.shape} {repacked.dtype}"
        )
    if torch.count_nonzero(repacked).item() != 0:
        raise RuntimeError("zero GPTQ-Marlin preflight input produced nonzero output")

    result = {
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "python": sys.version.split()[0],
        "packages": {
            "sglang": metadata.version("sglang"),
            "torch": torch.__version__,
            "ninja": ninja_package_version,
            **package_versions,
        },
        "toolchain": {
            "compiler_module": toolchain["compiler_module"],
            "cuda_module": toolchain["cuda_module"],
            "CRAY_CUDA_VERSION": cray_cuda_version,
            "CC": str(cc),
            "CC_version": cc_version,
            "CXX": str(cxx),
            "CXX_version": cxx_version,
            "NVCC_CCBIN": str(nvcc_ccbin),
            "CUDA_HOME": str(cuda_home),
            "cudart_link": str(cudart_link),
            "cudart_runtime": str(cudart_runtime),
            "cudart_runtime_version": cudart_runtime_version.value,
            "cudart_host_link_and_load": "passed",
            "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
            "nvcc": str(nvcc),
            "nvcc_release": nvcc_release,
            "nvcc_build_version": nvcc_build_version,
            "nvcc_version_output": nvcc_version.splitlines(),
            "ninja": ninja,
            "ninja_cli_version": ninja_cli_version,
            "cxx20_version_header": "passed",
            "TVM_FFI_CACHE_DIR": str(jit_cache_dir),
            "jit_cache_namespace": toolchain["jit_cache_namespace"],
        },
        "cuda": {
            "available": True,
            "device": torch.cuda.get_device_name(),
            "compute_capability": capability,
            "torch_runtime": torch.version.cuda,
        },
        "sglang_api": {
            "batch_generate_logprob_normalization": "passed",
            "logprob_start_len_semantics": "score token at start_len + 1",
        },
        "jit": {
            "module": "sglang.jit_kernel.gptq_marlin_repack",
            "compile_seconds": jit_compile_seconds,
            "entrypoint": "gptq_marlin_repack",
            "synthetic_execution": "passed",
            "input_shape": list(packed.shape),
            "output_shape": list(repacked.shape),
        },
        "sources": source_records,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SGLANG_WNA16_RUNTIME_PREFLIGHT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
