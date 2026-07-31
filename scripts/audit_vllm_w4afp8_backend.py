#!/usr/bin/env python3
"""Audit the pinned GH200 vLLM W4AFP8 kernel without loading a model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.vllm_w4afp8 import (  # noqa: E402
    EXPECTED_KERNEL,
    validate_export_config,
    validate_kernel_shapes,
)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit(config_path: Path) -> dict[str, Any]:
    import torch
    import vllm
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
        CutlassW4A8LinearKernel,
    )
    from vllm.scalar_type import scalar_types

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_export_config(config)
    runtime = config["runtime"]
    expected_capability = tuple(runtime["compute_capability"])
    if not torch.cuda.is_available():
        raise RuntimeError("W4AFP8 backend audit requires an allocated CUDA device")
    if torch.cuda.get_device_capability(0) != expected_capability:
        raise RuntimeError("W4AFP8 backend audit did not receive the pinned GH200")

    installed = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "vllm": metadata.version("vllm"),
        "compressed_tensors": metadata.version("compressed-tensors"),
    }
    if installed["torch"] != runtime["torch_version"]:
        raise RuntimeError(f"unexpected PyTorch version: {installed['torch']}")
    if installed["vllm"].split("+")[0] != runtime["vllm_version"]:
        raise RuntimeError(f"unexpected vLLM version: {installed['vllm']}")
    if installed["compressed_tensors"] != runtime["serving_compressed_tensors_version"]:
        raise RuntimeError(
            f"unexpected serving compressed-tensors: {installed['compressed_tensors']}"
        )

    shapes = [tuple(values) for values in config["kernel"]["required_linear_shapes"]]
    validate_kernel_shapes(shapes)
    selections = []
    for shape in shapes:
        layer_config = MPLinearLayerConfig(
            full_weight_shape=shape,
            partition_weight_shape=shape,
            weight_type=scalar_types.int4,
            act_type=torch.float8_e4m3fn,
            group_size=128,
            zero_points=False,
            has_g_idx=False,
            out_type=torch.bfloat16,
        )
        kernel = choose_mp_linear_kernel(layer_config)
        if kernel.__name__ != EXPECTED_KERNEL:
            raise RuntimeError(f"shape {shape} selected {kernel.__name__}")
        supported, reason = CutlassW4A8LinearKernel.can_implement(layer_config)
        if not supported or reason is not None:
            raise RuntimeError(f"shape {shape} failed CUTLASS audit: {reason}")
        selections.append({"shape": list(shape), "kernel": kernel.__name__})

    rejected = MPLinearLayerConfig(
        full_weight_shape=shapes[0],
        partition_weight_shape=shapes[0],
        weight_type=scalar_types.int4,
        act_type=torch.float8_e4m3fn,
        group_size=128,
        zero_points=False,
        has_g_idx=True,
        out_type=torch.bfloat16,
    )
    supported, reason = CutlassW4A8LinearKernel.can_implement(rejected)
    if supported or reason != "Act reordering not supported by CUTLASS W4A8":
        raise RuntimeError("runtime g_idx rejection contract changed")

    package_root = Path(vllm.__file__).resolve().parent
    source_paths = [
        package_root
        / "model_executor/kernels/linear/mixed_precision/cutlass.py",
        package_root
        / "model_executor/layers/quantization/compressed_tensors/schemes"
        / "compressed_tensors_w4a8_fp8.py",
    ]
    if not all(path.is_file() for path in source_paths):
        raise FileNotFoundError(f"installed vLLM W4AFP8 source is incomplete: {source_paths}")

    return {
        "status": "passed",
        "scope": (
            "pinned vLLM W4AFP8 capability audit only; no checkpoint, "
            "quality, or performance result"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "runtime": {
            **installed,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "selections": selections,
        "runtime_g_idx": {
            "supported": supported,
            "rejection": reason,
        },
        "source": {
            str(path.relative_to(package_root)): _sha256(path) for path in source_paths
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config.resolve())
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("VLLM_W4AFP8_BACKEND_AUDIT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
