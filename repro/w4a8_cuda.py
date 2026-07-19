"""CUDA binding for the owned W4A8 integer-accumulator correctness kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_NAME = "newsmallproject_w4a8_grouped_int32_v1"
_extension: Any = None


def load_extension(verbose: bool = False) -> Any:
    """Build/load the exact W4A8 correctness kernel for the active CUDA GPU."""
    global _extension
    if not torch.cuda.is_available():
        raise RuntimeError("W4A8 CUDA kernel requires an NVIDIA CUDA runtime")
    if _extension is None:
        _extension = load(
            name=_EXTENSION_NAME,
            sources=[
                str(PROJECT_ROOT / "csrc" / "w4a8_grouped_int32.cpp"),
                str(PROJECT_ROOT / "csrc" / "w4a8_grouped_int32.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    return _extension


def grouped_int32_matmul(
    activations: torch.Tensor, packed_weights: torch.Tensor, group_size: int, verbose: bool = False
) -> torch.Tensor:
    """Return `[tokens, out_features, groups_per_row]` exact int32 accumulators."""
    return load_extension(verbose=verbose).w4a8_grouped_int32(activations, packed_weights, group_size)
