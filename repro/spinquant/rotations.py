"""Trainable R1/R2 rotations and their offline linear-weight convention."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from repro.spinquant.stiefel import orthogonality_error
from repro.structured_hadamard import (
    normalized_structured_hadamard_matrix,
    supports_structured_hadamard,
)
from repro.torch_smoke import normalized_hadamard_matrix


def random_signed_hadamard(
    size: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create a reproducible normalized Hadamard matrix with random column signs."""

    if size < 1:
        raise ValueError("random signed Hadamard size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(dtype=dtype, device=device)
    target_device = torch.device(device)
    if not size & (size - 1):
        base = normalized_hadamard_matrix(size, dtype, target_device)
    elif supports_structured_hadamard(size):
        base = normalized_structured_hadamard_matrix(size, dtype, target_device)
    else:
        raise ValueError(
            "random signed Hadamard initialization requires a power-of-two "
            "or supported structured Hadamard size"
        )
    return base * signs


class SpinQuantRotations(nn.Module):
    """Global residual R1 and independently learned per-layer head-wise R2."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_layers: int,
        seed: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if hidden_size % head_dim:
            raise ValueError("hidden_size must be divisible by head_dim")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.r1 = nn.Parameter(
            random_signed_hadamard(
                hidden_size,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        )
        r2 = [
            random_signed_hadamard(
                head_dim,
                seed=seed + layer + 1,
                dtype=dtype,
                device=device,
            )
            for layer in range(num_layers)
        ]
        self.r2 = nn.Parameter(torch.stack(r2))

    def r2_block(self, layer: int, *, num_heads: int | None = None) -> Tensor:
        """Repeat one learned head rotation for Q heads or GQA KV heads."""

        if not 0 <= layer < self.num_layers:
            raise IndexError("layer index is out of range")
        if num_heads is None:
            num_heads = self.hidden_size // self.head_dim
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        return torch.block_diag(*([self.r2[layer]] * num_heads))

    def errors(self) -> Dict[str, float]:
        """Return maximum R1 and per-layer R2 orthogonality errors."""

        return {
            "r1": float(orthogonality_error(self.r1).detach()),
            "r2": float(orthogonality_error(self.r2).max().detach()),
        }


def fuse_value_output_pair(
    value_weight: Tensor,
    output_weight: Tensor,
    residual_rotation: Tensor,
    head_rotation_block: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse R1/R2 into a V/O pair using Hugging Face row-vector semantics.

    For ``x' = x @ R1`` and ``v' = v @ R2``, the equivalent weights are
    ``Wv' = R2.T @ Wv @ R1`` and ``Wo' = R1.T @ Wo @ R2``.
    """

    hidden_size = residual_rotation.shape[-1]
    expected = (hidden_size, hidden_size)
    for name, values in (
        ("value_weight", value_weight),
        ("output_weight", output_weight),
        ("residual_rotation", residual_rotation),
        ("head_rotation_block", head_rotation_block),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    value_rotated = head_rotation_block.transpose(-1, -2) @ value_weight @ residual_rotation
    output_rotated = (
        residual_rotation.transpose(-1, -2)
        @ output_weight
        @ head_rotation_block
    )
    return value_rotated, output_rotated
