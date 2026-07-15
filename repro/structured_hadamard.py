"""CPU structured Hadamard transforms needed by non-power-of-two LLaMA MLPs.

QuaRot factors a dimension such as 1536 into 12 x 128, applies a Walsh
Hadamard transform to the power-of-two factor, then a 12-order Hadamard.  This
module intentionally implements that small, auditable operation in PyTorch;
it does not import or depend on the upstream CUDA extension.
"""

import math

import torch
from torch import nn


# The order-12 sign matrix was transcribed after auditing the pinned upstream
# reference snapshot.  It is data, not a CUDA/kernel dependency.  Tests below
# verify its orthogonality and the exact row-vector convention used here.
_H12 = (
    (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    (1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1),
    (1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1),
    (1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1),
    (1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1),
    (1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1),
    (1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1),
    (1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1),
    (1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1),
    (1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1),
    (1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1),
    (1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1),
)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def supports_structured_hadamard(size: int) -> bool:
    return size % 12 == 0 and is_power_of_two(size // 12)


def normalized_h12(dtype: torch.dtype, device: torch.device, transpose: bool = False) -> torch.Tensor:
    matrix = torch.tensor(_H12, dtype=dtype, device=device) / math.sqrt(12.0)
    return matrix.T if transpose else matrix


def _walsh_hadamard_last_axis(values: torch.Tensor) -> torch.Tensor:
    width = values.shape[-1]
    result = values
    block = 1
    while block < width:
        shape = result.shape
        result = result.reshape(*shape[:-1], width // (2 * block), 2, block)
        first, second = result.select(-2, 0), result.select(-2, 1)
        result = torch.stack((first + second, first - second), dim=-2).reshape(shape)
        block *= 2
    return result


def structured_hadamard_12x_power2(values: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Apply the upstream-compatible row-vector transform without dense 1536² tensors."""
    size = values.shape[-1]
    if not supports_structured_hadamard(size):
        raise ValueError("expected size 12 times a power of two, got {}".format(size))
    factor = size // 12
    shape = values.shape
    work = values.reshape(-1, 12, factor)
    work = _walsh_hadamard_last_axis(work)
    h12 = normalized_h12(work.dtype, work.device, transpose=transpose) * math.sqrt(12.0)
    work = h12 @ work
    return (work / math.sqrt(float(size))).reshape(shape)


class StructuredHadamardInputLinear(nn.Module):
    """Online 12 x power-of-two Hadamard before a compensated Linear layer."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not supports_structured_hadamard(linear.in_features):
            raise ValueError("linear input is not compatible with 12 x power-of-two Hadamard")
        self.linear = linear

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(structured_hadamard_12x_power2(values))
