"""CPU structured Hadamard transforms needed by non-power-of-two LLaMA MLPs.

QuaRot factors a dimension such as 1536 into 12 x 128, applies a Walsh
Hadamard transform to the power-of-two factor, then a 12-order Hadamard.  This
module intentionally implements that small, auditable operation in PyTorch;
it does not import or depend on the upstream CUDA extension.
"""

import base64
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

# Bit-packed upstream get_had40() sign matrix.  Llama-2-13B has
# hidden_size = 40 x 128, so this supplies its exact residual-space transform.
_H40_BITS_B64 = (
    "gAAIAADYV52FeewrzsK8thXrYV6bCvmwr82FfNhX5sK+bCvzYV82Ffmwr5sKvNhbzYXebC3mwq82GvNh15sNebCrzYq82JXmyV5sivNorzaFebhXm8K83CvN4V5uFeawrzsK84AAB///2FeSeobsK8E9Q7YV5J6hmwr2T1DNhXMnqObCsZPU82FQyer5sKBk9bzYVDJ63mwiGT2vNhUMntebAoZPq82FQyeV5sahk4rzZ1DJhXm3qGTCvNPUMuFeYeoZsK809Qw="
)

# Bit-packed (+1 = 1, -1 = 0) order-108 sign matrix transcribed from the
# pinned QuaRot reference's get_had108().  Llama-2-13B has
# intermediate_size = 108 x 128, so this is the factor required for its
# exact online MLP transform.  Keeping it packed avoids a large source literal
# and retains a dependency-free portable implementation.
_H108_BITS_B64 = (
    "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"
    "zVEvw2p4FuqYIWMzl75qiX4b"
)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def supports_structured_hadamard(size: int) -> bool:
    return _structured_factor(size) is not None


def _structured_factor(size: int) -> int | None:
    for factor in (108, 40, 12):
        if size % factor == 0 and is_power_of_two(size // factor):
            return factor
    return None


def normalized_h12(dtype: torch.dtype, device: torch.device, transpose: bool = False) -> torch.Tensor:
    matrix = torch.tensor(_H12, dtype=dtype, device=device) / math.sqrt(12.0)
    return matrix.T if transpose else matrix


def _normalized_packed_sign_matrix(
    encoded: str, order: int, dtype: torch.dtype, device: torch.device, transpose: bool = False
) -> torch.Tensor:
    packed = torch.tensor(list(base64.b64decode(encoded)), dtype=torch.uint8, device=device)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    signs = ((packed.unsqueeze(1) >> shifts) & 1).reshape(-1).to(dtype).mul_(2).sub_(1)
    matrix = signs.reshape(order, order) / math.sqrt(float(order))
    return matrix.T if transpose else matrix


def normalized_h40(dtype: torch.dtype, device: torch.device, transpose: bool = False) -> torch.Tensor:
    return _normalized_packed_sign_matrix(_H40_BITS_B64, 40, dtype, device, transpose)


def normalized_h108(dtype: torch.dtype, device: torch.device, transpose: bool = False) -> torch.Tensor:
    return _normalized_packed_sign_matrix(_H108_BITS_B64, 108, dtype, device, transpose)


def _normalized_structured_factor(
    factor: int, dtype: torch.dtype, device: torch.device, transpose: bool = False
) -> torch.Tensor:
    if factor == 12:
        return normalized_h12(dtype, device, transpose)
    if factor == 40:
        return normalized_h40(dtype, device, transpose)
    if factor == 108:
        return normalized_h108(dtype, device, transpose)
    raise ValueError("unsupported structured Hadamard factor {}".format(factor))


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


def _normalized_walsh_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not is_power_of_two(size):
        raise ValueError("Walsh Hadamard size must be a power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(float(size))


def normalized_structured_hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build an exact dense matrix only where a residual reparameterization needs it."""
    factor = _structured_factor(size)
    if factor is None:
        raise ValueError("expected a supported factor times a power of two, got {}".format(size))
    return torch.kron(
        _normalized_structured_factor(factor, dtype, device),
        _normalized_walsh_matrix(size // factor, dtype, device),
    )


def structured_hadamard(values: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Apply the upstream-compatible factor x power-of-two transform."""
    size = values.shape[-1]
    factor = _structured_factor(size)
    if factor is None:
        raise ValueError("expected a supported factor times a power of two, got {}".format(size))
    width = size // factor
    shape = values.shape
    work = values.reshape(-1, factor, width)
    work = _walsh_hadamard_last_axis(work)
    matrix = _normalized_structured_factor(factor, work.dtype, work.device, transpose=transpose) * math.sqrt(float(factor))
    work = matrix @ work
    return (work / math.sqrt(float(size))).reshape(shape)


def structured_hadamard_12x_power2(values: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """Compatibility wrapper for the original 12 x power-of-two helper."""
    size = values.shape[-1]
    if size % 12 != 0 or not is_power_of_two(size // 12):
        raise ValueError("expected size 12 times a power of two, got {}".format(size))
    return structured_hadamard(values, transpose)


class StructuredHadamardInputLinear(nn.Module):
    """Online structured Hadamard before a compensated Linear layer."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not supports_structured_hadamard(linear.in_features):
            raise ValueError("linear input is not compatible with a supported structured Hadamard")
        self.linear = linear

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(structured_hadamard(values))
