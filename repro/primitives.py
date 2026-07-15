"""Reference math used to validate QuaRot before framework integration.

These functions deliberately use Python sequences rather than CUDA or a copied
upstream implementation. They provide transparent, deterministic correctness
oracles for the later PyTorch/CUDA paths; they are not performance kernels.
"""

import math
from typing import Iterable, List, Sequence, Tuple


def _require_bits(bits: int) -> None:
    if not isinstance(bits, int) or bits < 2 or bits > 16:
        raise ValueError("bits must be an integer in [2, 16]")


def _require_clip_ratio(clip_ratio: float) -> None:
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in (0, 1]")


def _round_nearest_even(value: float) -> int:
    """Match the nearest-even convention used by PyTorch round for finite values."""
    return int(round(value))


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def symmetric_quantize(
    values: Sequence[float], bits: int, clip_ratio: float = 1.0
) -> Tuple[List[int], float]:
    """Per-vector signed quantization compatible with QuaRot's symmetric QDQ."""
    _require_bits(bits)
    _require_clip_ratio(clip_ratio)
    if not values:
        raise ValueError("values must not be empty")

    maxq = 2 ** (bits - 1) - 1
    max_abs = max(abs(value) for value in values) * clip_ratio
    scale = max_abs / maxq if max_abs != 0.0 else 1.0
    qmin, qmax = -maxq - 1, maxq
    quantized = [_clamp(_round_nearest_even(value / scale), qmin, qmax) for value in values]
    return quantized, scale


def symmetric_dequantize(values: Sequence[int], scale: float) -> List[float]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return [value * scale for value in values]


def asymmetric_quantize(
    values: Sequence[float], bits: int, clip_ratio: float = 1.0
) -> Tuple[List[int], float, int]:
    """Per-vector affine quantization following the fake-quant reference policy."""
    _require_bits(bits)
    _require_clip_ratio(clip_ratio)
    if not values:
        raise ValueError("values must not be empty")

    maxq = 2**bits - 1
    xmin = min(min(values), 0.0) * clip_ratio
    xmax = max(max(values), 0.0) * clip_ratio
    if xmin == 0.0 and xmax == 0.0:
        xmin, xmax = -1.0, 1.0
    scale = (xmax - xmin) / maxq
    zero = _round_nearest_even(-xmin / scale)
    quantized = [_clamp(_round_nearest_even(value / scale) + zero, 0, maxq) for value in values]
    return quantized, scale, zero


def asymmetric_dequantize(values: Sequence[int], scale: float, zero: int) -> List[float]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return [scale * (value - zero) for value in values]


def pack_signed_int4(values: Sequence[int]) -> List[int]:
    """Pack signed two's-complement int4 values into bytes, two values per byte."""
    if len(values) % 2 != 0:
        raise ValueError("int4 packing requires an even number of values")
    if any(value < -8 or value > 7 for value in values):
        raise ValueError("int4 values must lie in [-8, 7]")

    packed = []
    for low, high in zip(values[0::2], values[1::2]):
        low_nibble = low & 0x0F
        high_nibble = high & 0x0F
        packed.append(low_nibble | (high_nibble << 4))
    return packed


def unpack_signed_int4(values: Sequence[int]) -> List[int]:
    """Invert :func:`pack_signed_int4`."""
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("packed values must lie in [0, 255]")

    unpacked = []
    for value in values:
        for nibble in (value & 0x0F, (value >> 4) & 0x0F):
            unpacked.append(nibble - 16 if nibble >= 8 else nibble)
    return unpacked


def is_power_of_two(size: int) -> bool:
    return size > 0 and (size & (size - 1)) == 0


def select_hadamard_plan(size: int) -> Tuple[int, int]:
    """Return ``(remainder_dimension, power_of_two_block)`` for known LLaMA shapes.

    QuaRot's practical transforms factor dimensions such as LLaMA-2 7B's MLP
    width (11008 = 172 * 64).  The non-power-of-two remainder requires a
    separately supplied orthogonal matrix in framework integration; this helper
    only validates the shape plan and never substitutes a different transform.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    for remainder in (172, 156, 140, 108, 60, 52, 40, 36, 28, 20, 12):
        if size % remainder == 0 and is_power_of_two(size // remainder):
            return remainder, size // remainder
    if is_power_of_two(size):
        return 1, size
    raise ValueError("no documented Hadamard factorization for size {}".format(size))


def normalized_hadamard(values: Sequence[float]) -> List[float]:
    """Apply an orthonormal Walsh-Hadamard transform using in-place butterflies."""
    if not is_power_of_two(len(values)):
        raise ValueError("Hadamard length must be a positive power of two")

    result = [float(value) for value in values]
    width = 1
    stage_scale = math.sqrt(0.5)
    while width < len(result):
        for start in range(0, len(result), 2 * width):
            for offset in range(width):
                left = result[start + offset]
                right = result[start + width + offset]
                result[start + offset] = (left + right) * stage_scale
                result[start + width + offset] = (left - right) * stage_scale
        width *= 2
    return result


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(left, right))


def matvec(weight: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    """Apply a row-major linear weight matrix as PyTorch ``F.linear`` does."""
    if not weight or not vector:
        raise ValueError("weight and vector must not be empty")
    if any(len(row) != len(vector) for row in weight):
        raise ValueError("weight shape is incompatible with vector")
    return [dot(row, vector) for row in weight]


def right_multiply(weight: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Return ``weight @ matrix`` for small reference matrices."""
    if not weight or not matrix:
        raise ValueError("matrices must not be empty")
    width = len(matrix)
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be square")
    if any(len(row) != width for row in weight):
        raise ValueError("weight and matrix shapes are incompatible")
    return [[sum(row[k] * matrix[k][col] for k in range(width)) for col in range(width)] for row in weight]


def hadamard_matrix(size: int) -> List[List[float]]:
    """Construct a small orthonormal Hadamard matrix for transformation tests."""
    if not is_power_of_two(size):
        raise ValueError("Hadamard size must be a positive power of two")
    return [normalized_hadamard([1.0 if row == col else 0.0 for col in range(size)]) for row in range(size)]
