"""SpinQuant activation QDQ primitives written from the paper protocol."""

from __future__ import annotations

import torch
from torch import Tensor

from repro.qdq import qdq_last_axis


def spinquant_activation_qdq(
    values: Tensor,
    bits: int,
    *,
    symmetric: bool,
    group_size: int = -1,
) -> Tensor:
    """Apply dynamic per-token activation QDQ at the requested granularity.

    For asymmetric ungrouped vectors, zero is included in the observed range.
    Positive ``group_size`` values use independent contiguous groups and an
    unconstrained zero point.  These details reproduce the mathematical
    SpinQuant activation-quantizer behavior without importing its code.
    """

    if bits >= 16:
        return values
    if bits < 2:
        raise ValueError("activation bits must be at least 2")
    width = values.shape[-1]
    if group_size == -1:
        grouped = values.reshape(-1, width)
    else:
        if group_size < 1 or width % group_size:
            raise ValueError("group_size must be -1 or divide the last axis")
        grouped = values.reshape(-1, width // group_size, group_size)

    if symmetric:
        quantized = qdq_last_axis(grouped, bits, symmetric=True)
        return quantized.reshape_as(values)

    minimum = grouped.amin(dim=-1, keepdim=True)
    maximum = grouped.amax(dim=-1, keepdim=True)
    if group_size == -1:
        minimum = torch.minimum(minimum, torch.zeros_like(minimum))
        maximum = torch.maximum(maximum, torch.zeros_like(maximum))

    qmax = 2**bits - 1
    all_zero = (minimum == 0) & (maximum == 0)
    safe_minimum = torch.where(all_zero, -torch.ones_like(minimum), minimum)
    safe_maximum = torch.where(all_zero, torch.ones_like(maximum), maximum)
    scale = (safe_maximum - safe_minimum) / qmax
    zero = torch.round(-safe_minimum / scale)
    quantized = torch.clamp(torch.round(grouped / scale) + zero, 0, qmax)
    return ((quantized - zero) * scale).reshape_as(values).to(dtype=values.dtype)


def ste_spinquant_activation_qdq(
    values: Tensor,
    bits: int,
    *,
    symmetric: bool,
    group_size: int = -1,
) -> Tensor:
    """SpinQuant activation QDQ with an identity straight-through gradient."""

    quantized = spinquant_activation_qdq(
        values,
        bits,
        symmetric=symmetric,
        group_size=group_size,
    )
    return values + (quantized - values).detach()
