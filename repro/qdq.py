"""Small, explicit QDQ operators shared by the portable fake-quant path."""

from typing import Final

import torch


_NO_QUANT_BITS: Final[int] = 16


def qdq_last_axis(values: torch.Tensor, bits: int, *, symmetric: bool = True, clip_ratio: float = 1.0) -> torch.Tensor:
    """Quantize/dequantize independently for every prefix of the last axis.

    The operator intentionally returns floating-point values: it is an
    algorithmic fake-quantization primitive, not packed storage or an integer
    kernel.  ``clip_ratio`` contracts the observed range around zero before
    quantization and must be in ``(0, 1]``.
    """
    if bits >= _NO_QUANT_BITS:
        return values
    if bits < 2:
        raise ValueError("bits must be at least 2")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in (0, 1]")

    if symmetric:
        maxq = 2 ** (bits - 1) - 1
        bound = values.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scale = torch.where(bound == 0, torch.ones_like(bound), bound / maxq)
        quantized = torch.clamp(torch.round(values / scale), -maxq - 1, maxq)
        return quantized * scale

    qmax = 2**bits - 1
    minimum = values.amin(dim=-1, keepdim=True) * clip_ratio
    maximum = values.amax(dim=-1, keepdim=True) * clip_ratio
    scale = (maximum - minimum) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.clamp(torch.round(-minimum / scale), 0, qmax)
    quantized = torch.clamp(torch.round(values / scale) + zero, 0, qmax)
    return (quantized - zero) * scale


def qdq_attention_tensor(
    values: torch.Tensor,
    bits: int,
    *,
    group_size: int,
    symmetric: bool,
    clip_ratio: float,
) -> torch.Tensor:
    """QDQ an attention tensor shaped ``[batch, heads, tokens, head_dim]``.

    QuaRot's K-cache options support token-wise quantization across all heads
    (``group_size=-1``) or independent per-head vectors
    (``group_size=head_dim``).  This helper keeps those two semantics explicit
    instead of silently treating both settings as per-head quantization.
    """
    if values.ndim != 4:
        raise ValueError("attention QDQ expects [batch, heads, tokens, head_dim]")
    if bits >= _NO_QUANT_BITS:
        return values
    batch, heads, tokens, head_dim = values.shape
    if group_size == -1:
        token_vectors = values.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
        quantized = qdq_last_axis(token_vectors, bits, symmetric=symmetric, clip_ratio=clip_ratio)
        return quantized.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    if group_size == head_dim:
        return qdq_last_axis(values, bits, symmetric=symmetric, clip_ratio=clip_ratio)
    raise ValueError("attention group_size must be -1 or head_dim")
