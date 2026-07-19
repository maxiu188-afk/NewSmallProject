"""W4A8 linear module backed by the owned CUDA integer-accumulator kernel."""

from __future__ import annotations

import torch
from torch import nn

from repro.w4a8_cuda import grouped_int32_matmul


def pack_w4_weight(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Return owned-layout packed W4 bytes and FP32 per-output/group scales."""
    if weight.ndim != 2 or weight.shape[1] % group_size or group_size % 2:
        raise ValueError("weight must be [out, in] with even group_size dividing in_features")
    out_features, in_features = weight.shape
    groups = in_features // group_size
    values = weight.float().reshape(out_features, groups, group_size)
    scales = values.abs().amax(dim=-1).div(7.0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.clamp(torch.round(values / scales.unsqueeze(-1)), -8, 7).to(torch.int8).reshape(
        out_features, in_features
    )
    low = (quantized[:, 0::2].to(torch.int16) & 0x0F).to(torch.uint8)
    high = (quantized[:, 1::2].to(torch.int16) & 0x0F).to(torch.uint8)
    return (low | (high << 4)).contiguous(), scales.float().contiguous()


def quantize_a8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric A8 quantization for the deployment module."""
    values = inputs.reshape(-1, inputs.shape[-1]).float()
    scales = values.abs().amax(dim=-1, keepdim=True).div(127.0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return torch.clamp(torch.round(values / scales), -128, 127).to(torch.int8).contiguous(), scales


def w4a8_reference_linear(inputs: torch.Tensor, packed_weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independent floating reference with the same packed values and scales."""
    tokens, in_features = inputs.reshape(-1, inputs.shape[-1]).shape
    out_features = packed_weight.shape[0]
    groups = scales.shape[1]
    low = (packed_weight & 0x0F).to(torch.int16)
    high = (packed_weight >> 4).to(torch.int16)
    weights = torch.empty((out_features, in_features), dtype=torch.int8, device=packed_weight.device)
    weights[:, 0::2] = torch.where(low >= 8, low - 16, low).to(torch.int8)
    weights[:, 1::2] = torch.where(high >= 8, high - 16, high).to(torch.int8)
    activations, activation_scales = quantize_a8(inputs)
    group_size = in_features // groups
    dequantized_weights = weights.reshape(out_features, groups, group_size).float() * scales.unsqueeze(-1)
    dequantized_activations = activations.reshape(tokens, groups, group_size).float() * activation_scales.view(
        tokens, 1, 1
    )
    return torch.einsum("tgi,ogi->to", dequantized_activations, dequantized_weights)


class W4A8Linear(nn.Module):
    """Inference-only W4A8 linear that calls the verified int32 CUDA kernel."""

    def __init__(self, packed_weight: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("bias", None if bias is None else bias.float())

    @classmethod
    def from_float(cls, linear: nn.Linear, group_size: int = 128) -> "W4A8Linear":
        packed_weight, scales = pack_w4_weight(linear.weight.detach(), group_size)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(packed_weight, scales, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape[:-1]
        activations, activation_scales = quantize_a8(inputs)
        groups = self.weight_scales.shape[1]
        group_size = activations.shape[1] // groups
        accumulators = grouped_int32_matmul(activations, self.packed_weight, group_size)
        output = (accumulators.float() * self.weight_scales.unsqueeze(0) * activation_scales.unsqueeze(-1)).sum(dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output.view(*original_shape, self.packed_weight.shape[0])
