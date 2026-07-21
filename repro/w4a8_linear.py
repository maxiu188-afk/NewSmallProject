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


def unpack_w4_weight(packed_weight: torch.Tensor) -> torch.Tensor:
    """Unpack owned signed-int4 bytes to an int8 matrix for an independent oracle."""
    if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
        raise ValueError("packed_weight must be a rank-two uint8 tensor")
    low = (packed_weight & 0x0F).to(torch.int16)
    high = (packed_weight >> 4).to(torch.int16)
    weights = torch.empty((packed_weight.shape[0], packed_weight.shape[1] * 2), dtype=torch.int8, device=packed_weight.device)
    weights[:, 0::2] = torch.where(low >= 8, low - 16, low).to(torch.int8)
    weights[:, 1::2] = torch.where(high >= 8, high - 16, high).to(torch.int8)
    return weights


def quantize_a8(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric A8 quantization for the deployment module."""
    values = inputs.reshape(-1, inputs.shape[-1]).float()
    scales = values.abs().amax(dim=-1, keepdim=True).div(127.0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return torch.clamp(torch.round(values / scales), -128, 127).to(torch.int8).contiguous(), scales


def w4a8_reference_linear(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    input_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent floating reference with the same packed values and scales."""
    original_shape = inputs.shape[:-1]
    if input_permutation is not None:
        inputs = inputs.index_select(-1, input_permutation)
    tokens, in_features = inputs.reshape(-1, inputs.shape[-1]).shape
    out_features = packed_weight.shape[0]
    groups = scales.shape[1]
    weights = unpack_w4_weight(packed_weight)
    if weights.shape != (out_features, in_features):
        raise ValueError("packed weight and input feature dimensions differ")
    activations, activation_scales = quantize_a8(inputs)
    group_size = in_features // groups
    # This must remain independent from the CUDA extension and avoid a float
    # GEMM (which may select TF32 on NVIDIA).  It exactly mirrors the declared
    # integer accumulation and post-group scaling semantics in Torch ops.
    accumulators = (
        activations.reshape(tokens, 1, groups, group_size).to(torch.int32)
        * weights.reshape(1, out_features, groups, group_size).to(torch.int32)
    ).sum(dim=-1, dtype=torch.int32)
    output = (accumulators.float() * scales.unsqueeze(0) * activation_scales.unsqueeze(-1)).sum(dim=-1)
    return output.view(*original_shape, out_features)


class W4A8Linear(nn.Module):
    """Inference-only W4A8 linear that calls the verified int32 CUDA kernel."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        input_permutation: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if packed_weight.ndim != 2 or scales.ndim != 2:
            raise ValueError("packed_weight and scales must be matrices")
        in_features = packed_weight.shape[1] * 2
        if in_features % scales.shape[1]:
            raise ValueError("scale groups must divide in_features")
        if input_permutation is not None:
            if input_permutation.ndim != 1 or input_permutation.numel() != in_features:
                raise ValueError("input_permutation must cover every input feature")
            if not torch.equal(torch.sort(input_permutation.cpu()).values, torch.arange(in_features)):
                raise ValueError("input_permutation must be a permutation of input features")
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("bias", None if bias is None else bias.float())
        self.register_buffer("input_permutation", input_permutation)
        self.output_dtype = output_dtype

    @classmethod
    def from_float(cls, linear: nn.Linear, group_size: int = 128) -> "W4A8Linear":
        packed_weight, scales = pack_w4_weight(linear.weight.detach(), group_size)
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(packed_weight, scales, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape[:-1]
        if self.input_permutation is not None:
            inputs = inputs.index_select(-1, self.input_permutation)
        activations, activation_scales = quantize_a8(inputs)
        groups = self.weight_scales.shape[1]
        group_size = activations.shape[1] // groups
        accumulators = grouped_int32_matmul(activations, self.packed_weight, group_size)
        output = (accumulators.float() * self.weight_scales.unsqueeze(0) * activation_scales.unsqueeze(-1)).sum(dim=-1)
        if self.bias is not None:
            output = output + self.bias
        if self.output_dtype is not None:
            output = output.to(self.output_dtype)
        return output.view(*original_shape, self.packed_weight.shape[0])


class PackedW4A8ReferenceLinear(nn.Module):
    """Floating oracle module for the exact packed W4/A8 representation.

    This is deliberately distinct from ``W4A8Linear``: it follows the same
    signed-int4 packing, scales, input permutation, and A8 rule, but performs
    the final product through the independent floating reference.  It is the
    immediate numerical reference for a transformer-level replacement test.
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        input_permutation: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scales", scales)
        self.register_buffer("bias", None if bias is None else bias.float())
        self.register_buffer("input_permutation", input_permutation)
        self.output_dtype = output_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = w4a8_reference_linear(inputs, self.packed_weight, self.weight_scales, self.input_permutation)
        if self.bias is not None:
            output = output + self.bias
        if self.output_dtype is not None:
            output = output.to(self.output_dtype)
        return output.view(*inputs.shape[:-1], self.packed_weight.shape[0])
