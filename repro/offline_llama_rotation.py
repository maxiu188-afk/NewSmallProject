"""Offline-only QuaRot-style reparameterization for standard Llama models.

This module deliberately excludes activation quantization, online MLP/QK
Hadamards, custom kernels, and serving code.  Its output keeps the ordinary
Transformers Llama module layout so a later, independently validated
quantizer can emit a format supported by vLLM.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from repro.structured_hadamard import (
    normalized_structured_hadamard_matrix,
    supports_structured_hadamard,
)
from repro.torch_smoke import normalized_hadamard_matrix


def _norm_epsilon(module: nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        if hasattr(module, name):
            return float(getattr(module, name))
    raise ValueError("unsupported RMSNorm module: missing epsilon attribute")


def _fuse_norm_scale(norm: nn.Module, linears: tuple[nn.Linear, ...]) -> None:
    if not hasattr(norm, "weight") or norm.weight is None:
        raise ValueError("offline rotation requires affine RMSNorm weights")
    scale = norm.weight.detach().clone()
    with torch.no_grad():
        for linear in linears:
            fused = linear.weight.float() * scale.float()
            linear.weight.copy_(fused.to(linear.weight.dtype))
        norm.weight.fill_(1)


def _residual_rotation(
    size: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, str]:
    if size > 0 and not size & (size - 1):
        return normalized_hadamard_matrix(size, dtype, device), "walsh"
    if supports_structured_hadamard(size):
        return normalized_structured_hadamard_matrix(size, dtype, device), "structured"
    raise ValueError(
        "offline residual rotation requires a power-of-two or supported structured Hadamard size"
    )


def _copy_matmul_(target: torch.Tensor, *factors: torch.Tensor) -> None:
    work = factors[0].float()
    for factor in factors[1:]:
        work = work @ factor.float()
    if work.shape != target.shape and work.numel() == target.numel():
        work = work.reshape(target.shape)
    target.copy_(work.to(target.dtype))


def _untie_output_head_if_needed(model: nn.Module) -> bool:
    input_weight = model.model.embed_tokens.weight
    output_weight = model.lm_head.weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        return False
    replacement = nn.Linear(
        model.lm_head.in_features,
        model.lm_head.out_features,
        bias=model.lm_head.bias is not None,
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(output_weight)
        if replacement.bias is not None:
            replacement.bias.copy_(model.lm_head.bias)
    model.lm_head = replacement
    model.config.tie_word_embeddings = False
    return True


@torch.inference_mode()
def apply_offline_llama_rotation(model: nn.Module, *, rotate_values: bool = True) -> Dict[str, Any]:
    """Apply only rotations that can remain fused in a standard Llama checkpoint.

    The residual-space normalized Hadamard is absorbed into embeddings, the
    language-model head, and every decoder input/output projection.  The
    optional per-head V/O Hadamard pair is also fully absorbed into weights.
    No runtime wrapper or non-standard module is installed.
    """

    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "llama":
        raise ValueError("offline rotation currently supports model_type=llama only")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("model does not expose the standard Llama decoder layout")

    hidden_size = int(config.hidden_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    if hidden_size % num_q_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // num_q_heads
    if head_dim & (head_dim - 1):
        raise ValueError("offline V/O rotation requires a power-of-two head_dim")

    parameter = next(model.parameters())
    device = parameter.device
    compute_dtype = torch.float32
    residual, residual_kind = _residual_rotation(hidden_size, compute_dtype, device)
    head = normalized_hadamard_matrix(head_dim, compute_dtype, device)
    q_head_block = torch.block_diag(*([head] * num_q_heads))
    kv_head_block = torch.block_diag(*([head] * num_kv_heads))

    output_head_was_untied = _untie_output_head_if_needed(model)
    for layer in model.model.layers:
        _fuse_norm_scale(
            layer.input_layernorm,
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
        )
        _fuse_norm_scale(
            layer.post_attention_layernorm,
            (layer.mlp.up_proj, layer.mlp.gate_proj),
        )
    _fuse_norm_scale(model.model.norm, (model.lm_head,))

    _copy_matmul_(model.model.embed_tokens.weight, model.model.embed_tokens.weight, residual)
    _copy_matmul_(model.lm_head.weight, model.lm_head.weight, residual)
    for layer in model.model.layers:
        attention, mlp = layer.self_attn, layer.mlp
        for linear in (attention.q_proj, attention.k_proj, mlp.up_proj, mlp.gate_proj):
            _copy_matmul_(linear.weight, linear.weight, residual)

        if rotate_values:
            _copy_matmul_(
                attention.v_proj.weight,
                kv_head_block,
                attention.v_proj.weight,
                residual,
            )
            _copy_matmul_(
                attention.o_proj.weight,
                residual.T,
                attention.o_proj.weight,
                q_head_block,
            )
            if attention.v_proj.bias is not None:
                _copy_matmul_(
                    attention.v_proj.bias,
                    kv_head_block,
                    attention.v_proj.bias.unsqueeze(1),
                )
        else:
            _copy_matmul_(attention.v_proj.weight, attention.v_proj.weight, residual)
            _copy_matmul_(attention.o_proj.weight, residual.T, attention.o_proj.weight)
        if attention.o_proj.bias is not None:
            _copy_matmul_(
                attention.o_proj.bias,
                residual.T,
                attention.o_proj.bias.unsqueeze(1),
            )

        _copy_matmul_(mlp.down_proj.weight, residual.T, mlp.down_proj.weight)
        if mlp.down_proj.bias is not None:
            _copy_matmul_(
                mlp.down_proj.bias,
                residual.T,
                mlp.down_proj.bias.unsqueeze(1),
            )

    return {
        "applied": True,
        "kind": "offline_fused_residual_hadamard",
        "residual_hadamard_kind": residual_kind,
        "model_type": "llama",
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "value_output_rotation": rotate_values,
        "output_head_was_untied": output_head_was_untied,
        "online_mlp_rotation": False,
        "post_rope_qk_rotation": False,
        "module_layout": "standard_transformers_llama",
    }


def assert_standard_llama_layout(model: nn.Module) -> None:
    """Reject runtime wrappers before exporting a serving-oriented checkpoint."""

    for layer in model.model.layers:
        modules = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        )
        if not all(isinstance(module, nn.Linear) for module in modules):
            raise ValueError("offline-rotated checkpoint contains a non-standard decoder linear")
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            _norm_epsilon(norm)
            if not torch.equal(norm.weight, torch.ones_like(norm.weight)):
                raise ValueError("RMSNorm scale was not fully fused")
    _norm_epsilon(model.model.norm)
    if not torch.equal(model.model.norm.weight, torch.ones_like(model.model.norm.weight)):
        raise ValueError("final RMSNorm scale was not fully fused")
