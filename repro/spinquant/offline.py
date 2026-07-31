"""Offline R1/R2 fusion and W4A16 fake quantization for standard Llama.

The learned rotations are absorbed into ordinary embedding, RMSNorm, and
Linear parameters. No runtime wrapper remains, so evaluation exercises the
installed Transformers Llama forward rather than copied model source.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch import Tensor, nn

from repro.qdq import qdq_last_axis
from repro.spinquant.llama_adapter import (
    _left_apply_head_rotation_transpose,
    _right_apply_head_rotation,
)
from repro.spinquant.rotations import SpinQuantRotations


def _copy_(target: Tensor, values: Tensor) -> None:
    if target.shape != values.shape:
        raise ValueError("offline SpinQuant transform changed a parameter shape")
    target.copy_(values.to(dtype=target.dtype))


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


def _fuse_norm_scale(norm: nn.Module, linears: Iterable[nn.Linear]) -> None:
    if not hasattr(norm, "weight") or norm.weight is None:
        raise ValueError("offline SpinQuant requires affine RMSNorm weights")
    scale = norm.weight.detach().float()
    for linear in linears:
        _copy_(linear.weight, linear.weight.float() * scale)
    norm.weight.fill_(1)


def _validate_layout(
    model: nn.Module,
    rotations: SpinQuantRotations,
) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "llama":
        raise ValueError("offline SpinQuant currently supports model_type=llama")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("model does not expose the standard Llama decoder layout")
    if int(getattr(config, "pretraining_tp", 1)) != 1:
        raise ValueError("offline SpinQuant requires pretraining_tp=1")
    hidden_size = int(config.hidden_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = hidden_size // num_q_heads
    if (
        rotations.hidden_size != hidden_size
        or rotations.head_dim != head_dim
        or rotations.num_layers != len(model.model.layers)
    ):
        raise ValueError("rotation shapes do not match the selected Llama model")
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
            raise ValueError("offline SpinQuant requires standard Llama linears")
    return num_q_heads, num_kv_heads, head_dim


@torch.inference_mode()
def apply_spinquant_llama_offline(
    model: nn.Module,
    rotations: SpinQuantRotations,
) -> Dict[str, Any]:
    """Absorb learned R1/R2 into a standard Transformers Llama model."""

    num_q_heads, num_kv_heads, head_dim = _validate_layout(model, rotations)
    r1 = rotations.r1.detach().float()
    output_head_was_untied = _untie_output_head_if_needed(model)

    for layer in model.model.layers:
        _fuse_norm_scale(
            layer.input_layernorm,
            (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ),
        )
        _fuse_norm_scale(
            layer.post_attention_layernorm,
            (layer.mlp.gate_proj, layer.mlp.up_proj),
        )
    _fuse_norm_scale(model.model.norm, (model.lm_head,))

    _copy_(
        model.model.embed_tokens.weight,
        model.model.embed_tokens.weight.float() @ r1,
    )
    _copy_(model.lm_head.weight, model.lm_head.weight.float() @ r1)

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        mlp = layer.mlp
        r2 = rotations.r2[layer_index].detach().float()
        for linear in (
            attention.q_proj,
            attention.k_proj,
            mlp.gate_proj,
            mlp.up_proj,
        ):
            _copy_(linear.weight, linear.weight.float() @ r1)

        value = _left_apply_head_rotation_transpose(
            attention.v_proj.weight.float(),
            r2,
            num_kv_heads,
        )
        _copy_(attention.v_proj.weight, value @ r1)

        output = _right_apply_head_rotation(
            attention.o_proj.weight.float(),
            r2,
            num_q_heads,
        )
        _copy_(attention.o_proj.weight, r1.transpose(-1, -2) @ output)
        _copy_(
            mlp.down_proj.weight,
            r1.transpose(-1, -2) @ mlp.down_proj.weight.float(),
        )

        if attention.v_proj.bias is not None:
            heads = attention.v_proj.bias.float().reshape(
                num_kv_heads,
                head_dim,
                1,
            )
            _copy_(
                attention.v_proj.bias,
                (
                    r2.transpose(-1, -2).unsqueeze(0) @ heads
                ).reshape_as(attention.v_proj.bias),
            )
        for writer in (attention.o_proj, mlp.down_proj):
            if writer.bias is not None:
                _copy_(
                    writer.bias,
                    r1.transpose(-1, -2) @ writer.bias.float(),
                )

    return {
        "applied": True,
        "model_type": "llama",
        "hidden_size": rotations.hidden_size,
        "head_dim": head_dim,
        "num_layers": rotations.num_layers,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "output_head_was_untied": output_head_was_untied,
        "online_rotation_modules": 0,
        "module_layout": "standard_transformers_llama",
    }


def _decoder_linears(model: nn.Module) -> Iterable[nn.Linear]:
    for layer in model.model.layers:
        yield layer.self_attn.q_proj
        yield layer.self_attn.k_proj
        yield layer.self_attn.v_proj
        yield layer.self_attn.o_proj
        yield layer.mlp.gate_proj
        yield layer.mlp.up_proj
        yield layer.mlp.down_proj


@torch.inference_mode()
def fake_quantize_llama_decoder_w4_(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
) -> Dict[str, Any]:
    """Apply one immutable group-wise weight QDQ pass to decoder linears."""

    if bits < 2 or bits >= 16:
        raise ValueError("decoder fake quantization requires bits in [2, 15]")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    count = 0
    for linear in _decoder_linears(model):
        weight = linear.weight
        width = weight.shape[-1]
        if width % group_size:
            raise ValueError("decoder weight width is not divisible by group_size")
        grouped = weight.float().reshape(
            *weight.shape[:-1],
            width // group_size,
            group_size,
        )
        quantized = qdq_last_axis(grouped, bits, symmetric=symmetric)
        weight.copy_(quantized.reshape_as(weight).to(dtype=weight.dtype))
        count += 1
    return {
        "applied": True,
        "runtime_form": "floating_qdq",
        "weight_bits": bits,
        "activation_bits": 16,
        "group_size": group_size,
        "symmetric": symmetric,
        "quantized_decoder_linears": count,
        "quantized_lm_head": False,
    }
