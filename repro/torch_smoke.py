"""A tiny random LLaMA equivalence smoke test using PyTorch and Transformers.

    This module creates a model from ``LlamaConfig`` only. It never calls
``from_pretrained`` and therefore never downloads model weights or a tokenizer.
The Q/K post-RoPE head rotation remains covered by the framework-free algebraic
test because the upstream implementation injects it into attention internals.
"""

import copy
import math
import os
from pathlib import Path
from typing import Dict

import torch
from torch import nn

# Transformers creates cache-path constants while importing. Keep even an empty
# local smoke cache inside the project rather than relying on a user-home path.
_CACHE_HOME = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_HOME))

from transformers import LlamaConfig, LlamaForCausalLM


class UnitRMSNorm(nn.Module):
    """Weight-free RMSNorm used after fusing the original affine scale."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        return (values * torch.rsqrt(variance + self.eps)).to(input_dtype)


class HadamardInputLinear(nn.Module):
    """Apply an online Hadamard transform before an already-compensated linear."""

    def __init__(self, linear: nn.Linear, hadamard: torch.Tensor, num_heads: int = 1) -> None:
        super().__init__()
        self.linear = linear
        self.num_heads = num_heads
        self.register_buffer("hadamard", hadamard)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.num_heads * self.hadamard.shape[0]:
            raise ValueError("Hadamard shape is incompatible with the linear input")
        if self.num_heads == 1:
            rotated = values @ self.hadamard
        else:
            shape = values.shape
            rotated = values.reshape(*shape[:-1], self.num_heads, self.hadamard.shape[0])
            rotated = rotated @ self.hadamard
            rotated = rotated.reshape(shape)
        return self.linear(rotated)


def normalized_hadamard_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(size)


def _fuse_rmsnorm_into_inputs(model: LlamaForCausalLM) -> None:
    """Fuse RMSNorm scale into following linears without using upstream helpers."""
    with torch.no_grad():
        for layer in model.model.layers:
            input_scale = layer.input_layernorm.weight.detach()
            for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                linear.weight.mul_(input_scale)
            layer.input_layernorm = UnitRMSNorm(layer.input_layernorm.variance_epsilon)

            mlp_scale = layer.post_attention_layernorm.weight.detach()
            for linear in (layer.mlp.up_proj, layer.mlp.gate_proj):
                linear.weight.mul_(mlp_scale)
            layer.post_attention_layernorm = UnitRMSNorm(layer.post_attention_layernorm.variance_epsilon)

        final_scale = model.model.norm.weight.detach()
        model.lm_head.weight.mul_(final_scale)
        model.model.norm = UnitRMSNorm(model.model.norm.variance_epsilon)


def rotate_tiny_llama_in_place(model: LlamaForCausalLM) -> torch.Tensor:
    """Apply the QuaRot identities needed by this tiny local LLaMA smoke model.

    This implementation is deliberately small and device-agnostic. It validates
    transformations against a random model graph; it is not a replacement for
    the upstream fake-quant or CUDA implementation.
    """
    config = model.config
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden size must divide attention heads")
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError("this smoke test intentionally uses no grouped-query attention")
    if config.intermediate_size & (config.intermediate_size - 1):
        raise ValueError("tiny smoke intermediate size must be a power of two")

    _fuse_rmsnorm_into_inputs(model)
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    hidden_rotation = normalized_hadamard_matrix(config.hidden_size, dtype, device)
    head_dim = config.hidden_size // config.num_attention_heads
    head_rotation = normalized_hadamard_matrix(head_dim, dtype, device)
    head_block_rotation = torch.block_diag(*([head_rotation] * config.num_attention_heads))
    intermediate_rotation = normalized_hadamard_matrix(config.intermediate_size, dtype, device)

    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight @ hidden_rotation)
        model.lm_head.weight.copy_(model.lm_head.weight @ hidden_rotation)
        for layer in model.model.layers:
            attention = layer.self_attn
            mlp = layer.mlp
            for linear in (attention.q_proj, attention.k_proj, mlp.up_proj, mlp.gate_proj):
                linear.weight.copy_(linear.weight @ hidden_rotation)
            attention.v_proj.weight.copy_(head_block_rotation @ attention.v_proj.weight @ hidden_rotation)
            attention.o_proj.weight.copy_(hidden_rotation.T @ attention.o_proj.weight @ head_block_rotation)
            mlp.down_proj.weight.copy_(hidden_rotation.T @ mlp.down_proj.weight @ intermediate_rotation)

            mlp.down_proj = HadamardInputLinear(mlp.down_proj, intermediate_rotation)
    return hidden_rotation


def run_tiny_llama_smoke() -> Dict[str, float]:
    """Run a deterministic two-layer random LLaMA equivalence check on CPU."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
    )
    reference = LlamaForCausalLM(config).float().eval()
    rotated = copy.deepcopy(reference).eval()
    hidden_rotation = rotate_tiny_llama_in_place(rotated)
    input_ids = torch.tensor([[1, 7, 11, 17, 23], [2, 3, 5, 7, 11]], dtype=torch.long)

    with torch.inference_mode():
        reference_output = reference(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        rotated_output = rotated(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    restored_hidden_errors = []
    for original, transformed in zip(reference_output.hidden_states, rotated_output.hidden_states):
        restored = transformed @ hidden_rotation
        restored_hidden_errors.append((original - restored).abs().max().item())
    return {
        "max_logit_error": (reference_output.logits - rotated_output.logits).abs().max().item(),
        "max_hidden_error": max(restored_hidden_errors),
        "num_hidden_state_tensors": float(len(restored_hidden_errors)),
    }
