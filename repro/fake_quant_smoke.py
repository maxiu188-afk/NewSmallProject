"""Local fake-quant smoke/ablation on the random tiny LLaMA model.

The implementation uses transparent symmetric QDQ: per-output-channel weights
and per-token activations. It is deliberately a correctness mechanism test,
not GPTQ, calibration, KV-cache quantization, or a paper-accuracy experiment.
"""

import copy
from contextlib import contextmanager
from typing import Dict, Iterator, List

import torch
from torch import nn

from repro.torch_smoke import rotate_tiny_llama_in_place
from transformers import LlamaConfig, LlamaForCausalLM


def symmetric_qdq(values: torch.Tensor, bits: int, reduction_dim: int) -> torch.Tensor:
    """Symmetric fake quantization with a scale per retained tensor slice."""
    if bits >= 16:
        return values
    if bits < 2:
        raise ValueError("bits must be at least 2")
    maxq = 2 ** (bits - 1) - 1
    scale = values.abs().amax(dim=reduction_dim, keepdim=True) / maxq
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(torch.round(values / scale), -maxq - 1, maxq)
    return quantized * scale


def quantize_linear_weights_in_place(model: nn.Module, bits: int) -> None:
    """Apply per-output-channel W-bit QDQ exactly once to every linear weight."""
    if bits >= 16:
        return
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(symmetric_qdq(module.weight, bits, reduction_dim=1))


@contextmanager
def quantize_linear_inputs(model: nn.Module, bits: int) -> Iterator[None]:
    """Temporarily apply per-token activation QDQ immediately before linears."""
    handles: List[torch.utils.hooks.RemovableHandle] = []
    if bits < 16:
        def hook(_module: nn.Module, inputs: tuple) -> tuple:
            return (symmetric_qdq(inputs[0], bits, reduction_dim=-1),) + inputs[1:]

        for module in model.modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
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


def _logits(model: LlamaForCausalLM, input_ids: torch.Tensor, activation_bits: int) -> torch.Tensor:
    with torch.inference_mode(), quantize_linear_inputs(model, activation_bits):
        return model(input_ids=input_ids, use_cache=False).logits


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    difference = (reference - candidate).abs()
    return {
        "mean_absolute_logit_error": difference.mean().item(),
        "max_absolute_logit_error": difference.max().item(),
    }


def run_tiny_fake_quant_ablation() -> Dict[str, Dict[str, float]]:
    """Run controlled F0–F4 QDQ cases on one deterministic random tiny model."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    input_ids = torch.tensor([[1, 7, 11, 17, 23], [2, 3, 5, 7, 11]], dtype=torch.long)
    baseline = LlamaForCausalLM(_tiny_config()).float().eval()
    with torch.inference_mode():
        reference_logits = baseline(input_ids=input_ids, use_cache=False).logits

    cases = {
        "F0-fp32": (copy.deepcopy(baseline).eval(), 16, 16, False),
        "F1-naive-w4": (copy.deepcopy(baseline).eval(), 4, 16, False),
        "F2-quarot-w4": (copy.deepcopy(baseline).eval(), 4, 16, True),
        "F3-naive-w4a4": (copy.deepcopy(baseline).eval(), 4, 4, False),
        "F4-quarot-w4a4": (copy.deepcopy(baseline).eval(), 4, 4, True),
    }
    results: Dict[str, Dict[str, float]] = {}
    for name, (model, weight_bits, activation_bits, rotate) in cases.items():
        if rotate:
            rotate_tiny_llama_in_place(model)
        quantize_linear_weights_in_place(model, weight_bits)
        logits = _logits(model, input_ids, activation_bits)
        result = _error(reference_logits, logits)
        result.update({
            "weight_bits": float(weight_bits),
            "activation_bits": float(activation_bits),
            "rotation_enabled": float(rotate),
        })
        results[name] = result
    return results
