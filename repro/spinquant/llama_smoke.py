"""End-to-end random Tiny Llama SpinQuant W4 fake-quantization smoke."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from repro.spinquant.llama_adapter import (
    SpinQuantFakeQuantSpec,
    apply_spinquant_llama_training_adapter,
)
from repro.spinquant.rotations import SpinQuantRotations
from repro.spinquant.stiefel import CayleySGD


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {"seed", "model", "data", "quantization", "optimization"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing Tiny Llama SpinQuant config keys: {sorted(missing)}")
    model = config["model"]
    if int(model["hidden_size"]) % int(model["num_attention_heads"]):
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if int(model["num_attention_heads"]) % int(model["num_key_value_heads"]):
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if int(config["optimization"]["steps"]) < 1:
        raise ValueError("optimization.steps must be positive")


def run_tiny_llama_spinquant_fake_quant(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Optimize R1/R2 against the logits of one frozen random Tiny Llama."""

    _validate_config(config)
    from transformers import LlamaConfig, LlamaForCausalLM

    seed = int(config["seed"])
    model_spec = config["model"]
    data_spec = config["data"]
    quant_spec = config["quantization"]
    optimization = config["optimization"]
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    llama_config = LlamaConfig(
        vocab_size=int(model_spec["vocab_size"]),
        hidden_size=int(model_spec["hidden_size"]),
        intermediate_size=int(model_spec["intermediate_size"]),
        num_hidden_layers=int(model_spec["num_hidden_layers"]),
        num_attention_heads=int(model_spec["num_attention_heads"]),
        num_key_value_heads=int(model_spec["num_key_value_heads"]),
        max_position_embeddings=int(model_spec["max_position_embeddings"]),
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=bool(model_spec["tie_word_embeddings"]),
        use_cache=False,
    )
    reference = LlamaForCausalLM(llama_config).float().eval()
    candidate = copy.deepcopy(reference).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    input_ids = torch.randint(
        0,
        llama_config.vocab_size,
        (
            int(data_spec["samples"]),
            int(data_spec["sequence_length"]),
        ),
        generator=generator,
    )
    with torch.no_grad():
        target_logits = reference(input_ids=input_ids, use_cache=False).logits.float()

    head_dim = llama_config.hidden_size // llama_config.num_attention_heads
    rotations = SpinQuantRotations(
        hidden_size=llama_config.hidden_size,
        head_dim=head_dim,
        num_layers=llama_config.num_hidden_layers,
        seed=seed + 100,
    )
    adapter = apply_spinquant_llama_training_adapter(
        candidate,
        rotations,
        SpinQuantFakeQuantSpec(
            weight_bits=int(quant_spec["weight_bits"]),
            activation_bits=int(quant_spec["activation_bits"]),
            weight_group_size=int(quant_spec["weight_group_size"]),
            weight_symmetric=bool(quant_spec["weight_symmetric"]),
            activation_symmetric=bool(quant_spec["activation_symmetric"]),
            quantize_lm_head=bool(quant_spec["quantize_lm_head"]),
        ),
    )

    steps = int(optimization["steps"])
    initial_lr = float(optimization["learning_rate"])
    optimizer = CayleySGD(
        rotations.parameters(),
        lr=initial_lr,
        method=str(optimization["cayley_method"]),
        fixed_point_steps=int(optimization["fixed_point_steps"]),
    )
    losses = []
    for step in range(steps):
        optimizer.param_groups[0]["lr"] = initial_lr * (1.0 - step / steps)
        optimizer.zero_grad(set_to_none=True)
        logits = candidate(input_ids=input_ids, use_cache=False).logits.float()
        loss = F.mse_loss(logits, target_logits)
        if not torch.isfinite(loss):
            raise RuntimeError("Tiny Llama SpinQuant smoke produced a non-finite loss")
        loss.backward()
        losses.append(float(loss.detach()))
        optimizer.step()

    with torch.no_grad():
        final_logits = candidate(input_ids=input_ids, use_cache=False).logits.float()
        final_loss = float(F.mse_loss(final_logits, target_logits))
    errors = rotations.errors()
    return {
        "status": "passed",
        "scope": (
            "random-config Transformers Llama R1/R2 W4 fake-quant mechanism smoke; "
            "no pretrained weights, WikiText-2, GPTQ, downstream task, packing, or deployment"
        ),
        "adapter": adapter,
        "seed": seed,
        "tokens": int(input_ids.numel()),
        "steps": steps,
        "initial_loss": losses[0],
        "best_training_loss": min(losses),
        "final_loss": final_loss,
        "loss_ratio": final_loss / losses[0],
        "orthogonality_error": errors,
        "losses": losses,
    }
