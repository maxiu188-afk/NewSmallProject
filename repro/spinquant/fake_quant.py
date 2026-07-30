"""Deterministic tiny SpinQuant fake-quantization optimization smoke."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from repro.qdq import qdq_last_axis
from repro.spinquant.rotations import SpinQuantRotations, fuse_value_output_pair
from repro.spinquant.stiefel import CayleySGD


def ste_qdq_last_axis(values: Tensor, bits: int, *, symmetric: bool = True) -> Tensor:
    """QDQ in the forward pass with an identity straight-through gradient."""

    quantized = qdq_last_axis(values, bits, symmetric=symmetric)
    return values + (quantized - values).detach()


def _deterministic_weights(
    num_layers: int,
    hidden_size: int,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    values = torch.randn(
        (num_layers, hidden_size, hidden_size),
        generator=generator,
        dtype=torch.float32,
    )
    outputs = torch.randn(
        (num_layers, hidden_size, hidden_size),
        generator=generator,
        dtype=torch.float32,
    )
    # Fixed channel imbalance makes rotation quality visible at four bits.
    scale = torch.logspace(-1.0, 1.0, hidden_size, dtype=torch.float32)
    values = values * scale.view(1, -1, 1)
    outputs = outputs * scale.flip(0).view(1, -1, 1)
    return values / hidden_size**0.5, outputs / hidden_size**0.5


def _reference_outputs(inputs: Tensor, value_weights: Tensor, output_weights: Tensor) -> Tensor:
    outputs = []
    for value_weight, output_weight in zip(value_weights, output_weights):
        outputs.append(F.linear(F.linear(inputs, value_weight), output_weight))
    return torch.stack(outputs, dim=0)


def _quantized_outputs(
    inputs: Tensor,
    value_weights: Tensor,
    output_weights: Tensor,
    rotations: SpinQuantRotations,
    *,
    weight_bits: int,
) -> Tensor:
    rotated_inputs = inputs @ rotations.r1
    outputs = []
    for layer, (value_weight, output_weight) in enumerate(
        zip(value_weights, output_weights)
    ):
        value_rotated, output_rotated = fuse_value_output_pair(
            value_weight,
            output_weight,
            rotations.r1,
            rotations.r2_block(layer),
        )
        value_quantized = ste_qdq_last_axis(value_rotated, weight_bits)
        output_quantized = ste_qdq_last_axis(output_rotated, weight_bits)
        rotated_output = F.linear(
            F.linear(rotated_inputs, value_quantized),
            output_quantized,
        )
        outputs.append(rotated_output @ rotations.r1.transpose(-1, -2))
    return torch.stack(outputs, dim=0)


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {"seed", "model", "quantization", "optimization"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing SpinQuant smoke config keys: {sorted(missing)}")
    model = config["model"]
    optimization = config["optimization"]
    if int(model["hidden_size"]) % int(model["head_dim"]):
        raise ValueError("hidden_size must be divisible by head_dim")
    if int(config["quantization"]["weight_bits"]) >= 16:
        raise ValueError("the learning smoke requires sub-16-bit weight fake quantization")
    if int(optimization["steps"]) < 1:
        raise ValueError("optimization.steps must be positive")


def run_tiny_spinquant_fake_quant(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Learn tiny R1/R2 rotations against a W4 fake-weight objective."""

    _validate_config(config)
    seed = int(config["seed"])
    model_config = config["model"]
    quantization = config["quantization"]
    optimization = config["optimization"]
    hidden_size = int(model_config["hidden_size"])
    head_dim = int(model_config["head_dim"])
    num_layers = int(model_config["num_layers"])
    samples = int(model_config["samples"])
    weight_bits = int(quantization["weight_bits"])
    steps = int(optimization["steps"])
    initial_lr = float(optimization["learning_rate"])

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.randn((samples, hidden_size), generator=generator)
    value_weights, output_weights = _deterministic_weights(
        num_layers,
        hidden_size,
        generator=generator,
    )
    reference = _reference_outputs(inputs, value_weights, output_weights)
    rotations = SpinQuantRotations(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_layers=num_layers,
        seed=seed + 100,
    )
    optimizer = CayleySGD(
        rotations.parameters(),
        lr=initial_lr,
        method=str(optimization["cayley_method"]),
        fixed_point_steps=int(optimization["fixed_point_steps"]),
    )

    losses = []
    for step in range(steps):
        fraction = 1.0 - step / steps
        optimizer.param_groups[0]["lr"] = initial_lr * fraction
        optimizer.zero_grad(set_to_none=True)
        candidate = _quantized_outputs(
            inputs,
            value_weights,
            output_weights,
            rotations,
            weight_bits=weight_bits,
        )
        loss = F.mse_loss(candidate, reference)
        if not torch.isfinite(loss):
            raise RuntimeError("SpinQuant tiny smoke produced a non-finite loss")
        loss.backward()
        losses.append(float(loss.detach()))
        optimizer.step()

    with torch.no_grad():
        final = _quantized_outputs(
            inputs,
            value_weights,
            output_weights,
            rotations,
            weight_bits=weight_bits,
        )
        final_loss = float(F.mse_loss(final, reference))
    errors = rotations.errors()
    return {
        "status": "passed",
        "scope": (
            "independent tiny W4 fake-weight optimization of R1/R2; "
            "not a pretrained-model, WikiText-2, GPTQ, packed-kernel, or deployment result"
        ),
        "seed": seed,
        "weight_bits": weight_bits,
        "steps": steps,
        "initial_loss": losses[0],
        "best_training_loss": min(losses),
        "final_loss": final_loss,
        "loss_ratio": final_loss / losses[0],
        "orthogonality_error": errors,
        "losses": losses,
    }
