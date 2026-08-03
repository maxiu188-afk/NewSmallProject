"""Config-driven R1/R2 optimization on a provided standard Llama model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from repro.spinquant.activation_qdq import (
    FP8_E4M3FN_MAX,
    FP8_MIN_SCALING_FACTOR,
)
from repro.spinquant.llama_adapter import (
    SpinQuantFakeQuantSpec,
    apply_spinquant_llama_training_adapter,
)
from repro.spinquant.rotations import SpinQuantRotations
from repro.spinquant.stiefel import CayleySGD


def _validate_rotation_objective(quantization: Mapping[str, Any]) -> str:
    """Validate the quantized network used to learn the rotations."""

    objective = str(quantization.get("rotation_objective", ""))
    if objective not in {"weight_qdq", "activation_qdq", "fp8_activation_qdq"}:
        raise ValueError(
            "quantization.rotation_objective must be weight_qdq, activation_qdq, "
            "or fp8_activation_qdq"
        )
    weight_bits = int(quantization["weight_bits"])
    activation_bits = int(quantization["activation_bits"])
    if objective == "weight_qdq" and weight_bits >= 16:
        raise ValueError("weight_qdq rotation learning requires sub-16-bit weights")
    if objective == "activation_qdq":
        if weight_bits != 16 or activation_bits >= 16:
            raise ValueError(
                "activation_qdq rotation learning requires 16-bit weights and "
                "sub-16-bit activations"
            )
        if bool(quantization.get("activation_clipping", False)):
            raise ValueError(
                "paper-aligned activation_qdq currently requires unclipped min-max QDQ"
            )
        if bool(quantization["activation_symmetric"]):
            raise ValueError(
                "paper-aligned activation_qdq requires asymmetric activations"
            )
        if int(quantization.get("activation_o_proj_group_size", -1)) < 1:
            raise ValueError(
                "paper-aligned activation_qdq requires grouped o_proj inputs"
            )
        if not bool(
            quantization.get("activation_ungrouped_include_zero", False)
        ):
            raise ValueError(
                "paper-aligned activation_qdq must include zero in ungrouped ranges"
            )
    if objective == "fp8_activation_qdq":
        if weight_bits != 16 or activation_bits != 8:
            raise ValueError(
                "fp8_activation_qdq requires 16-bit weights and 8-bit activations"
            )
        if str(quantization.get("activation_type", "")) != "float":
            raise ValueError("fp8_activation_qdq requires activation_type=float")
        if str(quantization.get("activation_dtype", "")) != "float8_e4m3fn":
            raise ValueError(
                "fp8_activation_qdq requires activation_dtype=float8_e4m3fn"
            )
        if not bool(quantization.get("activation_dynamic", False)):
            raise ValueError("fp8_activation_qdq requires dynamic activation scales")
        if str(quantization.get("activation_strategy", "")) != "token":
            raise ValueError("fp8_activation_qdq requires strategy=token")
        if not bool(quantization.get("activation_symmetric", False)):
            raise ValueError("fp8_activation_qdq requires symmetric activation scales")
        if bool(quantization.get("activation_clipping", False)):
            raise ValueError("fp8_activation_qdq does not support activation clipping")
        if int(quantization.get("activation_o_proj_group_size", -1)) != -1:
            raise ValueError(
                "fp8_activation_qdq requires ungrouped per-token o_proj inputs"
            )
        if str(quantization.get("activation_granularity", "")) != (
            "per_token_last_axis"
        ):
            raise ValueError(
                "fp8_activation_qdq requires per_token_last_axis granularity"
            )
        if str(quantization.get("activation_scale_dtype", "")) != "float32":
            raise ValueError("fp8_activation_qdq requires float32 scales")
        if float(quantization.get("activation_maximum", 0.0)) != FP8_E4M3FN_MAX:
            raise ValueError("fp8_activation_qdq requires an FP8 maximum of 448")
        minimum_scale = float(quantization.get("activation_minimum_scale", 0.0))
        if abs(minimum_scale - FP8_MIN_SCALING_FACTOR) > 1e-15:
            raise ValueError(
                "fp8_activation_qdq minimum scale does not match vLLM QuantFP8"
            )
        if str(quantization.get("runtime_form", "")) != (
            "vllm_dynamic_per_token_fp8_e4m3fn_qdq_with_ste"
        ):
            raise ValueError("fp8_activation_qdq runtime form is not frozen")
    return objective


def train_llama_rotations(
    model: nn.Module,
    sequences: Sequence[Sequence[int]],
    config: Mapping[str, Any],
) -> Tuple[SpinQuantRotations, Dict[str, Any]]:
    """Train only R1/R2 using causal-LM loss and fixed calibration tokens."""

    quantization = config["quantization"]
    rotation_objective = _validate_rotation_objective(quantization)
    optimization = config["optimization"]
    steps = int(optimization["steps"])
    accumulation = int(optimization["gradient_accumulation_steps"])
    if steps < 1 or accumulation < 1:
        raise ValueError("steps and gradient_accumulation_steps must be positive")
    required_sequences = steps * accumulation
    if len(sequences) != required_sequences:
        raise ValueError(
            f"expected exactly {required_sequences} calibration sequences, "
            f"received {len(sequences)}"
        )
    sequence_length = len(sequences[0])
    if sequence_length < 2 or any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("calibration sequences must share one length of at least two")

    config_object = model.config
    hidden_size = int(config_object.hidden_size)
    head_dim = hidden_size // int(config_object.num_attention_heads)
    parameter = next(model.parameters())
    device = parameter.device
    rotations = SpinQuantRotations(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_layers=int(config_object.num_hidden_layers),
        seed=int(config["seed"]),
        dtype=torch.float32,
        device=device,
    )
    adapter = apply_spinquant_llama_training_adapter(
        model,
        rotations,
        SpinQuantFakeQuantSpec(
            weight_bits=int(quantization["weight_bits"]),
            activation_bits=int(quantization["activation_bits"]),
            weight_group_size=int(quantization["weight_group_size"]),
            weight_symmetric=bool(quantization["weight_symmetric"]),
            activation_type=str(quantization.get("activation_type", "int")),
            activation_symmetric=bool(quantization["activation_symmetric"]),
            activation_o_proj_group_size=int(
                quantization.get("activation_o_proj_group_size", -1)
            ),
            quantize_lm_head=bool(quantization["quantize_lm_head"]),
        ),
    )
    if bool(optimization.get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    initial_lr = float(optimization["learning_rate"])
    optimizer = CayleySGD(
        rotations.parameters(),
        lr=initial_lr,
        method=str(optimization["cayley_method"]),
        fixed_point_steps=int(optimization["fixed_point_steps"]),
    )
    update_losses = []
    gradient_maxima = []
    cursor = 0
    for update in range(steps):
        optimizer.param_groups[0]["lr"] = initial_lr * (1.0 - update / steps)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(accumulation):
            input_ids = torch.tensor(
                [sequences[cursor]],
                dtype=torch.long,
                device=device,
            )
            cursor += 1
            output = model(
                input_ids=input_ids,
                labels=input_ids,
                use_cache=False,
            )
            loss: Tensor = output.loss.float() / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("SpinQuant rotation training produced a non-finite loss")
            loss.backward()
            accumulated_loss += float(loss.detach())
        gradient_maximum = max(
            float(parameter.grad.detach().abs().max())
            for parameter in rotations.parameters()
            if parameter.grad is not None
        )
        if not torch.isfinite(torch.tensor(gradient_maximum)) or gradient_maximum <= 0:
            raise RuntimeError("SpinQuant rotations received no finite non-zero gradient")
        gradient_maxima.append(gradient_maximum)
        update_losses.append(accumulated_loss)
        optimizer.step()

    errors = rotations.errors()
    return rotations, {
        "status": "passed",
        "scope": (
            "R1/R2 fake-quant rotation optimization only; "
            "no GPTQ, evaluation, packing, kernel, or deployment claim"
        ),
        "rotation_objective": rotation_objective,
        "adapter": adapter,
        "optimizer_steps": steps,
        "gradient_accumulation_steps": accumulation,
        "calibration_sequences": len(sequences),
        "sequence_length": sequence_length,
        "initial_update_loss": update_losses[0],
        "final_update_loss": update_losses[-1],
        "best_update_loss": min(update_losses),
        "update_losses": update_losses,
        "gradient_maxima": gradient_maxima,
        "orthogonality_error": errors,
    }
