"""Contracts for the isolated W4AFP8 min/max versus MSE diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from repro.vllm_w4afp8 import validate_checkpoint_quantization_config


MSE_VARIANTS = (
    "unrotated_mse_w4afp8",
    "quarot_mse_w4afp8",
)

PPL_MODELS = (
    "bf16",
    "unrotated_minmax_w4afp8",
    "quarot_minmax_w4afp8",
    *MSE_VARIANTS,
)


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject drift from the one-variable observer diagnostic."""

    if tuple(config["variants"]) != MSE_VARIANTS:
        raise ValueError(f"diagnostic variants must be {MSE_VARIANTS}")
    expected_quantization = {
        "algorithm": "GPTQ",
        "scheme": "W4AFP8",
        "targets": "Linear",
        "ignore": ["lm_head"],
        "weight_bits": 4,
        "weight_type": "int",
        "weight_strategy": "group",
        "weight_group_size": 128,
        "weight_block_size": 128,
        "dampening_frac": 0.01,
        "weight_symmetric": True,
        "weight_actorder": None,
        "weight_observer": "mse",
        "weight_observer_kwargs": {
            "maxshrink": 0.8,
            "grid": 100.0,
            "norm": 2.4,
            "patience": 100,
        },
        "resolved_weight_observers": ["mse", "memoryless_mse"],
        "weight_clipping": True,
        "activation_bits": 8,
        "activation_type": "float",
        "activation_strategy": "token",
        "activation_dynamic": True,
        "activation_symmetric": True,
        "format": "pack-quantized",
        "key_bits": 16,
        "value_bits": 16,
        "quantize_lm_head": False,
    }
    if config["quantization"] != expected_quantization:
        raise ValueError("MSE diagnostic quantization contract changed")
    calibration = config["calibration"]
    if (
        calibration["samples"] != 128
        or calibration["sequence_length"] != 2048
        or calibration["seed"] != 0
        or len(calibration["manifest_sha256"]) != 64
        or len(calibration["token_ids_sha256"]) != 64
    ):
        raise ValueError("MSE diagnostic calibration contract changed")
    tokens = config["tokens"]
    if (
        tokens["samples"] != 162
        or tokens["sequence_length"] != 2048
        or tokens["expected_scored_tokens"] != 331614
    ):
        raise ValueError("MSE diagnostic evaluation-token contract changed")


def validate_mse_checkpoint_quantization_config(
    metadata: Mapping[str, Any],
) -> str:
    """Require the ordinary W4AFP8 layout plus an MSE weight observer."""

    validate_checkpoint_quantization_config(metadata)
    groups = metadata.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("MSE checkpoint requires exactly one quantization group")
    group = next(iter(groups.values()))
    observer = group["weights"].get("observer")
    if observer not in {"mse", "memoryless_mse"}:
        raise ValueError(f"MSE checkpoint has unexpected weight observer: {observer}")
    return observer


def validate_minmax_checkpoint_quantization_config(
    metadata: Mapping[str, Any],
) -> str:
    """Require the accepted control layout plus a min/max weight observer."""

    validate_checkpoint_quantization_config(metadata)
    group = next(iter(metadata["config_groups"].values()))
    observer = group["weights"].get("observer")
    if observer not in {"minmax", "memoryless_minmax"}:
        raise ValueError(
            f"min/max checkpoint has unexpected weight observer: {observer}"
        )
    return observer
