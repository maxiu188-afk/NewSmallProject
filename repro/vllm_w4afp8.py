"""Pure validation helpers for the GH200 compressed-tensors W4AFP8 route."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


EXPECTED_VARIANTS = (
    "bf16",
    "unrotated_w4afp8",
    "quarot_w4afp8",
    "spinquant_w4afp8",
)
QUAROT_VARIANTS = EXPECTED_VARIANTS[:3]
SPINQUANT_VARIANTS = (
    EXPECTED_VARIANTS[0],
    EXPECTED_VARIANTS[1],
    EXPECTED_VARIANTS[3],
)
EXPECTED_KERNEL = "CutlassW4A8LinearKernel"
EXPECTED_SCHEME = "CompressedTensorsW4A8Fp8"


def validate_export_config(config: Mapping[str, Any]) -> None:
    """Reject protocol drift before a model or CUDA runtime is loaded."""

    model = config["model"]
    quant = config["quantization"]
    variants = tuple(config["variants"])
    if variants != EXPECTED_VARIANTS[1:]:
        raise ValueError(f"export variants must be {EXPECTED_VARIANTS[1:]}")
    expected_quant = {
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
        "weight_observer": "minmax",
        "weight_clipping": False,
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
    for key, expected in expected_quant.items():
        if quant.get(key) != expected:
            raise ValueError(f"quantization.{key} must be {expected!r}")
    if int(model["expected_decoder_linears"]) != int(model["expected_layers"]) * 7:
        raise ValueError("expected decoder-linear count must be seven per layer")
    validate_kernel_shapes(config["kernel"]["required_linear_shapes"])


def validate_kernel_shapes(shapes: Iterable[Iterable[int]]) -> None:
    """Validate the K/N alignment required by the Hopper CUTLASS kernel."""

    normalized = []
    for shape in shapes:
        values = tuple(int(value) for value in shape)
        if len(values) != 2 or min(values) <= 0:
            raise ValueError(f"invalid linear shape: {values}")
        if values[0] % 128 or values[1] % 128:
            raise ValueError(f"W4AFP8 K/N dimensions must divide 128: {values}")
        normalized.append(values)
    if not normalized:
        raise ValueError("at least one W4AFP8 linear shape is required")


def _one_config_group(quantization_config: Mapping[str, Any]) -> Mapping[str, Any]:
    groups = quantization_config.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("W4AFP8 checkpoint must contain exactly one config group")
    group = next(iter(groups.values()))
    if not isinstance(group, Mapping):
        raise ValueError("W4AFP8 config group is not an object")
    return group


def validate_checkpoint_quantization_config(
    quantization_config: Mapping[str, Any],
) -> None:
    """Validate the serialized compressed-tensors W4AFP8 numerical contract."""

    expected_top = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
    }
    for key, expected in expected_top.items():
        if quantization_config.get(key) != expected:
            raise ValueError(f"quantization_config.{key} must be {expected!r}")
    group = _one_config_group(quantization_config)
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, Mapping) or not isinstance(activations, Mapping):
        raise ValueError("W4AFP8 weights or input_activations metadata is missing")

    expected_weights = {
        "num_bits": 4,
        "type": "int",
        "strategy": "group",
        "group_size": 128,
        "symmetric": True,
        "dynamic": False,
    }
    expected_activations = {
        "num_bits": 8,
        "type": "float",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    for key, expected in expected_weights.items():
        if weights.get(key) != expected:
            raise ValueError(f"W4AFP8 weights.{key} must be {expected!r}")
    for key, expected in expected_activations.items():
        if activations.get(key) != expected:
            raise ValueError(f"W4AFP8 input_activations.{key} must be {expected!r}")

    # The selected weaker deployment protocol disables activation ordering.
    # GROUP/DYNAMIC ordering would additionally serialize runtime g_idx and is
    # rejected by CutlassW4A8LinearKernel.
    if weights.get("actorder") is not None:
        raise ValueError("W4AFP8 checkpoint requires actorder=None")


def validate_no_runtime_g_idx(tensor_names: Iterable[str]) -> None:
    offenders = sorted(
        name
        for name in tensor_names
        if name.endswith(".weight_g_idx")
        or name.endswith(".g_idx")
        or ".g_idx_sort_indices" in name
    )
    if offenders:
        raise ValueError(f"W4AFP8 checkpoint contains runtime g_idx tensors: {offenders[:3]}")
