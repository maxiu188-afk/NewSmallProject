"""Validation contracts for packed compressed-tensors W4A16 exports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


EXPECTED_WEIGHT_CONFIG = {
    "num_bits": 4,
    "type": "int",
    "strategy": "group",
    "group_size": 128,
    "symmetric": True,
    "dynamic": False,
    "actorder": "static",
    "observer": "memoryless_minmax",
}


def checkpoint_tree_sha256(root: Path) -> str:
    """Hash checkpoint file names, sizes, and contents in stable order."""

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("cannot hash an empty checkpoint tree")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_spinquant_export_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the dedicated SpinQuant W4A16 recipe drifts."""

    quant = config.get("quantization")
    if not isinstance(quant, Mapping):
        raise ValueError("W4A16 export config is missing quantization metadata")
    expected_quant = {
        "algorithm": "GPTQ",
        "scheme": "W4A16",
        "group_size": 128,
        "targets": "Linear",
        "ignore": ["lm_head"],
        "format": "pack-quantized",
    }
    for key, expected in expected_quant.items():
        if quant.get(key) != expected:
            raise ValueError(f"unexpected W4A16 {key}: {quant.get(key)!r}")

    rotation = config.get("rotation")
    if not isinstance(rotation, Mapping):
        raise ValueError("SpinQuant W4A16 export requires rotation metadata")
    required_text = (
        "spinquant_training_job_id",
        "spinquant_training_project_revision",
        "spinquant_evidence_label",
    )
    for key in required_text:
        if not isinstance(rotation.get(key), str) or not rotation[key]:
            raise ValueError(f"SpinQuant rotation field {key} is missing")
    required_hashes = (
        "spinquant_training_result_sha256",
        "spinquant_rotation_manifest_sha256",
        "spinquant_rotation_safetensors_sha256",
    )
    for key in required_hashes:
        value = rotation.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"SpinQuant rotation field {key} is not a SHA-256")
    tolerance = rotation.get("maximum_orthogonality_error")
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("maximum_orthogonality_error must be positive")


def validate_checkpoint_quantization_config(config: Mapping[str, Any]) -> None:
    """Require the exact accepted packed W4A16 runtime representation."""

    if (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "pack-quantized"
        or config.get("quantization_status") != "compressed"
        or config.get("ignore") != ["lm_head"]
    ):
        raise ValueError("checkpoint is not compressed-tensors packed W4A16")
    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("W4A16 checkpoint must contain one quantization group")
    group = next(iter(groups.values()))
    if not isinstance(group, Mapping):
        raise ValueError("W4A16 quantization group is malformed")
    if group.get("input_activations") is not None:
        raise ValueError("W4A16 checkpoint must not quantize input activations")
    weights = group.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("W4A16 checkpoint is missing weight metadata")
    for key, expected in EXPECTED_WEIGHT_CONFIG.items():
        if weights.get(key) != expected:
            raise ValueError(
                f"unexpected packed W4A16 weight field {key}: {weights.get(key)!r}"
            )
