"""SafeTensor rotation artifacts with checksummed JSON provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from repro.spinquant.rotations import SpinQuantRotations
from repro.spinquant.stiefel import orthogonality_error


FORMAT = "newsmallproject-spinquant-rotations-v1"
TENSOR_FILE = "rotations.safetensors"
MANIFEST_FILE = "rotation-manifest.json"


class RotationArtifactError(ValueError):
    """Raised when a saved rotation artifact fails validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_round_trip(values: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise RotationArtifactError("artifact metadata must be JSON serializable") from error


def save_rotation_artifact(
    output_dir: Path,
    rotations: SpinQuantRotations,
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Save only R1/R2 tensors and explicit provenance, never a pickle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / TENSOR_FILE
    manifest_path = output_dir / MANIFEST_FILE
    if tensor_path.exists() or manifest_path.exists():
        raise RotationArtifactError(
            "refusing to overwrite an existing rotation artifact"
        )
    tensors = {
        "r1": rotations.r1.detach().float().cpu().contiguous(),
        "r2": rotations.r2.detach().float().cpu().contiguous(),
    }
    save_file(tensors, str(tensor_path))
    errors = {
        "r1": float(orthogonality_error(tensors["r1"])),
        "r2": float(orthogonality_error(tensors["r2"]).max()),
    }
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "file": {
            "name": TENSOR_FILE,
            "sha256": _sha256(tensor_path),
        },
        "rotation": {
            "hidden_size": rotations.hidden_size,
            "head_dim": rotations.head_dim,
            "num_layers": rotations.num_layers,
            "r1_shape": list(tensors["r1"].shape),
            "r2_shape": list(tensors["r2"].shape),
            "dtype": "float32",
            "orthogonality_error": errors,
        },
        "provenance": _json_round_trip(provenance),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_rotation_artifact(
    manifest_path: Path,
    *,
    target: Optional[SpinQuantRotations] = None,
    maximum_orthogonality_error: float = 1e-4,
) -> Dict[str, Any]:
    """Validate and load a rotation artifact, optionally copying into a target."""

    if maximum_orthogonality_error <= 0:
        raise ValueError("maximum_orthogonality_error must be positive")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RotationArtifactError("rotation manifest is unreadable") from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != FORMAT
    ):
        raise RotationArtifactError("unsupported rotation artifact format")
    file_info = manifest.get("file")
    if not isinstance(file_info, dict) or file_info.get("name") != TENSOR_FILE:
        raise RotationArtifactError("rotation manifest has an unexpected tensor file")
    tensor_path = manifest_path.parent / TENSOR_FILE
    if not tensor_path.is_file() or _sha256(tensor_path) != file_info.get("sha256"):
        raise RotationArtifactError("rotation SafeTensor checksum mismatch")

    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"r1", "r2"}:
        raise RotationArtifactError("rotation artifact must contain only r1 and r2")
    r1, r2 = tensors["r1"], tensors["r2"]
    rotation_info = manifest.get("rotation")
    if not isinstance(rotation_info, dict):
        raise RotationArtifactError("rotation metadata is missing")
    expected_r1 = tuple(rotation_info.get("r1_shape", ()))
    expected_r2 = tuple(rotation_info.get("r2_shape", ()))
    if (
        tuple(r1.shape) != expected_r1
        or tuple(r2.shape) != expected_r2
        or r1.dtype != torch.float32
        or r2.dtype != torch.float32
    ):
        raise RotationArtifactError("rotation tensor shape or dtype mismatch")
    if (
        r1.ndim != 2
        or r1.shape[0] != r1.shape[1]
        or r2.ndim != 3
        or r2.shape[-1] != r2.shape[-2]
    ):
        raise RotationArtifactError("rotation tensors do not contain square matrices")
    observed_errors = {
        "r1": float(orthogonality_error(r1)),
        "r2": float(orthogonality_error(r2).max()),
    }
    if max(observed_errors.values()) > maximum_orthogonality_error:
        raise RotationArtifactError(
            "rotation artifact exceeds the orthogonality tolerance"
        )

    if target is not None:
        if tuple(target.r1.shape) != tuple(r1.shape) or tuple(target.r2.shape) != tuple(r2.shape):
            raise RotationArtifactError("target rotation shapes do not match artifact")
        with torch.no_grad():
            target.r1.copy_(r1.to(device=target.r1.device, dtype=target.r1.dtype))
            target.r2.copy_(r2.to(device=target.r2.device, dtype=target.r2.dtype))
    return {
        "manifest": manifest,
        "tensors": tensors,
        "observed_orthogonality_error": observed_errors,
    }
