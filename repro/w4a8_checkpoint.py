"""Self-describing, sharded GPTQ W4A8 decoder checkpoint support.

The writer is deliberately streaming: each GPTQ linear is moved to CPU and
saved as soon as it is quantized.  A checkpoint manifest is published only
after every expected decoder linear exists and passes structural validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn

from repro.gptq import GPTQPackedWeight
from repro.w4a8_linear import PackedW4A8ReferenceLinear, W4A8Linear


CHECKPOINT_FORMAT = "newsmallproject-gptq-w4a8-checkpoint-v1"
LINEAR_FORMAT = "newsmallproject-gptq-w4a8-linear-v1"
PACKED_LAYOUT = "packed-w4-row-major-le-nibble-v1"
MANIFEST_NAME = "manifest.json"


class W4A8CheckpointError(RuntimeError):
    """Raised when a sharded packed checkpoint violates its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoder_linear_names(model: nn.Module) -> list[str]:
    """Return every concrete decoder ``nn.Linear`` after QuaRot wrapping."""
    names = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    ]
    if not names:
        raise W4A8CheckpointError("model has no decoder nn.Linear modules")
    if len(names) != len(set(names)):
        raise W4A8CheckpointError("decoder linear names are not unique")
    return sorted(names)


def _safe_shard_name(index: int, tensor_name: str) -> str:
    encoded = tensor_name.replace(".", "__")
    if not encoded.replace("_", "").isalnum():
        raise W4A8CheckpointError("tensor name cannot be encoded safely: {}".format(tensor_name))
    return "{:04d}-{}.pt".format(index, encoded)


def validate_linear_artifact(artifact: Mapping[str, Any], expected_name: str | None = None) -> None:
    required = {
        "format",
        "layout",
        "tensor",
        "packed_weight",
        "weight_scales",
        "input_permutation",
        "bias",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise W4A8CheckpointError("packed linear misses fields: {}".format(", ".join(missing)))
    if artifact["format"] != LINEAR_FORMAT or artifact["layout"] != PACKED_LAYOUT:
        raise W4A8CheckpointError("unsupported packed linear format or layout")
    if expected_name is not None and artifact["tensor"] != expected_name:
        raise W4A8CheckpointError("packed linear tensor name does not match manifest")
    packed = artifact["packed_weight"]
    scales = artifact["weight_scales"]
    permutation = artifact["input_permutation"]
    bias = artifact["bias"]
    if not isinstance(packed, torch.Tensor) or packed.ndim != 2 or packed.dtype != torch.uint8:
        raise W4A8CheckpointError("packed_weight must be a rank-two uint8 tensor")
    if not isinstance(scales, torch.Tensor) or scales.ndim != 2 or scales.dtype != torch.float32:
        raise W4A8CheckpointError("weight_scales must be a rank-two float32 tensor")
    in_features = int(packed.shape[1]) * 2
    if int(packed.shape[0]) != int(scales.shape[0]) or in_features % int(scales.shape[1]):
        raise W4A8CheckpointError("packed weight and scale shapes are incompatible")
    if permutation is not None:
        if not isinstance(permutation, torch.Tensor) or permutation.ndim != 1:
            raise W4A8CheckpointError("input_permutation must be a rank-one tensor or null")
        values = permutation.to(dtype=torch.long, device="cpu")
        if values.numel() != in_features or not torch.equal(torch.sort(values).values, torch.arange(in_features)):
            raise W4A8CheckpointError("input_permutation is not a complete permutation")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1 or bias.numel() != packed.shape[0]:
            raise W4A8CheckpointError("bias shape is incompatible with packed weight")


class PackedW4A8CheckpointWriter:
    """Write independent tensor shards and atomically publish their manifest."""

    def __init__(
        self,
        output_dir: Path,
        expected_names: Iterable[str],
        source: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir
        self.expected_names = tuple(sorted(expected_names))
        if not self.expected_names or len(self.expected_names) != len(set(self.expected_names)):
            raise W4A8CheckpointError("expected tensor names must be non-empty and unique")
        self._indices = {name: index for index, name in enumerate(self.expected_names)}
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.source = dict(source)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / MANIFEST_NAME).exists():
            raise W4A8CheckpointError("checkpoint manifest already exists: {}".format(self.output_dir / MANIFEST_NAME))
        existing = list(self.output_dir.glob("*.pt"))
        if existing:
            raise W4A8CheckpointError("checkpoint directory contains incomplete tensor shards")

    def write(self, tensor_name: str, packed: GPTQPackedWeight, bias: torch.Tensor | None) -> None:
        if tensor_name not in self._indices:
            raise W4A8CheckpointError("unexpected packed tensor: {}".format(tensor_name))
        if tensor_name in self._entries:
            raise W4A8CheckpointError("packed tensor was written twice: {}".format(tensor_name))
        artifact = {
            "format": LINEAR_FORMAT,
            "layout": PACKED_LAYOUT,
            "tensor": tensor_name,
            "packed_weight": packed.packed_weight.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
            "weight_scales": packed.scales.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            "input_permutation": None
            if packed.input_permutation is None
            else packed.input_permutation.detach().to(device="cpu", dtype=torch.long).contiguous(),
            "bias": None if bias is None else bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        }
        validate_linear_artifact(artifact, tensor_name)
        filename = _safe_shard_name(self._indices[tensor_name], tensor_name)
        path = self.output_dir / filename
        if path.exists():
            raise W4A8CheckpointError("packed tensor shard already exists: {}".format(path))
        torch.save(artifact, path)
        self._entries[tensor_name] = {
            "file": filename,
            "sha256": sha256_file(path),
            "packed_weight_shape": list(artifact["packed_weight"].shape),
            "weight_scale_shape": list(artifact["weight_scales"].shape),
            "input_permutation": artifact["input_permutation"] is not None,
            "bias": artifact["bias"] is not None,
        }

    def finalize(self, metadata: Mapping[str, Any]) -> Path:
        missing = sorted(set(self.expected_names) - set(self._entries))
        if missing:
            raise W4A8CheckpointError("cannot finalize checkpoint; missing tensors: {}".format(", ".join(missing)))
        manifest = {
            "schema_version": 1,
            "format": CHECKPOINT_FORMAT,
            "layout": PACKED_LAYOUT,
            "source": self.source,
            "metadata": dict(metadata),
            "tensor_count": len(self._entries),
            "tensors": {name: self._entries[name] for name in self.expected_names},
        }
        temporary = self.output_dir / (MANIFEST_NAME + ".tmp")
        manifest_path = self.output_dir / MANIFEST_NAME
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        return manifest_path


def load_checkpoint_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != CHECKPOINT_FORMAT
        or manifest.get("layout") != PACKED_LAYOUT
    ):
        raise W4A8CheckpointError("unsupported packed checkpoint manifest")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors or manifest.get("tensor_count") != len(tensors):
        raise W4A8CheckpointError("checkpoint tensor index is missing or inconsistent")
    for name, entry in tensors.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise W4A8CheckpointError("checkpoint tensor index is malformed")
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise W4A8CheckpointError("checkpoint shard path must be a local filename")
        path = manifest_path.parent / filename
        if not path.is_file() or entry.get("sha256") != sha256_file(path):
            raise W4A8CheckpointError("checkpoint shard checksum mismatch: {}".format(name))
    return manifest


def load_linear_artifact(manifest_path: Path, manifest: Mapping[str, Any], tensor_name: str) -> Dict[str, Any]:
    entry = manifest.get("tensors", {}).get(tensor_name)
    if not isinstance(entry, Mapping):
        raise W4A8CheckpointError("tensor is absent from checkpoint: {}".format(tensor_name))
    path = manifest_path.parent / str(entry["file"])
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise W4A8CheckpointError("packed tensor shard is not an object: {}".format(tensor_name))
    validate_linear_artifact(artifact, tensor_name)
    if list(artifact["packed_weight"].shape) != entry.get("packed_weight_shape"):
        raise W4A8CheckpointError("packed tensor shape differs from manifest: {}".format(tensor_name))
    if list(artifact["weight_scales"].shape) != entry.get("weight_scale_shape"):
        raise W4A8CheckpointError("scale shape differs from manifest: {}".format(tensor_name))
    return artifact


def _replace_submodule(model: nn.Module, qualified_name: str, replacement: nn.Module) -> nn.Module:
    parent_name, child_name = qualified_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = getattr(parent, child_name)
    if not isinstance(original, nn.Module):
        raise W4A8CheckpointError("checkpoint target is not a module: {}".format(qualified_name))
    setattr(parent, child_name, replacement)
    return original


def install_checkpoint_linears(
    model: nn.Module,
    manifest_path: Path,
    *,
    implementation: str,
) -> list[str]:
    """Replace every indexed decoder linear with the oracle or CUDA module."""
    if implementation not in {"reference", "cuda"}:
        raise ValueError("implementation must be reference or cuda")
    manifest = load_checkpoint_manifest(manifest_path)
    expected = decoder_linear_names(model)
    indexed = sorted(manifest["tensors"])
    if indexed != expected:
        missing = sorted(set(expected) - set(indexed))
        extra = sorted(set(indexed) - set(expected))
        raise W4A8CheckpointError(
            "checkpoint/model decoder linears differ; missing={} extra={}".format(missing, extra)
        )
    module_class = PackedW4A8ReferenceLinear if implementation == "reference" else W4A8Linear
    device = next(model.parameters()).device
    for name in expected:
        original = model.get_submodule(name)
        if not isinstance(original, nn.Linear):
            raise W4A8CheckpointError("checkpoint target is not nn.Linear: {}".format(name))
        artifact = load_linear_artifact(manifest_path, manifest, name)
        expected_shape = (int(artifact["packed_weight"].shape[0]), int(artifact["packed_weight"].shape[1]) * 2)
        if tuple(original.weight.shape) != expected_shape:
            raise W4A8CheckpointError("checkpoint weight shape differs from model: {}".format(name))
        if (original.bias is None) != (artifact["bias"] is None):
            raise W4A8CheckpointError("checkpoint bias presence differs from model: {}".format(name))
        replacement = module_class(
            artifact["packed_weight"],
            artifact["weight_scales"],
            artifact["bias"],
            artifact["input_permutation"],
            output_dtype=original.weight.dtype,
        ).to(device).eval()
        _replace_submodule(model, name, replacement)
    return expected


def convert_reference_linears_to_cuda(model: nn.Module, tensor_names: Iterable[str]) -> list[str]:
    """Swap installed packed-reference modules for CUDA modules without reloading shards."""
    names = list(tensor_names)
    if not names or len(names) != len(set(names)):
        raise W4A8CheckpointError("reference tensor names must be non-empty and unique")
    for name in names:
        original = model.get_submodule(name)
        if not isinstance(original, PackedW4A8ReferenceLinear):
            raise W4A8CheckpointError("target is not an installed packed reference linear: {}".format(name))
        replacement = W4A8Linear(
            original.packed_weight,
            original.weight_scales,
            original.bias,
            original.input_permutation,
            output_dtype=original.output_dtype,
        ).to(original.packed_weight.device).eval()
        _replace_submodule(model, name, replacement)
    return names
