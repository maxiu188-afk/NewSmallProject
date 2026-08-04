#!/usr/bin/env python3
"""Validate immutable accepted QuaRot W4A16 inputs without rerunning export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    if not files:
        raise RuntimeError("checkpoint tree has no regular files")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{_sha256(path)}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_quantization_config(value: dict[str, Any]) -> None:
    if value.get("quant_method") != "compressed-tensors":
        raise RuntimeError("checkpoint quantization method changed")
    if value.get("format") != "pack-quantized":
        raise RuntimeError("checkpoint compression format changed")
    groups = value.get("config_groups")
    if not isinstance(groups, dict) or list(groups) != ["group_0"]:
        raise RuntimeError("checkpoint quantization groups changed")
    group = groups["group_0"]
    weights = group.get("weights")
    if not isinstance(weights, dict):
        raise RuntimeError("checkpoint weight scheme is missing")
    expected = {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
        "actorder": "static",
    }
    for key, expected_value in expected.items():
        if weights.get(key) != expected_value:
            raise RuntimeError(f"checkpoint weight scheme changed at {key}")
    if group.get("input_activations") is not None:
        raise RuntimeError("W4A16 checkpoint unexpectedly quantizes activations")
    if group.get("targets") != ["Linear"] or value.get("ignore") != ["lm_head"]:
        raise RuntimeError("checkpoint target coverage changed")


def _packed_linear_count(model_path: Path) -> int:
    from safetensors import safe_open

    count = 0
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            count += sum(key.endswith(".weight_packed") for key in handle.keys())
    return count


def validate(
    *,
    project_root: Path,
    config_path: Path,
    source_result_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = config["source_gate"]
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    current_revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if _sha256(source_result_path) != gate["result_sha256"]:
        raise RuntimeError("accepted inference result SHA-256 changed")
    if source.get("status") != "passed":
        raise RuntimeError("accepted inference result did not pass")
    if source.get("project_revision") != gate["source_revision"]:
        raise RuntimeError("accepted inference source revision changed")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "merge-base",
            "--is-ancestor",
            gate["source_revision"],
            current_revision,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("accepted W4A16 source revision is not an ancestor")
    source_model = source.get("models", {}).get("rotated_w4a16", {})
    if model_path.resolve() != Path(source_model.get("model_path", "")).resolve():
        raise RuntimeError("rotated W4A16 path differs from accepted inference")
    if source.get("vllm") != config["runtime"]["vllm_version"]:
        raise RuntimeError("accepted inference vLLM version changed")
    if source.get("torch") != config["runtime"]["torch_version"]:
        raise RuntimeError("accepted inference PyTorch version changed")
    if source.get("gpu") != config["runtime"]["gpu"]:
        raise RuntimeError("accepted inference GPU changed")
    if source.get("compute_capability") != config["runtime"]["compute_capability"]:
        raise RuntimeError("accepted inference compute capability changed")

    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    _validate_quantization_config(model_config.get("quantization_config", {}))
    packed_count = _packed_linear_count(model_path)
    if packed_count != 280:
        raise RuntimeError(f"expected 280 packed decoder linears, found {packed_count}")
    tree_sha256 = _tree_sha256(model_path)
    if tree_sha256 != gate["checkpoint_tree_sha256"]:
        raise RuntimeError("rotated W4A16 checkpoint tree SHA-256 changed")

    return {
        "status": "accepted",
        "current_revision": current_revision,
        "source_revision": gate["source_revision"],
        "source_result": str(source_result_path.resolve()),
        "source_result_sha256": _sha256(source_result_path),
        "model_path": str(model_path.resolve()),
        "checkpoint_tree_sha256": tree_sha256,
        "packed_decoder_linear_count": packed_count,
        "quantization": {
            "format": "pack-quantized",
            "weight_bits": 4,
            "activation_bits": 16,
            "group_size": 128,
            "actorder": "static",
            "lm_head_excluded": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        project_root=args.project_root.resolve(),
        config_path=args.config.resolve(),
        source_result_path=args.source_result.resolve(),
        model_path=args.model.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("QUAROT_W4A16_IMMUTABLE_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
