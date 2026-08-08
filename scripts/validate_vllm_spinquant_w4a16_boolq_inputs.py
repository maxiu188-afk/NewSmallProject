#!/usr/bin/env python3
"""Validate the accepted SpinQuant W4A16 gate before BoolQ evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.vllm_w4a16 import (  # noqa: E402
    checkpoint_tree_sha256,
    validate_checkpoint_quantization_config,
)
from scripts.run_vllm_w4a16_llama2_13b_ppl import (  # noqa: E402
    _checkpoint_metadata,
)


EXPECTED_MODELS = ("bf16", "spinquant_w4a16")
UNCHANGED_GATE_INPUTS = (
    "configs/deployment/vllm_spinquant_w4a16_llama2_13b_isambard.json",
    "repro/vllm_w4a16.py",
    "repro/spinquant/artifacts.py",
    "repro/spinquant/offline.py",
    "repro/spinquant/rotations.py",
    "scripts/export_vllm_w4a16_llama2_13b.py",
    "scripts/run_vllm_w4a16_llama2_13b_gate.py",
    "scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_gate.sbatch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str, check: bool = True):
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def validate(
    *,
    project_root: Path,
    config_path: Path,
    export_result_path: Path,
    inference_result_path: Path,
    source_manifest_path: Path,
    stdout_path: Path,
    models: dict[str, Path],
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = config["source_gate"]
    if gate.get("status") != "accepted":
        raise RuntimeError("SpinQuant W4A16 source gate is not frozen as accepted")
    if tuple(gate.get("expected_models", ())) != EXPECTED_MODELS:
        raise RuntimeError("SpinQuant W4A16 source-gate model order changed")

    bound_files = (
        (export_result_path, "export_result_sha256"),
        (inference_result_path, "inference_result_sha256"),
        (source_manifest_path, "source_manifest_sha256"),
        (stdout_path, "stdout_sha256"),
    )
    for path, hash_field in bound_files:
        if _sha256(path) != gate[hash_field]:
            raise RuntimeError(f"accepted SpinQuant W4A16 {hash_field} changed")

    stdout = stdout_path.read_text(encoding="utf-8")
    if gate["required_marker"] not in stdout:
        raise RuntimeError("accepted SpinQuant W4A16 gate marker is absent")
    if gate["required_kernel_evidence"] not in stdout:
        raise RuntimeError("accepted SpinQuant W4A16 Machete evidence is absent")

    export = json.loads(export_result_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_result_path.read_text(encoding="utf-8"))
    if export.get("status") != "passed" or export.get("mode") != "spinquant":
        raise RuntimeError("accepted SpinQuant W4A16 export result is incomplete")
    source_revision = export.get("project_revision")
    if inference.get("status") != "passed" or inference.get(
        "project_revision"
    ) != source_revision:
        raise RuntimeError("accepted SpinQuant W4A16 inference result is incomplete")
    if tuple(inference.get("models", {})) != EXPECTED_MODELS:
        raise RuntimeError("accepted SpinQuant W4A16 inference model order changed")
    if export.get("packed_decoder_linear_count") != 280:
        raise RuntimeError("accepted SpinQuant W4A16 export lacks 280 packed linears")
    if export.get("checkpoint_tree_sha256") != gate["checkpoint_tree_sha256"]:
        raise RuntimeError("accepted SpinQuant W4A16 tree hash changed in the result")
    validate_checkpoint_quantization_config(export["quantization_config"])

    manifest = source_manifest_path.read_text(encoding="utf-8")
    if f"git_revision={source_revision}\n" not in manifest:
        raise RuntimeError("SpinQuant W4A16 source manifest revision changed")

    current_revision = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    if _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        source_revision,
        current_revision,
        check=False,
    ).returncode != 0:
        raise RuntimeError("accepted SpinQuant W4A16 gate is not an ancestor of HEAD")
    changed = _git(
        project_root,
        "diff",
        "--name-only",
        f"{source_revision}..{current_revision}",
        "--",
        *UNCHANGED_GATE_INPUTS,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(f"SpinQuant W4A16 gate-producing inputs changed: {changed}")

    if tuple(models) != EXPECTED_MODELS:
        raise RuntimeError("requested SpinQuant W4A16 model order is incomplete")
    for name, path in models.items():
        accepted_path = Path(inference["models"][name]["model_path"]).resolve()
        if path.resolve() != accepted_path:
            raise RuntimeError(f"{name} path differs from the accepted gate")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")

    spinquant_path = models["spinquant_w4a16"].resolve()
    if spinquant_path != Path(export["checkpoint"]).resolve():
        raise RuntimeError("SpinQuant checkpoint differs between export and inference")
    observed_tree_sha256 = checkpoint_tree_sha256(spinquant_path)
    if observed_tree_sha256 != gate["checkpoint_tree_sha256"]:
        raise RuntimeError("SpinQuant W4A16 checkpoint tree changed after acceptance")
    checkpoint = _checkpoint_metadata(spinquant_path)
    if checkpoint["packed_decoder_linear_count"] != 280:
        raise RuntimeError("SpinQuant W4A16 checkpoint no longer has 280 packed linears")
    validate_checkpoint_quantization_config(checkpoint["quantization_config"])

    return {
        "status": "accepted",
        "current_revision": current_revision,
        "source_revision": source_revision,
        "source_job_id": gate["job_id"],
        "export_result_sha256": _sha256(export_result_path),
        "inference_result_sha256": _sha256(inference_result_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "stdout_sha256": _sha256(stdout_path),
        "checkpoint_tree_sha256": observed_tree_sha256,
        "models": {name: str(path.resolve()) for name, path in models.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export-result", type=Path, required=True)
    parser.add_argument("--inference-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()

    models = {}
    for value in args.model:
        if "=" not in value:
            parser.error(f"--model requires name=path: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            parser.error(f"repeated model: {name}")
        models[name] = Path(raw_path)

    result = validate(
        project_root=args.project_root.resolve(),
        config_path=args.config.resolve(),
        export_result_path=args.export_result.resolve(),
        inference_result_path=args.inference_result.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        stdout_path=args.stdout.resolve(),
        models=models,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("VLLM_SPINQUANT_W4A16_BOOLQ_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
