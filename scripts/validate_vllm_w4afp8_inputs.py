#!/usr/bin/env python3
"""Validate accepted W4AFP8 gate artifacts before quality or serving jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


UNCHANGED_GATE_INPUTS = (
    "configs/deployment/vllm_w4afp8_llama2_13b_isambard.json",
    "repro/vllm_w4afp8.py",
    "repro/offline_llama_rotation.py",
    "repro/spinquant/artifacts.py",
    "repro/spinquant/offline.py",
    "scripts/audit_vllm_w4afp8_backend.py",
    "scripts/export_vllm_w4afp8_llama2_13b.py",
    "scripts/run_vllm_w4afp8_llama2_13b_gate.py",
    "scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch",
)
EXPECTED_MODELS = {
    "bf16",
    "unrotated_w4afp8",
    "quarot_w4afp8",
    "spinquant_w4afp8",
}


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
    source_result_path: Path,
    capability_result_path: Path,
    models: dict[str, Path],
    smoke_result_path: Path | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_gate = config["source_gate"]
    if source_gate.get("status") != "accepted":
        raise RuntimeError("W4AFP8 source gate is not frozen as accepted")
    if set(source_gate["expected_models"]) != EXPECTED_MODELS:
        raise RuntimeError("W4AFP8 source-gate model set changed")
    if _sha256(source_result_path) != source_gate.get("result_sha256"):
        raise RuntimeError("accepted W4AFP8 inference result SHA-256 changed")
    if _sha256(capability_result_path) != source_gate.get("capability_result_sha256"):
        raise RuntimeError("accepted W4AFP8 capability result SHA-256 changed")

    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    capability = json.loads(capability_result_path.read_text(encoding="utf-8"))
    if source.get("status") != "passed" or set(source.get("models", {})) != EXPECTED_MODELS:
        raise RuntimeError("accepted W4AFP8 inference result is incomplete")
    if capability.get("status") != "passed" or {
        row["kernel"] for row in capability.get("selections", [])
    } != {"CutlassW4A8LinearKernel"}:
        raise RuntimeError("accepted W4AFP8 capability result did not select CUTLASS")
    for name in EXPECTED_MODELS - {"bf16"}:
        checkpoint = source["models"][name].get("checkpoint", {})
        if (
            checkpoint.get("packed_decoder_linear_count") != 280
            or checkpoint.get("runtime_g_idx_tensors") != 0
        ):
            raise RuntimeError(f"accepted {name} checkpoint metadata is invalid")

    current_revision = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    source_revision = source["project_revision"]
    if _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        source_revision,
        current_revision,
        check=False,
    ).returncode != 0:
        raise RuntimeError("accepted W4AFP8 gate revision is not an ancestor of HEAD")
    changed = _git(
        project_root,
        "diff",
        "--name-only",
        f"{source_revision}..{current_revision}",
        "--",
        *UNCHANGED_GATE_INPUTS,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(f"W4AFP8 gate-producing inputs changed: {changed}")

    if set(models) != EXPECTED_MODELS:
        raise RuntimeError("requested W4AFP8 model set is incomplete")
    for name, path in models.items():
        expected = Path(source["models"][name]["model_path"]).resolve()
        if path.resolve() != expected:
            raise RuntimeError(f"{name} path differs from the accepted gate")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")

    result: dict[str, Any] = {
        "current_revision": current_revision,
        "source_revision": source_revision,
        "source_result_sha256": _sha256(source_result_path),
        "capability_result_sha256": _sha256(capability_result_path),
        "models": {name: str(path.resolve()) for name, path in models.items()},
    }
    if smoke_result_path is not None:
        smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "passed" or set(smoke.get("models", {})) != EXPECTED_MODELS:
            raise RuntimeError("W4AFP8 service smoke is incomplete")
        if smoke.get("project_revision") != current_revision:
            raise RuntimeError("W4AFP8 smoke revision differs from benchmark revision")
        for name in EXPECTED_MODELS - {"bf16"}:
            evidence = smoke["models"][name].get("kernel_evidence", {})
            if not evidence.get("matched"):
                raise RuntimeError(f"{name} service smoke lacks W4AFP8 kernel evidence")
        result["smoke_result"] = str(smoke_result_path.resolve())
        result["smoke_result_sha256"] = _sha256(smoke_result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--capability-result", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--smoke-result", type=Path)
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
        source_result_path=args.source_result.resolve(),
        capability_result_path=args.capability_result.resolve(),
        models=models,
        smoke_result_path=(
            args.smoke_result.resolve() if args.smoke_result is not None else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("VLLM_W4AFP8_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
