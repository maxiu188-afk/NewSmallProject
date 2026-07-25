#!/usr/bin/env python3
"""Validate the accepted inference gate before Llama-2-13B serving jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Optional


UNCHANGED_INFERENCE_INPUTS = (
    "configs/deployment/vllm_w4a16_llama2_13b_isambard.json",
    "repro/offline_llama_rotation.py",
    "repro/structured_hadamard.py",
    "scripts/export_vllm_w4a16_llama2_13b.py",
    "scripts/run_vllm_w4a16_llama2_13b_gate.py",
)
EXPECTED_MODELS = {"bf16", "unrotated_w4a16", "rotated_w4a16"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(
    project_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
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
    models: dict[str, Path],
    smoke_result_path: Optional[Path],
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_gate = config["source_gate"]
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    current_revision = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    source_revision = source["project_revision"]

    if _sha256(source_result_path) != source_gate["result_sha256"]:
        raise RuntimeError("accepted inference result SHA-256 changed")
    if source["status"] != "passed" or set(source["models"]) != EXPECTED_MODELS:
        raise RuntimeError("accepted inference result is incomplete")
    if not all(
        source["comparisons"][name]["generated_tokens_equal"]
        for name in ("unrotated_w4a16_vs_bf16", "rotated_w4a16_vs_bf16")
    ):
        raise RuntimeError("accepted inference result did not preserve greedy tokens")
    if source["gpu"] != config["runtime"]["gpu"]:
        raise RuntimeError("accepted inference gate used a different GPU")
    if source["compute_capability"] != config["runtime"]["compute_capability"]:
        raise RuntimeError("accepted inference gate used a different compute capability")
    if source["vllm"] != config["runtime"]["vllm_version"]:
        raise RuntimeError("accepted inference gate used a different vLLM")
    if source["torch"] != config["runtime"]["torch_version"]:
        raise RuntimeError("accepted inference gate used a different PyTorch")

    ancestor = _git(
        project_root,
        "merge-base",
        "--is-ancestor",
        source_revision,
        current_revision,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("accepted inference revision is not an ancestor of HEAD")
    changed = _git(
        project_root,
        "diff",
        "--name-only",
        f"{source_revision}..{current_revision}",
        "--",
        *UNCHANGED_INFERENCE_INPUTS,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(f"inference-producing inputs changed: {changed}")

    if set(models) != EXPECTED_MODELS:
        raise RuntimeError(f"model set changed: {set(models)}")
    for name, path in models.items():
        expected = Path(source["models"][name]["model_path"]).resolve()
        if path.resolve() != expected:
            raise RuntimeError(f"{name} path does not match the accepted inference gate")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")

    result: dict[str, Any] = {
        "current_revision": current_revision,
        "source_revision": source_revision,
        "source_result_sha256": _sha256(source_result_path),
        "models": {name: str(path.resolve()) for name, path in models.items()},
    }
    if smoke_result_path is not None:
        smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
        if smoke["status"] != "passed":
            raise RuntimeError("service smoke did not pass")
        if smoke["project_revision"] != current_revision:
            raise RuntimeError("service smoke revision differs from benchmark revision")
        if set(smoke["models"]) != EXPECTED_MODELS:
            raise RuntimeError("service smoke model set is incomplete")
        if not smoke["generated_texts_equal"]:
            raise RuntimeError("service smoke returned different greedy text")
        for name in EXPECTED_MODELS:
            if len(smoke["models"][name]["served_model_ids"]) != 1:
                raise RuntimeError(f"{name} did not expose exactly one served model")
            usage = smoke["models"][name]["completion"]["usage"]
            if usage["completion_tokens"] != int(config["smoke"]["max_tokens"]):
                raise RuntimeError(f"{name} smoke completion length changed")
        result["smoke_result"] = str(smoke_result_path.resolve())
        result["smoke_result_sha256"] = _sha256(smoke_result_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()

    models = {}
    for value in args.model:
        if "=" not in value:
            parser.error(f"--model requires name=path: {value}")
        name, path = value.split("=", 1)
        if name in models:
            parser.error(f"repeated model: {name}")
        models[name] = Path(path)
    result = validate(
        project_root=args.project_root.resolve(),
        config_path=args.config.resolve(),
        source_result_path=args.source_result.resolve(),
        models=models,
        smoke_result_path=(
            args.smoke_result.resolve() if args.smoke_result is not None else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("VLLM_13B_SERVING_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
