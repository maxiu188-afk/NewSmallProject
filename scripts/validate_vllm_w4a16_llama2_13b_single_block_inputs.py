#!/usr/bin/env python3
"""Validate the accepted serving result and optional single-block smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_MODELS = ("bf16", "unrotated_w4a16", "rotated_w4a16")
EXPECTED_SERVING_CASES = ("latency_c1", "throughput_c8")
UNCHANGED_SOURCE_INPUTS = (
    "configs/deployment/vllm_w4a16_llama2_13b_isambard.json",
    "configs/deployment/vllm_w4a16_llama2_13b_serving_isambard.json",
    "repro/offline_llama_rotation.py",
    "repro/structured_hadamard.py",
    "scripts/export_vllm_w4a16_llama2_13b.py",
    "scripts/run_vllm_w4a16_llama2_13b_gate.py",
    "scripts/run_vllm_w4a16_llama2_13b_serving.py",
    "scripts/validate_vllm_w4a16_llama2_13b_serving_inputs.py",
    "scripts/run_isambard_vllm_w4a16_llama2_13b_service_smoke.sbatch",
    "scripts/run_isambard_vllm_w4a16_llama2_13b_serving_benchmark.sbatch",
)


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


def _check_serving_source(
    *,
    project_root: Path,
    config: dict[str, Any],
    source_result_path: Path,
    current_revision: str,
) -> dict[str, Any]:
    gate = config["source_gate"]
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    if _sha256(source_result_path) != gate["serving_benchmark_sha256"]:
        raise RuntimeError("accepted serving benchmark SHA-256 changed")
    if source["status"] != "passed":
        raise RuntimeError("accepted serving benchmark did not pass")
    if source["project_revision"] == current_revision:
        pass
    elif (
        _git(
            project_root,
            "merge-base",
            "--is-ancestor",
            source["project_revision"],
            current_revision,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("accepted serving revision is not an ancestor of HEAD")
    changed = _git(
        project_root,
        "diff",
        "--name-only",
        f"{source['project_revision']}..{current_revision}",
        "--",
        *UNCHANGED_SOURCE_INPUTS,
    ).stdout.splitlines()
    if changed:
        raise RuntimeError(f"serving-producing inputs changed: {changed}")

    if set(source["models"]) != set(EXPECTED_MODELS):
        raise RuntimeError("accepted serving model set changed")
    benchmark = source["benchmark_config"]
    expected_benchmark = {
        "input_length": 256,
        "output_length": 64,
        "num_prompts": 64,
        "num_warmups": 4,
        "ignore_eos": True,
        "save_detailed": True,
    }
    for key, expected in expected_benchmark.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(f"accepted serving setting changed: {key}")
    if [case["name"] for case in benchmark["cases"]] != list(EXPECTED_SERVING_CASES):
        raise RuntimeError("accepted serving cases changed")

    for model_name in EXPECTED_MODELS:
        model = source["models"][model_name]
        if set(model) != set(EXPECTED_SERVING_CASES):
            raise RuntimeError(f"{model_name} serving cases changed")
        for case_name in EXPECTED_SERVING_CASES:
            case = model[case_name]
            if case["metrics"]["request_throughput"] <= 0:
                raise RuntimeError(f"{model_name}/{case_name} has invalid throughput")
            if not all(math.isfinite(value) for value in case["metrics"].values()):
                raise RuntimeError(f"{model_name}/{case_name} has non-finite metrics")
            raw_path = Path(case["raw_result"])
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            if _sha256(raw_path) != case["raw_result_sha256"]:
                raise RuntimeError(f"{model_name}/{case_name} raw result changed")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if raw["project_revision"] != source["project_revision"]:
                raise RuntimeError(f"{model_name}/{case_name} raw revision changed")
            if raw["model_variant"] != model_name or raw["case"] != case_name:
                raise RuntimeError(f"{model_name}/{case_name} raw identity changed")
            if raw["completed"] != 64 or raw["failed"] != 0 or any(raw["errors"]):
                raise RuntimeError(f"{model_name}/{case_name} requests were incomplete")
            if set(raw["input_lens"]) != {256} or set(raw["output_lens"]) != {64}:
                raise RuntimeError(f"{model_name}/{case_name} request lengths changed")
    return source


def _check_smoke(
    *,
    smoke_result_path: Path,
    config: dict[str, Any],
    current_revision: str,
    models: dict[str, Path],
) -> None:
    smoke = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    if smoke["status"] != "passed" or smoke["protocol"] != "smoke":
        raise RuntimeError("single-block smoke did not pass")
    if smoke["project_revision"] != current_revision:
        raise RuntimeError("single-block smoke revision differs from formal revision")
    if smoke["source_gate"] != config["source_gate"]:
        raise RuntimeError("single-block smoke source gate changed")
    if list(smoke["models"]) != config["smoke"]["models"]:
        raise RuntimeError("single-block smoke model set changed")

    model_name = config["smoke"]["models"][0]
    model = smoke["models"][model_name]
    if Path(model["model_path"]).resolve() != models[model_name].resolve():
        raise RuntimeError("single-block smoke used a different W4A16 checkpoint")
    if model["status"] != "passed" or model["protocol"] != "smoke":
        raise RuntimeError("single-block smoke worker did not pass")
    if model["layer_inspection"]["packed_parameter_count"] <= 0:
        raise RuntimeError("single-block smoke did not execute a packed W4 layer")
    expected_cases = {case["name"]: case for case in config["smoke"]["cases"]}
    if set(model["cases"]) != set(expected_cases):
        raise RuntimeError("single-block smoke cases changed")
    repetitions = int(config["smoke"]["repetitions"])
    for case_name, case_config in expected_cases.items():
        case = model["cases"][case_name]
        expected_calls = (
            repetitions
            if case_config["phase"] == "prefill"
            else repetitions * int(case_config["decode_steps"])
        )
        if case["layer_call_elapsed_ms"]["count"] != expected_calls:
            raise RuntimeError(f"{case_name} returned the wrong layer-call count")
        if case["per_request_mean_layer_ms"]["count"] != repetitions:
            raise RuntimeError(f"{case_name} returned the wrong request count")
        if case["layer_call_elapsed_ms"]["minimum"] <= 0:
            raise RuntimeError(f"{case_name} returned non-positive timing")


def validate(
    *,
    project_root: Path,
    config_path: Path,
    source_result_path: Path,
    models: dict[str, Path],
    smoke_result_path: Path | None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    current_revision = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    if set(models) != set(EXPECTED_MODELS):
        raise RuntimeError("single-block model set changed")
    for name, path in models.items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} config is missing: {path / 'config.json'}")
    source = _check_serving_source(
        project_root=project_root,
        config=config,
        source_result_path=source_result_path,
        current_revision=current_revision,
    )
    if smoke_result_path is not None:
        _check_smoke(
            smoke_result_path=smoke_result_path,
            config=config,
            current_revision=current_revision,
            models=models,
        )
    return {
        "current_revision": current_revision,
        "serving_revision": source["project_revision"],
        "serving_result_sha256": _sha256(source_result_path),
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "smoke_result": (
            str(smoke_result_path.resolve()) if smoke_result_path is not None else None
        ),
        "smoke_result_sha256": (
            _sha256(smoke_result_path) if smoke_result_path is not None else None
        ),
    }


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
        name, raw_path = value.split("=", 1)
        if name in models:
            parser.error(f"repeated model: {name}")
        models[name] = Path(raw_path)
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
    print("VLLM_13B_SINGLE_BLOCK_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
