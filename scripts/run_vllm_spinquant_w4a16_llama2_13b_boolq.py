#!/usr/bin/env python3
"""Evaluate BF16 and SpinQuant-derived packed W4A16 on frozen BoolQ."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.vllm_w4a16 import validate_checkpoint_quantization_config  # noqa: E402
from scripts import run_vllm_w4afp8_llama2_13b_boolq as common  # noqa: E402


EXPECTED_MODELS = ("bf16", "spinquant_w4a16")

# Reuse the already accepted BoolQ prompt, token-boundary, and scoring code.
_choice_loglikelihood = common._choice_loglikelihood
_context = common._context
_encode_pair = common._encode_pair
_load_examples = common._load_examples
_one_model = common._one_model
_sha256 = common._sha256
_write_json = common._write_json


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_worker(
    *,
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    name: str,
    model_path: Path,
    max_examples: int | None,
    output_path: Path,
) -> dict[str, Any]:
    """Run each model in a fresh process so vLLM releases all CUDA state."""

    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--dataset-manifest",
        str(manifest_path),
        "--tokenizer-model",
        str(tokenizer_path),
        "--worker-name",
        name,
        "--worker-model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    if max_examples is not None:
        command.extend(["--max-examples", str(max_examples)])
    subprocess.run(command, check=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result.get("name") != name:
        raise RuntimeError(f"BoolQ worker returned the wrong model name for {name}")
    return result


def _parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model requires name=path: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            raise ValueError(f"repeated model: {name}")
        path = Path(raw_path).resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")
        models[name] = path
    if tuple(models) != EXPECTED_MODELS:
        raise ValueError(f"model order must be {EXPECTED_MODELS}")
    return models


def _accuracy_comparison(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | int]:
    if candidate["evaluated_examples"] != reference["evaluated_examples"]:
        raise RuntimeError("BoolQ comparison uses different example counts")
    delta = float(candidate["accuracy"]) - float(reference["accuracy"])
    return {
        "accuracy_delta": delta,
        "accuracy_point_delta": 100.0 * delta,
        "correct_delta": int(candidate["correct"]) - int(reference["correct"]),
    }


def run(
    *,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    models: dict[str, Path],
    max_examples: int | None,
    worker_output_dir: Path,
) -> dict[str, Any]:
    if config["source_gate"].get("status") != "accepted":
        raise RuntimeError("SpinQuant W4A16 source gate is still pending")
    if tuple(models) != EXPECTED_MODELS:
        raise ValueError(f"model order must be {EXPECTED_MODELS}")
    if tuple(config["source_gate"]["expected_models"]) != EXPECTED_MODELS:
        raise ValueError("source-gate model order drifted")

    examples, manifest = _load_examples(config, manifest_path, max_examples)
    results = {}
    for name, model_path in models.items():
        results[name] = _run_worker(
            config_path=config_path,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer_path,
            name=name,
            model_path=model_path,
            max_examples=max_examples,
            output_path=worker_output_dir / f"{name}.json",
        )

    runtimes = {
        json.dumps(model["runtime"], sort_keys=True) for model in results.values()
    }
    if len(runtimes) != 1:
        raise RuntimeError("BoolQ workers returned inconsistent runtime metadata")
    first_indices = [item["idx"] for item in results["bf16"]["example_metrics"]]
    for name, result in results.items():
        if [item["idx"] for item in result["example_metrics"]] != first_indices:
            raise RuntimeError(f"{name} evaluated a different BoolQ example order")

    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "examples_sha256": manifest["examples"]["sha256"],
        "evaluated_examples": len(examples),
        "evaluated_requests": len(examples) * len(config["protocol"]["choices"]),
        "protocol": config["protocol"],
        "engine": config["engine"],
        "runtime": json.loads(runtimes.pop()),
        "models": results,
        "comparisons": {
            "spinquant_w4a16_vs_bf16": _accuracy_comparison(
                results["spinquant_w4a16"], results["bf16"]
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    manifest_path = args.dataset_manifest.resolve()
    tokenizer_path = args.tokenizer_model.resolve()

    if args.worker_name is not None:
        if args.worker_name not in EXPECTED_MODELS or args.worker_model is None:
            parser.error("worker mode requires a known --worker-name and --worker-model")
        examples, _ = _load_examples(config, manifest_path, args.max_examples)
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            tokenizer_path,
            examples,
            config,
        )
        if args.worker_name != "bf16":
            validate_checkpoint_quantization_config(result["quantization_config"])
        _write_json(args.output.resolve(), result)
        print(f"VLLM_SPINQUANT_W4A16_BOOLQ_MODEL_PASSED={args.worker_name}")
        return 0

    models = _parse_models(args.model)
    result = run(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        tokenizer_path=tokenizer_path,
        models=models,
        max_examples=args.max_examples,
        worker_output_dir=args.output.resolve().parent
        / f"{args.output.stem}-models",
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_BOOLQ_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
