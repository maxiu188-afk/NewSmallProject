#!/usr/bin/env python3
"""Measure matched WikiText-2 PPL through four deployed W4AFP8 checkpoints."""

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

from repro.vllm_w4afp8 import (  # noqa: E402
    EXPECTED_VARIANTS,
    validate_checkpoint_quantization_config,
)
from scripts.run_vllm_w4a16_llama2_13b_ppl import (  # noqa: E402
    _comparison,
    _one_model,
    _revision,
    _sha256,
    _write_json,
)


def _load_sequences(
    config: dict[str, Any],
    manifest_path: Path,
    max_sequences: int | None,
) -> tuple[list[list[int]], dict[str, Any]]:
    spec = config["tokens"]
    if _sha256(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError("accepted PPL token manifest SHA-256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError("PPL token manifest did not pass")
    source_config = (PROJECT_ROOT / spec["source_config"]).resolve()
    if manifest.get("config_sha256") != _sha256(source_config):
        raise RuntimeError("PPL tokens used a different frozen source config")
    dataset = manifest.get("dataset", {})
    if dataset.get("token_ids_sha256") != spec["token_ids_sha256"]:
        raise RuntimeError("PPL token IDs changed")
    tokens_path = Path(manifest["tokens"]["path"])
    if _sha256(tokens_path) != manifest["tokens"]["sha256"]:
        raise RuntimeError("PPL token file SHA-256 changed")
    sequences = json.loads(tokens_path.read_text(encoding="utf-8"))
    if (
        len(sequences) != int(spec["samples"])
        or any(
            not isinstance(sequence, list)
            or len(sequence) != int(spec["sequence_length"])
            or not all(isinstance(token, int) for token in sequence)
            for sequence in sequences
        )
    ):
        raise RuntimeError("PPL token sequence shape or type changed")
    if max_sequences is not None:
        if max_sequences < 1 or max_sequences > len(sequences):
            raise ValueError("max_sequences is outside the materialized range")
        sequences = sequences[:max_sequences]
    return sequences, manifest


def _run_worker(
    *,
    config_path: Path,
    manifest_path: Path,
    name: str,
    model_path: Path,
    max_sequences: int | None,
    output_path: Path,
) -> dict[str, Any]:
    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--tokens-manifest",
        str(manifest_path),
        "--worker-name",
        name,
        "--worker-model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    if max_sequences is not None:
        command.extend(["--max-sequences", str(max_sequences)])
    subprocess.run(command, check=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result.get("name") != name:
        raise RuntimeError(f"PPL worker returned the wrong model name for {name}")
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
    if tuple(models) != EXPECTED_VARIANTS:
        raise ValueError(f"model order must be {EXPECTED_VARIANTS}")
    return models


def run(
    *,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    models: dict[str, Path],
    max_sequences: int | None,
    worker_output_dir: Path,
) -> dict[str, Any]:
    if config["source_gate"].get("status") != "accepted":
        raise RuntimeError("W4AFP8 source gate is still pending")
    if tuple(models) != EXPECTED_VARIANTS:
        raise ValueError(f"model order must be {EXPECTED_VARIANTS}")
    if tuple(config["source_gate"]["expected_models"]) != EXPECTED_VARIANTS:
        raise ValueError("source-gate model order drifted")
    sequences, manifest = _load_sequences(config, manifest_path, max_sequences)
    results = {}
    for name, model_path in models.items():
        results[name] = _run_worker(
            config_path=config_path,
            manifest_path=manifest_path,
            name=name,
            model_path=model_path,
            max_sequences=max_sequences,
            output_path=worker_output_dir / f"{name}.json",
        )
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("vLLM PPL workers returned inconsistent runtime metadata")
    comparisons = {
        f"{name}_vs_bf16": _comparison(results[name], results["bf16"])
        for name in EXPECTED_VARIANTS[1:]
    }
    comparisons.update(
        {
            "quarot_vs_unrotated_w4afp8": _comparison(
                results["quarot_w4afp8"], results["unrotated_w4afp8"]
            ),
            "spinquant_vs_unrotated_w4afp8": _comparison(
                results["spinquant_w4afp8"], results["unrotated_w4afp8"]
            ),
            "spinquant_vs_quarot_w4afp8": _comparison(
                results["spinquant_w4afp8"], results["quarot_w4afp8"]
            ),
        }
    )
    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "tokens_manifest": str(manifest_path),
        "tokens_manifest_sha256": _sha256(manifest_path),
        "token_ids_sha256": manifest["dataset"]["token_ids_sha256"],
        "evaluated_sequences": len(sequences),
        "scored_tokens": len(sequences) * (len(sequences[0]) - 1),
        "engine": config["engine"],
        "runtime": json.loads(runtimes.pop()),
        "models": results,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["source_gate"].get("status") != "accepted":
        raise RuntimeError("W4AFP8 source gate is still pending")
    manifest_path = args.tokens_manifest.resolve()

    if args.worker_name is not None:
        if args.worker_name not in EXPECTED_VARIANTS or args.worker_model is None:
            parser.error("worker mode requires a known --worker-name and --worker-model")
        sequences, _ = _load_sequences(config, manifest_path, args.max_sequences)
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            sequences,
            config,
        )
        if args.worker_name != "bf16":
            validate_checkpoint_quantization_config(result["quantization_config"])
        _write_json(args.output.resolve(), result)
        print(f"VLLM_W4AFP8_PPL_MODEL_PASSED={args.worker_name}")
        return 0

    models = _parse_models(args.model)
    result = run(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        models=models,
        max_sequences=args.max_sequences,
        worker_output_dir=args.output.resolve().parent / f"{args.output.stem}-models",
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
