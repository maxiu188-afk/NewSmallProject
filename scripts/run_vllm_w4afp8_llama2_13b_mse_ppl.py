#!/usr/bin/env python3
"""Evaluate the isolated W4AFP8 min/max versus MSE observer diagnostic."""

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

from repro.vllm_w4afp8_mse_diagnostic import (
    PPL_MODELS,
    validate_config,
    validate_minmax_checkpoint_quantization_config,
    validate_mse_checkpoint_quantization_config,
)
from scripts.run_vllm_w4a16_llama2_13b_ppl import (
    _comparison,
    _one_model,
    _revision,
    _sha256,
    _write_json,
)
from scripts.run_vllm_w4afp8_llama2_13b_ppl import _load_sequences


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
    if tuple(models) != PPL_MODELS:
        raise ValueError(f"model order must be {PPL_MODELS}")
    return models


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


def run(
    *,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    models: dict[str, Path],
    max_sequences: int | None,
    worker_output_dir: Path,
) -> dict[str, Any]:
    validate_config(config)
    if tuple(models) != PPL_MODELS:
        raise ValueError(f"model order must be {PPL_MODELS}")
    sequences, manifest = _load_sequences(config, manifest_path, max_sequences)
    results = {
        name: _run_worker(
            config_path=config_path,
            manifest_path=manifest_path,
            name=name,
            model_path=model_path,
            max_sequences=max_sequences,
            output_path=worker_output_dir / f"{name}.json",
        )
        for name, model_path in models.items()
    }
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("vLLM PPL workers returned inconsistent runtime metadata")
    comparisons = {
        f"{name}_vs_bf16": _comparison(results[name], results["bf16"])
        for name in PPL_MODELS[1:]
    }
    comparisons.update(
        {
            "minmax_quarot_vs_unrotated": _comparison(
                results["quarot_minmax_w4afp8"],
                results["unrotated_minmax_w4afp8"],
            ),
            "mse_quarot_vs_unrotated": _comparison(
                results["quarot_mse_w4afp8"],
                results["unrotated_mse_w4afp8"],
            ),
            "unrotated_mse_vs_minmax": _comparison(
                results["unrotated_mse_w4afp8"],
                results["unrotated_minmax_w4afp8"],
            ),
            "quarot_mse_vs_minmax": _comparison(
                results["quarot_mse_w4afp8"],
                results["quarot_minmax_w4afp8"],
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
    validate_config(config)
    manifest_path = args.tokens_manifest.resolve()

    if args.worker_name is not None:
        if args.worker_name not in PPL_MODELS or args.worker_model is None:
            parser.error("worker mode requires a known --worker-name and --worker-model")
        sequences, _ = _load_sequences(config, manifest_path, args.max_sequences)
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            sequences,
            config,
        )
        if args.worker_name != "bf16":
            quantization_config = result["quantization_config"]
            if "_mse_" in args.worker_name:
                validate_mse_checkpoint_quantization_config(quantization_config)
            else:
                validate_minmax_checkpoint_quantization_config(quantization_config)
        _write_json(args.output.resolve(), result)
        print(f"VLLM_W4AFP8_MSE_PPL_MODEL_PASSED={args.worker_name}")
        return 0

    result = run(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        models=_parse_models(args.model),
        max_sequences=args.max_sequences,
        worker_output_dir=args.output.resolve().parent / f"{args.output.stem}-models",
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
