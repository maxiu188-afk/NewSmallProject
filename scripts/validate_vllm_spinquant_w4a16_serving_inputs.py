#!/usr/bin/env python3
"""Validate the accepted SpinQuant W4A16 gate before serving measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_w4a16_llama2_13b_serving import (  # noqa: E402
    _expected_models,
    _validate_config,
)
from scripts.validate_vllm_spinquant_w4a16_boolq_inputs import (  # noqa: E402
    validate as validate_source_gate,
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
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if _expected_models(config) != ("bf16", "spinquant_w4a16"):
        raise RuntimeError("SpinQuant W4A16 serving model order changed")
    if config.get("execution", {}).get("kind") != "formal_only":
        raise RuntimeError("SpinQuant W4A16 serving must remain formal-only")
    if config["kernel_gate"].get("quantized_models") != ["spinquant_w4a16"]:
        raise RuntimeError("SpinQuant W4A16 serving kernel gate changed")
    result = validate_source_gate(
        project_root=project_root,
        config_path=config_path,
        export_result_path=export_result_path,
        inference_result_path=inference_result_path,
        source_manifest_path=source_manifest_path,
        stdout_path=stdout_path,
        models=models,
    )
    result["execution"] = config["execution"]
    result["serving_protocol"] = {
        "input_length": config["benchmark"]["input_length"],
        "output_length": config["benchmark"]["output_length"],
        "num_prompts": config["benchmark"]["num_prompts"],
        "num_warmups": config["benchmark"]["num_warmups"],
        "cases": config["benchmark"]["cases"],
        "kv_cache_memory_bytes": config["server"]["kv_cache_memory_bytes"],
    }
    return result


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
    print("VLLM_SPINQUANT_W4A16_SERVING_INPUTS_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
