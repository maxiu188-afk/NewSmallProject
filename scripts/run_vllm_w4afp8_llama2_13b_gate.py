#!/usr/bin/env python3
"""Run fresh-process vLLM load/inference for selected W4AFP8 variants."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from safetensors import safe_open

from repro.vllm_w4afp8 import (  # noqa: E402
    EXPECTED_VARIANTS,
    QUAROT_VARIANTS,
    SPINQUANT_VARIANTS,
    validate_checkpoint_quantization_config,
    validate_no_runtime_g_idx,
)
from scripts.run_vllm_w4a16_llama2_13b_gate import (  # noqa: E402
    _compare,
    _one_model,
    _revision,
    _run_model_in_subprocess,
    _write_json,
)


VARIANT_SETS = {
    "quarot": QUAROT_VARIANTS,
    "spinquant": SPINQUANT_VARIANTS,
    "joint": EXPECTED_VARIANTS,
}

VARIANT_SCOPES = {
    "quarot": "QuaRot-only W4AFP8 export/load correctness gate; no SpinQuant, PPL, or performance claim",
    "spinquant": "SpinQuant INT8-trained rotation transfer to W4AFP8 export/load correctness gate; no PPL or performance claim",
}


def _parse_models(
    values: list[str],
    expected_variants: tuple[str, ...] = EXPECTED_VARIANTS,
) -> dict[str, Path]:
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
    if tuple(models) != expected_variants:
        raise ValueError(f"model order must be {expected_variants}")
    return models


def _validate_quantized_checkpoint(path: Path) -> dict[str, Any]:
    model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    quantization_config = model_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError(f"checkpoint lacks quantization_config: {path}")
    validate_checkpoint_quantization_config(quantization_config)
    tensor_names = []
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor_names.extend(handle.keys())
    validate_no_runtime_g_idx(tensor_names)
    packed = sum(name.endswith(".weight_packed") for name in tensor_names)
    if packed != 280:
        raise RuntimeError(f"checkpoint contains {packed} packed linears, expected 280")
    return {
        "packed_decoder_linear_count": packed,
        "runtime_g_idx_tensors": 0,
        "quantization_config": quantization_config,
    }


def run(
    *,
    config: dict[str, Any],
    config_path: Path,
    models: dict[str, Path],
    worker_output_dir: Path,
    expected_variants: tuple[str, ...] = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard gate requires the native vLLM sampler fallback")
    if tuple(models) != expected_variants:
        raise ValueError(f"model order must be {expected_variants}")
    variant_set = next(
        name for name, variants in VARIANT_SETS.items() if variants == expected_variants
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        models["bf16"],
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = tokenizer(
        config["inference"]["prompt"],
        add_special_tokens=True,
    )["input_ids"]
    checkpoint_metadata = {
        name: _validate_quantized_checkpoint(path)
        for name, path in models.items()
        if name != "bf16"
    }
    results = {}
    for name, path in models.items():
        results[name] = _run_model_in_subprocess(
            config_path,
            name,
            path,
            prompt,
            worker_output_dir / f"{name}.json",
        )
        if name in checkpoint_metadata:
            results[name]["checkpoint"] = checkpoint_metadata[name]
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("vLLM workers returned inconsistent runtime metadata")
    return {
        "status": "passed",
        "variant_set": variant_set,
        "scope": VARIANT_SCOPES.get(variant_set, config["scope"]),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "runtime": json.loads(runtimes.pop()),
        "vllm_use_flashinfer_sampler": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "prompt": config["inference"]["prompt"],
        "prompt_token_ids": prompt,
        "models": results,
        "comparisons": {
            f"{name}_vs_bf16": _compare(results[name], results["bf16"])
            for name in expected_variants[1:]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--variant-set", choices=tuple(VARIANT_SETS), default="joint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.worker_name is not None:
        if args.worker_model is None or args.prompt_token_ids is None:
            parser.error("--worker-name requires --worker-model and --prompt-token-ids")
        prompt = json.loads(args.prompt_token_ids)
        if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
            parser.error("--prompt-token-ids must encode a JSON list of integers")
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            prompt,
            config,
        )
        _write_json(args.output.resolve(), result)
        print(f"VLLM_W4AFP8_MODEL_PASSED={args.worker_name}")
        return 0

    result = run(
        config=config,
        config_path=config_path,
        models=_parse_models(args.model, VARIANT_SETS[args.variant_set]),
        worker_output_dir=args.output.resolve().parent / f"{args.output.stem}-models",
        expected_variants=VARIANT_SETS[args.variant_set],
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
