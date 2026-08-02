#!/usr/bin/env python3
"""Export one isolated MSE-clipped Llama-2-13B W4AFP8 checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import struct
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro.vllm_w4afp8 import validate_no_runtime_g_idx
from repro.vllm_w4afp8_mse_diagnostic import (
    MSE_VARIANTS,
    validate_config,
    validate_mse_checkpoint_quantization_config,
)
from scripts.export_vllm_w4afp8_llama2_13b import (
    _apply_variant_rotation,
    _checkpoint_tensor_names,
    _revision,
    _sha256,
    _snapshot,
)


BASE_VARIANTS = {
    "unrotated_mse_w4afp8": "unrotated_w4afp8",
    "quarot_mse_w4afp8": "quarot_w4afp8",
}


def _calibration_token_sha256(dataset: Any) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        for token in row["input_ids"]:
            digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def _resolved_observer(recipe: GPTQModifier, quant: dict[str, Any]) -> str:
    resolved = recipe.resolve_quantization_config()
    groups = tuple(resolved.config_groups.values())
    if len(groups) != 1:
        raise RuntimeError(f"expected one W4AFP8 recipe group, found {len(groups)}")
    weights = groups[0].weights
    if weights.group_size != int(quant["weight_group_size"]):
        raise RuntimeError(f"unexpected resolved GPTQ group size: {weights.group_size}")
    if weights.actorder is not None:
        raise RuntimeError(f"unexpected resolved GPTQ actorder: {weights.actorder}")
    if weights.observer not in set(quant["resolved_weight_observers"]):
        raise RuntimeError(f"unexpected resolved GPTQ observer: {weights.observer}")
    return str(weights.observer)


def export(
    *,
    config_path: Path,
    calibration_dir: Path,
    output_dir: Path,
    report_path: Path,
    variant: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Llama-2-13B W4AFP8 MSE export requires CUDA")
    if variant not in MSE_VARIANTS:
        raise ValueError(f"variant must be one of {MSE_VARIANTS}")
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic state: {output_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    model_spec = config["model"]
    calibration_spec = config["calibration"]
    quant = config["quantization"]
    runtime = config["runtime"]

    manifest_path = calibration_dir / "manifest.json"
    if _sha256(manifest_path) != calibration_spec["manifest_sha256"]:
        raise RuntimeError("accepted calibration manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_from_disk(calibration_dir / "dataset")
    if (
        manifest["status"] != "passed"
        or len(dataset) != int(calibration_spec["samples"])
        or manifest["dataset"]["token_ids_sha256"]
        != calibration_spec["token_ids_sha256"]
        or _calibration_token_sha256(dataset)
        != calibration_spec["token_ids_sha256"]
    ):
        raise RuntimeError("accepted calibration tokens changed")

    installed = {
        "llmcompressor": metadata.version("llmcompressor"),
        "compressed-tensors": metadata.version("compressed-tensors"),
    }
    if installed["llmcompressor"] != runtime["llmcompressor_version"]:
        raise RuntimeError(f"unexpected llmcompressor: {installed['llmcompressor']}")
    if (
        installed["compressed-tensors"]
        != runtime["quantizer_compressed_tensors_version"]
    ):
        raise RuntimeError(
            f"unexpected compressed-tensors: {installed['compressed-tensors']}"
        )

    torch.manual_seed(int(calibration_spec["seed"]))
    torch.cuda.reset_peak_memory_stats()
    snapshot = _snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        device_map="cuda",
    ).eval()
    if model.config.architectures != [model_spec["expected_architecture"]]:
        raise RuntimeError(f"unexpected model architecture: {model.config.architectures}")
    if int(model.config.num_hidden_layers) != int(model_spec["expected_layers"]):
        raise RuntimeError("unexpected decoder layer count")

    prompt = tokenizer(
        config["inference"]["prompt"],
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    with torch.inference_mode():
        baseline_logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
    rotation = _apply_variant_rotation(
        model,
        BASE_VARIANTS[variant],
        config,
        None,
    )
    with torch.inference_mode():
        transformed_logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
    rotation_error = float((baseline_logits - transformed_logits).abs().max().item())
    if not torch.isfinite(transformed_logits).all() or rotation_error > 0.25:
        raise RuntimeError(f"offline rotation validation failed: {rotation_error}")

    diagnostic_scheme = QuantizationScheme(
        targets=[quant["targets"]],
        weights=QuantizationArgs(
            num_bits=int(quant["weight_bits"]),
            type=quant["weight_type"],
            strategy=quant["weight_strategy"],
            group_size=int(quant["weight_group_size"]),
            symmetric=bool(quant["weight_symmetric"]),
            dynamic=False,
            actorder=quant["weight_actorder"],
            observer=quant["weight_observer"],
            observer_kwargs=quant["weight_observer_kwargs"],
        ),
        input_activations=QuantizationArgs(
            num_bits=int(quant["activation_bits"]),
            type=quant["activation_type"],
            strategy=quant["activation_strategy"],
            symmetric=bool(quant["activation_symmetric"]),
            dynamic=bool(quant["activation_dynamic"]),
            observer=None,
        ),
    )
    recipe = GPTQModifier(
        config_groups={"group_0": diagnostic_scheme},
        ignore=quant["ignore"],
        block_size=int(quant["weight_block_size"]),
        dampening_frac=float(quant["dampening_frac"]),
        actorder=quant["weight_actorder"],
    )
    resolved_observer = _resolved_observer(recipe, quant)
    print(f"VLLM_W4AFP8_MSE_RESOLVED_WEIGHT_OBSERVER={resolved_observer}")
    oneshot(
        model=model,
        dataset=dataset,
        processor=tokenizer,
        recipe=recipe,
        batch_size=1,
        max_seq_length=int(calibration_spec["sequence_length"]),
        num_calibration_samples=int(calibration_spec["samples"]),
        shuffle_calibration_samples=False,
        pipeline="sequential",
    )

    incomplete.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        incomplete,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size="5GB",
        quantization_format=quant["format"],
    )
    tokenizer.save_pretrained(incomplete)
    saved_config = json.loads((incomplete / "config.json").read_text(encoding="utf-8"))
    quantization_config = saved_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError("saved MSE checkpoint lacks quantization_config")
    checkpoint_observer = validate_mse_checkpoint_quantization_config(
        quantization_config
    )
    tensor_names = _checkpoint_tensor_names(incomplete)
    validate_no_runtime_g_idx(tensor_names)
    packed_linears = [name for name in tensor_names if name.endswith(".weight_packed")]
    if len(packed_linears) != int(model_spec["expected_decoder_linears"]):
        raise RuntimeError(
            f"expected {model_spec['expected_decoder_linears']} packed linears, "
            f"found {len(packed_linears)}"
        )
    incomplete.rename(output_dir)

    result = {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "calibration_manifest_sha256": _sha256(manifest_path),
        "calibration_token_ids_sha256": _calibration_token_sha256(dataset),
        "variant": variant,
        "base_variant": BASE_VARIANTS[variant],
        "rotation": rotation,
        "prequant_rotation_max_absolute_logit_error": rotation_error,
        "requested_weight_observer": quant["weight_observer"],
        "resolved_recipe_weight_observer": resolved_observer,
        "checkpoint_weight_observer": checkpoint_observer,
        "quantization_config": quantization_config,
        "runtime_g_idx_tensors": 0,
        "packed_decoder_linear_count": len(packed_linears),
        "checkpoint_bytes": sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        ),
        "checkpoint": str(output_dir),
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **installed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=MSE_VARIANTS, required=True)
    args = parser.parse_args()
    export(
        config_path=args.config.resolve(),
        calibration_dir=args.calibration_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve(),
        variant=args.variant,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
