#!/usr/bin/env python3
"""Export one pinned Llama-2-13B compressed-tensors W4AFP8 checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_from_disk
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro.offline_llama_rotation import (
    apply_offline_llama_rotation,
    assert_standard_llama_layout,
)
from repro.spinquant.artifacts import load_rotation_artifact
from repro.spinquant.offline import apply_spinquant_llama_offline
from repro.spinquant.rotations import SpinQuantRotations
from repro.vllm_w4afp8 import (
    EXPECTED_VARIANTS,
    validate_checkpoint_quantization_config,
    validate_export_config,
    validate_no_runtime_g_idx,
)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(config: dict[str, Any]) -> Path:
    hub = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    repo_dir = "models--" + config["model"]["id"].replace("/", "--")
    return hub / repo_dir / "snapshots" / config["model"]["revision"]


def _checkpoint_tensor_names(output_dir: Path) -> list[str]:
    names = []
    for path in sorted(output_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            names.extend(tensors.keys())
    return sorted(names)


def _spinquant_rotations(model: Any, seed: int) -> SpinQuantRotations:
    config = model.config
    return SpinQuantRotations(
        hidden_size=int(config.hidden_size),
        head_dim=int(config.hidden_size) // int(config.num_attention_heads),
        num_layers=int(config.num_hidden_layers),
        seed=seed,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )


@torch.inference_mode()
def _apply_variant_rotation(
    model: Any,
    variant: str,
    config: dict[str, Any],
    rotation_manifest: Path | None,
) -> dict[str, Any]:
    if variant == "unrotated_w4afp8":
        if rotation_manifest is not None:
            raise ValueError("unrotated export does not accept a rotation manifest")
        return {"applied": False, "kind": "none"}
    if variant == "quarot_w4afp8":
        if rotation_manifest is not None:
            raise ValueError("QuaRot-style export does not accept a learned rotation")
        summary = apply_offline_llama_rotation(model)
        assert_standard_llama_layout(model)
        return {**summary, "applied": True, "kind": "quarot_style_fixed_offline"}
    if variant != "spinquant_w4afp8":
        raise ValueError(f"unsupported W4AFP8 variant: {variant}")
    if rotation_manifest is None:
        raise ValueError("SpinQuant W4AFP8 export requires --rotation-manifest")
    rotation_spec = config["rotation"]
    if _sha256(rotation_manifest) != rotation_spec["spinquant_rotation_manifest_sha256"]:
        raise RuntimeError("SpinQuant rotation manifest SHA-256 changed")
    rotations = _spinquant_rotations(model, int(rotation_spec["seed"]))
    loaded = load_rotation_artifact(
        rotation_manifest,
        target=rotations,
        maximum_orthogonality_error=float(
            rotation_spec["maximum_orthogonality_error"]
        ),
    )
    if (
        loaded["manifest"]["file"]["sha256"]
        != rotation_spec["spinquant_rotation_safetensors_sha256"]
    ):
        raise RuntimeError("SpinQuant rotation SafeTensor SHA-256 changed")
    summary = apply_spinquant_llama_offline(model, rotations)
    del rotations
    assert_standard_llama_layout(model)
    return {
        "applied": True,
        "kind": "spinquant_learned_offline",
        "evidence_label": rotation_spec["spinquant_evidence_label"],
        "training_job_id": rotation_spec["spinquant_training_job_id"],
        "observed_orthogonality_error": loaded["observed_orthogonality_error"],
        **summary,
    }


def export(
    *,
    config_path: Path,
    calibration_dir: Path,
    output_dir: Path,
    report_path: Path,
    variant: str,
    rotation_manifest: Path | None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Llama-2-13B W4AFP8 export requires an allocated CUDA device")
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or incomplete.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint state: {output_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_export_config(config)
    if variant not in EXPECTED_VARIANTS[1:]:
        raise ValueError(f"variant must be one of {EXPECTED_VARIANTS[1:]}")
    model_spec = config["model"]
    quant = config["quantization"]
    calibration_spec = config["calibration"]
    runtime = config["runtime"]
    calibration_manifest_path = calibration_dir / "manifest.json"
    calibration_manifest = json.loads(
        calibration_manifest_path.read_text(encoding="utf-8")
    )
    if calibration_manifest["project_revision"] != _revision():
        raise RuntimeError("calibration manifest revision does not match source revision")
    if calibration_manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("calibration manifest used a different W4AFP8 config")
    dataset = load_from_disk(calibration_dir / "dataset")
    if len(dataset) != int(calibration_spec["samples"]):
        raise RuntimeError("calibration dataset sample count mismatch")

    installed = {
        "llmcompressor": metadata.version("llmcompressor"),
        "compressed_tensors": metadata.version("compressed-tensors"),
    }
    if installed["llmcompressor"] != runtime["llmcompressor_version"]:
        raise RuntimeError(f"unexpected llmcompressor: {installed['llmcompressor']}")
    if (
        installed["compressed_tensors"]
        != runtime["quantizer_compressed_tensors_version"]
    ):
        raise RuntimeError(
            f"unexpected quantizer compressed-tensors: {installed['compressed_tensors']}"
        )

    torch.manual_seed(int(calibration_spec["seed"]))
    torch.cuda.reset_peak_memory_stats()
    snapshot = _snapshot(config)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
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
    rotation = _apply_variant_rotation(model, variant, config, rotation_manifest)
    with torch.inference_mode():
        transformed_logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
    rotation_error = float((baseline_logits - transformed_logits).abs().max().item())
    if not torch.isfinite(transformed_logits).all():
        raise RuntimeError("offline-transformed BF16 logits are not finite")
    if rotation_error > float(config["rotation"]["max_absolute_logit_error_tolerance"]):
        raise RuntimeError(f"offline rotation error exceeds tolerance: {rotation_error}")

    recipe = GPTQModifier(
        targets=quant["targets"],
        scheme=quant["scheme"],
        ignore=quant["ignore"],
        block_size=int(quant["weight_block_size"]),
        dampening_frac=float(quant["dampening_frac"]),
        actorder=quant["weight_actorder"],
    )
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
        raise RuntimeError("saved W4AFP8 checkpoint lacks quantization_config")
    validate_checkpoint_quantization_config(quantization_config)
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
        "scope": (
            "complete Llama-2-13B compressed-tensors W4AFP8 export; vLLM "
            "execution, deployed PPL, and performance are separate"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "calibration_manifest_sha256": _sha256(calibration_manifest_path),
        "variant": variant,
        "model": model_spec,
        "rotation": rotation,
        "prequant_rotation_max_absolute_logit_error": rotation_error,
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
    parser.add_argument("--variant", choices=EXPECTED_VARIANTS[1:], required=True)
    parser.add_argument("--rotation-manifest", type=Path)
    args = parser.parse_args()
    export(
        config_path=args.config.resolve(),
        calibration_dir=args.calibration_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve(),
        variant=args.variant,
        rotation_manifest=(
            args.rotation_manifest.resolve()
            if args.rotation_manifest is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
