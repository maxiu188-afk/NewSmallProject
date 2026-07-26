#!/usr/bin/env python3
"""Export one pinned Llama-2-13B GPTQ W4A16 checkpoint for vLLM."""

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


def _snapshot(config: dict) -> Path:
    hub = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    repo_dir = "models--" + config["model"]["id"].replace("/", "--")
    return hub / repo_dir / "snapshots" / config["model"]["revision"]


def _packed_linears(output_dir: Path) -> list[str]:
    packed = []
    for path in output_dir.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            packed.extend(key for key in tensors.keys() if key.endswith(".weight_packed"))
    return sorted(packed)


def export(
    config_path: Path,
    calibration_dir: Path,
    output_dir: Path,
    report_path: Path,
    mode: str,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Llama-2-13B W4A16 export requires an allocated CUDA device")
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".incomplete").exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint state: {output_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = config["model"]
    quant = config["quantization"]
    calibration_spec = config["calibration"]
    snapshot = _snapshot(config)
    calibration_manifest = json.loads(
        (calibration_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if calibration_manifest["project_revision"] != _revision():
        raise RuntimeError("calibration manifest revision does not match source revision")
    dataset = load_from_disk(calibration_dir / "dataset")
    if len(dataset) != int(calibration_spec["samples"]):
        raise RuntimeError("calibration dataset sample count mismatch")

    torch.manual_seed(int(calibration_spec["seed"]))
    torch.cuda.reset_peak_memory_stats()
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
        baseline_last_logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()

    rotation = {"applied": False, "kind": "none"}
    rotation_error = None
    if mode == "rotated":
        rotation = apply_offline_llama_rotation(model)
        assert_standard_llama_layout(model)
        with torch.inference_mode():
            rotated_last_logits = model(input_ids=prompt, use_cache=False).logits[:, -1].float().cpu()
        rotation_error = float((baseline_last_logits - rotated_last_logits).abs().max().item())
        if not torch.isfinite(rotated_last_logits).all():
            raise RuntimeError("offline-rotated BF16 logits are not finite")
        tolerance = float(config["rotation"]["max_absolute_logit_error_tolerance"])
        if rotation_error > tolerance:
            raise RuntimeError(
                f"offline rotation error {rotation_error} exceeds tolerance {tolerance}"
            )
    elif mode != "unrotated":
        raise ValueError(f"unsupported mode: {mode}")

    recipe = GPTQModifier(
        targets=quant["targets"],
        scheme=quant["scheme"],
        ignore=quant["ignore"],
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

    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        incomplete,
        save_compressed=True,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(incomplete)
    saved_config = json.loads((incomplete / "config.json").read_text(encoding="utf-8"))
    quantization_config = saved_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError("saved checkpoint lacks quantization_config")
    if (
        quantization_config.get("quant_method") != "compressed-tensors"
        or quantization_config.get("format") != quant["format"]
        or quantization_config.get("quantization_status") != "compressed"
    ):
        raise RuntimeError(f"unexpected quantization metadata: {quantization_config}")
    groups = list(quantization_config.get("config_groups", {}).values())
    if (
        len(groups) != 1
        or groups[0].get("weights", {}).get("num_bits") != 4
        or groups[0].get("weights", {}).get("group_size") != int(quant["group_size"])
    ):
        raise RuntimeError(f"checkpoint is not group-128 W4: {quantization_config}")
    packed_linears = _packed_linears(incomplete)
    if len(packed_linears) != int(model_spec["expected_decoder_linears"]):
        raise RuntimeError(
            f"expected {model_spec['expected_decoder_linears']} packed linears, "
            f"found {len(packed_linears)}"
        )
    incomplete.rename(output_dir)

    result = {
        "status": "passed",
        "scope": "complete Llama-2-13B compressed-tensors GPTQ W4A16 export; vLLM execution is separate",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "calibration_manifest_sha256": _sha256(calibration_dir / "manifest.json"),
        "mode": mode,
        "model": model_spec,
        "rotation": rotation,
        "prequant_rotation_max_absolute_logit_error": rotation_error,
        "quantization_config": quantization_config,
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
            "llmcompressor": metadata.version("llmcompressor"),
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
    parser.add_argument("--mode", choices=("unrotated", "rotated"), required=True)
    args = parser.parse_args()
    export(
        args.config.resolve(),
        args.calibration_dir.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
