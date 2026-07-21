#!/usr/bin/env python3
"""Stream a complete QuaRot/GPTQ Llama decoder into a sharded W4A8 checkpoint."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch

from repro.gptq import GPTQSettings, quantize_llama_weights_gptq
from repro.quarot_pipeline import (
    _calibration_config,
    apply_llama_quarot,
    load_model_and_tokenizer,
    load_pipeline_config,
    resolve_device,
    token_batches,
    validate_pipeline_config,
)
from repro.w4a8_checkpoint import PackedW4A8CheckpointWriter, decoder_linear_names, sha256_file


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def run(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("full GPTQ W4A8 checkpoint export requires CUDA; CPU or MPS fallback is not permitted")
    config = load_pipeline_config(config_path)
    validate_pipeline_config(config)
    weight_config = config["experiment"].get("weight_quantization", {})
    if weight_config.get("method") != "gptq":
        raise RuntimeError("checkpoint export configuration must select GPTQ")
    if int(config["experiment"]["quantization"]["w_bits"]) != 4:
        raise RuntimeError("checkpoint export requires w_bits=4")
    torch.manual_seed(int(config["experiment"].get("seed", 0)))
    device = resolve_device(config["runtime"])
    model, tokenizer = load_model_and_tokenizer(config, device)
    calibration = list(token_batches(_calibration_config(config), model, tokenizer))
    if not calibration:
        raise RuntimeError("checkpoint export received no calibration batches")
    rotation = apply_llama_quarot(
        model, config["experiment"]["rotation"], config["experiment"]["quantization"]
    )
    names = decoder_linear_names(model)
    expected_per_layer = 7
    layer_count = int(model.config.num_hidden_layers)
    if len(names) != layer_count * expected_per_layer:
        raise RuntimeError(
            "expected {} decoder linears for {} Llama layers, found {}".format(
                layer_count * expected_per_layer, layer_count, len(names)
            )
        )
    writer = PackedW4A8CheckpointWriter(
        output_dir,
        names,
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "project_revision": _project_revision(),
            "model": dict(config["model"]),
        },
    )
    progress = {"written": 0}

    def write_packed(name: str, packed: Any) -> None:
        linear = model.get_submodule(name)
        if not isinstance(linear, torch.nn.Linear):
            raise RuntimeError("GPTQ capture target is not nn.Linear: {}".format(name))
        writer.write(name, packed, linear.bias)
        progress["written"] += 1
        print(
            "W4A8_CHECKPOINT_SHARD={}/{} {}".format(progress["written"], len(names), name),
            flush=True,
        )

    summary = quantize_llama_weights_gptq(
        model,
        calibration,
        GPTQSettings(
            bits=4,
            group_size=int(weight_config.get("group_size", 128)),
            damp_percent=float(weight_config.get("damp_percent", 0.01)),
            block_size=int(weight_config.get("block_size", 128)),
            act_order=bool(weight_config.get("act_order", True)),
            symmetric=bool(weight_config.get("symmetric", True)),
        ),
        capture_packed_linears=names,
        packed_weight_callback=write_packed,
    )
    runtime = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    manifest_path = writer.finalize(
        {
            "rotation": rotation,
            "gptq_settings": dict(weight_config),
            "gptq_summary": summary,
            "decoder_layers": layer_count,
            "decoder_linears": len(names),
            "decoder_linear_precision": "packed signed W4 with FP32 group scales and per-linear act-order permutation",
            "activation_precision": "per-token symmetric A8 at runtime",
            "key_value_precision": "BF16",
            "embedding_and_lm_head_precision": "BF16",
            "runtime": runtime,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    return {
        "scope": "complete sharded GPTQ W4 checkpoint for every Llama decoder linear; correctness, PPL, KV4, and performance require separate gates",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tensor_count": len(names),
        "runtime": runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.config, args.output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        print("FULL GPTQ W4A8 CHECKPOINT EXPORT FAILED: {}".format(error), file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
