#!/usr/bin/env python3
"""Export one formally GPTQ-quantized Llama linear in the owned W4A8 layout.

The exporter preserves GPTQ act-order grouping: the saved input permutation is
applied before A8 quantization, so the packed W4 groups retain the exact column
order that determined their GPTQ scales.
"""

import argparse
import datetime as dt
import hashlib
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
from repro.quarot_pipeline import _calibration_config, apply_llama_quarot, load_model_and_tokenizer, load_pipeline_config, resolve_device, token_batches, validate_pipeline_config
from repro.w4a8_linear import W4A8Linear, unpack_w4_weight, w4a8_reference_linear


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _dequantize_to_original_order(packed_weight: torch.Tensor, scales: torch.Tensor, permutation: torch.Tensor | None) -> torch.Tensor:
    integers = unpack_w4_weight(packed_weight)
    group_size = integers.shape[1] // scales.shape[1]
    packed_order = integers.reshape(integers.shape[0], scales.shape[1], group_size).float() * scales.unsqueeze(-1)
    packed_order = packed_order.reshape_as(integers)
    if permutation is None:
        return packed_order
    original_order = torch.empty_like(packed_order)
    original_order[:, permutation] = packed_order
    return original_order


def run(config_path: Path, tensor_name: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal GPTQ export requires CUDA; CPU or MPS fallback is not permitted")
    config = load_pipeline_config(config_path)
    validate_pipeline_config(config)
    if config["experiment"].get("weight_quantization", {}).get("method") != "gptq":
        raise RuntimeError("export configuration must select GPTQ weight quantization")
    if int(config["experiment"]["quantization"]["w_bits"]) != 4:
        raise RuntimeError("owned W4A8 exporter only supports w_bits=4")
    torch.manual_seed(int(config["experiment"].get("seed", 0)))
    device = resolve_device(config["runtime"])
    model, tokenizer = load_model_and_tokenizer(config, device)
    available_linears = {
        "model.layers.0." + name
        for name, module in model.model.layers[0].named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if tensor_name not in available_linears:
        raise RuntimeError(
            "requested tensor is not a capturable layer-0 linear; available names: {}".format(
                ", ".join(sorted(available_linears))
            )
        )
    calibration = list(token_batches(_calibration_config(config), model, tokenizer))
    if not calibration:
        raise RuntimeError("GPTQ export received no calibration batches")
    rotation = apply_llama_quarot(model, config["experiment"]["rotation"], config["experiment"]["quantization"])
    weight_config = config["experiment"]["weight_quantization"]
    captures = {}
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
        capture_packed_linears=[tensor_name],
        captured_packed_weights=captures,
    )
    if tensor_name not in captures:
        raise RuntimeError("GPTQ did not capture requested tensor {}".format(tensor_name))
    packed = captures[tensor_name]
    linear = dict(model.named_modules()).get(tensor_name)
    if not isinstance(linear, torch.nn.Linear):
        raise RuntimeError("requested tensor is not a Llama nn.Linear after GPTQ: {}".format(tensor_name))
    inputs = torch.randn((1, 2, linear.in_features), dtype=torch.bfloat16, device=device)
    deployment = W4A8Linear(packed.packed_weight, packed.scales, linear.bias, packed.input_permutation).to(device).eval()
    with torch.inference_mode():
        packed_reference = w4a8_reference_linear(inputs, packed.packed_weight, packed.scales, packed.input_permutation)
        deployed = deployment(inputs)
        reconstructed_weight = _dequantize_to_original_order(
            packed.packed_weight, packed.scales, packed.input_permutation
        )
    kernel_error = (deployed - packed_reference).abs()
    weight_error = (reconstructed_weight - linear.weight.float()).abs()
    if not torch.allclose(deployed, packed_reference, atol=1e-4, rtol=1e-5):
        raise RuntimeError("owned W4A8 kernel output differs from packed floating oracle")
    if not torch.allclose(reconstructed_weight, linear.weight.float(), atol=2e-3, rtol=1e-4):
        raise RuntimeError("exported W4 weights do not reconstruct the GPTQ linear")
    artifact = {
        "format": "newsmallproject-gptq-w4a8-linear-v1",
        "layout": "packed-w4-row-major-le-nibble-v1",
        "tensor": tensor_name,
        "packed_weight": packed.packed_weight.detach().cpu(),
        "weight_scales": packed.scales.detach().cpu(),
        "input_permutation": None if packed.input_permutation is None else packed.input_permutation.detach().cpu(),
        "bias": None if linear.bias is None else linear.bias.detach().float().cpu(),
    }
    result = {
        "scope": "formal F4 GPTQ W4 export of one real Llama linear plus owned W4A8 correctness check; not a packed full-model, PPL, KV4, or performance result",
        "model": dict(config["model"]),
        "source": {"config": str(config_path), "config_sha256": _sha256(config_path), "project_revision": _project_revision()},
        "tensor": {"name": tensor_name, "in_features": linear.in_features, "out_features": linear.out_features},
        "rotation": rotation,
        "gptq": {"settings": dict(weight_config), "summary": {key: summary[key] for key in ("layers", "linear_layers", "calibration_sequences", "calibration_sequence_length")}, "tensor_summary": summary["per_linear"][tensor_name]},
        "artifact": {"format": artifact["format"], "layout": artifact["layout"], "weight_dtype": "signed int4 packed in uint8", "scale_dtype": "float32", "input_permutation": packed.input_permutation is not None, "packed_weight_shape": list(packed.packed_weight.shape), "weight_scale_shape": list(packed.scales.shape)},
        "runtime": {"device": torch.cuda.get_device_name(device), "compute_capability": list(torch.cuda.get_device_capability(device)), "torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "verification": {"w4a8_kernel_vs_packed_oracle_max_absolute_error": float(kernel_error.max().item()), "packed_weight_vs_gptq_weight_max_absolute_error": float(weight_error.max().item())},
    }
    return artifact, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tensor", default="model.layers.0.self_attn.attention.q_proj")
    args = parser.parse_args()
    try:
        artifact, result = run(args.config, args.tensor)
    except (OSError, RuntimeError, ValueError) as error:
        print("FORMAL GPTQ W4 EXPORT FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.artifact)
    result["artifact"]["sha256"] = _sha256(args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
