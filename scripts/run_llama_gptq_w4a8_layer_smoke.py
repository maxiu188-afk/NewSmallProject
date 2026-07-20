#!/usr/bin/env python3
"""Compare a real Llama layer with packed-GPTQ oracle and CUDA W4A8 q_proj.

Only the selected q_proj is W4A8; every other model weight and the K/V cache
remain BF16.  This is a Phase-2 fixed-token correctness gate, not full-model
quantization, PPL, KV4, or performance evidence.
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
from torch import nn

from repro.quarot_pipeline import apply_llama_quarot, load_model_and_tokenizer, load_pipeline_config, resolve_device, validate_pipeline_config
from repro.w4a8_linear import PackedW4A8ReferenceLinear, W4A8Linear


ARTIFACT_FORMAT = "newsmallproject-gptq-w4a8-linear-v1"
ARTIFACT_LAYOUT = "packed-w4-row-major-le-nibble-v1"
LAYER_ATOL = 2e-4
LAYER_RTOL = 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _hidden_states(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    hidden_states = getattr(output, "hidden_states", None)
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    raise RuntimeError("decoder layer returned an unsupported output type")


def _load_artifact(artifact_path: Path, report_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_hash = _sha256(artifact_path)
    if report.get("artifact", {}).get("sha256") != artifact_hash:
        raise RuntimeError("artifact SHA-256 does not match its report")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    required = {"format", "layout", "tensor", "packed_weight", "weight_scales", "input_permutation", "bias"}
    missing = sorted(required - set(artifact))
    if missing:
        raise RuntimeError("packed artifact misses fields: {}".format(", ".join(missing)))
    if artifact["format"] != ARTIFACT_FORMAT or artifact["layout"] != ARTIFACT_LAYOUT:
        raise RuntimeError("unsupported packed artifact format or layout")
    return artifact


def _replace_submodule(model: nn.Module, qualified_name: str, replacement: nn.Module) -> nn.Module:
    parent_name, child_name = qualified_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original = getattr(parent, child_name)
    if not isinstance(original, nn.Module):
        raise RuntimeError("target is not a module: {}".format(qualified_name))
    setattr(parent, child_name, replacement)
    return original


def _run_and_capture(model: nn.Module, layer: nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(_hidden_states(output).detach().float().cpu())

    handle = layer.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits.detach().float().cpu()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected one captured decoder-layer output, received {}".format(len(captured)))
    return captured[0], logits


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    error = (reference - candidate).abs()
    return {
        "shape": list(candidate.shape),
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "allclose": bool(torch.allclose(reference, candidate, atol=LAYER_ATOL, rtol=LAYER_RTOL)),
    }


def run(config_path: Path, artifact_path: Path, report_path: Path, sequence_length: int) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Llama layer W4A8 smoke requires CUDA; CPU or MPS fallback is not permitted")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    config = load_pipeline_config(config_path)
    validate_pipeline_config(config)
    artifact = _load_artifact(artifact_path, report_path)
    tensor_name = artifact["tensor"]
    if not isinstance(tensor_name, str):
        raise RuntimeError("artifact tensor name must be a string")
    if artifact["packed_weight"].dtype != torch.uint8 or artifact["weight_scales"].dtype != torch.float32:
        raise RuntimeError("artifact tensor dtypes are incompatible with owned W4A8")
    torch.manual_seed(int(config["experiment"].get("seed", 0)))
    device = resolve_device(config["runtime"])
    model, _ = load_model_and_tokenizer(config, device)
    rotation = apply_llama_quarot(model, config["experiment"]["rotation"], config["experiment"]["quantization"])
    if artifact["tensor"] not in dict(model.named_modules()):
        raise RuntimeError("artifact target is absent after QuaRot rotation: {}".format(tensor_name))
    target = model.get_submodule(tensor_name)
    if not isinstance(target, nn.Linear):
        raise RuntimeError("artifact target is not an nn.Linear after QuaRot rotation")
    if tuple(target.weight.shape) != (artifact["packed_weight"].shape[0], artifact["packed_weight"].shape[1] * 2):
        raise RuntimeError("artifact weight shape does not match the selected Llama linear")
    if artifact["bias"] is not None:
        raise RuntimeError("this Llama q_proj is expected to be bias-free")
    input_ids = torch.arange(sequence_length, device=device, dtype=torch.long).view(1, -1) % model.config.vocab_size
    reference_linear = PackedW4A8ReferenceLinear(
        artifact["packed_weight"], artifact["weight_scales"], artifact["bias"], artifact["input_permutation"]
    ).to(device).eval()
    original = _replace_submodule(model, tensor_name, reference_linear)
    layer = model.model.layers[0]
    reference_layer, reference_logits = _run_and_capture(model, layer, input_ids)
    deployment_linear = W4A8Linear(
        artifact["packed_weight"], artifact["weight_scales"], artifact["bias"], artifact["input_permutation"]
    ).to(device).eval()
    _replace_submodule(model, tensor_name, deployment_linear)
    candidate_layer, candidate_logits = _run_and_capture(model, layer, input_ids)
    del original, reference_linear, deployment_linear, model
    torch.cuda.empty_cache()
    layer_comparison = _comparison(reference_layer, candidate_layer)
    logits_comparison = _comparison(reference_logits, candidate_logits)
    if not layer_comparison["allclose"] or not logits_comparison["allclose"]:
        raise RuntimeError("W4A8 Llama layer output differs from the packed floating oracle")
    return {
        "scope": "fixed-token Llama layer correctness: one formal GPTQ packed q_proj is W4A8; all other weights and K/V remain BF16; not a packed full-model, PPL, KV4, or performance result",
        "source": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "artifact_report": str(report_path),
            "project_revision": _project_revision(),
        },
        "model": dict(config["model"]),
        "rotation": rotation,
        "replacement": {
            "tensor": tensor_name,
            "weight_precision": "signed W4 packed in uint8 with FP32 group scales",
            "activation_precision": "per-token symmetric A8",
            "other_linear_precision": "BF16",
            "key_value_precision": "BF16",
            "input_permutation_applied_before_a8": artifact["input_permutation"] is not None,
        },
        "input": {"shape": list(input_ids.shape), "sequence": "arange token ids modulo vocab", "seed": int(config["experiment"].get("seed", 0))},
        "tolerance": {"atol": LAYER_ATOL, "rtol": LAYER_RTOL},
        "layer_output": layer_comparison,
        "logits": logits_comparison,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()
    try:
        result = run(args.config, args.artifact, args.artifact_report, args.sequence_length)
    except (OSError, RuntimeError, ValueError) as error:
        print("LLAMA GPTQ W4A8 LAYER SMOKE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
