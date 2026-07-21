#!/usr/bin/env python3
"""Compare all packed Llama decoder linears through oracle and CUDA W4A8 paths."""

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
from torch import nn

from repro.quarot_pipeline import (
    apply_llama_quarot,
    load_model_and_tokenizer,
    load_pipeline_config,
    resolve_device,
    validate_pipeline_config,
)
from repro.w4a8_checkpoint import (
    convert_reference_linears_to_cuda,
    install_checkpoint_linears,
    load_checkpoint_manifest,
    sha256_file,
)


DEFAULT_ATOL = 2e-4
DEFAULT_RTOL = 1e-5


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
    hidden = getattr(output, "hidden_states", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise RuntimeError("decoder layer returned an unsupported output")


def _run_and_capture(model: nn.Module, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    captured: list[torch.Tensor] = []
    handles = []
    for layer in model.model.layers:
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(_hidden_states(output).detach().float().cpu())
        handles.append(layer.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(model.model.layers):
        raise RuntimeError("did not capture every decoder layer output")
    return captured, logits


def _comparison(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> Dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"shape": list(candidate.shape), "allclose": False, "reason": "shape mismatch"}
    error = (reference - candidate).abs()
    return {
        "shape": list(candidate.shape),
        "finite": bool(torch.isfinite(candidate).all()),
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
    }


def _greedy_generation_smoke(
    model: nn.Module, input_ids: torch.Tensor, new_tokens: int
) -> tuple[torch.Tensor, bool]:
    generated = input_ids
    past_key_values = None
    next_input = input_ids
    finite = True
    with torch.inference_mode():
        for _ in range(new_tokens):
            output = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
            logits = output.logits[:, -1, :]
            finite = finite and bool(torch.isfinite(logits).all())
            next_input = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_input), dim=1)
            past_key_values = output.past_key_values
    return generated, finite


def run(
    config_path: Path,
    manifest_path: Path,
    sequence_length: int,
    generation_tokens: int,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("full-model W4A8 smoke requires CUDA; CPU or MPS fallback is not permitted")
    if sequence_length <= 0 or generation_tokens <= 0:
        raise ValueError("sequence_length and generation_tokens must be positive")
    config = load_pipeline_config(config_path)
    validate_pipeline_config(config)
    manifest = load_checkpoint_manifest(manifest_path)
    source = manifest.get("source", {})
    if source.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("checkpoint source config SHA-256 differs from selected config")
    if source.get("model") != config["model"]:
        raise RuntimeError("checkpoint model provenance differs from selected config")
    if source.get("project_revision") != _project_revision():
        raise RuntimeError("checkpoint project revision differs from the current checkout")
    torch.manual_seed(int(config["experiment"].get("seed", 0)))
    device = resolve_device(config["runtime"])
    model, _ = load_model_and_tokenizer(config, device)
    rotation = apply_llama_quarot(
        model, config["experiment"]["rotation"], config["experiment"]["quantization"]
    )
    input_ids = torch.arange(sequence_length, device=device, dtype=torch.long).view(1, -1) % model.config.vocab_size
    names = install_checkpoint_linears(model, manifest_path, implementation="reference")
    reference_layers, reference_logits = _run_and_capture(model, input_ids)
    convert_reference_linears_to_cuda(model, names)
    candidate_layers, candidate_logits = _run_and_capture(model, input_ids)
    layer_comparisons = [
        _comparison(reference, candidate, atol, rtol)
        for reference, candidate in zip(reference_layers, candidate_layers)
    ]
    logits_comparison = _comparison(reference_logits, candidate_logits, atol, rtol)
    generated, generation_finite = _greedy_generation_smoke(model, input_ids, generation_tokens)
    passed = bool(
        all(item.get("allclose", False) and item.get("finite", False) for item in layer_comparisons)
        and logits_comparison.get("allclose", False)
        and logits_comparison.get("finite", False)
        and generation_finite
    )
    return {
        "scope": "all Llama decoder linears use packed GPTQ W4 and per-token A8; embedding/lm_head and K/V remain BF16; not a PPL, KV4, performance, or memory result",
        "source": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_manifest": str(manifest_path),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "project_revision": _project_revision(),
        },
        "model": dict(config["model"]),
        "rotation": rotation,
        "precision": {
            "decoder_linears": "packed signed W4 with FP32 group scales; per-token symmetric A8",
            "decoder_linear_count": len(names),
            "embedding_and_lm_head": "BF16",
            "key_value_cache": "BF16",
        },
        "input": {"shape": list(input_ids.shape), "sequence": "arange token ids modulo vocab"},
        "tolerance": {"atol": atol, "rtol": rtol},
        "layer_outputs": layer_comparisons,
        "logits": logits_comparison,
        "generation": {
            "new_tokens": generation_tokens,
            "output_shape": list(generated.shape),
            "token_ids": generated.detach().cpu().tolist(),
            "finite_logits": generation_finite,
        },
        "passed": passed,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()
    try:
        result = run(
            args.config,
            args.manifest,
            args.sequence_length,
            args.generation_tokens,
            args.atol,
            args.rtol,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print("FULL-MODEL W4A8 SMOKE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        print("FULL-MODEL W4A8 SMOKE FAILED: numerical gate did not pass", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
