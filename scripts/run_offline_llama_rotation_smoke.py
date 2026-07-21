#!/usr/bin/env python3
"""Validate a standard-layout offline QuaRot rotation on a tiny Llama model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from repro.offline_llama_rotation import apply_offline_llama_rotation, assert_standard_llama_layout


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _cached_last_logits(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
    return model(
        input_ids=input_ids[:, -1:],
        past_key_values=prefix.past_key_values,
        use_cache=True,
    ).logits


def run_smoke() -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    model = LlamaForCausalLM(config).float().eval()
    input_ids = torch.arange(16, dtype=torch.long).view(1, -1) % config.vocab_size
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits
        baseline_cached = _cached_last_logits(model, input_ids)
    rotation = apply_offline_llama_rotation(model)
    assert_standard_llama_layout(model)
    with torch.inference_mode():
        rotated = model(input_ids=input_ids, use_cache=False).logits
        rotated_cached = _cached_last_logits(model, input_ids)

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, safe_serialization=True)
        reloaded = LlamaForCausalLM.from_pretrained(directory, local_files_only=True).float().eval()
        assert_standard_llama_layout(reloaded)
        with torch.inference_mode():
            reloaded_logits = reloaded(input_ids=input_ids, use_cache=False).logits
            reloaded_cached = _cached_last_logits(reloaded, input_ids)

    direct_error = float((baseline - rotated).abs().max().item())
    reload_error = float((baseline - reloaded_logits).abs().max().item())
    cached_error = float((baseline_cached - rotated_cached).abs().max().item())
    cached_reload_error = float((baseline_cached - reloaded_cached).abs().max().item())
    tolerance = 2e-5
    errors = (direct_error, reload_error, cached_error, cached_reload_error)
    return {
        "status": "passed" if max(errors) <= tolerance else "failed",
        "scope": "tiny standard-layout offline rotation equivalence; not quantized or vLLM execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "rotation": rotation,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(baseline.shape),
        "direct_max_absolute_error": direct_error,
        "reload_max_absolute_error": reload_error,
        "cached_decode_max_absolute_error": cached_error,
        "cached_reload_max_absolute_error": cached_reload_error,
        "tolerance": tolerance,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
