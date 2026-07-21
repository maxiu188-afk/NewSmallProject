#!/usr/bin/env python3
"""Load an exported upstream QuaRot checkpoint and run fixed-token W4A4KV4 inference."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "QuaRot"))

import quarot  # noqa: E402
from e2e.quantized_llama import modeling_llama  # noqa: E402


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(checkpoint: Path):
    config = modeling_llama.QuarotLlamaConfig.from_pretrained(checkpoint)
    config._attn_implementation = "flash_attention_2"
    return modeling_llama.QuarotLlamaForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )


@torch.inference_mode()
def _generate(model, input_ids: torch.Tensor, decode_tokens: int):
    model._expected_max_length = input_ids.shape[1] + decode_tokens
    output = model(input_ids, use_cache=True)
    prefill_finite = bool(torch.isfinite(output.logits).all().item())
    cache = output.past_key_values
    generated = []
    logits_finite = prefill_finite
    next_id = output.logits[:, -1:].argmax(dim=-1)
    for _ in range(decode_tokens):
        generated.append(int(next_id.item()))
        output = model(next_id, past_key_values=cache, use_cache=True)
        logits_finite = logits_finite and bool(torch.isfinite(output.logits).all().item())
        next_id = output.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    return {
        "generated_token_ids": generated,
        "logits_finite": logits_finite,
        "cache_length": int(cache.length),
    }


def run(checkpoint: Path, prefill_tokens: int, decode_tokens: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("upstream QuaRot checkpoint smoke requires NVIDIA CUDA")
    checkpoint = checkpoint.resolve()
    device = torch.device("cuda:0")
    model = _load(checkpoint)
    linears = [module for module in model.modules() if isinstance(module, quarot.nn.Linear4bit)]
    packed_checks = {
        "linear_count_is_280": len(linears) == 280,
        "all_weights_are_uint8": all(module.weight.dtype == torch.uint8 for module in linears),
        "all_scales_are_finite": all(bool(torch.isfinite(module.weight_scales).all().item()) for module in linears),
        "all_scales_are_positive": all(bool((module.weight_scales > 0).all().item()) for module in linears),
    }

    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).eval()
    input_ids = torch.arange(1, prefill_tokens + 1, device=device, dtype=torch.long).view(1, -1)
    first = _generate(model, input_ids, decode_tokens)
    second = _generate(model, input_ids, decode_tokens)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    checks = {
        **packed_checks,
        "first_run_finite": first["logits_finite"],
        "second_run_finite": second["logits_finite"],
        "cache_length_matches": first["cache_length"] == prefill_tokens + decode_tokens
        and second["cache_length"] == prefill_tokens + decode_tokens,
        "greedy_tokens_repeat_exactly": first["generated_token_ids"] == second["generated_token_ids"],
    }
    index_path = checkpoint / "model.safetensors.index.json"
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "full exported Llama-2-13B upstream QuaRot W4A4 linear plus KV4 fixed-token load/generation smoke; not PPL or performance evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "revisions": {
            "project": _revision(PROJECT_ROOT),
            "upstream_quarot": _revision(PROJECT_ROOT / "QuaRot"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "index_sha256": _sha256(index_path),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "input": {"batch_size": 1, "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens},
        "packed_linear_count": len(linears),
        "first_run": first,
        "second_run": second,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=2)
    args = parser.parse_args()
    result = run(args.checkpoint, args.prefill_tokens, args.decode_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
