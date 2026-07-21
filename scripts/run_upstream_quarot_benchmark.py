#!/usr/bin/env python3
"""Benchmark exported upstream QuaRot W4A4KV4 against its FP16 backend."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import statistics
import subprocess
import sys
import time
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


def _summary(samples_ms: list[float]) -> dict:
    return {
        "samples_ms": samples_ms,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _load_fp16(model_id: str):
    return modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )


def _load_w4a4kv4(checkpoint: Path):
    config = modeling_llama.QuarotLlamaConfig.from_pretrained(checkpoint)
    config._attn_implementation = "flash_attention_2"
    return modeling_llama.QuarotLlamaForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )


@torch.inference_mode()
def _prefill(model, input_ids: torch.Tensor, total_length: int):
    model._expected_max_length = total_length
    return model(input_ids, use_cache=True)


@torch.inference_mode()
def _run_e2e(model, input_ids: torch.Tensor, decode_tokens: int) -> bool:
    output = _prefill(model, input_ids, input_ids.shape[1] + decode_tokens)
    finite = bool(torch.isfinite(output.logits).all().item())
    cache = output.past_key_values
    next_id = output.logits[:, -1:].argmax(dim=-1)
    for _ in range(decode_tokens):
        output = model(next_id, past_key_values=cache, use_cache=True)
        finite = finite and bool(torch.isfinite(output.logits).all().item())
        next_id = output.logits[:, -1:].argmax(dim=-1)
    return finite


def _timed(callable_) -> tuple[float, object]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0, output


@torch.inference_mode()
def _benchmark_loaded_model(
    model,
    mode: str,
    device: torch.device,
    prefill_tokens: int,
    decode_tokens: int,
    warmups: int,
    repeats: int,
) -> dict:
    model = model.to(device).eval()
    input_ids = torch.arange(1, prefill_tokens + 1, device=device, dtype=torch.long).view(1, -1)
    resident_allocated = torch.cuda.memory_allocated(device)
    resident_reserved = torch.cuda.memory_reserved(device)

    for _ in range(warmups):
        output = _prefill(model, input_ids, prefill_tokens + decode_tokens)
        next_id = output.logits[:, -1:].argmax(dim=-1)
        for _ in range(decode_tokens):
            output = model(next_id, past_key_values=output.past_key_values, use_cache=True)
            next_id = output.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    prefill_samples = []
    prefill_finite = True
    for _ in range(repeats):
        elapsed, output = _timed(lambda: _prefill(model, input_ids, prefill_tokens + decode_tokens))
        prefill_samples.append(elapsed)
        prefill_finite = prefill_finite and bool(torch.isfinite(output.logits).all().item())
        del output
    prefill_peak_allocated = torch.cuda.max_memory_allocated(device)
    prefill_peak_reserved = torch.cuda.max_memory_reserved(device)

    decode_samples = []
    decode_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        output = _prefill(model, input_ids, prefill_tokens + decode_tokens)
        cache = output.past_key_values
        next_id = output.logits[:, -1:].argmax(dim=-1)

        def decode_loop():
            nonlocal output, next_id
            for _ in range(decode_tokens):
                output = model(next_id, past_key_values=cache, use_cache=True)
                next_id = output.logits[:, -1:].argmax(dim=-1)
            return output

        elapsed, output = _timed(decode_loop)
        decode_samples.append(elapsed / decode_tokens)
        decode_finite = decode_finite and bool(torch.isfinite(output.logits).all().item())
        del output, cache
    decode_peak_allocated = torch.cuda.max_memory_allocated(device)
    decode_peak_reserved = torch.cuda.max_memory_reserved(device)

    e2e_samples = []
    e2e_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        elapsed, finite = _timed(lambda: _run_e2e(model, input_ids, decode_tokens))
        e2e_samples.append(elapsed)
        e2e_finite = e2e_finite and bool(finite)
    e2e_peak_allocated = torch.cuda.max_memory_allocated(device)
    e2e_peak_reserved = torch.cuda.max_memory_reserved(device)

    packed_linears = sum(isinstance(module, quarot.nn.Linear4bit) for module in model.modules())
    checks = {
        "prefill_logits_finite": prefill_finite,
        "decode_logits_finite": decode_finite,
        "e2e_logits_finite": e2e_finite,
        "packed_linear_count_matches_mode": packed_linears == (280 if mode == "w4a4kv4" else 0),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "packed_linear_count": packed_linears,
        "memory_bytes": {
            "model_resident_allocated": resident_allocated,
            "model_resident_reserved": resident_reserved,
            "prefill_peak_allocated": prefill_peak_allocated,
            "prefill_peak_reserved": prefill_peak_reserved,
            "decode_peak_allocated": decode_peak_allocated,
            "decode_peak_reserved": decode_peak_reserved,
            "e2e_peak_allocated": e2e_peak_allocated,
            "e2e_peak_reserved": e2e_peak_reserved,
        },
        "prefill": _summary(prefill_samples),
        "decode_per_token": _summary(decode_samples),
        "e2e": _summary(e2e_samples),
    }


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("upstream QuaRot benchmark requires NVIDIA CUDA")
    if args.prefill_tokens <= 0 or args.decode_tokens <= 0 or args.warmups < 1 or args.repeats < 2:
        raise ValueError("require positive token counts, at least one warmup, and at least two repeats")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    checkpoint = args.checkpoint.resolve()
    modes = {}
    for mode in ("fp16", "w4a4kv4"):
        _cleanup()
        model = _load_fp16(args.model_id) if mode == "fp16" else _load_w4a4kv4(checkpoint)
        modes[mode] = _benchmark_loaded_model(
            model,
            mode,
            device,
            args.prefill_tokens,
            args.decode_tokens,
            args.warmups,
            args.repeats,
        )
        del model
        _cleanup()

    fp16 = modes["fp16"]
    quant = modes["w4a4kv4"]
    comparison = {
        "prefill_speedup_fp16_over_w4a4kv4": fp16["prefill"]["median_ms"] / quant["prefill"]["median_ms"],
        "decode_speedup_fp16_over_w4a4kv4": fp16["decode_per_token"]["median_ms"] / quant["decode_per_token"]["median_ms"],
        "e2e_speedup_fp16_over_w4a4kv4": fp16["e2e"]["median_ms"] / quant["e2e"]["median_ms"],
        "resident_memory_ratio_fp16_over_w4a4kv4": fp16["memory_bytes"]["model_resident_allocated"]
        / quant["memory_bytes"]["model_resident_allocated"],
    }
    return {
        "status": "passed" if all(result["status"] == "passed" for result in modes.values()) else "failed",
        "scope": "matched official QuaRot FP16 backend versus exported official QuaRot W4A4 linear plus KV4 fixed-token inference",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "revisions": {
            "project": _revision(PROJECT_ROOT),
            "upstream_quarot": _revision(PROJECT_ROOT / "QuaRot"),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "input": {
            "model_id": args.model_id,
            "checkpoint": str(checkpoint),
            "batch_size": 1,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "modes": modes,
        "comparison": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
