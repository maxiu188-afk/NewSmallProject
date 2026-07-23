#!/usr/bin/env python3
"""Run matched vLLM offline inference for three complete Llama-2-13B models."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _one_model(name: str, model_path: Path, prompt: list[int], config: dict) -> dict:
    inference = config["inference"]
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=int(inference["max_model_len"]),
        gpu_memory_utilization=float(inference["gpu_memory_utilization"]),
        enforce_eager=True,
        skip_tokenizer_init=True,
        disable_log_stats=True,
        max_logprobs=int(inference["max_logprobs"]),
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=int(inference["max_new_tokens"]),
        logprobs=int(inference["max_logprobs"]),
    )
    outputs = engine.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    first = completion.logprobs[0]
    logprobs = {str(token): float(value.logprob) for token, value in first.items()}
    if not logprobs or not all(math.isfinite(value) for value in logprobs.values()):
        raise RuntimeError(f"{name} returned missing or non-finite logprobs")
    result = {
        "name": name,
        "model_path": str(model_path),
        "generated_token_ids": list(completion.token_ids),
        "first_token_logprobs": logprobs,
    }
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _compare(left: dict, right: dict) -> dict:
    common = sorted(set(left["first_token_logprobs"]) & set(right["first_token_logprobs"]))
    if not common:
        raise RuntimeError("model pair returned no common first-token logprobs")
    return {
        "generated_tokens_equal": left["generated_token_ids"] == right["generated_token_ids"],
        "common_logprob_tokens": len(common),
        "max_absolute_logprob_error": max(
            abs(left["first_token_logprobs"][token] - right["first_token_logprobs"][token])
            for token in common
        ),
    }


def run(
    config: dict,
    original: Path,
    unrotated: Path,
    rotated: Path,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("full-model vLLM gate requires an allocated CUDA device")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard gate requires the native vLLM sampler fallback")
    tokenizer = AutoTokenizer.from_pretrained(
        original,
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = tokenizer(
        config["inference"]["prompt"],
        add_special_tokens=True,
    )["input_ids"]
    models = {
        name: _one_model(name, path, prompt, config)
        for name, path in (
            ("bf16", original),
            ("unrotated_w4a16", unrotated),
            ("rotated_w4a16", rotated),
        )
    }
    return {
        "status": "passed",
        "scope": "matched Llama-2-13B vLLM offline load and inference; not quality or performance evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "vllm": metadata.version("vllm"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "vllm_use_flashinfer_sampler": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "prompt": config["inference"]["prompt"],
        "prompt_token_ids": prompt,
        "models": models,
        "comparisons": {
            "unrotated_w4a16_vs_bf16": _compare(models["unrotated_w4a16"], models["bf16"]),
            "rotated_w4a16_vs_bf16": _compare(models["rotated_w4a16"], models["bf16"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--unrotated", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(
        config,
        args.original.resolve(),
        args.unrotated.resolve(),
        args.rotated.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
