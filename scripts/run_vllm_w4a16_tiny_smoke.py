#!/usr/bin/env python3
"""Load three tiny Llama checkpoints through one frozen vLLM runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import subprocess

import torch
from vllm import LLM, SamplingParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _one_model(name: str, model_path: Path, prompt: list[int], max_tokens: int) -> dict:
    engine = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=128,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        skip_tokenizer_init=True,
        disable_log_stats=True,
        max_logprobs=128,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=128)
    outputs = engine.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
    completion = outputs[0].outputs[0]
    first = completion.logprobs[0]
    logprobs = {str(token): float(value.logprob) for token, value in first.items()}
    if not logprobs or not all(math.isfinite(value) for value in logprobs.values()):
        raise AssertionError(f"{name} returned missing or non-finite logprobs")
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


def _pair(left: dict, right: dict) -> dict:
    common = sorted(set(left["first_token_logprobs"]) & set(right["first_token_logprobs"]))
    error = max(
        abs(left["first_token_logprobs"][token] - right["first_token_logprobs"][token])
        for token in common
    )
    return {
        "generated_tokens_equal": left["generated_token_ids"] == right["generated_token_ids"],
        "common_logprob_tokens": len(common),
        "max_absolute_logprob_error": error,
    }


def run(checkpoint_root: Path, config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires an allocated CUDA device")
    tiny = config["tiny_smoke"]
    prompt = tiny["prompt_token_ids"]
    max_tokens = tiny["max_new_tokens"]
    models = {
        name: _one_model(name, checkpoint_root / directory, prompt, max_tokens)
        for name, directory in (
            ("bf16", "bf16"),
            ("unrotated_w4a16", "unrotated-w4a16"),
            ("rotated_w4a16", "rotated-w4a16"),
        )
    }
    return {
        "status": "passed",
        "scope": "tiny vLLM load and deterministic offline inference; not quality or performance evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "vllm": metadata.version("vllm"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "prompt_token_ids": prompt,
        "models": models,
        "comparisons": {
            "unrotated_w4a16_vs_bf16": _pair(models["unrotated_w4a16"], models["bf16"]),
            "rotated_w4a16_vs_bf16": _pair(models["rotated_w4a16"], models["bf16"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/deployment/vllm_w4a16_isambard.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(args.checkpoint_root.resolve(), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
