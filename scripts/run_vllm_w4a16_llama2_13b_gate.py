#!/usr/bin/env python3
"""Run matched vLLM offline inference for complete Llama-2-13B models."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys


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
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("full-model vLLM gate requires an allocated CUDA device")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard gate requires the native vLLM sampler fallback")
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
        "runtime": {
            "vllm": metadata.version("vllm"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_model_in_subprocess(
    config_path: Path,
    name: str,
    model_path: Path,
    prompt: list[int],
    output: Path,
) -> dict:
    output.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--worker-name",
            name,
            "--worker-model",
            str(model_path),
            "--prompt-token-ids",
            json.dumps(prompt),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("name") != name:
        raise RuntimeError(f"worker returned the wrong model name for {name}")
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


def run_model_set(
    config: dict,
    config_path: Path,
    original: Path,
    model_paths: list[tuple[str, Path]],
    worker_output_dir: Path,
) -> dict:
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard gate requires the native vLLM sampler fallback")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        original,
        local_files_only=True,
        trust_remote_code=False,
    )
    prompt = tokenizer(
        config["inference"]["prompt"],
        add_special_tokens=True,
    )["input_ids"]
    models = {}
    if not model_paths or model_paths[0] != ("bf16", original):
        raise ValueError("matched W4A16 gate must start with the BF16 control")
    if len({name for name, _path in model_paths}) != len(model_paths):
        raise ValueError("matched W4A16 gate model names must be unique")
    for name, path in model_paths:
        models[name] = _run_model_in_subprocess(
            config_path,
            name,
            path,
            prompt,
            worker_output_dir / f"{name}.json",
        )
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in models.values()}
    if len(runtimes) != 1:
        raise RuntimeError("vLLM workers returned inconsistent runtime metadata")
    runtime = json.loads(runtimes.pop())
    comparisons = {
        f"{name}_vs_bf16": _compare(models[name], models["bf16"])
        for name, _path in model_paths
        if name != "bf16"
    }
    return {
        "status": "passed",
        "scope": "matched Llama-2-13B vLLM offline load and inference; not quality or performance evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        **runtime,
        "vllm_use_flashinfer_sampler": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "prompt": config["inference"]["prompt"],
        "prompt_token_ids": prompt,
        "models": models,
        "comparisons": comparisons,
    }


def run(
    config: dict,
    config_path: Path,
    original: Path,
    unrotated: Path,
    rotated: Path,
    worker_output_dir: Path,
) -> dict:
    """Preserve the accepted three-model QuaRot W4A16 gate interface."""

    return run_model_set(
        config,
        config_path,
        original,
        [
            ("bf16", original),
            ("unrotated_w4a16", unrotated),
            ("rotated_w4a16", rotated),
        ],
        worker_output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--unrotated", type=Path)
    parser.add_argument("--rotated", type=Path)
    parser.add_argument("--spinquant", type=Path)
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker_name is not None:
        if args.worker_model is None or args.prompt_token_ids is None:
            parser.error("--worker-name requires --worker-model and --prompt-token-ids")
        prompt = json.loads(args.prompt_token_ids)
        if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
            parser.error("--prompt-token-ids must encode a JSON list of integers")
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            prompt,
            config,
        )
        _write_json(args.output, result)
        print(f"VLLM_13B_MODEL_PASSED={args.worker_name}")
        return 0
    if args.original is None:
        parser.error("parent mode requires --original")
    worker_output_dir = args.output.parent / f"{args.output.stem}-models"
    if args.spinquant is not None:
        if args.unrotated is not None or args.rotated is not None:
            parser.error("--spinquant cannot be combined with --unrotated/--rotated")
        result = run_model_set(
            config,
            args.config.resolve(),
            args.original.resolve(),
            [
                ("bf16", args.original.resolve()),
                ("spinquant_w4a16", args.spinquant.resolve()),
            ],
            worker_output_dir,
        )
    else:
        if args.unrotated is None or args.rotated is None:
            parser.error(
                "QuaRot parent mode requires --unrotated and --rotated"
            )
        result = run(
            config,
            args.config.resolve(),
            args.original.resolve(),
            args.unrotated.resolve(),
            args.rotated.resolve(),
            worker_output_dir,
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
