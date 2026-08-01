#!/usr/bin/env python3
"""Measure matched WikiText-2 PPL through three deployed vLLM checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = ("bf16", "unrotated_w4a16", "rotated_w4a16")


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_sequences(
    config: dict,
    manifest_path: Path,
    max_sequences: int | None,
) -> tuple[list[list[int]], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise RuntimeError("PPL token manifest did not pass")
    if manifest["config_sha256"] != _sha256(Path(config["_config_path"])):
        raise RuntimeError("PPL token manifest used a different config")
    tokens_path = Path(manifest["tokens"]["path"])
    if _sha256(tokens_path) != manifest["tokens"]["sha256"]:
        raise RuntimeError("PPL token file SHA-256 changed")
    sequences = json.loads(tokens_path.read_text(encoding="utf-8"))
    spec = config["evaluation"]
    if len(sequences) != int(spec["samples"]):
        raise RuntimeError("PPL token sequence count changed")
    if any(
        not isinstance(sequence, list)
        or len(sequence) != int(spec["sequence_length"])
        or not all(isinstance(token, int) for token in sequence)
        for sequence in sequences
    ):
        raise RuntimeError("PPL token sequence shape or type changed")
    if max_sequences is not None:
        if max_sequences <= 0 or max_sequences > len(sequences):
            raise ValueError("max_sequences is outside the materialized range")
        sequences = sequences[:max_sequences]
    return sequences, manifest


def _sequence_nll(output: Any, expected_tokens: list[int]) -> dict[str, float | int]:
    if list(output.prompt_token_ids) != expected_tokens:
        raise RuntimeError("vLLM returned prompt tokens in a different order")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_tokens):
        raise RuntimeError("vLLM returned incomplete prompt logprobs")
    if prompt_logprobs[0] is not None:
        raise RuntimeError("the first prompt token unexpectedly has a logprob")

    total_nll = 0.0
    for position, token_id in enumerate(expected_tokens[1:], start=1):
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"missing target token logprob at position {position}: {token_id}"
            )
        logprob = float(candidates[token_id].logprob)
        if not math.isfinite(logprob):
            raise RuntimeError(f"non-finite logprob at position {position}")
        total_nll -= logprob
    tokens = len(expected_tokens) - 1
    return {
        "nll": total_nll,
        "tokens": tokens,
        "mean_nll": total_nll / tokens,
    }


def _checkpoint_metadata(model_path: Path) -> dict:
    from safetensors import safe_open

    packed = 0
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            packed += sum(key.endswith(".weight_packed") for key in handle.keys())
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    return {
        "packed_decoder_linear_count": packed,
        "quantization_config": model_config.get("quantization_config"),
    }


def _one_model(
    name: str,
    model_path: Path,
    sequences: list[list[int]],
    config: dict,
) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("deployed PPL requires an allocated CUDA device")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard PPL requires the native vLLM sampler fallback")
    engine_config = config["engine"]
    engine = LLM(
        model=str(model_path),
        dtype=engine_config["dtype"],
        max_model_len=int(engine_config["max_model_len"]),
        max_num_seqs=int(engine_config["max_num_seqs"]),
        gpu_memory_utilization=float(engine_config["gpu_memory_utilization"]),
        enforce_eager=bool(engine_config["enforce_eager"]),
        enable_prefix_caching=bool(engine_config["enable_prefix_caching"]),
        skip_tokenizer_init=True,
        disable_log_stats=bool(engine_config["disable_log_stats"]),
        max_logprobs=int(engine_config["prompt_logprobs"]),
        seed=int(engine_config["seed"]),
    )
    params = SamplingParams(
        temperature=float(engine_config["temperature"]),
        max_tokens=int(engine_config["max_tokens"]),
        prompt_logprobs=int(engine_config["prompt_logprobs"]),
        ignore_eos=True,
        detokenize=False,
    )
    prompts = [{"prompt_token_ids": sequence} for sequence in sequences]
    outputs = engine.generate(prompts, params, use_tqdm=False)
    if len(outputs) != len(sequences):
        raise RuntimeError("vLLM returned the wrong number of PPL outputs")

    sequence_metrics = [
        {"index": index, **_sequence_nll(output, sequence)}
        for index, (output, sequence) in enumerate(zip(outputs, sequences))
    ]
    total_nll = sum(float(item["nll"]) for item in sequence_metrics)
    total_tokens = sum(int(item["tokens"]) for item in sequence_metrics)
    if total_tokens <= 0 or not math.isfinite(total_nll):
        raise RuntimeError("deployed PPL aggregation is empty or non-finite")
    mean_nll = total_nll / total_tokens
    checkpoint = _checkpoint_metadata(model_path)
    if name == "bf16" and checkpoint["packed_decoder_linear_count"] != 0:
        raise RuntimeError("BF16 checkpoint unexpectedly contains packed W4 linears")
    if name != "bf16" and checkpoint["packed_decoder_linear_count"] != 280:
        raise RuntimeError(f"{name} does not contain 280 packed W4 linears")
    return {
        "name": name,
        "model_path": str(model_path),
        **checkpoint,
        "evaluated_sequences": len(sequences),
        "tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "sequence_metrics": sequence_metrics,
        "runtime": {
            "vllm": metadata.version("vllm"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }


def _run_worker(
    *,
    config_path: Path,
    manifest_path: Path,
    name: str,
    model_path: Path,
    max_sequences: int | None,
    output_path: Path,
) -> dict:
    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--tokens-manifest",
        str(manifest_path),
        "--worker-name",
        name,
        "--worker-model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    if max_sequences is not None:
        command.extend(["--max-sequences", str(max_sequences)])
    subprocess.run(command, check=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result.get("name") != name:
        raise RuntimeError(f"PPL worker returned the wrong model name for {name}")
    return result


def _comparison(candidate: dict, reference: dict) -> dict[str, float]:
    if candidate["tokens"] != reference["tokens"]:
        raise RuntimeError("PPL comparison uses different token counts")
    return {
        "mean_nll_delta": float(candidate["mean_nll"]) - float(reference["mean_nll"]),
        "perplexity_delta": float(candidate["perplexity"]) - float(reference["perplexity"]),
        "perplexity_ratio": float(candidate["perplexity"]) / float(reference["perplexity"]),
    }


def run(
    *,
    config: dict,
    config_path: Path,
    manifest_path: Path,
    models: dict[str, Path],
    max_sequences: int | None,
    worker_output_dir: Path,
) -> dict:
    if tuple(models) != EXPECTED_MODELS:
        raise ValueError(f"model order must be {EXPECTED_MODELS}")
    sequences, manifest = _load_sequences(config, manifest_path, max_sequences)
    results = {}
    for name, model_path in models.items():
        results[name] = _run_worker(
            config_path=config_path,
            manifest_path=manifest_path,
            name=name,
            model_path=model_path,
            max_sequences=max_sequences,
            output_path=worker_output_dir / f"{name}.json",
        )
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("vLLM PPL workers returned inconsistent runtime metadata")
    runtime = json.loads(runtimes.pop())
    return {
        "status": "passed",
        "scope": "matched WikiText-2 PPL through deployed BF16 and packed GPTQ W4A16 vLLM checkpoints",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "tokens_manifest": str(manifest_path),
        "tokens_manifest_sha256": _sha256(manifest_path),
        "token_ids_sha256": manifest["dataset"]["token_ids_sha256"],
        "evaluated_sequences": len(sequences),
        "scored_tokens": len(sequences) * (len(sequences[0]) - 1),
        "engine": config["engine"],
        **runtime,
        "models": results,
        "comparisons": {
            "unrotated_w4a16_vs_bf16": _comparison(
                results["unrotated_w4a16"], results["bf16"]
            ),
            "rotated_w4a16_vs_bf16": _comparison(
                results["rotated_w4a16"], results["bf16"]
            ),
            "rotated_vs_unrotated_w4a16": _comparison(
                results["rotated_w4a16"], results["unrotated_w4a16"]
            ),
        },
    }


def _parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model requires name=path: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            raise ValueError(f"repeated model: {name}")
        path = Path(raw_path).resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")
        models[name] = path
    if tuple(models) != EXPECTED_MODELS:
        raise ValueError(f"model order must be {EXPECTED_MODELS}")
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    manifest_path = args.tokens_manifest.resolve()

    if args.worker_name is not None:
        if args.worker_name not in EXPECTED_MODELS or args.worker_model is None:
            parser.error("worker mode requires a known --worker-name and --worker-model")
        sequences, _ = _load_sequences(config, manifest_path, args.max_sequences)
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            sequences,
            config,
        )
        _write_json(args.output.resolve(), result)
        print(f"VLLM_13B_PPL_MODEL_PASSED={args.worker_name}")
        return 0

    models = _parse_models(args.model)
    result = run(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        models=models,
        max_sequences=args.max_sequences,
        worker_output_dir=args.output.resolve().parent
        / f"{args.output.stem}-models",
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
