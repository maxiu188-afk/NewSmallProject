#!/usr/bin/env python3
"""Evaluate matched deployed Llama-2-13B checkpoints on zero-shot BoolQ."""

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
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.vllm_w4afp8 import (  # noqa: E402
    EXPECTED_VARIANTS,
    validate_checkpoint_quantization_config,
)
from scripts.run_vllm_w4a16_llama2_13b_ppl import (  # noqa: E402
    _checkpoint_metadata,
    _revision,
    _sha256,
    _write_json,
)


def _load_examples(
    config: dict[str, Any],
    manifest_path: Path,
    max_examples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError("BoolQ manifest did not pass")
    if manifest.get("config_sha256") != _sha256(Path(config["_config_path"])):
        raise RuntimeError("BoolQ manifest used a different config")
    spec = config["dataset"]
    dataset = manifest.get("dataset", {})
    if dataset.get("revision") != spec["revision"]:
        raise RuntimeError("BoolQ dataset revision changed")
    if dataset.get("fingerprint") != spec["expected_fingerprint"]:
        raise RuntimeError("BoolQ dataset fingerprint changed")
    examples_path = Path(manifest["examples"]["path"])
    examples_sha256 = _sha256(examples_path)
    if examples_sha256 != manifest["examples"]["sha256"]:
        raise RuntimeError("BoolQ examples no longer match their manifest")
    if examples_sha256 != spec["expected_examples_sha256"]:
        raise RuntimeError("BoolQ examples SHA-256 changed")

    examples = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(examples) != int(spec["expected_rows"]):
        raise RuntimeError("BoolQ example count changed")
    indices = [item.get("idx") for item in examples]
    if len(set(indices)) != len(indices):
        raise RuntimeError("BoolQ example indices are not unique")
    if any(
        not isinstance(item.get("passage"), str)
        or not isinstance(item.get("question"), str)
        or item.get("label") not in (0, 1)
        for item in examples
    ):
        raise RuntimeError("BoolQ example schema changed")
    if max_examples is not None:
        if max_examples < 1 or max_examples > len(examples):
            raise ValueError("max_examples is outside the materialized range")
        examples = examples[:max_examples]
    return examples, manifest


def _context(example: dict[str, Any], protocol: dict[str, Any]) -> str:
    return protocol["prompt_template"].format(
        passage=example["passage"],
        question=example["question"],
    )


def _encode_pair(
    tokenizer: Any,
    context: str,
    continuation: str,
    max_model_len: int,
) -> tuple[list[int], int]:
    # Match lm-eval HFLM._encode_pair at the pinned QuaRot harness commit.
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    continuation_start = len(context_tokens)
    continuation_tokens = whole[continuation_start:]
    if not context_tokens or not continuation_tokens:
        raise RuntimeError("BoolQ context or continuation tokenization is empty")
    if len(whole) > max_model_len:
        raise RuntimeError(
            f"BoolQ request length {len(whole)} exceeds max_model_len {max_model_len}"
        )
    return whole, continuation_start


def _choice_loglikelihood(
    output: Any,
    expected_tokens: list[int],
    continuation_start: int,
) -> float:
    if list(output.prompt_token_ids) != expected_tokens:
        raise RuntimeError("vLLM returned BoolQ prompt tokens in a different order")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_tokens):
        raise RuntimeError("vLLM returned incomplete BoolQ prompt logprobs")
    if continuation_start <= 0 or continuation_start >= len(expected_tokens):
        raise RuntimeError("invalid BoolQ continuation boundary")

    total = 0.0
    for position in range(continuation_start, len(expected_tokens)):
        token_id = expected_tokens[position]
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"missing BoolQ target token logprob at position {position}: {token_id}"
            )
        logprob = float(candidates[token_id].logprob)
        if not math.isfinite(logprob):
            raise RuntimeError(f"non-finite BoolQ logprob at position {position}")
        total += logprob
    return total


def _one_model(
    name: str,
    model_path: Path,
    tokenizer_path: Path,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("deployed BoolQ evaluation requires an allocated CUDA device")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard BoolQ requires the native vLLM sampler fallback")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    protocol = config["protocol"]
    choices = protocol["choices"]
    delimiter = protocol["target_delimiter"]
    engine_config = config["engine"]
    requests: list[dict[str, Any]] = []
    prompt_inputs = []
    for example_position, example in enumerate(examples):
        context = _context(example, protocol)
        for choice_position, choice in enumerate(choices):
            tokens, continuation_start = _encode_pair(
                tokenizer,
                context,
                f"{delimiter}{choice}",
                int(engine_config["max_model_len"]),
            )
            requests.append(
                {
                    "example_position": example_position,
                    "choice_position": choice_position,
                    "tokens": tokens,
                    "continuation_start": continuation_start,
                }
            )
            prompt_inputs.append({"prompt_token_ids": tokens})

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
    outputs = engine.generate(prompt_inputs, params, use_tqdm=False)
    if len(outputs) != len(requests):
        raise RuntimeError("vLLM returned the wrong number of BoolQ outputs")

    scores = [[None for _ in choices] for _ in examples]
    choice_token_counts = [[None for _ in choices] for _ in examples]
    max_prompt_tokens = 0
    for request, output in zip(requests, outputs):
        loglikelihood = _choice_loglikelihood(
            output,
            request["tokens"],
            request["continuation_start"],
        )
        example_position = request["example_position"]
        choice_position = request["choice_position"]
        scores[example_position][choice_position] = loglikelihood
        choice_token_counts[example_position][choice_position] = (
            len(request["tokens"]) - request["continuation_start"]
        )
        max_prompt_tokens = max(max_prompt_tokens, len(request["tokens"]))

    example_metrics = []
    correct = 0
    for example, example_scores, token_counts in zip(
        examples, scores, choice_token_counts
    ):
        if any(value is None for value in example_scores):
            raise RuntimeError(f"incomplete BoolQ scores for idx {example['idx']}")
        prediction = max(range(len(example_scores)), key=example_scores.__getitem__)
        is_correct = prediction == example["label"]
        correct += int(is_correct)
        example_metrics.append(
            {
                "idx": example["idx"],
                "label": example["label"],
                "prediction": prediction,
                "correct": is_correct,
                "choice_loglikelihoods": example_scores,
                "choice_token_counts": token_counts,
            }
        )

    checkpoint = _checkpoint_metadata(model_path)
    if name == "bf16" and checkpoint["packed_decoder_linear_count"] != 0:
        raise RuntimeError("BF16 checkpoint unexpectedly contains packed W4 linears")
    if name != "bf16" and checkpoint["packed_decoder_linear_count"] != 280:
        raise RuntimeError(f"{name} does not contain 280 packed W4 linears")
    return {
        "name": name,
        "model_path": str(model_path),
        **checkpoint,
        "evaluated_examples": len(examples),
        "evaluated_requests": len(requests),
        "correct": correct,
        "accuracy": correct / len(examples),
        "max_prompt_tokens": max_prompt_tokens,
        "example_metrics": example_metrics,
        "runtime": {
            "vllm": metadata.version("vllm"),
            "torch": torch.__version__,
            "transformers": metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "tokenizer_class": tokenizer.__class__.__name__,
        },
    }


def _run_worker(
    *,
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    name: str,
    model_path: Path,
    max_examples: int | None,
    output_path: Path,
) -> dict[str, Any]:
    output_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--dataset-manifest",
        str(manifest_path),
        "--tokenizer-model",
        str(tokenizer_path),
        "--worker-name",
        name,
        "--worker-model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    if max_examples is not None:
        command.extend(["--max-examples", str(max_examples)])
    subprocess.run(command, check=True)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if result.get("name") != name:
        raise RuntimeError(f"BoolQ worker returned the wrong model name for {name}")
    return result


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
    if tuple(models) != EXPECTED_VARIANTS:
        raise ValueError(f"model order must be {EXPECTED_VARIANTS}")
    return models


def _accuracy_comparison(candidate: dict[str, Any], reference: dict[str, Any]) -> dict:
    if candidate["evaluated_examples"] != reference["evaluated_examples"]:
        raise RuntimeError("BoolQ comparison uses different example counts")
    return {
        "accuracy_delta": float(candidate["accuracy"]) - float(reference["accuracy"]),
        "accuracy_point_delta": 100.0
        * (float(candidate["accuracy"]) - float(reference["accuracy"])),
        "correct_delta": int(candidate["correct"]) - int(reference["correct"]),
    }


def run(
    *,
    config: dict[str, Any],
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    models: dict[str, Path],
    max_examples: int | None,
    worker_output_dir: Path,
) -> dict[str, Any]:
    if config["source_gate"].get("status") != "accepted":
        raise RuntimeError("W4AFP8 source gate is still pending")
    if tuple(models) != EXPECTED_VARIANTS:
        raise ValueError(f"model order must be {EXPECTED_VARIANTS}")
    if tuple(config["source_gate"]["expected_models"]) != EXPECTED_VARIANTS:
        raise ValueError("source-gate model order drifted")
    examples, manifest = _load_examples(config, manifest_path, max_examples)
    results = {}
    for name, model_path in models.items():
        results[name] = _run_worker(
            config_path=config_path,
            manifest_path=manifest_path,
            tokenizer_path=tokenizer_path,
            name=name,
            model_path=model_path,
            max_examples=max_examples,
            output_path=worker_output_dir / f"{name}.json",
        )
    runtimes = {json.dumps(model["runtime"], sort_keys=True) for model in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("BoolQ workers returned inconsistent runtime metadata")
    first_indices = [item["idx"] for item in results["bf16"]["example_metrics"]]
    for name, result in results.items():
        if [item["idx"] for item in result["example_metrics"]] != first_indices:
            raise RuntimeError(f"{name} evaluated a different BoolQ example order")

    comparisons = {
        f"{name}_vs_bf16": _accuracy_comparison(results[name], results["bf16"])
        for name in EXPECTED_VARIANTS[1:]
    }
    comparisons.update(
        {
            "quarot_vs_unrotated_w4afp8": _accuracy_comparison(
                results["quarot_w4afp8"], results["unrotated_w4afp8"]
            ),
            "spinquant_vs_unrotated_w4afp8": _accuracy_comparison(
                results["spinquant_w4afp8"], results["unrotated_w4afp8"]
            ),
            "spinquant_vs_quarot_w4afp8": _accuracy_comparison(
                results["spinquant_w4afp8"], results["quarot_w4afp8"]
            ),
        }
    )
    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "examples_sha256": manifest["examples"]["sha256"],
        "evaluated_examples": len(examples),
        "evaluated_requests": len(examples) * len(config["protocol"]["choices"]),
        "protocol": config["protocol"],
        "engine": config["engine"],
        "runtime": json.loads(runtimes.pop()),
        "models": results,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--worker-name")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    manifest_path = args.dataset_manifest.resolve()
    tokenizer_path = args.tokenizer_model.resolve()

    if args.worker_name is not None:
        if args.worker_name not in EXPECTED_VARIANTS or args.worker_model is None:
            parser.error("worker mode requires a known --worker-name and --worker-model")
        examples, _ = _load_examples(config, manifest_path, args.max_examples)
        result = _one_model(
            args.worker_name,
            args.worker_model.resolve(),
            tokenizer_path,
            examples,
            config,
        )
        if args.worker_name != "bf16":
            validate_checkpoint_quantization_config(result["quantization_config"])
        _write_json(args.output.resolve(), result)
        print(f"VLLM_W4AFP8_BOOLQ_MODEL_PASSED={args.worker_name}")
        return 0

    models = _parse_models(args.model)
    result = run(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        tokenizer_path=tokenizer_path,
        models=models,
        max_examples=args.max_examples,
        worker_output_dir=args.output.resolve().parent / f"{args.output.stem}-models",
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_BOOLQ_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
