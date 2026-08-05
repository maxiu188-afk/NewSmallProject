#!/usr/bin/env python3
"""Record matched BoolQ scoring for one QuaRot W4A16 checkpoint across backends."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sglang_vllm_llama2_13b_smoke import (  # noqa: E402
    _Server,
    _absolute_executable,
    _log_excerpt,
    _runtime_probe,
    _sha256,
    _write_json,
)
from scripts.run_vllm_w4afp8_llama2_13b_boolq import (  # noqa: E402
    _context,
    _encode_pair,
    _load_examples,
    _one_model,
)


MODEL_NAME = "quarot_w4a16"


def _validate_config(config: dict[str, Any]) -> None:
    if config["source_gate"].get("status") != "accepted":
        raise ValueError("QuaRot W4A16 source gate is not accepted")
    if config["source_gate"].get("expected_model") != "rotated_w4a16":
        raise ValueError("source gate no longer selects rotated W4A16")
    if config.get("model") != MODEL_NAME:
        raise ValueError("smoke must contain only the QuaRot W4A16 checkpoint")
    if config.get("backends") != ["vllm", "sglang"]:
        raise ValueError("backend order drifted")
    if config["protocol"].get("choices") != ["no", "yes"]:
        raise ValueError("BoolQ choices drifted")
    if int(config["smoke"]["examples"]) != 32:
        raise ValueError("result-gated BoolQ smoke must use 32 examples")
    if int(config["smoke"]["expected_requests_per_backend"]) != 64:
        raise ValueError("BoolQ smoke must score both choices per example")
    if config["sglang"].get("attention_backend") != "flashinfer":
        raise ValueError("SGLang must not retry the unavailable default FA3 backend")
    if config["sglang"].get("offline_quantization_argument") is not None:
        raise ValueError("the accepted checkpoint must load without requantization")


def _requests(tokenizer: Any, examples: list[dict[str, Any]], config: dict[str, Any]):
    requests = []
    protocol = config["protocol"]
    for example_position, example in enumerate(examples):
        context = _context(example, protocol)
        for choice_position, choice in enumerate(protocol["choices"]):
            tokens, continuation_start = _encode_pair(
                tokenizer,
                context,
                f"{protocol['target_delimiter']}{choice}",
                int(config["engine"]["max_model_len"]),
            )
            requests.append(
                {
                    "example_position": example_position,
                    "choice_position": choice_position,
                    "tokens": tokens,
                    "continuation_start": continuation_start,
                }
            )
    return requests


def _sglang_loglikelihood(
    meta_info: dict[str, Any], expected_tokens: list[int], continuation_start: int
) -> float:
    values = meta_info.get("input_token_logprobs")
    if not isinstance(values, list):
        raise RuntimeError("SGLang did not return input_token_logprobs")
    expected_continuation = expected_tokens[continuation_start:]
    if len(values) != len(expected_continuation):
        raise RuntimeError(
            "SGLang continuation logprob length differs from the tokenized request"
        )
    total = 0.0
    observed_ids = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise RuntimeError("SGLang returned a malformed input-token logprob")
        logprob, token_id = item[0], item[1]
        if logprob is None or not math.isfinite(float(logprob)):
            raise RuntimeError("SGLang returned a non-finite continuation logprob")
        observed_ids.append(int(token_id))
        total += float(logprob)
    if observed_ids != expected_continuation:
        raise RuntimeError("SGLang returned continuation token IDs in a different order")
    return total


def _metrics(
    examples: list[dict[str, Any]], requests: list[dict[str, Any]], scores: list[float]
) -> dict[str, Any]:
    if len(requests) != len(scores):
        raise RuntimeError("request and score counts differ")
    per_choice = [[None, None] for _ in examples]
    for request, score in zip(requests, scores):
        per_choice[request["example_position"]][request["choice_position"]] = score
    rows = []
    correct = 0
    for example, choice_scores in zip(examples, per_choice):
        if any(value is None for value in choice_scores):
            raise RuntimeError("BoolQ example has incomplete choice scores")
        prediction = max(range(2), key=choice_scores.__getitem__)
        is_correct = prediction == int(example["label"])
        correct += int(is_correct)
        rows.append(
            {
                "idx": example["idx"],
                "label": example["label"],
                "prediction": prediction,
                "correct": is_correct,
                "choice_loglikelihoods": choice_scores,
            }
        )
    return {
        "evaluated_examples": len(examples),
        "evaluated_requests": len(requests),
        "correct": correct,
        "accuracy": correct / len(examples),
        "example_metrics": rows,
    }


def _http_json(url: str, payload: dict[str, Any], timeout: float = 900.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.loads(response.read())


def _run_sglang(
    *,
    model_path: Path,
    tokenizer_path: Path,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    sglang_python: Path,
    log_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    requests = _requests(tokenizer, examples, config)
    scores = []
    served_name = "llama2-13b-sglang-quarot-w4a16"
    server: _Server | None = None
    with _Server(
        backend="sglang",
        executable=_absolute_executable(sglang_python),
        model_path=model_path,
        served_name=served_name,
        config=config,
        log_path=log_path,
    ) as server:
        batch_size = int(config["smoke"]["batch_size"])
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            response = _http_json(
                f"{server.base_url}/generate",
                {
                    "input_ids": [item["tokens"] for item in batch],
                    "sampling_params": {
                        "temperature": float(config["engine"]["temperature"]),
                        "max_new_tokens": int(config["engine"]["max_tokens"]),
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": [
                        item["continuation_start"] for item in batch
                    ],
                    "top_logprobs_num": 0,
                    "return_text_in_logprobs": False,
                },
            )
            if not isinstance(response, list) or len(response) != len(batch):
                raise RuntimeError("SGLang batch response has the wrong shape")
            for item, output in zip(batch, response):
                if not isinstance(output, dict) or not isinstance(
                    output.get("meta_info"), dict
                ):
                    raise RuntimeError("SGLang response is missing meta_info")
                scores.append(
                    _sglang_loglikelihood(
                        output["meta_info"],
                        item["tokens"],
                        item["continuation_start"],
                    )
                )
    assert server is not None
    return {
        "status": "passed",
        "backend": "sglang",
        **_metrics(examples, requests, scores),
        "startup_seconds": server.startup_seconds,
        "baseline_gpu_memory_used_mib": server.baseline_memory_mib,
        "ready_gpu_memory_used_mib": server.ready_memory_mib,
        "released_gpu_memory_used_mib": server.released_memory_mib,
        "server_log": str(log_path),
        "server_log_sha256": _sha256(log_path),
        "quantization_log_excerpt": _log_excerpt(
            log_path, ["compressed", "w4a16", "gptq", "kernel"]
        ),
    }


def _run_vllm_worker(
    *,
    script: Path,
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    model_path: Path,
    output_path: Path,
    max_examples: int,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
        "--dataset-manifest",
        str(manifest_path),
        "--tokenizer-model",
        str(tokenizer_path),
        "--model-path",
        str(model_path),
        "--max-examples",
        str(max_examples),
        "--worker-vllm",
        "--output",
        str(output_path),
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["status"] = "passed"
    result["backend"] = "vllm"
    result["worker_log"] = str(log_path)
    result["worker_log_sha256"] = _sha256(log_path)
    return result


def _comparison(cases: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if any(cases[name].get("status") != "passed" for name in ("vllm", "sglang")):
        return None
    vllm_rows = cases["vllm"]["example_metrics"]
    sglang_rows = cases["sglang"]["example_metrics"]
    if [row["idx"] for row in vllm_rows] != [row["idx"] for row in sglang_rows]:
        raise RuntimeError("backends evaluated a different BoolQ order")
    disagreements = [
        left["idx"]
        for left, right in zip(vllm_rows, sglang_rows)
        if left["prediction"] != right["prediction"]
    ]
    score_deltas = [
        abs(float(left_score) - float(right_score))
        for left, right in zip(vllm_rows, sglang_rows)
        for left_score, right_score in zip(
            left["choice_loglikelihoods"], right["choice_loglikelihoods"]
        )
    ]
    return {
        "prediction_disagreement_count": len(disagreements),
        "prediction_disagreement_indices": disagreements,
        "accuracy_delta_sglang_minus_vllm": (
            float(cases["sglang"]["accuracy"]) - float(cases["vllm"]["accuracy"])
        ),
        "max_absolute_choice_loglikelihood_delta": max(score_deltas),
        "mean_absolute_choice_loglikelihood_delta": sum(score_deltas) / len(score_deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path)
    parser.add_argument("--max-examples", type=int, required=True)
    parser.add_argument("--worker-vllm", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    _validate_config(config)
    examples, manifest = _load_examples(
        config, args.dataset_manifest.resolve(), args.max_examples
    )
    model_path = args.model_path.resolve()
    tokenizer_path = args.tokenizer_model.resolve()

    if args.worker_vllm:
        result = _one_model(MODEL_NAME, model_path, tokenizer_path, examples, config)
        _write_json(args.output.resolve(), result)
        return 0

    if args.sglang_python is None or args.log_dir is None:
        parser.error("parent mode requires --sglang-python and --log-dir")
    sglang_python = _absolute_executable(args.sglang_python)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict[str, Any]] = {}
    try:
        cases["vllm"] = _run_vllm_worker(
            script=Path(__file__).resolve(),
            config_path=config_path,
            manifest_path=args.dataset_manifest.resolve(),
            tokenizer_path=tokenizer_path,
            model_path=model_path,
            output_path=log_dir / "vllm-result.json",
            max_examples=args.max_examples,
            log_path=log_dir / "vllm.log",
        )
    except Exception as error:
        cases["vllm"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        cases["sglang"] = _run_sglang(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            examples=examples,
            config=config,
            sglang_python=sglang_python,
            log_path=log_dir / "sglang.log",
        )
    except Exception as error:
        sglang_log = log_dir / "sglang.log"
        cases["sglang"] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "server_log": str(sglang_log),
            "server_log_sha256": _sha256(sglang_log) if sglang_log.is_file() else None,
            "log_excerpt": _log_excerpt(sglang_log, ["error", "exception", "kernel"]),
        }

    result = {
        "status": "recorded",
        "comparison_status": (
            "both_backends_scored" if all(
                case.get("status") == "passed" for case in cases.values()
            ) else "incomplete"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest.resolve()),
        "examples_sha256": manifest["examples"]["sha256"],
        "model": MODEL_NAME,
        "model_path": str(model_path),
        "source_gate": config["source_gate"],
        "evaluated_examples": len(examples),
        "expected_requests_per_backend": len(examples) * 2,
        "protocol": config["protocol"],
        "runtimes": {
            "sglang": _runtime_probe(sglang_python, ["sglang", "torch"]),
            "environment": {
                "CUDA_HOME": os.environ.get("CUDA_HOME"),
                "SGLANG_ENABLE_JIT_DEEPGEMM": os.environ.get(
                    "SGLANG_ENABLE_JIT_DEEPGEMM"
                ),
            },
        },
        "cases": cases,
        "comparison": _comparison(cases),
        "scope": config["scope"],
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
