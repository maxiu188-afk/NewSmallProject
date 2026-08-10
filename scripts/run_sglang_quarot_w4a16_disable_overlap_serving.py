#!/usr/bin/env python3
"""Run the protocol-matched SGLang no-overlap third arm."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_sglang_vllm_llama2_13b_smoke as smoke_runtime  # noqa: E402
import scripts.run_sglang_vllm_quarot_w4a16_serving as serving  # noqa: E402


EXPECTED_CASES = [
    {"name": "latency_c1", "max_concurrency": 1},
    {"name": "throughput_c8", "max_concurrency": 8},
]
EXPECTED_SERVER_ARGUMENT = "--disable-overlap-schedule"
METRICS: dict[str, tuple[str, ...]] = {
    "request_throughput_per_second": (
        "metrics",
        "request_throughput_per_second",
    ),
    "input_token_throughput_per_second": (
        "metrics",
        "input_token_throughput_per_second",
    ),
    "output_token_throughput_per_second": (
        "metrics",
        "output_token_throughput_per_second",
    ),
    "total_token_throughput_per_second": (
        "metrics",
        "total_token_throughput_per_second",
    ),
    "ttft_p50_ms": ("metrics", "ttft", "p50_ms"),
    "ttft_p95_ms": ("metrics", "ttft", "p95_ms"),
    "ttft_p99_ms": ("metrics", "ttft", "p99_ms"),
    "tpot_p50_ms": ("metrics", "tpot", "p50_ms"),
    "tpot_p95_ms": ("metrics", "tpot", "p95_ms"),
    "tpot_p99_ms": ("metrics", "tpot", "p99_ms"),
    "e2e_p50_ms": ("metrics", "e2e", "p50_ms"),
    "e2e_p95_ms": ("metrics", "e2e", "p95_ms"),
    "e2e_p99_ms": ("metrics", "e2e", "p99_ms"),
    "itl_p50_ms": ("metrics", "stream_event_itl", "p50_ms"),
    "itl_p95_ms": ("metrics", "stream_event_itl", "p95_ms"),
    "itl_p99_ms": ("metrics", "stream_event_itl", "p99_ms"),
    "startup_seconds": ("startup_seconds",),
    "ready_gpu_memory_used_mib": ("ready_gpu_memory_used_mib",),
    "peak_gpu_memory_used_mib": ("peak_gpu_memory_used_mib",),
    "released_gpu_memory_used_mib": ("released_gpu_memory_used_mib",),
}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("status") != "authorized_not_submitted":
        raise ValueError("no-overlap third arm is not authorized")
    if config.get("model") != serving.MODEL_NAME:
        raise ValueError("model route drifted")
    experiment = config.get("experiment", {})
    if experiment.get("backend") != "sglang":
        raise ValueError("third arm backend drifted")
    if experiment.get("variant") != "disable_overlap_schedule":
        raise ValueError("third arm variant drifted")
    if experiment.get("server_argument") != EXPECTED_SERVER_ARGUMENT:
        raise ValueError("third arm server argument drifted")
    if config.get("reference_formal_job", {}).get("job_id") != "5960180":
        raise ValueError("third arm reference job drifted")
    benchmark = config.get("benchmark", {})
    expected = {
        "protocol": "repository_openai_completions_stream_v1",
        "endpoint": "/v1/completions",
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream_include_usage": True,
        "validation_requests": 1,
        "warmup_requests": 4,
        "measured_requests": 64,
        "repetitions": 3,
        "metric_percentiles": [50, 95, 99],
        "memory_sample_interval_seconds": 0.2,
    }
    for key, value in expected.items():
        if benchmark.get(key) != value:
            raise ValueError(f"benchmark field drifted: {key}")
    if benchmark.get("cases") != EXPECTED_CASES:
        raise ValueError("concurrency matrix drifted")
    server = config.get("server", {})
    if server.get("kv_cache_dtype") != "bfloat16":
        raise ValueError("third arm requires BF16 KV cache")
    if int(server.get("kv_cache_memory_bytes", 0)) != 8 * 1024**3:
        raise ValueError("third arm requires the frozen 8 GiB KV budget")
    if not server.get("disable_prefix_cache"):
        raise ValueError("third arm requires prefix caching disabled")
    if not server.get("disable_chunked_prefill"):
        raise ValueError("third arm requires chunked prefill disabled")
    if config.get("request_corpus", {}).get("file_sha256") != (
        "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1"
    ):
        raise ValueError("request corpus drifted")


def _server_command_with_disabled_overlap(
    *,
    backend: str,
    executable: Path,
    model_path: Path,
    served_name: str,
    config: dict[str, Any],
) -> list[str]:
    if backend != "sglang":
        raise ValueError("no-overlap third arm may only launch SGLang")
    command = _ORIGINAL_SERVER_COMMAND(
        backend=backend,
        executable=executable,
        model_path=model_path,
        served_name=served_name,
        config=config,
    )
    if EXPECTED_SERVER_ARGUMENT in command:
        raise RuntimeError("base SGLang command already disables overlap scheduling")
    command.append(EXPECTED_SERVER_ARGUMENT)
    if command.count(EXPECTED_SERVER_ARGUMENT) != 1:
        raise RuntimeError("no-overlap server argument was not added exactly once")
    return command


_ORIGINAL_SERVER_COMMAND = smoke_runtime._server_command


@contextmanager
def _disabled_overlap_command_scope() -> Iterator[None]:
    original_smoke = smoke_runtime._server_command
    original_serving = serving._server_command
    if original_smoke is not _ORIGINAL_SERVER_COMMAND:
        raise RuntimeError("shared SGLang server command was unexpectedly patched")
    smoke_runtime._server_command = _server_command_with_disabled_overlap
    serving._server_command = _server_command_with_disabled_overlap
    try:
        yield
    finally:
        serving._server_command = original_serving
        smoke_runtime._server_command = original_smoke


def _run_case(**kwargs: Any) -> dict[str, Any]:
    with _disabled_overlap_command_scope():
        record = serving._run_case(**kwargs)
    command = record.get("server_command", [])
    if command.count(EXPECTED_SERVER_ARGUMENT) != 1:
        raise RuntimeError(
            "measured cell did not retain the no-overlap server argument"
        )
    return record


def _value(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _distribution(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError("third-arm summary requires all three repetitions")
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _aggregate(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(repetitions) != 3:
        return {"status": "incomplete", "cases": {}}
    summaries: dict[str, Any] = {}
    for case in EXPECTED_CASES:
        name = case["name"]
        records = [item["cases"][name] for item in repetitions]
        if any(record.get("status") != "passed" for record in records):
            summaries[name] = {"status": "incomplete"}
            continue
        summaries[name] = {
            "status": "complete",
            "repetitions": 3,
            "metrics": {
                metric: _distribution([_value(record, path) for record in records])
                for metric, path in METRICS.items()
            },
        }
    complete = all(item["status"] == "complete" for item in summaries.values())
    return {
        "status": "third_arm_complete" if complete else "incomplete",
        "spread_definition": "min_max_and_range_across_all_three_repetitions",
        "pairing_boundary": "not_paired_with_reference_job_5960180",
        "cases": summaries,
    }


def _failure_record(
    *, case: dict[str, Any], error: Exception, log_dir: Path, raw_dir: Path
) -> dict[str, Any]:
    log_path = log_dir / f"sglang-{case['name']}-server.log"
    raw_path = raw_dir / f"sglang-{case['name']}.json"
    return {
        "status": "failed",
        "backend": "sglang",
        "case": case,
        "error_type": type(error).__name__,
        "error": str(error),
        "server_log": str(log_path),
        "server_log_sha256": serving._sha256(log_path) if log_path.is_file() else None,
        "raw_result": str(raw_path) if raw_path.is_file() else None,
        "raw_result_sha256": serving._sha256(raw_path) if raw_path.is_file() else None,
        "log_excerpt": serving._log_excerpt(
            log_path,
            ["error", "exception", "traceback", "kernel", "timeout"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-record", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--sglang-preflight", type=Path, required=True)
    parser.add_argument("--request-corpus", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    input_record_path = args.input_record.resolve()
    input_record = json.loads(input_record_path.read_text(encoding="utf-8"))
    if input_record.get("status") != "accepted":
        raise RuntimeError("third-arm input gate did not pass")
    if input_record.get("reference_formal_job_id") != "5960180":
        raise RuntimeError("third-arm input gate used another reference job")

    model_path = args.model_path.resolve()
    tokenizer_path = args.tokenizer_model.resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    if not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(tokenizer_path / "tokenizer.json")
    sglang_python = serving._absolute_executable(args.sglang_python)
    if not sglang_python.is_file():
        raise FileNotFoundError(sglang_python)

    preflight_path = args.sglang_preflight.resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("SGLang runtime preflight did not pass")
    if preflight.get("config_sha256") != serving._sha256(config_path):
        raise RuntimeError("SGLang preflight used another effective config")

    corpus_path = args.request_corpus.resolve()
    corpus = serving._build_request_corpus(tokenizer_path, config)
    serving._write_corpus(corpus_path, corpus)
    if serving._sha256(corpus_path) != config["request_corpus"]["file_sha256"]:
        raise RuntimeError("request corpus hash drifted")

    log_root = args.log_dir.resolve()
    raw_root = args.raw_dir.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    repetitions: list[dict[str, Any]] = []
    for index in range(1, int(config["benchmark"]["repetitions"]) + 1):
        repetition_log_dir = log_root / f"repetition-{index}"
        repetition_raw_dir = raw_root / f"repetition-{index}"
        repetition_log_dir.mkdir(parents=True, exist_ok=False)
        repetition_raw_dir.mkdir(parents=True, exist_ok=False)
        cases: dict[str, dict[str, Any]] = {}
        for case in config["benchmark"]["cases"]:
            try:
                cases[case["name"]] = _run_case(
                    backend="sglang",
                    executable=sglang_python,
                    model_path=model_path,
                    corpus=corpus,
                    config=config,
                    case=case,
                    log_dir=repetition_log_dir,
                    raw_dir=repetition_raw_dir,
                )
            except Exception as error:
                cases[case["name"]] = _failure_record(
                    case=case,
                    error=error,
                    log_dir=repetition_log_dir,
                    raw_dir=repetition_raw_dir,
                )
        repetitions.append({"repetition": index, "cases": cases})

    aggregate = _aggregate(repetitions)
    complete = aggregate["status"] == "third_arm_complete"
    result = {
        "status": "passed" if complete else "incomplete",
        "comparison_status": (
            "protocol_matched_historical_third_arm_complete"
            if complete
            else "incomplete"
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
        "config_sha256": serving._sha256(config_path),
        "input_record": {
            "path": str(input_record_path),
            "sha256": serving._sha256(input_record_path),
            "record": input_record,
        },
        "model": serving.MODEL_NAME,
        "model_path": str(model_path),
        "experiment": config["experiment"],
        "reference_formal_job": config["reference_formal_job"],
        "request_corpus": {
            "path": str(corpus_path),
            "sha256": serving._sha256(corpus_path),
            "prompt_list_sha256": config["request_corpus"]["prompt_list_sha256"],
            "token_id_list_sha256": config["request_corpus"]["token_id_list_sha256"],
        },
        "benchmark": config["benchmark"],
        "runtime": {
            "sglang": serving._runtime_probe(sglang_python, ["sglang", "torch"]),
            "sglang_preflight": {
                "path": str(preflight_path),
                "sha256": serving._sha256(preflight_path),
                "record": preflight,
            },
            "environment": {
                "CUDA_HOME": os.environ.get("CUDA_HOME"),
                "SGLANG_ENABLE_JIT_DEEPGEMM": os.environ.get(
                    "SGLANG_ENABLE_JIT_DEEPGEMM"
                ),
            },
        },
        "repetitions": repetitions,
        "aggregate": aggregate,
        "comparison_boundary": (
            "May be placed beside job 5960180 as a protocol-matched third arm; "
            "it is not a contemporaneous paired repetition with either "
            "reference backend."
        ),
        "scope": config["scope"],
    }
    serving._write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
