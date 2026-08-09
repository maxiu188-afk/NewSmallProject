#!/usr/bin/env python3
"""Run the paired three-repetition vLLM-versus-SGLang formal serving matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sglang_vllm_quarot_w4a16_serving import (  # noqa: E402
    MODEL_NAME,
    _absolute_executable,
    _build_request_corpus,
    _log_excerpt,
    _run_case,
    _runtime_probe,
    _sha256,
    _write_corpus,
    _write_json,
)


EXPECTED_CASES = [
    {"name": "latency_c1", "max_concurrency": 1},
    {"name": "throughput_c8", "max_concurrency": 8},
]
EXPECTED_BACKEND_ORDERS = [
    ["vllm", "sglang"],
    ["sglang", "vllm"],
    ["vllm", "sglang"],
]
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


def _validate_formal_config(config: dict[str, Any]) -> None:
    if config.get("status") != "accepted_serving_smoke_bound_formal_matrix_authorized":
        raise ValueError("formal matrix is not authorized")
    if config.get("model") != MODEL_NAME:
        raise ValueError("formal matrix model drifted")
    if config.get("backends") != ["vllm", "sglang"]:
        raise ValueError("formal matrix backend set drifted")
    gate = config.get("serving_smoke_gate", {})
    if gate.get("status") != "accepted" or gate.get("job_id") != "5952554":
        raise ValueError("formal matrix is not bound to accepted smoke 5952554")
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
            raise ValueError(f"formal benchmark field drifted: {key}")
    if benchmark.get("cases") != EXPECTED_CASES:
        raise ValueError("formal concurrency matrix drifted")
    if benchmark.get("backend_orders") != EXPECTED_BACKEND_ORDERS:
        raise ValueError("formal paired backend order drifted")
    server = config.get("server", {})
    if server.get("kv_cache_dtype") != "bfloat16":
        raise ValueError("formal comparison requires BF16 KV cache")
    if int(server.get("kv_cache_memory_bytes", 0)) != 8 * 1024**3:
        raise ValueError("formal comparison requires the frozen 8 GiB KV budget")
    if not server.get("disable_prefix_cache"):
        raise ValueError("formal comparison requires prefix caching disabled")
    if not server.get("disable_chunked_prefill"):
        raise ValueError("formal comparison requires chunked prefill disabled")
    corpus = config.get("request_corpus", {})
    if corpus.get("file_sha256") != (
        "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1"
    ):
        raise ValueError("formal request corpus drifted")
    if int(corpus.get("requests", 0)) != 64:
        raise ValueError("formal request corpus size drifted")


def _value(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty repetition series")
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def _paired_comparison(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Any] = {}
    for case in EXPECTED_CASES:
        name = case["name"]
        left = cases[f"vllm:{name}"]
        right = cases[f"sglang:{name}"]
        if left.get("status") != "passed" or right.get("status") != "passed":
            by_case[name] = {"status": "incomplete"}
            continue
        ratios = {
            metric: _value(right, path) / _value(left, path)
            for metric, path in METRICS.items()
        }
        by_case[name] = {
            "status": "paired",
            "ratio_definition": "sglang_over_vllm",
            "ratios": ratios,
        }
    complete = all(item["status"] == "paired" for item in by_case.values())
    return {
        "status": "paired" if complete else "incomplete",
        "cases": by_case,
    }


def _aggregate(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for case in EXPECTED_CASES:
        name = case["name"]
        cell: dict[str, Any] = {
            "repetitions": len(repetitions),
            "backends": {},
            "ratios_sglang_over_vllm": {},
        }
        for backend in ("vllm", "sglang"):
            records = [item["cases"][f"{backend}:{name}"] for item in repetitions]
            if any(record.get("status") != "passed" for record in records):
                cell["backends"][backend] = {"status": "incomplete"}
                continue
            cell["backends"][backend] = {
                "status": "complete",
                "metrics": {
                    metric: _distribution([_value(record, path) for record in records])
                    for metric, path in METRICS.items()
                },
            }
        comparisons = [item["comparison"]["cases"][name] for item in repetitions]
        if all(item.get("status") == "paired" for item in comparisons):
            cell["ratios_sglang_over_vllm"] = {
                metric: _distribution([item["ratios"][metric] for item in comparisons])
                for metric in METRICS
            }
            cell["status"] = "complete"
        else:
            cell["status"] = "incomplete"
        summary[name] = cell
    return {
        "status": (
            "formal_matrix_complete"
            if all(item["status"] == "complete" for item in summary.values())
            else "incomplete"
        ),
        "spread_definition": "min_max_and_range_across_all_three_repetitions",
        "cases": summary,
    }


def _failure_record(
    *, backend: str, case: dict[str, Any], error: Exception, log_dir: Path, raw_dir: Path
) -> dict[str, Any]:
    log_path = log_dir / f"{backend}-{case['name']}-server.log"
    raw_path = raw_dir / f"{backend}-{case['name']}.json"
    return {
        "status": "failed",
        "backend": backend,
        "case": case,
        "error_type": type(error).__name__,
        "error": str(error),
        "server_log": str(log_path),
        "server_log_sha256": _sha256(log_path) if log_path.is_file() else None,
        "raw_result": str(raw_path) if raw_path.is_file() else None,
        "raw_result_sha256": _sha256(raw_path) if raw_path.is_file() else None,
        "log_excerpt": _log_excerpt(
            log_path,
            [
                "error",
                "exception",
                "traceback",
                "kernel",
                "readiness_timeout",
                "timeout_diagnostic",
                "waiting for application startup",
            ],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--formal-input-record", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vllm-executable", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--sglang-preflight", type=Path, required=True)
    parser.add_argument("--request-corpus", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_formal_config(config)
    input_record_path = args.formal_input_record.resolve()
    input_record = json.loads(input_record_path.read_text(encoding="utf-8"))
    if input_record.get("status") != "accepted":
        raise RuntimeError("formal input gate did not pass")
    if input_record.get("serving_smoke_job_id") != "5952554":
        raise RuntimeError("formal input gate used another smoke")

    model_path = args.model_path.resolve()
    tokenizer_path = args.tokenizer_model.resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(model_path / "config.json")
    if not (tokenizer_path / "tokenizer.json").is_file():
        raise FileNotFoundError(tokenizer_path / "tokenizer.json")
    vllm_executable = _absolute_executable(args.vllm_executable)
    sglang_python = _absolute_executable(args.sglang_python)
    for executable in (vllm_executable, sglang_python):
        if not executable.is_file():
            raise FileNotFoundError(executable)

    preflight_path = args.sglang_preflight.resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("formal SGLang runtime preflight did not pass")
    if preflight.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("formal SGLang preflight used another effective config")

    corpus_path = args.request_corpus.resolve()
    corpus = _build_request_corpus(tokenizer_path, config)
    _write_corpus(corpus_path, corpus)
    if _sha256(corpus_path) != config["request_corpus"]["file_sha256"]:
        raise RuntimeError("formal request corpus hash drifted")

    log_root = args.log_dir.resolve()
    raw_root = args.raw_dir.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    executables = {"vllm": vllm_executable, "sglang": sglang_python}
    repetitions: list[dict[str, Any]] = []
    for index, backend_order in enumerate(
        config["benchmark"]["backend_orders"], start=1
    ):
        repetition_log_dir = log_root / f"repetition-{index}"
        repetition_raw_dir = raw_root / f"repetition-{index}"
        repetition_log_dir.mkdir(parents=True, exist_ok=False)
        repetition_raw_dir.mkdir(parents=True, exist_ok=False)
        cases: dict[str, dict[str, Any]] = {}
        for case in config["benchmark"]["cases"]:
            for backend in backend_order:
                key = f"{backend}:{case['name']}"
                try:
                    cases[key] = _run_case(
                        backend=backend,
                        executable=executables[backend],
                        model_path=model_path,
                        corpus=corpus,
                        config=config,
                        case=case,
                        log_dir=repetition_log_dir,
                        raw_dir=repetition_raw_dir,
                    )
                except Exception as error:
                    cases[key] = _failure_record(
                        backend=backend,
                        case=case,
                        error=error,
                        log_dir=repetition_log_dir,
                        raw_dir=repetition_raw_dir,
                    )
        repetitions.append(
            {
                "repetition": index,
                "backend_order": backend_order,
                "cases": cases,
                "comparison": _paired_comparison(cases),
            }
        )

    aggregate = _aggregate(repetitions)
    complete = aggregate["status"] == "formal_matrix_complete"
    result = {
        "status": "passed" if complete else "incomplete",
        "comparison_status": aggregate["status"],
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
        "formal_input_record": {
            "path": str(input_record_path),
            "sha256": _sha256(input_record_path),
            "record": input_record,
        },
        "model": MODEL_NAME,
        "model_path": str(model_path),
        "source_gate": config["source_gate"],
        "compatibility_gate": config["compatibility_gate"],
        "serving_smoke_gate": config["serving_smoke_gate"],
        "request_corpus": {
            "path": str(corpus_path),
            "sha256": _sha256(corpus_path),
            "prompt_list_sha256": config["request_corpus"]["prompt_list_sha256"],
            "token_id_list_sha256": config["request_corpus"][
                "token_id_list_sha256"
            ],
        },
        "benchmark": config["benchmark"],
        "runtimes": {
            "vllm": _runtime_probe(
                vllm_executable.parent / "python", ["vllm", "torch"]
            ),
            "sglang": _runtime_probe(sglang_python, ["sglang", "torch"]),
            "sglang_preflight": {
                "path": str(preflight_path),
                "sha256": _sha256(preflight_path),
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
        "scope": config["scope"],
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
