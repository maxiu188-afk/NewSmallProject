#!/usr/bin/env python3
"""Run one backend-neutral serving benchmark for the accepted QuaRot W4A16 model."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sglang_vllm_llama2_13b_smoke import (  # noqa: E402
    _Server,
    _absolute_executable,
    _log_excerpt,
    _runtime_probe,
    _server_command,
    _sha256,
    _write_json,
)
from scripts.run_vllm_w4a16_llama2_13b_serving import (  # noqa: E402
    _gpu_memory_used_mib,
    _request_json,
)


MODEL_NAME = "quarot_w4a16"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_json_list(value: list[Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if config["source_gate"].get("status") != "accepted":
        raise ValueError("serving smoke requires the accepted W4A16 source gate")
    if config["compatibility_gate"].get("status") != "accepted":
        raise ValueError("serving smoke requires accepted cross-backend compatibility")
    if config.get("model") != MODEL_NAME:
        raise ValueError("serving smoke must contain only the QuaRot W4A16 checkpoint")
    if config.get("backends") != ["vllm", "sglang"]:
        raise ValueError("backend order drifted")

    server = config["server"]
    if server.get("host") != "127.0.0.1":
        raise ValueError("comparison servers must bind only to 127.0.0.1")
    if int(server["max_model_len"]) < 320:
        raise ValueError("server context is smaller than the 256+64 workload")
    if int(server["max_running_requests"]) != 8:
        raise ValueError("server concurrency capacity must remain eight")
    if int(server["kv_cache_memory_bytes"]) != 8 * 1024**3:
        raise ValueError("comparison requires the accepted 8 GiB KV budget")
    expected_tokens = int(server["kv_cache_memory_bytes"]) // int(
        server["kv_bytes_per_token"]
    )
    if int(server["max_total_tokens"]) != expected_tokens:
        raise ValueError("SGLang token pool no longer matches the 8 GiB KV budget")
    if server.get("kv_cache_dtype") != "bfloat16":
        raise ValueError("comparison requires BF16 KV cache")
    if not server.get("disable_prefix_cache"):
        raise ValueError("prefix caching must remain disabled")
    if not server.get("disable_chunked_prefill"):
        raise ValueError("chunked prefill must remain disabled")
    if server.get("served_model_id_policy") != "checkpoint_path":
        raise ValueError("serving smoke must use the shared checkpoint-path model ID")
    ready_timeouts = server.get("ready_timeout_seconds_by_backend")
    if ready_timeouts != {"vllm": 900, "sglang": 1200}:
        raise ValueError("backend-specific readiness budgets drifted")

    benchmark = config["benchmark"]
    if benchmark.get("protocol") != "repository_openai_completions_stream_v1":
        raise ValueError("benchmark client protocol drifted")
    if benchmark.get("endpoint") != "/v1/completions":
        raise ValueError("benchmark endpoint drifted")
    if float(benchmark["temperature"]) != 0.0:
        raise ValueError("benchmark must use greedy generation")
    if not benchmark.get("ignore_eos"):
        raise ValueError("benchmark must force the output length")
    if int(benchmark["validation_requests"]) != 1:
        raise ValueError("smoke must retain one unmeasured validation request")
    if int(benchmark["warmup_requests"]) != 4:
        raise ValueError("smoke must retain four warm-up requests")
    if int(benchmark["measured_requests"]) != 8:
        raise ValueError("result-gated smoke must measure exactly eight requests")
    if benchmark.get("cases") != [{"name": "smoke_c1", "max_concurrency": 1}]:
        raise ValueError("result-gated serving smoke case matrix drifted")
    if benchmark.get("metric_percentiles") != [50, 95, 99]:
        raise ValueError("serving percentile set drifted")

    corpus = config["request_corpus"]
    expected_corpus = {
        "algorithm": "vllm-random-dataset-0.25.1-compatible",
        "seed": 0,
        "requests": 64,
        "input_tokens_with_special": 256,
        "input_tokens_without_special": 255,
        "output_tokens": 64,
        "prefix_len": 0,
        "range_ratio": 0.0,
    }
    for key, value in expected_corpus.items():
        if corpus.get(key) != value:
            raise ValueError(f"request corpus field drifted: {key}")
    legacy = corpus["legacy_vllm_evidence"]
    if legacy.get("prompts_retained") is not False:
        raise ValueError("legacy serving provenance must not claim retained prompts")
    if (
        legacy.get("output_hash_policy")
        != "diagnostic_only_not_corpus_identity"
    ):
        raise ValueError("legacy output-hash evidence policy drifted")
    if config["sglang"].get("attention_backend") != "flashinfer":
        raise ValueError("SGLang must use the accepted FlashInfer attention path")
    if config["sglang"].get("offline_quantization_argument") is not None:
        raise ValueError("accepted checkpoint must load without requantization")
    if config["sglang"].get("enable_jit_deep_gemm") is not False:
        raise ValueError("unused JIT DeepGEMM must remain disabled")


def _build_request_corpus(
    tokenizer_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np
    from transformers import AutoTokenizer

    spec = config["request_corpus"]
    observed_files = {}
    for name, expected_sha256 in spec["tokenizer_file_sha256"].items():
        path = tokenizer_path / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected_sha256:
            raise RuntimeError(f"tokenizer file hash drifted: {name}")
        observed_files[name] = observed

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    if type(tokenizer).__name__ != spec["tokenizer_class"]:
        raise RuntimeError("tokenizer class drifted")
    if int(tokenizer.vocab_size) != int(spec["tokenizer_vocab_size"]):
        raise RuntimeError("tokenizer vocabulary size drifted")
    if list(tokenizer.all_special_ids) != spec["tokenizer_all_special_ids"]:
        raise RuntimeError("tokenizer special-token IDs drifted")

    rng = np.random.default_rng(int(spec["seed"]))
    request_count = int(spec["requests"])
    input_with_special = int(spec["input_tokens_with_special"])
    output_tokens = int(spec["output_tokens"])
    special_count = int(tokenizer.num_special_tokens_to_add())
    input_without_special = max(0, input_with_special - special_count)
    if input_without_special != int(spec["input_tokens_without_special"]):
        raise RuntimeError("tokenizer special-token count drifted")

    input_lengths = rng.integers(
        input_without_special, input_without_special + 1, size=request_count
    )
    output_lengths = rng.integers(
        output_tokens, output_tokens + 1, size=request_count
    )
    offsets = rng.integers(0, tokenizer.vocab_size, size=request_count)
    all_tokens = np.arange(tokenizer.vocab_size)
    allowed_tokens = np.array(
        list(set(all_tokens) - set(tokenizer.all_special_ids))
    )

    requests = []
    for index in range(request_count):
        target_length = int(input_lengths[index])
        token_ids = allowed_tokens[
            (
                int(offsets[index])
                + index
                + np.arange(target_length)
            )
            % len(allowed_tokens)
        ].tolist()
        remaining_retries = 10
        while True:
            prompt = tokenizer.decode(token_ids)
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if remaining_retries <= 0 or len(token_ids) == target_length:
                break
            if len(token_ids) < target_length:
                token_ids.extend(
                    rng.integers(
                        0,
                        tokenizer.vocab_size,
                        size=target_length - len(token_ids),
                    ).tolist()
                )
            else:
                token_ids = token_ids[:target_length]
            remaining_retries -= 1
        if len(token_ids) != target_length:
            raise RuntimeError(
                f"request {index} did not reach its frozen token length"
            )
        if tokenizer.encode(prompt, add_special_tokens=False) != token_ids:
            raise RuntimeError(f"request {index} no-special token IDs drifted")
        if len(tokenizer.encode(prompt, add_special_tokens=True)) != input_with_special:
            raise RuntimeError(f"request {index} total input length drifted")
        requests.append(
            {
                "request_id": f"random-{index:03d}",
                "prompt": prompt,
                "prompt_token_ids": token_ids,
                "expected_input_tokens": input_with_special,
                "expected_output_tokens": int(output_lengths[index]),
            }
        )

    corpus = {
        "schema_version": 1,
        "algorithm": {
            "name": spec["algorithm"],
            "seed": int(spec["seed"]),
            "requests": request_count,
            "input_tokens_with_special": input_with_special,
            "input_tokens_without_special": input_without_special,
            "output_tokens": output_tokens,
            "prefix_len": int(spec["prefix_len"]),
            "range_ratio": float(spec["range_ratio"]),
        },
        "tokenizer": {
            "revision": spec["tokenizer_revision"],
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "all_special_ids": list(tokenizer.all_special_ids),
            "files": observed_files,
        },
        "requests": requests,
    }
    if _sha256_json_list([item["prompt"] for item in requests]) != spec[
        "prompt_list_sha256"
    ]:
        raise RuntimeError("frozen prompt-list hash drifted")
    if _sha256_json_list([item["prompt_token_ids"] for item in requests]) != spec[
        "token_id_list_sha256"
    ]:
        raise RuntimeError("frozen token-ID-list hash drifted")
    if hashlib.sha256(_canonical_json_bytes(corpus)).hexdigest() != spec[
        "file_sha256"
    ]:
        raise RuntimeError("frozen request-corpus file hash drifted")
    return corpus


def _write_corpus(path: Path, corpus: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing request corpus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(corpus))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


async def _stream_completion(
    *,
    session: Any,
    url: str,
    served_name: str,
    request_spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    payload = {
        "model": served_name,
        "prompt": request_spec["prompt"],
        "repetition_penalty": float(benchmark["repetition_penalty"]),
        "temperature": float(benchmark["temperature"]),
        "max_tokens": int(request_spec["expected_output_tokens"]),
        "ignore_eos": bool(benchmark["ignore_eos"]),
        "stream": True,
        "stream_options": {
            "include_usage": bool(benchmark["stream_include_usage"])
        },
    }
    started = time.perf_counter()
    choice_timestamps: list[float] = []
    generated_parts: list[str] = []
    usage: dict[str, Any] | None = None
    done_seen = False
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(
                f"completion returned HTTP {response.status}: {body[-2000:]}"
            )
        while True:
            raw_line = await response.content.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise RuntimeError(f"unexpected streaming line: {line[:200]}")
            body = line[5:].strip()
            if body == "[DONE]":
                done_seen = True
                continue
            chunk = json.loads(body)
            if chunk.get("error") is not None:
                raise RuntimeError(f"stream returned an error: {chunk['error']}")
            choices = chunk.get("choices")
            if choices:
                if len(choices) != 1:
                    raise RuntimeError("stream returned multiple choices")
                choice_timestamps.append(time.perf_counter())
                text = choices[0].get("text")
                if text is not None and not isinstance(text, str):
                    raise RuntimeError("stream returned non-string completion text")
                generated_parts.append(text or "")
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    finished = time.perf_counter()
    if not done_seen:
        raise RuntimeError("stream ended without [DONE]")
    if not choice_timestamps:
        raise RuntimeError("stream returned no completion choices")
    if not isinstance(usage, dict):
        raise RuntimeError("stream omitted requested final usage")

    expected_input = int(request_spec["expected_input_tokens"])
    expected_output = int(request_spec["expected_output_tokens"])
    if int(usage.get("prompt_tokens", -1)) != expected_input:
        raise RuntimeError(
            "server prompt-token usage differs from the frozen request: "
            f"observed={usage.get('prompt_tokens')} expected={expected_input}"
        )
    if int(usage.get("completion_tokens", -1)) != expected_output:
        raise RuntimeError(
            "server completion-token usage differs from the forced output: "
            f"observed={usage.get('completion_tokens')} expected={expected_output}"
        )
    total_tokens = usage.get("total_tokens")
    if total_tokens is not None and int(total_tokens) != expected_input + expected_output:
        raise RuntimeError("server total-token usage is inconsistent")

    ttft = choice_timestamps[0] - started
    e2e = finished - started
    if ttft <= 0 or e2e < ttft:
        raise RuntimeError("client timing record is invalid")
    itls = [
        right - left for left, right in zip(choice_timestamps, choice_timestamps[1:])
    ]
    if any(value < 0 for value in itls):
        raise RuntimeError("stream event timestamps are not monotonic")
    generated_text = "".join(generated_parts)
    return {
        "request_id": request_spec["request_id"],
        "status": "passed",
        "prompt_tokens": expected_input,
        "completion_tokens": expected_output,
        "total_tokens": expected_input + expected_output,
        "started_perf_counter": started,
        "finished_perf_counter": finished,
        "ttft_seconds": ttft,
        "e2e_seconds": e2e,
        "tpot_seconds": (e2e - ttft) / (expected_output - 1),
        "stream_choice_events": len(choice_timestamps),
        "itl_seconds": itls,
        "usage": usage,
        "generated_text": generated_text,
        "generated_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
    }


async def _run_http_workload(
    *,
    base_url: str,
    served_name: str,
    corpus: dict[str, Any],
    config: dict[str, Any],
    max_concurrency: int,
) -> dict[str, Any]:
    import aiohttp

    endpoint = config["benchmark"]["endpoint"]
    url = f"{base_url}{endpoint}"
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
    )
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    requests = corpus["requests"]
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        validation = await _stream_completion(
            session=session,
            url=url,
            served_name=served_name,
            request_spec=requests[0],
            config=config,
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited(request_spec: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await _stream_completion(
                    session=session,
                    url=url,
                    served_name=served_name,
                    request_spec=request_spec,
                    config=config,
                )

        warmup_count = int(config["benchmark"]["warmup_requests"])
        warmups = await asyncio.gather(
            *[limited(requests[0]) for _ in range(warmup_count)]
        )

        measured_count = int(config["benchmark"]["measured_requests"])
        phase_started = time.perf_counter()
        measured = await asyncio.gather(
            *[limited(request) for request in requests[:measured_count]]
        )
        phase_finished = time.perf_counter()
    return {
        "validation": validation,
        "warmups": warmups,
        "measured": measured,
        "measurement_duration_seconds": phase_finished - phase_started,
    }


def _summarize_workload(
    record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    measured = record["measured"]
    duration = float(record["measurement_duration_seconds"])
    if duration <= 0:
        raise RuntimeError("measurement duration is invalid")
    if len(measured) != int(config["benchmark"]["measured_requests"]):
        raise RuntimeError("measured request count drifted")
    if any(item.get("status") != "passed" for item in measured):
        raise RuntimeError("measured workload contains a failed request")

    ttft = [float(item["ttft_seconds"]) * 1000 for item in measured]
    tpot = [float(item["tpot_seconds"]) * 1000 for item in measured]
    e2e = [float(item["e2e_seconds"]) * 1000 for item in measured]
    itl = [
        float(value) * 1000
        for item in measured
        for value in item["itl_seconds"]
    ]
    input_tokens = sum(int(item["prompt_tokens"]) for item in measured)
    output_tokens = sum(int(item["completion_tokens"]) for item in measured)
    percentiles = config["benchmark"]["metric_percentiles"]

    def distribution(values: list[float]) -> dict[str, float]:
        result = {
            "mean_ms": sum(values) / len(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for percentile in percentiles:
            result[f"p{percentile}_ms"] = _percentile(values, float(percentile))
        return result

    return {
        "completed": len(measured),
        "failed": 0,
        "duration_seconds": duration,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "request_throughput_per_second": len(measured) / duration,
        "input_token_throughput_per_second": input_tokens / duration,
        "output_token_throughput_per_second": output_tokens / duration,
        "total_token_throughput_per_second": (input_tokens + output_tokens)
        / duration,
        "ttft": distribution(ttft),
        "tpot": distribution(tpot),
        "e2e": distribution(e2e),
        "stream_event_itl": distribution(itl) if itl else None,
    }


def _sample_memory(
    stop: threading.Event,
    interval_seconds: float,
    samples: list[dict[str, Any]],
    errors: list[str],
) -> None:
    while not stop.is_set():
        try:
            samples.append(
                {
                    "perf_counter": time.perf_counter(),
                    "gpu_memory_used_mib": _gpu_memory_used_mib(),
                }
            )
        except BaseException as error:
            errors.append(repr(error))
            return
        stop.wait(interval_seconds)


def _legacy_vllm_output_diagnostic(
    measured: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    generated = [item["generated_text"] for item in measured]
    observed = _sha256_json_list(generated)
    expected = config["request_corpus"]["legacy_vllm_evidence"][
        "first_eight_generated_texts_sha256"
    ]
    return {
        "policy": "diagnostic_only_not_corpus_identity",
        "status": "matched" if observed == expected else "mismatched",
        "observed_first_eight_generated_texts_sha256": observed,
        "expected_first_eight_generated_texts_sha256": expected,
        "accepted_job": config["request_corpus"]["legacy_vllm_evidence"],
    }


def _run_case(
    *,
    backend: str,
    executable: Path,
    model_path: Path,
    corpus: dict[str, Any],
    config: dict[str, Any],
    case: dict[str, Any],
    log_dir: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    served_name = str(model_path)
    case_name = str(case["name"])
    log_path = log_dir / f"{backend}-{case_name}-server.log"
    raw_path = raw_dir / f"{backend}-{case_name}.json"
    command = _server_command(
        backend=backend,
        executable=executable,
        model_path=model_path,
        served_name=served_name,
        config=config,
    )
    server: _Server | None = None
    memory_samples: list[dict[str, Any]] = []
    memory_errors: list[str] = []
    workload: dict[str, Any] | None = None
    with _Server(
        backend=backend,
        executable=executable,
        model_path=model_path,
        served_name=served_name,
        config=config,
        log_path=log_path,
    ) as server:
        models_response = _request_json(f"{server.base_url}/v1/models")
        model_ids = [entry["id"] for entry in models_response.get("data", [])]
        if model_ids != [served_name]:
            raise RuntimeError(f"unexpected served model IDs: {model_ids}")

        stop = threading.Event()
        sampler = threading.Thread(
            target=_sample_memory,
            args=(
                stop,
                float(config["benchmark"]["memory_sample_interval_seconds"]),
                memory_samples,
                memory_errors,
            ),
            daemon=True,
        )
        sampler.start()
        try:
            workload = asyncio.run(
                _run_http_workload(
                    base_url=server.base_url,
                    served_name=served_name,
                    corpus=corpus,
                    config=config,
                    max_concurrency=int(case["max_concurrency"]),
                )
            )
        finally:
            stop.set()
            sampler.join(timeout=5)
        if sampler.is_alive():
            raise RuntimeError("GPU memory sampler did not stop")
        if memory_errors:
            raise RuntimeError(f"GPU memory sampling failed: {memory_errors}")
        if not memory_samples:
            raise RuntimeError("GPU memory sampler recorded no observations")

    assert server is not None and workload is not None
    legacy_diagnostic = (
        _legacy_vllm_output_diagnostic(workload["measured"], config)
        if backend == "vllm"
        else None
    )
    raw_record = {
        "backend": backend,
        "case": case,
        "server_command": command,
        "served_model_ids": model_ids,
        "workload": workload,
        "memory_samples": memory_samples,
        "legacy_vllm_output_diagnostic": legacy_diagnostic,
    }
    _write_json(raw_path, raw_record)
    peak_memory = max(
        [server.ready_memory_mib]
        + [int(item["gpu_memory_used_mib"]) for item in memory_samples]
    )
    return {
        "status": "passed",
        "backend": backend,
        "case": case,
        "metrics": _summarize_workload(workload, config),
        "startup_seconds": server.startup_seconds,
        "baseline_gpu_memory_used_mib": server.baseline_memory_mib,
        "ready_gpu_memory_used_mib": server.ready_memory_mib,
        "peak_gpu_memory_used_mib": peak_memory,
        "released_gpu_memory_used_mib": server.released_memory_mib,
        "server_command": command,
        "server_log": str(log_path),
        "server_log_sha256": _sha256(log_path),
        "raw_result": str(raw_path),
        "raw_result_sha256": _sha256(raw_path),
        "legacy_vllm_output_diagnostic": legacy_diagnostic,
        "kernel_log_excerpt": _log_excerpt(
            log_path,
            [
                "compressed",
                "w4a16",
                "gptq",
                "marlin",
                "attention backend",
                "flashinfer",
                "prefix cach",
                "chunked prefill",
                "kv cache",
            ],
        ),
    }


def _comparison(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = cases["vllm:smoke_c1"]["metrics"]
    right = cases["sglang:smoke_c1"]["metrics"]
    return {
        "status": "smoke_observation_not_formal_performance_evidence",
        "request_throughput_ratio_sglang_over_vllm": (
            right["request_throughput_per_second"]
            / left["request_throughput_per_second"]
        ),
        "p50_ttft_ratio_sglang_over_vllm": (
            right["ttft"]["p50_ms"] / left["ttft"]["p50_ms"]
        ),
        "p50_tpot_ratio_sglang_over_vllm": (
            right["tpot"]["p50_ms"] / left["tpot"]["p50_ms"]
        ),
        "p50_e2e_ratio_sglang_over_vllm": (
            right["e2e"]["p50_ms"] / left["e2e"]["p50_ms"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    _validate_config(config)
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
        raise RuntimeError("SGLang runtime preflight did not pass")
    if preflight.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("SGLang runtime preflight used a different config")

    corpus_path = args.request_corpus.resolve()
    corpus = _build_request_corpus(tokenizer_path, config)
    _write_corpus(corpus_path, corpus)
    if _sha256(corpus_path) != config["request_corpus"]["file_sha256"]:
        raise RuntimeError("written request corpus hash drifted")

    log_dir = args.log_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    runtimes = {
        "vllm": _runtime_probe(vllm_executable.parent / "python", ["vllm", "torch"]),
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
    }
    executables = {"vllm": vllm_executable, "sglang": sglang_python}
    cases: dict[str, dict[str, Any]] = {}
    for case in config["benchmark"]["cases"]:
        for backend in config["backends"]:
            key = f"{backend}:{case['name']}"
            try:
                cases[key] = _run_case(
                    backend=backend,
                    executable=executables[backend],
                    model_path=model_path,
                    corpus=corpus,
                    config=config,
                    case=case,
                    log_dir=log_dir,
                    raw_dir=raw_dir,
                )
            except Exception as error:
                log_path = log_dir / f"{backend}-{case['name']}-server.log"
                raw_path = raw_dir / f"{backend}-{case['name']}.json"
                cases[key] = {
                    "status": "failed",
                    "backend": backend,
                    "case": case,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "server_log": str(log_path),
                    "server_log_sha256": (
                        _sha256(log_path) if log_path.is_file() else None
                    ),
                    "raw_result": str(raw_path) if raw_path.is_file() else None,
                    "raw_result_sha256": (
                        _sha256(raw_path) if raw_path.is_file() else None
                    ),
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

    complete = all(case.get("status") == "passed" for case in cases.values())

    result = {
        "status": "recorded",
        "comparison_status": (
            "both_backends_benchmark_client_passed" if complete else "incomplete"
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
        "model": MODEL_NAME,
        "model_path": str(model_path),
        "source_gate": config["source_gate"],
        "compatibility_gate": config["compatibility_gate"],
        "request_corpus": {
            "path": str(corpus_path),
            "sha256": _sha256(corpus_path),
            "prompt_list_sha256": config["request_corpus"]["prompt_list_sha256"],
            "token_id_list_sha256": config["request_corpus"][
                "token_id_list_sha256"
            ],
        },
        "benchmark": config["benchmark"],
        "runtimes": runtimes,
        "cases": cases,
        "comparison": _comparison(cases) if complete else None,
        "scope": config["scope"],
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
