#!/usr/bin/env python3
"""Run matched Llama-2-13B vLLM service smoke or serving benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = ("bf16", "unrotated_w4a16", "rotated_w4a16")


def _expected_models(config: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(config["source_gate"]["expected_models"])
    if len(values) < 2 or values[0] != "bf16" or len(set(values)) != len(values):
        raise ValueError("source gate requires unique models with bf16 first")
    return values


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_models(
    values: list[str],
    expected_models: tuple[str, ...] = EXPECTED_MODELS,
) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use name=path syntax: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            raise ValueError(f"repeated model name: {name}")
        models[name] = Path(raw_path).resolve()
    if tuple(models) != expected_models:
        raise ValueError(
            f"models must be supplied in this order: {', '.join(expected_models)}"
        )
    for name, path in models.items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} config is missing: {path / 'config.json'}")
    return models


def _validate_config(config: dict[str, Any]) -> None:
    server = config["server"]
    benchmark = config["benchmark"]
    source_gate = config["source_gate"]
    _expected_models(config)
    if server["host"] != "127.0.0.1":
        raise ValueError("service jobs must bind only to 127.0.0.1")
    if int(server["max_model_len"]) < (
        int(benchmark["input_length"]) + int(benchmark["output_length"])
    ):
        raise ValueError("max_model_len is smaller than the benchmark request")
    if int(server["max_num_seqs"]) < max(
        int(case["max_concurrency"]) for case in benchmark["cases"]
    ):
        raise ValueError("max_num_seqs is smaller than a benchmark concurrency")
    if int(server["kv_cache_memory_bytes"]) != 8 * 1024**3:
        raise ValueError("the matched benchmark requires an explicit 8 GiB KV cache")
    if benchmark["request_rate"] != "inf":
        raise ValueError("the frozen benchmark uses a closed-loop concurrency limit")
    if [case["name"] for case in benchmark["cases"]] != [
        "latency_c1",
        "throughput_c8",
    ]:
        raise ValueError("benchmark case matrix drifted")


def _request_json(
    url: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        body = response.read()
    if not body:
        return {}
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return value


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(1.0)
        return candidate.connect_ex((host, port)) != 0


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return "<server log is absent>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _gpu_memory_used_mib() -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    rows = []
    for line in output:
        index, uuid, memory = [field.strip() for field in line.split(",", 2)]
        rows.append({"index": index, "uuid": uuid, "memory": int(memory)})
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(rows) == 1:
        return int(rows[0]["memory"])
    if len(visible) != 1:
        raise RuntimeError(
            f"cannot identify one allocated GPU from CUDA_VISIBLE_DEVICES={visible} "
            f"and nvidia-smi rows={rows}"
        )
    matches = [
        row
        for row in rows
        if row["index"] == visible[0] or row["uuid"] == visible[0]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"allocated GPU selector {visible[0]} did not match nvidia-smi rows={rows}"
        )
    return int(matches[0]["memory"])


def _wait_for_server(
    process: subprocess.Popen[Any],
    base_url: str,
    timeout_seconds: int,
    log_path: Path,
) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"vLLM server exited with {return_code} before readiness:\n"
                f"{_log_tail(log_path)}"
            )
        try:
            _request_json(f"{base_url}/health", timeout=5.0)
            return time.monotonic() - started
        except (OSError, RuntimeError, json.JSONDecodeError):
            time.sleep(2.0)
    raise TimeoutError(
        f"vLLM server did not become ready in {timeout_seconds}s:\n{_log_tail(log_path)}"
    )


def _stop_server(
    process: subprocess.Popen[Any],
    baseline_memory_mib: int,
    timeout_seconds: int,
) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    deadline = time.monotonic() + timeout_seconds
    observed = _gpu_memory_used_mib()
    while observed > baseline_memory_mib + 256 and time.monotonic() < deadline:
        time.sleep(1.0)
        observed = _gpu_memory_used_mib()
    if observed > baseline_memory_mib + 256:
        raise RuntimeError(
            f"GPU memory did not return to baseline after server exit: "
            f"baseline={baseline_memory_mib} MiB observed={observed} MiB"
        )
    return observed


def _server_command(
    vllm_executable: Path,
    model_path: Path,
    served_name: str,
    config: dict[str, Any],
) -> list[str]:
    server = config["server"]
    command = [
        str(vllm_executable),
        "serve",
        str(model_path),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--served-model-name",
        served_name,
        "--dtype",
        str(server["dtype"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--kv-cache-memory-bytes",
        str(server["kv_cache_memory_bytes"]),
        "--seed",
        str(server["seed"]),
        "--generation-config",
        str(server["generation_config"]),
    ]
    if server["enforce_eager"]:
        command.append("--enforce-eager")
    if server["disable_log_stats"]:
        command.append("--disable-log-stats")
    return command


class _RunningServer:
    def __init__(
        self,
        *,
        vllm_executable: Path,
        model_path: Path,
        served_name: str,
        config: dict[str, Any],
        log_path: Path,
    ) -> None:
        self.vllm_executable = vllm_executable
        self.model_path = model_path
        self.served_name = served_name
        self.config = config
        self.log_path = log_path
        self.process: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None
        self.baseline_memory_mib = 0
        self.ready_memory_mib = 0
        self.released_memory_mib = 0
        self.startup_seconds = 0.0

    def __enter__(self) -> "_RunningServer":
        server = self.config["server"]
        host = str(server["host"])
        port = int(server["port"])
        if not _port_is_free(host, port):
            raise RuntimeError(f"server port is already in use: {host}:{port}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_memory_mib = _gpu_memory_used_mib()
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            _server_command(
                self.vllm_executable,
                self.model_path,
                self.served_name,
                self.config,
            ),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        base_url = f"http://{host}:{port}"
        try:
            self.startup_seconds = _wait_for_server(
                self.process,
                base_url,
                int(server["ready_timeout_seconds"]),
                self.log_path,
            )
            self.ready_memory_mib = _gpu_memory_used_mib()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.process is not None:
                self.released_memory_mib = _stop_server(
                    self.process,
                    self.baseline_memory_mib,
                    int(self.config["server"]["stop_timeout_seconds"]),
                )
        finally:
            if self.log_handle is not None:
                self.log_handle.close()

    @property
    def base_url(self) -> str:
        server = self.config["server"]
        return f"http://{server['host']}:{server['port']}"


def _server_record(
    server: _RunningServer,
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    record = {
        "startup_seconds": server.startup_seconds,
        "baseline_gpu_memory_used_mib": server.baseline_memory_mib,
        "ready_gpu_memory_used_mib": server.ready_memory_mib,
        "server_log": str(server.log_path),
        "server_log_sha256": _sha256(server.log_path),
    }
    kernel_gate = config.get("kernel_gate")
    if isinstance(kernel_gate, dict) and model_name in kernel_gate.get(
        "quantized_models", []
    ):
        pattern = str(kernel_gate["required_quantized_log_pattern"])
        log_text = server.log_path.read_text(encoding="utf-8", errors="replace")
        if pattern not in log_text:
            raise RuntimeError(
                f"{model_name} did not prove the required kernel in its server log"
            )
        record["kernel_evidence"] = {
            "required_pattern": pattern,
            "matched": True,
        }
    return record


def _run_smoke(
    config: dict[str, Any],
    models: dict[str, Path],
    vllm_executable: Path,
    log_dir: Path,
) -> dict[str, Any]:
    smoke = config["smoke"]
    results: dict[str, Any] = {}
    generated_texts = set()
    for name, model_path in models.items():
        served_name = f"llama2-13b-{name.replace('_', '-')}"
        log_path = log_dir / f"{name}.log"
        with _RunningServer(
            vllm_executable=vllm_executable,
            model_path=model_path,
            served_name=served_name,
            config=config,
            log_path=log_path,
        ) as server:
            models_response = _request_json(f"{server.base_url}/v1/models")
            ids = [entry["id"] for entry in models_response.get("data", [])]
            if ids != [served_name]:
                raise RuntimeError(f"{name} returned unexpected served model ids: {ids}")
            completion = _request_json(
                f"{server.base_url}/v1/completions",
                {
                    "model": served_name,
                    "prompt": smoke["prompt"],
                    "temperature": smoke["temperature"],
                    "max_tokens": smoke["max_tokens"],
                    "ignore_eos": smoke["ignore_eos"],
                    "stream": False,
                },
                timeout=120.0,
            )
            choices = completion.get("choices", [])
            if len(choices) != 1 or not isinstance(choices[0].get("text"), str):
                raise RuntimeError(f"{name} returned an invalid completion: {completion}")
            usage = completion.get("usage", {})
            if usage.get("completion_tokens") != int(smoke["max_tokens"]):
                raise RuntimeError(f"{name} returned the wrong completion length: {usage}")
            generated_texts.add(choices[0]["text"])
        results[name] = {
            **_server_record(server, config, name),
            "released_gpu_memory_used_mib": server.released_memory_mib,
            "served_model_ids": ids,
            "completion": completion,
        }
    generated_texts_equal = len(generated_texts) == 1
    if smoke.get("require_equal_generated_texts", True) and not generated_texts_equal:
        raise RuntimeError("service smoke models returned different greedy text")
    return {
        "status": "passed",
        "scope": (
            f"{len(models)} complete Llama-2-13B OpenAI-compatible service "
            "endpoint checks; not performance evidence"
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "source_gate": config["source_gate"],
        "server_config": config["server"],
        "smoke_config": smoke,
        "generated_texts_equal": generated_texts_equal,
        "models": results,
    }


def _sample_memory(stop: threading.Event, samples: list[int], errors: list[str]) -> None:
    while not stop.is_set():
        try:
            samples.append(_gpu_memory_used_mib())
        except BaseException as error:
            errors.append(repr(error))
            return
        stop.wait(0.5)


def _benchmark_command(
    *,
    vllm_executable: Path,
    tokenizer: Path,
    served_name: str,
    base_url: str,
    config: dict[str, Any],
    case: dict[str, Any],
    raw_result_dir: Path,
    raw_result_name: str,
    model_name: str,
) -> list[str]:
    benchmark = config["benchmark"]
    command = [
        str(vllm_executable),
        "bench",
        "serve",
        "--backend",
        str(benchmark["backend"]),
        "--base-url",
        base_url,
        "--endpoint",
        str(benchmark["endpoint"]),
        "--model",
        served_name,
        "--tokenizer",
        str(tokenizer),
        "--dataset-name",
        str(benchmark["dataset_name"]),
        "--input-len",
        str(benchmark["input_length"]),
        "--output-len",
        str(benchmark["output_length"]),
        "--num-prompts",
        str(benchmark["num_prompts"]),
        "--num-warmups",
        str(benchmark["num_warmups"]),
        "--request-rate",
        str(benchmark["request_rate"]),
        "--max-concurrency",
        str(case["max_concurrency"]),
        "--seed",
        str(benchmark["seed"]),
        "--percentile-metrics",
        ",".join(benchmark["percentile_metrics"]),
        "--metric-percentiles",
        ",".join(str(value) for value in benchmark["metric_percentiles"]),
        "--result-dir",
        str(raw_result_dir),
        "--result-filename",
        raw_result_name,
        "--save-result",
        "--disable-tqdm",
    ]
    if benchmark["ignore_eos"]:
        command.append("--ignore-eos")
    if benchmark["save_detailed"]:
        command.append("--save-detailed")
    command.extend(
        [
            "--metadata",
            f"project_revision={_revision()}",
            f"model_variant={model_name}",
            f"case={case['name']}",
        ]
    )
    return command


def _validate_raw_benchmark(
    raw: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    if raw.get("completed") != int(benchmark["num_prompts"]):
        raise RuntimeError(f"benchmark completed count is invalid: {raw.get('completed')}")
    if raw.get("failed") != 0:
        raise RuntimeError(f"benchmark contains failed requests: {raw.get('failed')}")
    fields = (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
    )
    metrics = {}
    for field in fields:
        value = raw.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"benchmark metric {field} is invalid: {value}")
        metrics[field] = value
    for field in (
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "p50_e2el_ms",
        "p90_e2el_ms",
        "p99_e2el_ms",
    ):
        value = raw.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"benchmark percentile {field} is invalid: {value}")
        metrics[field] = value
    return metrics


def _comparisons(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    def compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
        return {
            "request_throughput_ratio": (
                candidate["metrics"]["request_throughput"]
                / baseline["metrics"]["request_throughput"]
            ),
            "output_throughput_ratio": (
                candidate["metrics"]["output_throughput"]
                / baseline["metrics"]["output_throughput"]
            ),
            "median_ttft_ratio": (
                candidate["metrics"]["median_ttft_ms"]
                / baseline["metrics"]["median_ttft_ms"]
            ),
            "median_tpot_ratio": (
                candidate["metrics"]["median_tpot_ms"]
                / baseline["metrics"]["median_tpot_ms"]
            ),
            "ready_gpu_memory_ratio": (
                candidate["ready_gpu_memory_used_mib"]
                / baseline["ready_gpu_memory_used_mib"]
            ),
        }

    comparisons: dict[str, Any] = {}
    for case in config["benchmark"]["cases"]:
        case_name = case["name"]
        baseline = results["bf16"][case_name]
        per_case = {}
        for name in _expected_models(config)[1:]:
            candidate = results[name][case_name]
            per_case[f"{name}_vs_bf16"] = compare(candidate, baseline)
        for pair in config.get("comparison_pairs", []):
            candidate_name = pair["candidate"]
            reference_name = pair["reference"]
            per_case[pair["name"]] = compare(
                results[candidate_name][case_name],
                results[reference_name][case_name],
            )
        comparisons[case_name] = per_case
    return comparisons


def _run_benchmark(
    config: dict[str, Any],
    models: dict[str, Path],
    vllm_executable: Path,
    tokenizer: Path,
    log_dir: Path,
    raw_result_dir: Path,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    raw_result_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name, model_path in models.items():
        results[name] = {}
        served_name = f"llama2-13b-{name.replace('_', '-')}"
        for case in benchmark["cases"]:
            case_name = case["name"]
            server_log = log_dir / f"{name}-{case_name}-server.log"
            client_log = log_dir / f"{name}-{case_name}-client.log"
            raw_name = f"{name}-{case_name}.json"
            raw_path = raw_result_dir / raw_name
            raw_path.unlink(missing_ok=True)
            with _RunningServer(
                vllm_executable=vllm_executable,
                model_path=model_path,
                served_name=served_name,
                config=config,
                log_path=server_log,
            ) as server:
                samples = [server.ready_memory_mib]
                errors: list[str] = []
                stop = threading.Event()
                sampler = threading.Thread(
                    target=_sample_memory,
                    args=(stop, samples, errors),
                    daemon=True,
                )
                sampler.start()
                client_log.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with client_log.open("w", encoding="utf-8") as handle:
                        subprocess.run(
                            _benchmark_command(
                                vllm_executable=vllm_executable,
                                tokenizer=tokenizer,
                                served_name=served_name,
                                base_url=server.base_url,
                                config=config,
                                case=case,
                                raw_result_dir=raw_result_dir,
                                raw_result_name=raw_name,
                                model_name=name,
                            ),
                            check=True,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                finally:
                    stop.set()
                    sampler.join(timeout=10)
                if sampler.is_alive() or errors:
                    raise RuntimeError(f"GPU memory sampler failed: {errors}")
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                metrics = _validate_raw_benchmark(raw, benchmark)
            results[name][case_name] = {
                **_server_record(server, config, name),
                "released_gpu_memory_used_mib": server.released_memory_mib,
                "case": case,
                "metrics": metrics,
                "peak_gpu_memory_used_mib": max(samples),
                "gpu_memory_samples": len(samples),
                "client_log": str(client_log),
                "client_log_sha256": _sha256(client_log),
                "raw_result": str(raw_path),
                "raw_result_sha256": _sha256(raw_path),
            }
    return {
        "status": "passed",
        "scope": "matched full-model vLLM HTTP serving performance on one GH200; not quality or single-block evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "source_gate": config["source_gate"],
        "server_config": config["server"],
        "benchmark_config": benchmark,
        "models": results,
        "comparisons": _comparisons(results, config),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "benchmark"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vllm-executable", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw-result-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    models = _parse_models(args.model, _expected_models(config))
    if not args.vllm_executable.is_file():
        raise FileNotFoundError(args.vllm_executable)
    if not (args.tokenizer / "tokenizer.json").is_file():
        raise FileNotFoundError(args.tokenizer / "tokenizer.json")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("Isambard serving requires the native vLLM sampler fallback")

    if args.mode == "smoke":
        result = _run_smoke(
            config,
            models,
            args.vllm_executable.resolve(),
            args.log_dir.resolve(),
        )
    else:
        if args.raw_result_dir is None:
            parser.error("benchmark mode requires --raw-result-dir")
        result = _run_benchmark(
            config,
            models,
            args.vllm_executable.resolve(),
            args.tokenizer.resolve(),
            args.log_dir.resolve(),
            args.raw_result_dir.resolve(),
        )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
