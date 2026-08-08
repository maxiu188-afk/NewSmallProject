#!/usr/bin/env python3
"""Record exact-checkpoint vLLM/SGLang compatibility on one allocated GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_w4a16_llama2_13b_serving import (  # noqa: E402
    _gpu_memory_used_mib,
    _port_is_free,
    _request_json,
)


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


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_config(config: dict[str, Any]) -> None:
    if config["source_gate"].get("status") != "accepted":
        raise ValueError("comparison smoke requires an accepted source gate")
    if config["models"] != ["bf16", "fp8_targeted_w4afp8"]:
        raise ValueError("comparison smoke model matrix drifted")
    if config["backends"] != ["vllm", "sglang"]:
        raise ValueError("comparison smoke backend matrix drifted")
    server = config["server"]
    if server["host"] != "127.0.0.1":
        raise ValueError("comparison servers must bind only to 127.0.0.1")
    if int(server["max_model_len"]) < 320:
        raise ValueError("comparison context is smaller than the formal workload")
    if int(server["max_running_requests"]) != 8:
        raise ValueError("comparison smoke must preserve concurrency capacity 8")
    if int(server["kv_cache_memory_bytes"]) != 8 * 1024**3:
        raise ValueError("comparison requires the accepted 8 GiB KV budget")
    expected_tokens = int(server["kv_cache_memory_bytes"]) // int(
        server["kv_bytes_per_token"]
    )
    if int(server["max_total_tokens"]) != expected_tokens:
        raise ValueError("SGLang token pool does not match the 8 GiB KV derivation")
    if config["sglang"].get("offline_quantization_argument") is not None:
        raise ValueError("exact offline checkpoint smoke must not override quantization")
    if config["sglang"].get("enable_jit_deep_gemm") is not False:
        raise ValueError("comparison smoke must disable unused JIT DeepGEMM")


def _parse_models(values: list[str], expected: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use name=path syntax: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            raise ValueError(f"repeated model name: {name}")
        models[name] = Path(raw_path).resolve()
    if list(models) != expected:
        raise ValueError(f"models must be supplied in this order: {expected}")
    for name, path in models.items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} config is missing: {path / 'config.json'}")
    return models


def _absolute_executable(path: Path) -> Path:
    """Make a command path absolute without resolving a virtualenv symlink."""
    return Path(os.path.abspath(path))


def _server_command(
    *,
    backend: str,
    executable: Path,
    model_path: Path,
    served_name: str,
    config: dict[str, Any],
) -> list[str]:
    server = config["server"]
    use_explicit_served_name = (
        server.get("served_model_id_policy", "explicit_alias")
        != "checkpoint_path"
    )
    if backend == "vllm":
        command = [
            str(executable),
            "serve",
            str(model_path),
            "--host",
            str(server["host"]),
            "--port",
            str(server["port"]),
            "--dtype",
            str(server["dtype"]),
            "--max-model-len",
            str(server["max_model_len"]),
            "--max-num-seqs",
            str(server["max_running_requests"]),
            "--kv-cache-memory-bytes",
            str(server["kv_cache_memory_bytes"]),
            "--kv-cache-dtype",
            str(server["kv_cache_dtype"]),
            "--seed",
            str(server["seed"]),
            "--generation-config",
            str(config["vllm"]["generation_config"]),
        ]
        if use_explicit_served_name:
            command.extend(["--served-model-name", served_name])
        if config["vllm"]["disable_log_stats"]:
            command.append("--disable-log-stats")
        if server.get("disable_prefix_cache"):
            command.append("--no-enable-prefix-caching")
        if server.get("disable_chunked_prefill"):
            command.append("--no-enable-chunked-prefill")
        return command
    if backend != "sglang":
        raise ValueError(f"unsupported backend: {backend}")
    command = [
        str(executable),
        "-m",
        str(config["sglang"]["module"]),
        "--model-path",
        str(model_path),
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
        "--dtype",
        str(server["dtype"]),
        "--context-length",
        str(server["max_model_len"]),
        "--max-running-requests",
        str(server["max_running_requests"]),
        "--max-total-tokens",
        str(server["max_total_tokens"]),
        "--kv-cache-dtype",
        str(server["kv_cache_dtype"]),
        "--random-seed",
        str(server["seed"]),
        "--sampling-defaults",
        str(config["sglang"]["sampling_defaults"]),
    ]
    if use_explicit_served_name:
        command.extend(["--served-model-name", served_name])
    if server["disable_prefix_cache"]:
        command.append("--disable-radix-cache")
    if server["disable_chunked_prefill"]:
        command.extend(["--chunked-prefill-size", "-1"])
    attention_backend = config["sglang"].get("attention_backend")
    if attention_backend is not None:
        command.extend(["--attention-backend", str(attention_backend)])
    quantization = config["sglang"].get("offline_quantization_argument")
    if quantization is not None:
        command.extend(["--quantization", str(quantization)])
    return command


def _runtime_probe(python: Path, packages: list[str]) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform;"
        f"names={packages!r};"
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{n:m.version(n) for n in names}}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _log_excerpt(path: Path, patterns: list[str]) -> list[str]:
    if not path.is_file():
        return []
    lowered = [pattern.lower() for pattern in patterns]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line for line in lines if any(pattern in line.lower() for pattern in lowered)][
        -80:
    ]


class _Server:
    def __init__(
        self,
        *,
        backend: str,
        executable: Path,
        model_path: Path,
        served_name: str,
        config: dict[str, Any],
        log_path: Path,
    ) -> None:
        self.backend = backend
        self.executable = executable
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

    def _ready_timeout_seconds(self) -> int:
        server = self.config["server"]
        per_backend = server.get("ready_timeout_seconds_by_backend", {})
        return int(per_backend.get(self.backend, server["ready_timeout_seconds"]))

    def _append_timeout_diagnostics(self, timeout_seconds: int) -> None:
        if self.process is None or self.log_handle is None:
            return
        self.log_handle.flush()
        header = (
            "\nSGLANG_VLLM_SERVER_READINESS_TIMEOUT "
            f"backend={self.backend} pid={self.process.pid} "
            f"timeout_seconds={timeout_seconds}\n"
        )
        self.log_handle.write(header)
        commands = [
            (
                "process_session",
                [
                    "ps",
                    "-ww",
                    "-o",
                    "pid,ppid,pgid,sid,stat,etime,pcpu,pmem,rss,vsz,wchan:32,cmd",
                    "--sid",
                    str(self.process.pid),
                ],
            ),
            (
                "listening_port",
                [
                    "ss",
                    "-ltnp",
                    "sport",
                    "=",
                    f":{self.config['server']['port']}",
                ],
            ),
            (
                "gpu_processes",
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
            ),
        ]
        py_spy = self.executable.parent / "py-spy"
        if py_spy.is_file():
            commands.append(
                ("python_stack", [str(py_spy), "dump", "--pid", str(self.process.pid)])
            )
        for label, command in commands:
            self.log_handle.write(f"SGLANG_VLLM_TIMEOUT_DIAGNOSTIC_BEGIN={label}\n")
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.log_handle.write(f"returncode={completed.returncode}\n")
                self.log_handle.write(completed.stdout)
                self.log_handle.write(completed.stderr)
            except Exception as error:
                self.log_handle.write(f"diagnostic_error={error!r}\n")
            self.log_handle.write(f"SGLANG_VLLM_TIMEOUT_DIAGNOSTIC_END={label}\n")
        self.log_handle.flush()

        # SGLang installs a SIGQUIT diagnostic handler that attempts to dump
        # its live scheduler stacks. At this point readiness has already timed
        # out, so trigger it before the normal process-group cleanup.
        if self.backend == "sglang" and self.process.poll() is None:
            try:
                os.kill(self.process.pid, signal.SIGQUIT)
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            finally:
                self.log_handle.flush()

    @property
    def base_url(self) -> str:
        server = self.config["server"]
        return f"http://{server['host']}:{server['port']}"

    def __enter__(self) -> "_Server":
        server = self.config["server"]
        if not _port_is_free(str(server["host"]), int(server["port"])):
            raise RuntimeError(f"server port is in use: {self.base_url}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_memory_mib = _gpu_memory_used_mib()
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            _server_command(
                backend=self.backend,
                executable=self.executable,
                model_path=self.model_path,
                served_name=self.served_name,
                config=self.config,
            ),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        timeout_seconds = self._ready_timeout_seconds()
        deadline = started + timeout_seconds
        try:
            while time.monotonic() < deadline:
                return_code = self.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"{self.backend} exited before readiness with {return_code}"
                    )
                try:
                    _request_json(f"{self.base_url}/health", timeout=5.0)
                    self.startup_seconds = time.monotonic() - started
                    self.ready_memory_mib = _gpu_memory_used_mib()
                    return self
                except (OSError, RuntimeError, json.JSONDecodeError):
                    time.sleep(2.0)
            self._append_timeout_diagnostics(timeout_seconds)
            raise TimeoutError(
                f"{self.backend} did not become ready within {timeout_seconds} seconds"
            )
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=int(self.config["server"]["stop_timeout_seconds"]))
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=15)
            deadline = time.monotonic() + int(
                self.config["server"]["stop_timeout_seconds"]
            )
            observed = _gpu_memory_used_mib()
            while observed > self.baseline_memory_mib + 256 and time.monotonic() < deadline:
                time.sleep(1.0)
                observed = _gpu_memory_used_mib()
            self.released_memory_mib = observed
            if observed > self.baseline_memory_mib + 256:
                raise RuntimeError(
                    f"{self.backend} GPU memory did not recover: "
                    f"baseline={self.baseline_memory_mib} observed={observed}"
                )
        finally:
            if self.log_handle is not None:
                self.log_handle.close()


def _run_case(
    *,
    backend: str,
    model_name: str,
    model_path: Path,
    executable: Path,
    config: dict[str, Any],
    log_dir: Path,
) -> dict[str, Any]:
    served_name = f"llama2-13b-{backend}-{model_name.replace('_', '-')}"
    log_path = log_dir / f"{backend}-{model_name}.log"
    command = _server_command(
        backend=backend,
        executable=executable,
        model_path=model_path,
        served_name=served_name,
        config=config,
    )
    server: _Server | None = None
    try:
        with _Server(
            backend=backend,
            executable=executable,
            model_path=model_path,
            served_name=served_name,
            config=config,
            log_path=log_path,
        ) as server:
            model_response = _request_json(f"{server.base_url}/v1/models")
            model_ids = [entry["id"] for entry in model_response.get("data", [])]
            if len(model_ids) != 1:
                raise RuntimeError(f"unexpected model ids: {model_ids}")
            completion = _request_json(
                f"{server.base_url}/v1/completions",
                {
                    "model": model_ids[0],
                    "prompt": config["smoke"]["prompt"],
                    "temperature": config["smoke"]["temperature"],
                    "max_tokens": config["smoke"]["max_tokens"],
                    "ignore_eos": config["smoke"]["ignore_eos"],
                    "stream": False,
                },
                timeout=180.0,
            )
            choices = completion.get("choices", [])
            if len(choices) != 1 or not isinstance(choices[0].get("text"), str):
                raise RuntimeError("completion response is structurally invalid")
            if completion.get("usage", {}).get("completion_tokens") != int(
                config["smoke"]["max_tokens"]
            ):
                raise RuntimeError("completion did not return the forced token count")
        assert server is not None
        patterns = list(config[backend]["quantized_log_patterns"])
        return {
            "status": "passed",
            "backend": backend,
            "model": model_name,
            "model_path": str(model_path),
            "server_command": command,
            "served_model_ids": model_ids,
            "completion": completion,
            "startup_seconds": server.startup_seconds,
            "baseline_gpu_memory_used_mib": server.baseline_memory_mib,
            "ready_gpu_memory_used_mib": server.ready_memory_mib,
            "released_gpu_memory_used_mib": server.released_memory_mib,
            "server_log": str(log_path),
            "server_log_sha256": _sha256(log_path),
            "quantization_log_excerpt": (
                _log_excerpt(log_path, patterns)
                if model_name == "fp8_targeted_w4afp8"
                else []
            ),
        }
    except Exception as error:
        return {
            "status": "failed",
            "backend": backend,
            "model": model_name,
            "model_path": str(model_path),
            "server_command": command,
            "error_type": type(error).__name__,
            "error": str(error),
            "server_log": str(log_path),
            "server_log_sha256": _sha256(log_path) if log_path.is_file() else None,
            "quantization_log_excerpt": _log_excerpt(
                log_path, list(config[backend]["quantized_log_patterns"])
            ),
        }


def _compatibility_status(
    cases: dict[str, dict[str, Any]],
    generated_texts_equal: dict[str, bool] | None = None,
    *,
    require_equal_text: bool = False,
) -> str:
    bf16 = all(
        cases[f"{backend}:bf16"]["status"] == "passed"
        for backend in ("vllm", "sglang")
    )
    w4afp8 = all(
        cases[f"{backend}:fp8_targeted_w4afp8"]["status"] == "passed"
        for backend in ("vllm", "sglang")
    )
    if bf16 and w4afp8:
        if require_equal_text and (
            generated_texts_equal is None
            or not all(generated_texts_equal.values())
        ):
            return "checkpoint_generation_mismatch"
        return "exact_checkpoint_tracks_passed"
    if bf16:
        return "bf16_passed_w4afp8_unresolved"
    return "incomplete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vllm-executable", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    models = _parse_models(args.model, config["models"])
    if not args.vllm_executable.is_file():
        raise FileNotFoundError(args.vllm_executable)
    if not args.sglang_python.is_file():
        raise FileNotFoundError(args.sglang_python)

    runtimes = {
        "vllm": _runtime_probe(args.vllm_executable.parent / "python", ["vllm", "torch"]),
        "sglang": _runtime_probe(args.sglang_python, ["sglang", "torch"]),
    }
    runtimes["sglang"]["environment"] = {
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "SGLANG_ENABLE_JIT_DEEPGEMM": os.environ.get(
            "SGLANG_ENABLE_JIT_DEEPGEMM"
        ),
    }
    executables = {
        "vllm": _absolute_executable(args.vllm_executable),
        "sglang": _absolute_executable(args.sglang_python),
    }
    cases: dict[str, dict[str, Any]] = {}
    for backend in config["backends"]:
        for model_name, model_path in models.items():
            key = f"{backend}:{model_name}"
            cases[key] = _run_case(
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                executable=executables[backend],
                config=config,
                log_dir=args.log_dir.resolve(),
            )

    texts: dict[str, dict[str, str]] = {}
    for model_name in config["models"]:
        texts[model_name] = {}
        for backend in config["backends"]:
            case = cases[f"{backend}:{model_name}"]
            if case["status"] == "passed":
                texts[model_name][backend] = case["completion"]["choices"][0]["text"]
    text_equal = {
        model_name: len(values) == 2 and len(set(values.values())) == 1
        for model_name, values in texts.items()
    }
    result = {
        "status": "recorded",
        "compatibility_status": _compatibility_status(
            cases,
            text_equal,
            require_equal_text=bool(
                config["smoke"]["require_equal_text_per_checkpoint"]
            ),
        ),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "source_gate": config["source_gate"],
        "runtimes": runtimes,
        "server_config": config["server"],
        "smoke_config": config["smoke"],
        "generated_texts_equal_per_checkpoint": text_equal,
        "cases": cases,
        "scope": config["scope"],
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
