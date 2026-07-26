#!/usr/bin/env python3
"""Time one real vLLM Llama decoder block during full-model execution."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = ("bf16", "unrotated_w4a16", "rotated_w4a16")
HOOK_EVENTS_ATTRIBUTE = "_newsmallproject_block_timing_events"
HOOK_PENDING_ATTRIBUTE = "_newsmallproject_block_timing_pending"
HOOK_HANDLES_ATTRIBUTE = "_newsmallproject_block_timing_handles"


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_config(config: dict[str, Any]) -> None:
    engine = config["engine"]
    if config["source_gate"]["expected_models"] != list(EXPECTED_MODELS):
        raise ValueError("the accepted serving model set drifted")
    if engine["block_index"] != 0:
        raise ValueError("the diagnostic must measure Llama decoder layer 0")
    if engine["enforce_eager"] is not True:
        raise ValueError("Python layer hooks require enforce_eager=true")
    if engine["enable_prefix_caching"] is not False:
        raise ValueError("prefix caching would invalidate repeated prefill timing")
    if engine["kv_cache_memory_bytes"] != 16 * 1024**3:
        raise ValueError("the diagnostic requires an explicit 16 GiB KV cache")
    if engine["max_num_batched_tokens"] < 8 * 2048:
        raise ValueError("the largest prefill must fit in one scheduler step")
    if config["smoke"]["models"] != ["unrotated_w4a16"]:
        raise ValueError("the hook smoke must execute one real W4A16 layer")
    if config["formal"]["models"] != list(EXPECTED_MODELS):
        raise ValueError("the formal model order drifted")
    for protocol_name in ("smoke", "formal"):
        protocol = config[protocol_name]
        for case in protocol["cases"]:
            phase = case["phase"]
            if phase not in ("prefill", "decode"):
                raise ValueError(f"unsupported phase: {phase}")
            if phase == "prefill" and case["decode_steps"] != 0:
                raise ValueError(f"prefill case has decode work: {case}")
            if phase == "decode" and case["decode_steps"] <= 0:
                raise ValueError(f"decode case has no decode work: {case}")
            sequence = case["prompt_tokens"] + case["decode_steps"] + 1
            if sequence > engine["max_model_len"]:
                raise ValueError(f"case exceeds max_model_len: {case}")
            scheduled = case["batch_size"] * case["prompt_tokens"]
            if scheduled > engine["max_num_batched_tokens"]:
                raise ValueError(f"case requires chunked prefill: {case}")


def _locate_layer(model: Any, block_index: int) -> Any:
    language_model = getattr(model, "model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError(f"cannot locate Llama layers under {type(model).__name__}")
    if not 0 <= block_index < len(layers):
        raise RuntimeError(f"block index {block_index} is outside {len(layers)} layers")
    return layers[block_index]


def _install_timing_hook(model: Any, block_index: int) -> dict[str, Any]:
    import torch

    layer = _locate_layer(model, block_index)
    if hasattr(layer, HOOK_HANDLES_ATTRIBUTE):
        raise RuntimeError("timing hooks were already installed")
    setattr(layer, HOOK_EVENTS_ATTRIBUTE, [])
    setattr(layer, HOOK_PENDING_ATTRIBUTE, None)

    def before(module: Any, arguments: tuple[Any, ...]) -> None:
        if len(arguments) < 2:
            raise RuntimeError("Llama layer hook did not receive hidden_states")
        if getattr(module, HOOK_PENDING_ATTRIBUTE) is not None:
            raise RuntimeError("nested Llama layer timing invocation")
        hidden_states = arguments[1]
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        setattr(
            module,
            HOOK_PENDING_ATTRIBUTE,
            {
                "start": start,
                "token_rows": int(hidden_states.shape[0]),
                "hidden_shape": list(hidden_states.shape),
                "hidden_dtype": str(hidden_states.dtype),
            },
        )

    def after(module: Any, arguments: tuple[Any, ...], output: Any) -> None:
        pending = getattr(module, HOOK_PENDING_ATTRIBUTE)
        if pending is None:
            raise RuntimeError("Llama layer timing end event has no start event")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        pending["end"] = end
        getattr(module, HOOK_EVENTS_ATTRIBUTE).append(pending)
        setattr(module, HOOK_PENDING_ATTRIBUTE, None)

    handles = (
        layer.register_forward_pre_hook(before),
        layer.register_forward_hook(after),
    )
    setattr(layer, HOOK_HANDLES_ATTRIBUTE, handles)

    parameters = list(layer.named_parameters())
    buffers = list(layer.named_buffers())
    packed = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "bytes": int(parameter.numel() * parameter.element_size()),
        }
        for name, parameter in parameters
        if name.endswith("weight_packed")
    ]
    quant_methods = {}
    for name, module in layer.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_methods[name] = type(quant_method).__name__
    return {
        "model_class": type(model).__name__,
        "layer_class": type(layer).__name__,
        "block_index": block_index,
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for _, parameter in parameters
        ),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for _, buffer in buffers),
        "packed_parameters": packed,
        "packed_parameter_count": len(packed),
        "packed_parameter_bytes": sum(record["bytes"] for record in packed),
        "quant_method_classes": quant_methods,
    }


def _collect_timing_events(model: Any, block_index: int) -> list[dict[str, Any]]:
    import torch

    layer = _locate_layer(model, block_index)
    if getattr(layer, HOOK_PENDING_ATTRIBUTE, None) is not None:
        raise RuntimeError("cannot collect timing events during a layer invocation")
    torch.cuda.synchronize()
    pending = getattr(layer, HOOK_EVENTS_ATTRIBUTE, None)
    if pending is None:
        raise RuntimeError("timing hooks are not installed")
    records = []
    for event in pending:
        elapsed = float(event["start"].elapsed_time(event["end"]))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid CUDA event duration: {elapsed}")
        records.append(
            {
                "elapsed_ms": elapsed,
                "token_rows": event["token_rows"],
                "hidden_shape": event["hidden_shape"],
                "hidden_dtype": event["hidden_dtype"],
            }
        )
    pending.clear()
    return records


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("cannot summarize empty samples")
    ordered = sorted(samples)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "population_stddev": statistics.pstdev(samples),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "p90": ordered[p90_index],
    }


def _prompt_batch(
    *,
    batch_size: int,
    prompt_tokens: int,
    vocab_size: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    if vocab_size <= 4:
        raise ValueError("vocabulary is too small")
    prompts = []
    for batch_index in range(batch_size):
        ids = [1]
        ids.extend(
            3 + ((seed + batch_index * 977 + position * 37) % (vocab_size - 3))
            for position in range(prompt_tokens - 1)
        )
        prompts.append({"prompt_token_ids": ids})
    return prompts


def _validate_events(
    events: list[dict[str, Any]],
    case: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch_size = int(case["batch_size"])
    prompt_rows = batch_size * int(case["prompt_tokens"])
    prefill = [event for event in events if event["token_rows"] == prompt_rows]
    decode = [event for event in events if event["token_rows"] == batch_size]
    if case["phase"] == "prefill":
        if len(prefill) != 1 or decode:
            raise RuntimeError(
                f"{case['name']} expected one prefill event and no decode events: {events}"
            )
    else:
        if len(prefill) != 1 or len(decode) != int(case["decode_steps"]):
            raise RuntimeError(
                f"{case['name']} expected one prefill plus "
                f"{case['decode_steps']} decode events: {events}"
            )
    if len(prefill) + len(decode) != len(events):
        raise RuntimeError(f"{case['name']} returned unclassified layer events: {events}")
    return prefill, decode


def _run_request(
    llm: Any,
    sampling_params_class: Any,
    *,
    prompts: list[dict[str, list[int]]],
    case: dict[str, Any],
    block_index: int,
) -> tuple[list[dict[str, Any]], float, list[list[int]]]:
    max_tokens = 1 if case["phase"] == "prefill" else int(case["decode_steps"]) + 1
    params = sampling_params_class(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    generated = [list(output.outputs[0].token_ids) for output in outputs]
    if len(generated) != int(case["batch_size"]):
        raise RuntimeError(f"{case['name']} returned the wrong batch size")
    if any(len(tokens) != max_tokens for tokens in generated):
        raise RuntimeError(f"{case['name']} returned the wrong generated length")
    events_by_worker = llm.apply_model(
        functools.partial(_collect_timing_events, block_index=block_index)
    )
    if len(events_by_worker) != 1:
        raise RuntimeError(f"expected one vLLM worker, observed {len(events_by_worker)}")
    return events_by_worker[0], wall_ms, generated


def _run_case(
    llm: Any,
    sampling_params_class: Any,
    *,
    case: dict[str, Any],
    protocol: dict[str, Any],
    block_index: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    prompts = _prompt_batch(
        batch_size=int(case["batch_size"]),
        prompt_tokens=int(case["prompt_tokens"]),
        vocab_size=vocab_size,
        seed=seed,
    )
    for _ in range(int(protocol["warmup_requests"])):
        events, _, _ = _run_request(
            llm,
            sampling_params_class,
            prompts=prompts,
            case=case,
            block_index=block_index,
        )
        _validate_events(events, case)

    layer_call_samples: list[float] = []
    request_mean_samples: list[float] = []
    wall_samples: list[float] = []
    generated_hash_material = []
    event_shapes = set()
    for _ in range(int(protocol["repetitions"])):
        events, wall_ms, generated = _run_request(
            llm,
            sampling_params_class,
            prompts=prompts,
            case=case,
            block_index=block_index,
        )
        prefill, decode = _validate_events(events, case)
        selected = prefill if case["phase"] == "prefill" else decode
        selected_elapsed = [event["elapsed_ms"] for event in selected]
        layer_call_samples.extend(selected_elapsed)
        request_mean_samples.append(statistics.fmean(selected_elapsed))
        wall_samples.append(wall_ms)
        generated_hash_material.append(generated)
        event_shapes.update(
            (tuple(event["hidden_shape"]), event["hidden_dtype"]) for event in selected
        )

    work_items = (
        int(case["batch_size"]) * int(case["prompt_tokens"])
        if case["phase"] == "prefill"
        else int(case["batch_size"])
    )
    work_rate_samples = [
        work_items * 1000.0 / elapsed for elapsed in layer_call_samples
    ]
    return {
        "case": case,
        "warmup_requests": int(protocol["warmup_requests"]),
        "repetitions": int(protocol["repetitions"]),
        "layer_call_elapsed_ms_samples": layer_call_samples,
        "layer_call_elapsed_ms": _summary(layer_call_samples),
        "per_request_mean_layer_ms_samples": request_mean_samples,
        "per_request_mean_layer_ms": _summary(request_mean_samples),
        "full_model_generation_wall_ms_samples": wall_samples,
        "full_model_generation_wall_ms": _summary(wall_samples),
        "work_items_per_layer_call": work_items,
        "work_items_per_second_samples": work_rate_samples,
        "work_items_per_second": _summary(work_rate_samples),
        "selected_event_shapes": [
            {"shape": list(shape), "dtype": dtype}
            for shape, dtype in sorted(event_shapes)
        ],
        "generated_token_ids_by_repetition": generated_hash_material,
    }


def _run_model(
    *,
    model_name: str,
    model_path: Path,
    config: dict[str, Any],
    protocol_name: str,
) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("single-block timing requires an allocated CUDA GPU")
    runtime = config["runtime"]
    if metadata.version("vllm") != runtime["vllm_version"]:
        raise RuntimeError("vLLM version differs from the frozen protocol")
    if torch.__version__ != runtime["torch_version"]:
        raise RuntimeError("PyTorch version differs from the frozen protocol")
    if torch.version.cuda != runtime["cuda_runtime"]:
        raise RuntimeError("CUDA runtime differs from the frozen protocol")
    if torch.cuda.get_device_name(0) != runtime["gpu"]:
        raise RuntimeError("GPU differs from the frozen protocol")
    if list(torch.cuda.get_device_capability(0)) != runtime["compute_capability"]:
        raise RuntimeError("compute capability differs from the frozen protocol")
    if (
        os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
        != runtime["vllm_use_flashinfer_sampler"]
    ):
        raise RuntimeError("vLLM sampler fallback setting differs from the protocol")
    engine = config["engine"]
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    llm = LLM(
        model=str(model_path),
        dtype=str(engine["dtype"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        kv_cache_memory_bytes=int(engine["kv_cache_memory_bytes"]),
        seed=int(engine["seed"]),
        enforce_eager=bool(engine["enforce_eager"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        skip_tokenizer_init=bool(engine["skip_tokenizer_init"]),
        disable_log_stats=bool(engine["disable_log_stats"]),
    )
    inspections = llm.apply_model(
        functools.partial(
            _install_timing_hook,
            block_index=int(engine["block_index"]),
        )
    )
    if len(inspections) != 1:
        raise RuntimeError(f"expected one vLLM worker, observed {len(inspections)}")
    inspection = inspections[0]
    if model_name == "bf16" and inspection["packed_parameter_count"] != 0:
        raise RuntimeError("BF16 block unexpectedly contains packed W4 parameters")
    if model_name != "bf16" and inspection["packed_parameter_count"] == 0:
        raise RuntimeError(f"{model_name} block contains no packed W4 parameters")

    protocol = config[protocol_name]
    cases = {}
    for case_index, case in enumerate(protocol["cases"]):
        cases[case["name"]] = _run_case(
            llm,
            SamplingParams,
            case=case,
            protocol=protocol,
            block_index=int(engine["block_index"]),
            vocab_size=int(model_config["vocab_size"]),
            seed=int(engine["seed"]) + case_index * 100003,
        )
    return {
        "status": "passed",
        "model_name": model_name,
        "model_path": str(model_path),
        "protocol": protocol_name,
        "layer_inspection": inspection,
        "cases": cases,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "vllm": metadata.version("vllm"),
        },
    }


def _run_worker(
    *,
    script: Path,
    config_path: Path,
    protocol: str,
    model_name: str,
    model_path: Path,
    output: Path,
) -> dict[str, Any]:
    output.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--protocol",
            protocol,
            "--worker-model-name",
            model_name,
            "--worker-model-path",
            str(model_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["model_name"] != model_name or result["protocol"] != protocol:
        raise RuntimeError(f"worker result identity mismatch: {result}")
    return result


def _comparisons(
    models: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if "bf16" not in models:
        return {}
    comparisons = {}
    for case in config["formal"]["cases"]:
        case_name = case["name"]
        baseline = models["bf16"]["cases"][case_name]
        per_case = {}
        for model_name in EXPECTED_MODELS[1:]:
            candidate = models[model_name]["cases"][case_name]
            per_case[f"{model_name}_vs_bf16"] = {
                "median_layer_elapsed_ratio": (
                    candidate["layer_call_elapsed_ms"]["median"]
                    / baseline["layer_call_elapsed_ms"]["median"]
                ),
                "median_layer_speedup_bf16_over_candidate": (
                    baseline["layer_call_elapsed_ms"]["median"]
                    / candidate["layer_call_elapsed_ms"]["median"]
                ),
                "median_work_rate_ratio": (
                    candidate["work_items_per_second"]["median"]
                    / baseline["work_items_per_second"]["median"]
                ),
                "layer_parameter_bytes_ratio": (
                    models[model_name]["layer_inspection"]["parameter_bytes"]
                    / models["bf16"]["layer_inspection"]["parameter_bytes"]
                ),
            }
        comparisons[case_name] = per_case
    return comparisons


def _parent_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    protocol: str,
    models: dict[str, Path],
    worker_dir: Path,
) -> dict[str, Any]:
    worker_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name, model_path in models.items():
        results[model_name] = _run_worker(
            script=Path(__file__).resolve(),
            config_path=config_path,
            protocol=protocol,
            model_name=model_name,
            model_path=model_path,
            output=worker_dir / f"{model_name}.json",
        )
    runtimes = {
        json.dumps(result["runtime"], sort_keys=True) for result in results.values()
    }
    if len(runtimes) != 1:
        raise RuntimeError("single-block workers returned inconsistent runtimes")
    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "protocol": protocol,
        "source_gate": config["source_gate"],
        "engine": config["engine"],
        "runtime": json.loads(runtimes.pop()),
        "models": results,
        "comparisons": _comparisons(results, config),
    }


def _parse_models(values: list[str], expected: list[str]) -> dict[str, Path]:
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use name=path syntax: {value}")
        name, raw_path = value.split("=", 1)
        if name in models:
            raise ValueError(f"repeated model name: {name}")
        models[name] = Path(raw_path).resolve()
    if list(models) != expected:
        raise ValueError(f"expected model order {expected}, observed {list(models)}")
    for name, path in models.items():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{name} config is missing: {path / 'config.json'}")
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--worker-model-name")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    if args.worker_model_name is not None:
        if args.worker_model_path is None:
            parser.error("--worker-model-name requires --worker-model-path")
        result = _run_model(
            model_name=args.worker_model_name,
            model_path=args.worker_model_path.resolve(),
            config=config,
            protocol_name=args.protocol,
        )
        _write_json(args.output, result)
        print(f"VLLM_13B_SINGLE_BLOCK_MODEL_PASSED={args.worker_model_name}")
        return 0

    if not args.model or args.worker_dir is None:
        parser.error("parent mode requires --model and --worker-dir")
    models = _parse_models(args.model, config[args.protocol]["models"])
    result = _parent_run(
        config=config,
        config_path=args.config.resolve(),
        protocol=args.protocol,
        models=models,
        worker_dir=args.worker_dir.resolve(),
    )
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
