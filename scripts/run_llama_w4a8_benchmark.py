#!/usr/bin/env python3
"""Run matched QuaRot BF16 and packed-W4A8 deployment benchmarks on CUDA."""

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from torch import nn

from repro.quarot_pipeline import (
    apply_llama_quarot,
    load_model_and_tokenizer,
    load_pipeline_config,
    resolve_device,
    validate_pipeline_config,
)
from repro.w4a8_benchmark import compare_modes, summarize_samples, validate_benchmark_config
from repro.w4a8_checkpoint import install_checkpoint_linears, sha256_file


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON input must contain an object: {}".format(path))
    return value


def _effective_config(config_path: Path, correctness_result: str | None) -> Dict[str, Any]:
    config = _load_json(config_path)
    if correctness_result is not None:
        config["correctness_result"] = correctness_result
    validate_benchmark_config(config)
    return config


def _resolve_project_path(value: str) -> Path:
    path = PROJECT_ROOT / value
    if not path.is_file():
        raise RuntimeError("required project input is missing: {}".format(path))
    return path


def _validate_linked_evidence(config: Mapping[str, Any]) -> Dict[str, Any]:
    pipeline_path = _resolve_project_path(str(config["pipeline_config"]))
    manifest_path = _resolve_project_path(str(config["checkpoint_manifest"]))
    correctness_path = _resolve_project_path(str(config["correctness_result"]))
    pipeline = load_pipeline_config(pipeline_path)
    validate_pipeline_config(pipeline)
    manifest = _load_json(manifest_path)
    correctness = _load_json(correctness_path)
    if manifest.get("tensor_count") != 280:
        raise RuntimeError("benchmark requires a 280-linear checkpoint manifest")
    if not correctness.get("passed") or correctness.get("precision", {}).get("decoder_linear_count") != 280:
        raise RuntimeError("linked full-model correctness gate did not pass")
    manifest_hash = sha256_file(manifest_path)
    if correctness.get("source", {}).get("checkpoint_manifest_sha256") != manifest_hash:
        raise RuntimeError("correctness result does not reference the selected checkpoint manifest")
    if correctness.get("source", {}).get("config_sha256") != sha256_file(pipeline_path):
        raise RuntimeError("correctness result does not reference the selected pipeline config")
    if manifest.get("source", {}).get("model") != pipeline["model"] or correctness.get("model") != pipeline["model"]:
        raise RuntimeError("model provenance differs across benchmark inputs")
    if correctness.get("source", {}).get("project_revision") != manifest.get("source", {}).get("project_revision"):
        raise RuntimeError("checkpoint and correctness project revisions differ")
    return {
        "pipeline_path": pipeline_path,
        "manifest_path": manifest_path,
        "correctness_path": correctness_path,
        "pipeline": pipeline,
        "manifest_sha256": manifest_hash,
        "correctness_sha256": sha256_file(correctness_path),
    }


def _measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    work_items: int,
) -> Dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
            del output
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wall_start = time.perf_counter()
            output = function()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            samples.append(
                {
                    "cuda_ms": float(start.elapsed_time(end)),
                    "wall_ms": wall_ms,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
            del output
    return summarize_samples(samples, work_items)


def _measure_decode(
    model: nn.Module,
    prefix: torch.Tensor,
    next_token: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    samples = []
    with torch.inference_mode():
        for sample_index in range(warmup + repeats):
            prefix_output = model(input_ids=prefix, use_cache=True)
            past_key_values = prefix_output.past_key_values
            del prefix_output
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wall_start = time.perf_counter()
            output = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            if sample_index == 0 and not torch.isfinite(output.logits).all():
                raise RuntimeError("decode benchmark produced non-finite logits")
            if sample_index >= warmup:
                samples.append(
                    {
                        "cuda_ms": float(start.elapsed_time(end)),
                        "wall_ms": wall_ms,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    }
                )
            del output, past_key_values
    return summarize_samples(samples, int(next_token.shape[0]))


def _greedy_generate(
    model: nn.Module, input_ids: torch.Tensor, new_tokens: int, *, validate_finite: bool = False
) -> torch.Tensor:
    generated = input_ids
    next_input = input_ids
    past_key_values = None
    for _ in range(new_tokens):
        output = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
        if validate_finite and not torch.isfinite(output.logits).all():
            raise RuntimeError("generation benchmark produced non-finite logits")
        next_input = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_input), dim=1)
        past_key_values = output.past_key_values
    return generated


def _deterministic_input(tokens: int, width: int, dtype: torch.dtype, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((tokens, width), dtype=torch.float32, generator=generator).to(device=device, dtype=dtype)


def _benchmark_mode(
    mode: str,
    config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    pipeline = evidence["pipeline"]
    torch.manual_seed(int(config["seed"]))
    model, tokenizer = load_model_and_tokenizer(pipeline, device)
    rotation = apply_llama_quarot(
        model, pipeline["experiment"]["rotation"], pipeline["experiment"]["quantization"]
    )
    installed = []
    if mode == "w4a8":
        installed = install_checkpoint_linears(model, evidence["manifest_path"], implementation="cuda")
        if len(installed) != 280:
            raise RuntimeError("W4A8 benchmark did not install every decoder linear")
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    static_memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }
    result: Dict[str, Any] = {
        "mode": mode,
        "rotation": rotation,
        "decoder_linear_precision": "BF16" if mode == "bf16" else "packed GPTQ W4 with per-token A8",
        "embedding_lm_head_kv_precision": "BF16",
        "installed_decoder_linears": len(installed),
        "static_model_memory": static_memory,
        "linear": {},
        "prefill": {},
        "decode": {},
        "generation": {},
    }
    dtype = next(model.parameters()).dtype
    vocab_size = int(model.config.vocab_size)

    linear_config = config["linear"]
    for target_index, target_name in enumerate(linear_config["targets"]):
        module = model.get_submodule(target_name)
        in_features = int(getattr(module, "in_features", 0) or getattr(module, "packed_weight").shape[1] * 2)
        out_features = int(getattr(module, "out_features", 0) or getattr(module, "packed_weight").shape[0])
        for tokens in linear_config["token_counts"]:
            inputs = _deterministic_input(
                int(tokens), in_features, dtype, int(config["seed"]) + target_index * 10000 + int(tokens), device
            )
            with torch.inference_mode():
                check = module(inputs)
            if tuple(check.shape) != (int(tokens), out_features) or not torch.isfinite(check).all():
                raise RuntimeError("linear benchmark produced invalid output: {}".format(target_name))
            del check
            key = "{}|tokens={}".format(target_name, tokens)
            result["linear"][key] = _measure(
                lambda module=module, inputs=inputs: module(inputs),
                warmup=int(linear_config["warmup"]),
                repeats=int(linear_config["repeats"]),
                work_items=int(tokens),
            )
            result["linear"][key]["shape"] = {
                "tokens": int(tokens), "in_features": in_features, "out_features": out_features
            }
            print("BENCHMARK_PROGRESS mode={} section=linear case={}".format(mode, key), flush=True)
            del inputs

    prefill_config = config["prefill"]
    for sequence_length in prefill_config["sequence_lengths"]:
        input_ids = torch.arange(int(sequence_length), device=device, dtype=torch.long).view(1, -1) % vocab_size
        with torch.inference_mode():
            check = model(input_ids=input_ids, use_cache=False).logits
        if not torch.isfinite(check).all():
            raise RuntimeError("prefill benchmark produced non-finite logits")
        del check
        key = "batch=1|sequence={}".format(sequence_length)
        result["prefill"][key] = _measure(
            lambda model=model, input_ids=input_ids: model(input_ids=input_ids, use_cache=False),
            warmup=int(prefill_config["warmup"]),
            repeats=int(prefill_config["repeats"]),
            work_items=int(sequence_length),
        )
        print("BENCHMARK_PROGRESS mode={} section=prefill case={}".format(mode, key), flush=True)
        del input_ids

    decode_config = config["decode"]
    for context_length in decode_config["context_lengths"]:
        prefix = torch.arange(int(context_length), device=device, dtype=torch.long).view(1, -1) % vocab_size
        next_token = torch.tensor([[int(context_length) % vocab_size]], dtype=torch.long, device=device)
        key = "batch=1|context={}".format(context_length)
        result["decode"][key] = _measure_decode(
            model,
            prefix,
            next_token,
            warmup=int(decode_config["warmup"]),
            repeats=int(decode_config["repeats"]),
        )
        print("BENCHMARK_PROGRESS mode={} section=decode case={}".format(mode, key), flush=True)
        del prefix, next_token

    generation_config = config["generation"]
    for prompt_length in generation_config["prompt_lengths"]:
        input_ids = torch.arange(int(prompt_length), device=device, dtype=torch.long).view(1, -1) % vocab_size
        new_tokens = int(generation_config["new_tokens"])
        with torch.inference_mode():
            check = _greedy_generate(model, input_ids, new_tokens, validate_finite=True)
        if check.shape[1] != int(prompt_length) + new_tokens:
            raise RuntimeError("generation benchmark produced an invalid token count")
        del check
        key = "batch=1|prompt={}|new_tokens={}".format(prompt_length, new_tokens)
        result["generation"][key] = _measure(
            lambda model=model, input_ids=input_ids, new_tokens=new_tokens: _greedy_generate(
                model, input_ids, new_tokens
            ),
            warmup=int(generation_config["warmup"]),
            repeats=int(generation_config["repeats"]),
            work_items=new_tokens,
        )
        print("BENCHMARK_PROGRESS mode={} section=generation case={}".format(mode, key), flush=True)
        del input_ids

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["memory_after_release"] = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }
    return result


def run(
    config_path: Path,
    partial_path: Path | None = None,
    correctness_result: str | None = None,
) -> Dict[str, Any]:
    config = _effective_config(config_path, correctness_result)
    evidence = _validate_linked_evidence(config)
    if not torch.cuda.is_available():
        raise RuntimeError("deployment benchmarking requires NVIDIA CUDA; CPU or MPS fallback is not permitted")
    device = resolve_device(evidence["pipeline"]["runtime"])
    result: Dict[str, Any] = {
        "scope": "matched QuaRot BF16 versus full-decoder W4A8 with BF16 embedding/lm_head/KV; no KV4 claim",
        "status": "running",
        "source": {
            "benchmark_config": str(config_path),
            "benchmark_config_sha256": sha256_file(config_path),
            "pipeline_config": str(evidence["pipeline_path"]),
            "pipeline_config_sha256": sha256_file(evidence["pipeline_path"]),
            "checkpoint_manifest": str(evidence["manifest_path"]),
            "checkpoint_manifest_sha256": evidence["manifest_sha256"],
            "correctness_result": str(evidence["correctness_path"]),
            "correctness_result_sha256": evidence["correctness_sha256"],
            "project_revision": _project_revision(),
        },
        "protocol": config,
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "driver": _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "nvcc": _command_output(["/usr/local/cuda/bin/nvcc", "--version"]),
        },
        "modes": {},
    }
    for mode in config["modes"]:
        result["modes"][mode] = _benchmark_mode(mode, config, evidence, device)
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["comparison"] = compare_modes(result["modes"]["bf16"], result["modes"]["w4a8"])
    result["status"] = "complete"
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--correctness-result",
        help="Project-relative platform result override; workload settings remain config-driven",
    )
    args = parser.parse_args()
    try:
        config = _effective_config(args.config, args.correctness_result)
        evidence = _validate_linked_evidence(config)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "valid": True,
                        "manifest_sha256": evidence["manifest_sha256"],
                        "correctness_sha256": evidence["correctness_sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None:
            raise RuntimeError("--output is required unless --validate-only is used")
        partial_path = args.output.with_name(args.output.stem + ".partial.json")
        result = run(args.config, partial_path, args.correctness_result)
    except (OSError, RuntimeError, ValueError) as error:
        print("W4A8 BENCHMARK FAILED: {}".format(error), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
