#!/usr/bin/env python3
"""Run the paper-aligned official QuaRot single-decoder-block benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.upstream_quarot_layer_benchmark import (  # noqa: E402
    EXPECTED_PATCHED_PATHS,
    OfficialLayerBenchmarkError,
    compare_case,
    load_plan,
    select_groups,
    sha256_file,
    summarize_samples,
)


class CaseModeOutOfMemory(RuntimeError):
    """Carry the exact metric that exceeded GPU capacity without aborting the matrix."""

    def __init__(self, metric: str, message: str) -> None:
        super().__init__(message)
        self.metric = metric


def _run_command(command: list[str], cwd: Path) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.rstrip()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_evidence(plan: dict[str, Any], source: Path, plan_path: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise OfficialLayerBenchmarkError(f"upstream QuaRot checkout is missing: {source}")
    revision = _run_command(["git", "rev-parse", "HEAD"], source)
    if revision != plan["source"]["upstream_commit"]:
        raise OfficialLayerBenchmarkError(f"unexpected upstream revision: {revision}")
    patched_paths = sorted(
        line for line in _run_command(["git", "diff", "--name-only"], source).splitlines() if line
    )
    if patched_paths != EXPECTED_PATCHED_PATHS:
        raise OfficialLayerBenchmarkError(
            "upstream tracked changes must be exactly the SM89, cache, and RoPE adapters; observed {}".format(
                patched_paths
            )
        )
    patched_hashes = {relative: sha256_file(source / relative) for relative in patched_paths}
    if patched_hashes != plan["source"]["expected_patched_sha256"]:
        raise OfficialLayerBenchmarkError(
            f"patched upstream file hashes do not match the frozen adapters: {patched_hashes}"
        )
    submodule_status = _run_command(["git", "submodule", "status", "--recursive"], source).splitlines()
    if not submodule_status or any(not line.startswith(" ") for line in submodule_status):
        raise OfficialLayerBenchmarkError("upstream submodules are missing or not at their recorded revisions")
    observed_submodules = {}
    for line in submodule_status:
        fields = line[1:].split()
        if len(fields) >= 2:
            observed_submodules[fields[1]] = fields[0]
    if observed_submodules != plan["source"]["expected_submodules"]:
        raise OfficialLayerBenchmarkError(f"upstream submodule revisions changed: {observed_submodules}")
    patch_records = []
    for relative in plan["source"]["required_compatibility_patches"]:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise OfficialLayerBenchmarkError(f"required compatibility patch is missing: {relative}")
        patch_records.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "project_revision": _run_command(["git", "rev-parse", "HEAD"], PROJECT_ROOT),
        "project_status": _run_command(["git", "status", "--short"], PROJECT_ROOT).splitlines(),
        "upstream_revision": revision,
        "upstream_tracked_changes": patched_paths,
        "upstream_patched_sha256": patched_hashes,
        "upstream_submodules": submodule_status,
        "compatibility_patches": patch_records,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
    }


def _runtime_evidence(plan: dict[str, Any], torch: Any, transformers: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise OfficialLayerBenchmarkError("official QuaRot layer timing requires NVIDIA CUDA; no fallback is allowed")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    target = plan["target"]
    try:
        flash_attn_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError as error:
        raise OfficialLayerBenchmarkError("flash-attn is not installed in the benchmark environment") from error
    observed = {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn_version,
        "python": sys.version,
    }
    checks = {
        "gpu_matches": observed["gpu"] == target["gpu_name"],
        "compute_capability_matches": observed["compute_capability"] == target["compute_capability"],
        "torch_matches": observed["torch"] == target["torch"],
        "torch_cuda_matches": observed["torch_cuda"] == target["torch_cuda"],
        "transformers_matches": observed["transformers"] == target["transformers"],
        "flash_attn_matches": observed["flash_attn"] == target["flash_attn"],
    }
    if not all(checks.values()):
        raise OfficialLayerBenchmarkError(f"runtime does not match the frozen RTX 6000 Ada plan: {observed}")
    return {"observed": observed, "checks": checks}


def _cleanup(torch: Any) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _build_cache(
    quarot: Any,
    torch: Any,
    batch_size: int,
    length: int,
    *,
    disable_quant: bool,
    num_key_value_heads: int,
    hidden_size: int,
) -> Any:
    return quarot.transformers.MultiLayerPagedKVCache4Bit(
        batch_size=batch_size,
        page_size=length,
        max_seq_len=length,
        device=torch.device("cuda:0"),
        n_layers=1,
        num_heads=num_key_value_heads,
        head_dim=hidden_size // num_key_value_heads,
        disable_quant=disable_quant,
        hadamard_dtype=None if disable_quant else torch.float16,
    )


def _load_layer(
    mode: str,
    plan: dict[str, Any],
    torch: Any,
    transformers: Any,
    modeling_llama: Any,
    quarot: Any,
    local_files_only: bool,
) -> tuple[Any, Callable[[int, int], Any], int, int]:
    source = plan["source"]
    common = {
        "revision": source["model_revision"],
        "local_files_only": local_files_only,
    }
    if mode == "w4a4kv4":
        config = transformers.AutoConfig.from_pretrained(
            source["model_id"], attn_implementation="flash_attention_2", **common
        )
        config._attn_implementation = "flash_attention_2"
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            with transformers.modeling_utils.no_init_weights():
                model = modeling_llama.QuarotLlamaForCausalLM(config=config)
        finally:
            torch.set_default_dtype(previous_dtype)
        disable_quant = False
    elif mode == "fp16":
        model = modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
            source["model_id"],
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            **common,
        )
        config = model.config
        disable_quant = True
    else:
        raise OfficialLayerBenchmarkError(f"unsupported mode: {mode}")

    layer = model.model.layers[0]
    hidden_size = int(config.hidden_size)
    num_key_value_heads = int(config.num_key_value_heads)
    del model
    _cleanup(torch)
    layer = layer.to(torch.device("cuda:0")).eval()
    cache_builder = functools.partial(
        _build_cache,
        quarot,
        torch,
        disable_quant=disable_quant,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
    )
    return layer, cache_builder, hidden_size, num_key_value_heads


def _benchmark_callable(
    callable_: Callable[[], Any], protocol: dict[str, int], torch: Any, work_items: int
) -> dict[str, Any]:
    elapsed_samples: list[float] = []
    memory_samples: list[int] = []
    for _ in range(protocol["repetitions"]):
        output = None
        for _ in range(protocol["warmup_steps"]):
            output = callable_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(protocol["bench_steps"]):
            output = callable_()
        torch.cuda.synchronize()
        elapsed_samples.append((time.perf_counter() - started) * 1000.0 / protocol["bench_steps"])
        memory_samples.append(torch.cuda.max_memory_allocated())
        del output
    return summarize_samples(elapsed_samples, memory_samples, work_items)


def _position_ids(torch: Any, batch_size: int, start: int, length: int, device: Any) -> Any:
    return torch.arange(start, start + length, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)


def _layer_device(layer: Any) -> Any:
    for tensor in layer.parameters():
        return tensor.device
    for tensor in layer.buffers():
        return tensor.device
    raise OfficialLayerBenchmarkError("decoder layer has no parameters or buffers")


def _run_prefill(
    layer: Any,
    cache_builder: Callable[[int, int], Any],
    case: dict[str, Any],
    protocol: dict[str, int],
    hidden_size: int,
    torch: Any,
) -> dict[str, Any]:
    batch_size, prefill = case["batch_size"], case["prefill_tokens"]
    device = _layer_device(layer)
    inputs = torch.rand((batch_size, prefill, hidden_size), dtype=torch.float16, device=device)
    positions = _position_ids(torch, batch_size, 0, prefill, device)
    cache = cache_builder(batch_size, prefill)

    def prefill_once() -> Any:
        cache.length = 0
        cache._needs_init[0] = True
        return layer(inputs, position_ids=positions, past_key_value=cache)

    return _benchmark_callable(prefill_once, protocol, torch, batch_size * prefill)


def _run_decode(
    layer: Any,
    cache_builder: Callable[[int, int], Any],
    case: dict[str, Any],
    protocol: dict[str, int],
    hidden_size: int,
    torch: Any,
) -> dict[str, Any]:
    batch_size, prefill, decode = case["batch_size"], case["prefill_tokens"], case["decode_tokens"]
    device = _layer_device(layer)
    inputs = torch.rand((batch_size, prefill, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((batch_size, 1, hidden_size), dtype=torch.float16, device=device)
    prefill_positions = _position_ids(torch, batch_size, 0, prefill, device)
    cache = cache_builder(batch_size, prefill + decode)
    layer(inputs, position_ids=prefill_positions, past_key_value=cache)

    def decode_once() -> Any:
        cache.length = prefill
        output = None
        for index in range(decode):
            positions = _position_ids(torch, batch_size, prefill + index, 1, device)
            output = layer(next_input, position_ids=positions, past_key_value=cache)
        return output

    result = _benchmark_callable(decode_once, protocol, torch, batch_size * decode)
    result["elapsed_ms_per_decode_token_from_mean"] = result["elapsed_ms"]["mean"] / decode
    return result


def _run_e2e(
    layer: Any,
    cache_builder: Callable[[int, int], Any],
    case: dict[str, Any],
    protocol: dict[str, int],
    hidden_size: int,
    torch: Any,
) -> dict[str, Any]:
    batch_size, prefill, decode = case["batch_size"], case["prefill_tokens"], case["decode_tokens"]
    device = _layer_device(layer)
    inputs = torch.rand((batch_size, prefill, hidden_size), dtype=torch.float16, device=device)
    next_input = torch.rand((batch_size, 1, hidden_size), dtype=torch.float16, device=device)
    prefill_positions = _position_ids(torch, batch_size, 0, prefill, device)
    cache = cache_builder(batch_size, prefill + decode)

    def e2e_once() -> Any:
        cache.length = 0
        cache._needs_init[0] = True
        output = layer(inputs, position_ids=prefill_positions, past_key_value=cache)
        for index in range(decode):
            positions = _position_ids(torch, batch_size, prefill + index, 1, device)
            output = layer(next_input, position_ids=positions, past_key_value=cache)
        return output

    return _benchmark_callable(e2e_once, protocol, torch, batch_size * (prefill + decode))


def _case_result(
    layer: Any,
    cache_builder: Callable[[int, int], Any],
    case: dict[str, Any],
    protocol: dict[str, int],
    hidden_size: int,
    torch: Any,
) -> dict[str, Any]:
    torch.manual_seed(case["seed"])
    torch.cuda.manual_seed_all(case["seed"])
    runners = {"prefill": _run_prefill, "decode": _run_decode, "e2e": _run_e2e}
    result = {}
    for metric in case["metrics"]:
        try:
            result[metric] = runners[metric](
                layer, cache_builder, case, protocol, hidden_size, torch
            )
        except torch.cuda.OutOfMemoryError as error:
            raise CaseModeOutOfMemory(metric, str(error)) from error
        _cleanup(torch)
    return result


def _completion_status(
    cases: dict[str, dict[str, Any]], expected_modes: list[str]
) -> tuple[str, int]:
    expected = set(expected_modes)
    for case_id, case in cases.items():
        observed = set(case.get("modes", {})) | set(case.get("oom_by_mode", {}))
        if observed != expected:
            raise OfficialLayerBenchmarkError(
                f"case {case_id} is incomplete: observed modes {sorted(observed)}, "
                f"expected {sorted(expected)}"
            )
    oom_count = sum(len(case.get("oom_by_mode", {})) for case in cases.values())
    return ("completed_with_oom" if oom_count else "passed", oom_count)


def execute(
    plan: dict[str, Any],
    plan_path: Path,
    source: Path,
    groups: list[dict[str, Any]],
    output: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(source.resolve()))
    import torch
    import transformers
    import quarot
    from e2e.quantized_llama import modeling_llama

    evidence = _source_evidence(plan, source, plan_path)
    runtime = _runtime_evidence(plan, torch, transformers)
    selected_cases = [
        (group["name"], plan["protocols"][group["protocol"]], case)
        for group in groups
        for case in group["cases"]
    ]
    result: dict[str, Any] = {
        "status": "running",
        "scope": plan["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": evidence,
        "runtime": runtime,
        "selected_groups": [group["name"] for group in groups],
        "mode_order": plan["modes"],
        "paper_alignment": {
            "performance_scope": "one Llama-2 decoder transformer block",
            "paper_gpu": "NVIDIA RTX 3090",
            "run_gpu": plan["target"]["gpu_name"],
            "prefill_primary": "sequence length 2048 at batch 1/4/16/64",
            "decode_memory_primary": (
                "50 decode tokens at batch 1/16 and context 256/512/1024/2048/4096"
            ),
            "layer_e2e": "additional output from the official layer benchmark path, not full-model e2e",
            "table17_caption_note": (
                "the paper table says single token while Figure 4 says decoding 50 tokens; this plan follows Figure 4"
            ),
        },
        "cases": {
            case["id"]: {
                "group": group_name,
                "protocol": protocol,
                "input": case,
                "modes": {},
            }
            for group_name, protocol, case in selected_cases
        },
        "checks": {
            "single_decoder_block_only": True,
            "official_upstream_backend": True,
            "paper_gpu_exact": False,
            "full_model": False,
        },
    }
    _atomic_write(output, result)
    try:
        for mode in plan["modes"]:
            layer, cache_builder, hidden_size, num_key_value_heads = _load_layer(
                mode, plan, torch, transformers, modeling_llama, quarot, local_files_only
            )
            packed_linears = sum(isinstance(module, quarot.nn.Linear4bit) for module in layer.modules())
            expected_linears = 7 if mode == "w4a4kv4" else 0
            if packed_linears != expected_linears:
                raise OfficialLayerBenchmarkError(
                    f"{mode} layer contains {packed_linears} packed linears; expected {expected_linears}"
                )
            for _group_name, protocol, case in selected_cases:
                case_payload = result["cases"][case["id"]]
                case_payload["layer"] = {
                    "hidden_size": hidden_size,
                    "num_key_value_heads": num_key_value_heads,
                    "packed_linear_count_by_mode": case_payload.get("layer", {}).get(
                        "packed_linear_count_by_mode", {}
                    ),
                }
                case_payload["layer"]["packed_linear_count_by_mode"][mode] = packed_linears
                try:
                    case_payload["modes"][mode] = _case_result(
                        layer, cache_builder, case, protocol, hidden_size, torch
                    )
                except CaseModeOutOfMemory as error:
                    case_payload.setdefault("oom_by_mode", {})[mode] = {
                        "type": "OutOfMemoryError",
                        "metric": error.metric,
                        "message": str(error),
                    }
                    _cleanup(torch)
                if set(case_payload["modes"]) == set(plan["modes"]):
                    case_payload["comparison"] = compare_case(
                        case_payload["modes"]["fp16"], case_payload["modes"]["w4a4kv4"]
                    )
                _atomic_write(output, result)
            del layer, cache_builder
            _cleanup(torch)
        result["status"], result["oom_case_mode_count"] = _completion_status(
            result["cases"], plan["modes"]
        )
        result["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_write(output, result)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _atomic_write(output, result)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "configs/deployment/quarot_official_single_block_rtx6000ada.json",
    )
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "QuaRot-single-block")
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        plan = load_plan(args.plan)
        groups = select_groups(plan, args.groups)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "plan_valid",
                        "plan": str(args.plan),
                        "groups": [group["name"] for group in groups],
                        "case_count": sum(len(group["cases"]) for group in groups),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None:
            parser.error("--output is required unless --validate-only is used")
        if args.output.exists() and not args.overwrite:
            raise OfficialLayerBenchmarkError(f"output already exists: {args.output}")
        result = execute(
            plan,
            args.plan,
            args.source,
            groups,
            args.output,
            args.local_files_only,
        )
        print(f"OFFICIAL_QUAROT_SINGLE_BLOCK_STATUS={result['status']}")
        print(f"OFFICIAL_QUAROT_SINGLE_BLOCK_RESULT={args.output.resolve()}")
        return 0 if result["status"] == "passed" else 1
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"OFFICIAL QUAROT SINGLE-BLOCK BENCHMARK FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
