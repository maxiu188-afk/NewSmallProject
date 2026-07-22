"""Pure validation and reporting helpers for the official QuaRot layer benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


EXPECTED_UPSTREAM_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
EXPECTED_MODEL_ID = "meta-llama/Llama-2-7b-hf"
EXPECTED_MODEL_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
EXPECTED_MODES = ["w4a4kv4", "fp16"]
EXPECTED_PATCHED_PATHS = [
    "e2e/quantized_llama/modeling_llama.py",
    "quarot/transformers/kv_cache.py",
    "setup.py",
]
EXPECTED_PATCHED_SHA256 = {
    "e2e/quantized_llama/modeling_llama.py": "dee2f59c9c74a2a4799402cfe91d8dfad8087eb0b25f8d9a1ae9aad04a488610",
    "quarot/transformers/kv_cache.py": "cd6a3a7544ae47f31b4db05ac25532883720e3100ace25e5db09d60896fe4e31",
    "setup.py": "31752374109751793e05a873137d849ca0472382175e1c322489deccca297a0e",
}
EXPECTED_SUBMODULES = {
    "third-party/cutlass": "ffa34e70756b0bc744e1dfcc115b5a991a68f132",
    "third-party/fast-hadamard-transform": "4ea722e434e3d4f2a14522341959ebdbe62be2de",
    "third-party/nvbench": "d8dced8a64d9ce305add92fa6d274fd49b569b7e",
}
VALID_METRICS = {"prefill", "decode", "e2e"}


class OfficialLayerBenchmarkError(ValueError):
    """Raised when the single-block experiment contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OfficialLayerBenchmarkError(f"{label} must be a positive integer")
    return value


def _case_rows(plan: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    protocols = plan["protocols"]
    for group in plan["groups"]:
        protocol_name = group["protocol"]
        if protocol_name not in protocols:
            raise OfficialLayerBenchmarkError(f"group {group.get('name')} names an unknown protocol")
        for case in group["cases"]:
            yield group["name"], protocols[protocol_name], case


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise OfficialLayerBenchmarkError("schema_version must be 1")
    source = plan.get("source", {})
    if source.get("upstream_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise OfficialLayerBenchmarkError("the pinned upstream QuaRot revision changed")
    if source.get("model_id") != EXPECTED_MODEL_ID or source.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise OfficialLayerBenchmarkError("the pinned Llama-2-7B source changed")
    if sorted(source.get("expected_patched_paths", [])) != EXPECTED_PATCHED_PATHS:
        raise OfficialLayerBenchmarkError(
            "expected patched paths must contain only the SM89, cache, and RoPE adapters"
        )
    if source.get("expected_patched_sha256") != EXPECTED_PATCHED_SHA256:
        raise OfficialLayerBenchmarkError("expected patched-file hashes changed")
    if source.get("expected_submodules") != EXPECTED_SUBMODULES:
        raise OfficialLayerBenchmarkError("expected upstream submodule revisions changed")
    required_patches = source.get("required_compatibility_patches", [])
    if required_patches != [
        "patches/quarot-sm89-build.patch",
        "patches/quarot-transformers-4.38-cache.patch",
        "patches/quarot-transformers-4.38-rope.patch",
    ]:
        raise OfficialLayerBenchmarkError("required compatibility patches changed")
    if plan.get("modes") != EXPECTED_MODES:
        raise OfficialLayerBenchmarkError("mode order must match the upstream INT4 then FP16 benchmark")

    target = plan.get("target", {})
    if target.get("gpu_name") != "NVIDIA RTX 6000 Ada Generation" or target.get(
        "compute_capability"
    ) != [8, 9]:
        raise OfficialLayerBenchmarkError("this plan is restricted to one NVIDIA RTX 6000 Ada")
    if target.get("torch") != "2.2.1+cu121" or target.get("torch_cuda") != "12.1":
        raise OfficialLayerBenchmarkError("the official environment must use the pinned cu121 PyTorch pair")
    if target.get("transformers") != "4.38.0":
        raise OfficialLayerBenchmarkError("the upstream Transformers pin changed")
    if target.get("flash_attn") != "2.5.6":
        raise OfficialLayerBenchmarkError("the FlashAttention compatibility pin changed")

    protocols = plan.get("protocols")
    groups = plan.get("groups")
    if not isinstance(protocols, dict) or not isinstance(groups, list) or not groups:
        raise OfficialLayerBenchmarkError("protocols and non-empty groups are required")
    for name, protocol in protocols.items():
        for field in ("warmup_steps", "bench_steps", "repetitions"):
            _positive_int(protocol.get(field), f"protocol {name}.{field}")

    group_names: list[str] = []
    case_ids: list[str] = []
    for group in groups:
        name = group.get("name")
        if not isinstance(name, str) or not name or name in group_names:
            raise OfficialLayerBenchmarkError("group names must be unique non-empty strings")
        group_names.append(name)
        if not isinstance(group.get("cases"), list) or not group["cases"]:
            raise OfficialLayerBenchmarkError(f"group {name} must contain cases")
        for case in group["cases"]:
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise OfficialLayerBenchmarkError("case ids must be unique non-empty strings")
            case_ids.append(case_id)
            metrics = case.get("metrics")
            if not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics)):
                raise OfficialLayerBenchmarkError(f"case {case_id} metrics must be unique and non-empty")
            if not set(metrics).issubset(VALID_METRICS):
                raise OfficialLayerBenchmarkError(f"case {case_id} contains an unsupported metric")
            _positive_int(case.get("batch_size"), f"case {case_id}.batch_size")
            _positive_int(case.get("prefill_tokens"), f"case {case_id}.prefill_tokens")
            _positive_int(case.get("seed"), f"case {case_id}.seed")
            decode_tokens = case.get("decode_tokens")
            if not isinstance(decode_tokens, int) or isinstance(decode_tokens, bool) or decode_tokens < 0:
                raise OfficialLayerBenchmarkError(f"case {case_id}.decode_tokens must be a non-negative integer")
            if ({"decode", "e2e"} & set(metrics)) and decode_tokens <= 0:
                raise OfficialLayerBenchmarkError(f"case {case_id} needs decode_tokens for decode/e2e")

    if group_names != ["smoke", "paper_prefill", "paper_decode_memory"]:
        raise OfficialLayerBenchmarkError("the expected smoke/prefill/decode group order changed")
    official = protocols.get("official", {})
    if official != {"warmup_steps": 3, "bench_steps": 10, "repetitions": 10}:
        raise OfficialLayerBenchmarkError("the official 3 warm-up, 10-step, 10-repeat protocol changed")

    prefill_cases = next(group["cases"] for group in groups if group["name"] == "paper_prefill")
    observed_prefill = [(case["batch_size"], case["prefill_tokens"], case["metrics"]) for case in prefill_cases]
    expected_prefill = [(batch, 2048, ["prefill"]) for batch in (1, 4, 16, 64)]
    if observed_prefill != expected_prefill:
        raise OfficialLayerBenchmarkError("paper prefill grid must be sequence 2048 and batch 1/4/16/64")

    decode_cases = next(group["cases"] for group in groups if group["name"] == "paper_decode_memory")
    observed_decode = [
        (case["batch_size"], case["prefill_tokens"], case["decode_tokens"], case["metrics"])
        for case in decode_cases
    ]
    expected_decode = [
        (batch, length, 50, ["decode", "e2e"])
        for batch in (1, 16)
        for length in (256, 512, 1024, 2048, 4096)
    ]
    if observed_decode != expected_decode:
        raise OfficialLayerBenchmarkError("paper decode/memory grid changed")


def select_groups(plan: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    available = {group["name"]: group for group in plan["groups"]}
    names = requested or list(available)
    if len(names) != len(set(names)):
        raise OfficialLayerBenchmarkError("group selection contains duplicates")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise OfficialLayerBenchmarkError(f"unknown benchmark groups: {', '.join(unknown)}")
    return [available[name] for name in names]


def summarize_samples(elapsed_ms: list[float], peak_allocated_bytes: list[int], work_items: int) -> dict[str, Any]:
    if not elapsed_ms or len(elapsed_ms) != len(peak_allocated_bytes):
        raise OfficialLayerBenchmarkError("timing and memory samples must be non-empty and matched")
    if work_items <= 0:
        raise OfficialLayerBenchmarkError("work_items must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in elapsed_ms):
        raise OfficialLayerBenchmarkError("timing samples must be finite and positive")
    if any(not isinstance(value, int) or value <= 0 for value in peak_allocated_bytes):
        raise OfficialLayerBenchmarkError("peak-memory samples must be positive integers")
    return {
        "sample_count": len(elapsed_ms),
        "elapsed_ms_samples": elapsed_ms,
        "peak_allocated_bytes_samples": peak_allocated_bytes,
        "elapsed_ms": {
            "mean": statistics.fmean(elapsed_ms),
            "median": statistics.median(elapsed_ms),
            "minimum": min(elapsed_ms),
            "maximum": max(elapsed_ms),
        },
        "peak_allocated_bytes": {
            "mean": statistics.fmean(peak_allocated_bytes),
            "maximum": max(peak_allocated_bytes),
        },
        "work_items_per_second_from_mean": work_items * 1000.0 / statistics.fmean(elapsed_ms),
    }


def compare_case(fp16: dict[str, Any], w4a4kv4: dict[str, Any]) -> dict[str, Any]:
    if set(fp16) != set(w4a4kv4):
        raise OfficialLayerBenchmarkError("mode metric sets differ")
    comparison: dict[str, Any] = {}
    for metric in fp16:
        fp_metric = fp16[metric]
        w4_metric = w4a4kv4[metric]
        comparison[metric] = {
            "speedup_fp16_over_w4a4kv4_mean": fp_metric["elapsed_ms"]["mean"]
            / w4_metric["elapsed_ms"]["mean"],
            "speedup_fp16_over_w4a4kv4_median": fp_metric["elapsed_ms"]["median"]
            / w4_metric["elapsed_ms"]["median"],
            "memory_saving_fp16_over_w4a4kv4_mean": fp_metric["peak_allocated_bytes"]["mean"]
            / w4_metric["peak_allocated_bytes"]["mean"],
        }
    return comparison
