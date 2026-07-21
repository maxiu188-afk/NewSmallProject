"""Configuration and result helpers for matched BF16/W4A8 benchmarks."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, Mapping, Sequence


class W4A8BenchmarkError(ValueError):
    """Raised when a deployment benchmark configuration is unsafe or incomplete."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise W4A8BenchmarkError("{} must be a positive integer".format(name))
    return value


def _positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise W4A8BenchmarkError("{} must be a non-empty list".format(name))
    result = [_positive_int(item, "{} entry".format(name)) for item in value]
    if result != sorted(set(result)):
        raise W4A8BenchmarkError("{} must be sorted and unique".format(name))
    return result


def validate_benchmark_config(config: Mapping[str, Any]) -> None:
    required = {
        "benchmark_version",
        "pipeline_config",
        "checkpoint_manifest",
        "correctness_result",
        "seed",
        "modes",
        "linear",
        "prefill",
        "decode",
        "generation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise W4A8BenchmarkError("benchmark config misses fields: {}".format(", ".join(missing)))
    if config["benchmark_version"] != 1:
        raise W4A8BenchmarkError("unsupported benchmark_version")
    for name in ("pipeline_config", "checkpoint_manifest", "correctness_result"):
        value = config[name]
        if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
            raise W4A8BenchmarkError("{} must be a project-relative POSIX path".format(name))
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise W4A8BenchmarkError("seed must be an integer")
    if config["modes"] != ["bf16", "w4a8"]:
        raise W4A8BenchmarkError("modes must be exactly ['bf16', 'w4a8'] for a matched comparison")

    linear = config["linear"]
    if not isinstance(linear, Mapping):
        raise W4A8BenchmarkError("linear must be an object")
    targets = linear.get("targets")
    if not isinstance(targets, list) or not targets or any(not isinstance(item, str) or not item for item in targets):
        raise W4A8BenchmarkError("linear.targets must be a non-empty string list")
    if len(targets) != len(set(targets)):
        raise W4A8BenchmarkError("linear.targets must be unique")
    _positive_int_list(linear.get("token_counts"), "linear.token_counts")

    for section_name, grid_name in (
        ("prefill", "sequence_lengths"),
        ("decode", "context_lengths"),
        ("generation", "prompt_lengths"),
    ):
        section = config[section_name]
        if not isinstance(section, Mapping):
            raise W4A8BenchmarkError("{} must be an object".format(section_name))
        _positive_int_list(section.get(grid_name), "{}.{}".format(section_name, grid_name))
        _positive_int(section.get("warmup"), "{}.warmup".format(section_name))
        _positive_int(section.get("repeats"), "{}.repeats".format(section_name))
    _positive_int(linear.get("warmup"), "linear.warmup")
    _positive_int(linear.get("repeats"), "linear.repeats")
    _positive_int(config["generation"].get("new_tokens"), "generation.new_tokens")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise W4A8BenchmarkError("cannot summarize an empty sample list")
    if not 0.0 <= fraction <= 1.0:
        raise W4A8BenchmarkError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: Iterable[Mapping[str, Any]], work_items: int) -> Dict[str, Any]:
    rows = [dict(sample) for sample in samples]
    if not rows:
        raise W4A8BenchmarkError("cannot summarize empty benchmark samples")
    _positive_int(work_items, "work_items")
    cuda_ms = [float(row["cuda_ms"]) for row in rows]
    wall_ms = [float(row["wall_ms"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in cuda_ms + wall_ms):
        raise W4A8BenchmarkError("benchmark samples must contain finite positive timings")
    peak_allocated = [int(row["peak_allocated_bytes"]) for row in rows]
    peak_reserved = [int(row["peak_reserved_bytes"]) for row in rows]
    mean_cuda = statistics.fmean(cuda_ms)
    return {
        "samples": rows,
        "sample_count": len(rows),
        "cuda_ms": {
            "mean": mean_cuda,
            "median": statistics.median(cuda_ms),
            "p95": percentile(cuda_ms, 0.95),
            "minimum": min(cuda_ms),
            "maximum": max(cuda_ms),
            "stdev": statistics.stdev(cuda_ms) if len(cuda_ms) > 1 else 0.0,
        },
        "wall_ms": {
            "mean": statistics.fmean(wall_ms),
            "median": statistics.median(wall_ms),
            "p95": percentile(wall_ms, 0.95),
            "minimum": min(wall_ms),
            "maximum": max(wall_ms),
        },
        "work_items": work_items,
        "work_items_per_second": work_items / (mean_cuda / 1000.0),
        "peak_allocated_bytes": max(peak_allocated),
        "peak_reserved_bytes": max(peak_reserved),
    }


def compare_modes(bf16: Mapping[str, Any], w4a8: Mapping[str, Any]) -> Dict[str, Any]:
    """Build matched speed and memory ratios for each benchmark case."""
    comparison: Dict[str, Any] = {}
    for section in ("linear", "prefill", "decode", "generation"):
        bf16_cases = bf16.get(section)
        w4a8_cases = w4a8.get(section)
        if not isinstance(bf16_cases, Mapping) or not isinstance(w4a8_cases, Mapping):
            raise W4A8BenchmarkError("missing matched {} benchmark results".format(section))
        if set(bf16_cases) != set(w4a8_cases):
            raise W4A8BenchmarkError("{} benchmark case sets differ".format(section))
        section_result = {}
        for name in sorted(bf16_cases):
            baseline = bf16_cases[name]
            candidate = w4a8_cases[name]
            baseline_ms = float(baseline["cuda_ms"]["mean"])
            candidate_ms = float(candidate["cuda_ms"]["mean"])
            section_result[name] = {
                "bf16_cuda_ms_mean": baseline_ms,
                "w4a8_cuda_ms_mean": candidate_ms,
                "w4a8_speedup_over_bf16": baseline_ms / candidate_ms,
                "bf16_peak_allocated_bytes": int(baseline["peak_allocated_bytes"]),
                "w4a8_peak_allocated_bytes": int(candidate["peak_allocated_bytes"]),
                "w4a8_peak_allocated_ratio_to_bf16": int(candidate["peak_allocated_bytes"])
                / int(baseline["peak_allocated_bytes"]),
            }
        comparison[section] = section_result
    return comparison
