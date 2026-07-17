#!/usr/bin/env python3
"""Verify and summarize a matched LLaMA RTN W4A4 control/candidate pair."""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping


class ComparisonError(ValueError):
    """Raised when two result files do not form a fair RTN W4A4 comparison."""


def _load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ComparisonError("{} must contain a JSON object".format(path))
    return value


def _nested(result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = result.get(name)
    if not isinstance(value, Mapping):
        raise ComparisonError("result is missing object {}".format(name))
    return value


def _require_equal(left: Mapping[str, Any], right: Mapping[str, Any], name: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if left.get(key) != right.get(key):
            raise ComparisonError("{} differs for {}: {!r} != {!r}".format(name, key, left.get(key), right.get(key)))


def compare(naive: Mapping[str, Any], quarot: Mapping[str, Any]) -> Dict[str, Any]:
    naive_model, quarot_model = _nested(naive, "model"), _nested(quarot, "model")
    naive_data, quarot_data = _nested(naive, "data"), _nested(quarot, "data")
    naive_experiment, quarot_experiment = _nested(naive, "experiment"), _nested(quarot, "experiment")
    _require_equal(naive_model, quarot_model, "model", ("kind", "id", "revision", "dtype"))
    _require_equal(
        naive_data,
        quarot_data,
        "data",
        ("source", "id", "subset", "split", "revision", "sequence_length", "batch_size", "max_samples", "batches"),
    )
    _require_equal(naive_experiment, quarot_experiment, "experiment", ("seed",))

    naive_quantization, quarot_quantization = _nested(naive, "quantization"), _nested(quarot, "quantization")
    expected_quantization = {"w_bits": 4, "a_bits": 4, "k_bits": 16, "v_bits": 16}
    for label, quantization in (("naive", naive_quantization), ("quarot", quarot_quantization)):
        for key, expected in expected_quantization.items():
            if quantization.get(key) != expected:
                raise ComparisonError("{} result has {}={!r}; expected {!r}".format(label, key, quantization.get(key), expected))
    naive_rotation, quarot_rotation = _nested(naive, "rotation"), _nested(quarot, "rotation")
    if naive_rotation.get("reason") != "residual_mode=none":
        raise ComparisonError("naive result is not the unrotated control")
    if not quarot_rotation.get("applied") or quarot_rotation.get("residual_mode") != "hadamard":
        raise ComparisonError("QuaRot result is not a Hadamard-rotated candidate")
    if naive.get("kv_cache_simulated") or quarot.get("kv_cache_simulated"):
        raise ComparisonError("F3/F4 RTN W4A4 must leave KV quantization disabled")

    def metrics(result: Mapping[str, Any], label: str) -> Mapping[str, Any]:
        candidate, error = _nested(result, "candidate"), _nested(result, "logit_error")
        for key in ("perplexity", "mean_nll", "tokens"):
            if not math.isfinite(float(candidate.get(key, math.nan))):
                raise ComparisonError("{} candidate {} is not finite".format(label, key))
        for key in ("mean_absolute", "max_absolute"):
            if not math.isfinite(float(error.get(key, math.nan))):
                raise ComparisonError("{} logit error {} is not finite".format(label, key))
        return candidate

    naive_candidate, quarot_candidate = metrics(naive, "naive"), metrics(quarot, "quarot")
    return {
        "comparison": "matched RTN fake-quant W4A4: naive F3 versus QuaRot F4",
        "passed": True,
        "model": dict(naive_model),
        "data": dict(naive_data),
        "naive": {"perplexity": naive_candidate["perplexity"], "mean_nll": naive_candidate["mean_nll"]},
        "quarot": {"perplexity": quarot_candidate["perplexity"], "mean_nll": quarot_candidate["mean_nll"]},
        "delta_quarot_minus_naive": {
            "perplexity": float(quarot_candidate["perplexity"]) - float(naive_candidate["perplexity"]),
            "mean_nll": float(quarot_candidate["mean_nll"]) - float(naive_candidate["mean_nll"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naive", type=Path, required=True, help="F3 naive W4A4 result JSON")
    parser.add_argument("--quarot", type=Path, required=True, help="F4 QuaRot W4A4 result JSON")
    parser.add_argument("--output", type=Path, help="Optional comparison JSON path")
    args = parser.parse_args()
    try:
        summary = compare(_load(args.naive), _load(args.quarot))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
    except (OSError, ValueError, TypeError, ComparisonError) as error:
        print("RTN COMPARISON FAILED: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
