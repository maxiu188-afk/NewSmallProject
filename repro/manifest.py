"""Offline validation and provenance capture for QuaRot experiments."""

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


REQUIRED_TOP_LEVEL = {
    "experiment_id",
    "track",
    "status",
    "model",
    "quantization",
    "calibration",
    "evaluation",
    "runtime",
    "provenance",
}
VALID_TRACKS = {"algorithm", "deployment"}
VALID_EXECUTION_MODES = {
    "bf16",
    "fp16",
    "fake_quant",
    "packed_weight",
    "int4_gemm",
    "int4_kv_cache",
}


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration is incomplete or contradictory."""


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(config: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ConfigValidationError("{} must be an object".format(field))
    return value


def _require_fields(section: Mapping[str, Any], section_name: str, fields: set) -> None:
    missing = sorted(fields - set(section))
    if missing:
        raise ConfigValidationError("{} is missing: {}".format(section_name, ", ".join(missing)))


def validate_config(config: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if missing:
        raise ConfigValidationError("missing top-level fields: {}".format(", ".join(missing)))
    if config["track"] not in VALID_TRACKS:
        raise ConfigValidationError("track must be one of {}".format(sorted(VALID_TRACKS)))
    if not isinstance(config["experiment_id"], str) or not config["experiment_id"]:
        raise ConfigValidationError("experiment_id must be a non-empty string")

    model = _require_mapping(config, "model")
    quantization = _require_mapping(config, "quantization")
    calibration = _require_mapping(config, "calibration")
    evaluation = _require_mapping(config, "evaluation")
    runtime = _require_mapping(config, "runtime")
    provenance = _require_mapping(config, "provenance")
    _require_fields(model, "model", {"id", "revision", "dtype"})
    _require_fields(quantization, "quantization", {"execution_mode", "w_bits", "a_bits", "k_bits", "v_bits"})
    _require_fields(calibration, "calibration", {"dataset", "revision", "nsamples", "seed", "sequence_length"})
    _require_fields(evaluation, "evaluation", {"dataset", "revision", "sequence_length"})
    _require_fields(runtime, "runtime", {"environment", "device_class"})
    _require_fields(provenance, "provenance", {"upstream_commit"})

    mode = quantization["execution_mode"]
    if mode not in VALID_EXECUTION_MODES:
        raise ConfigValidationError("unknown execution_mode: {}".format(mode))
    for name in ("w_bits", "a_bits", "k_bits", "v_bits"):
        bits = quantization[name]
        if not isinstance(bits, int) or bits < 2 or bits > 16:
            raise ConfigValidationError("{} must be an integer in [2, 16]".format(name))
    if calibration["nsamples"] < 0 or calibration["sequence_length"] <= 0:
        raise ConfigValidationError("calibration nsamples and sequence_length must be non-negative/positive")
    if evaluation["sequence_length"] <= 0:
        raise ConfigValidationError("evaluation sequence_length must be positive")
    if config["track"] == "algorithm" and mode in {"int4_gemm", "int4_kv_cache"}:
        raise ConfigValidationError("algorithm track cannot claim a real CUDA execution mode")
    if config["track"] == "deployment" and runtime["device_class"] != "nvidia_cuda":
        raise ConfigValidationError("deployment track requires device_class nvidia_cuda")


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_run_manifest(config: Mapping[str, Any], command: str) -> Dict[str, Any]:
    """Return a serializable run record without executing a model or benchmark."""
    validate_config(config)
    project_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "config": dict(config),
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "project_revision": _git_revision(project_root),
            "upstream_revision_observed": _git_revision(project_root / "QuaRot"),
        },
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
