#!/usr/bin/env python3
"""Run the deterministic, kernel-free W4A8 packed-reference smoke."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.packed_w4 import (
    FORMAT_VERSION,
    LLAMA2_13B_LINEAR_SHAPES,
    dequantize_w4a8_matvec,
    load_packed_w4_artifact,
    quantize_w4_rows,
    validate_w4a8_linear_shape,
    write_packed_w4_artifact,
)


def _digest_accumulators(accumulators) -> str:
    encoded = json.dumps(accumulators, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(artifact_dir: Path) -> dict:
    """Exercise the format on a small matrix and validate planned Llama shapes."""
    out_features, in_features, group_size = 8, 128, 32
    rows = [
        [((row * 31 + column * 17) % 29 - 14) / 5.0 for column in range(in_features)]
        for row in range(out_features)
    ]
    activations = [((index * 11) % 23 - 11) / 4.0 for index in range(in_features)]
    matrix = quantize_w4_rows(rows, group_size=group_size)
    manifest_path = write_packed_w4_artifact(
        artifact_dir,
        "deterministic_w4_reference",
        matrix,
        {"kind": "kernel_free_reference", "seed": 0, "project_root": str(PROJECT_ROOT)},
    )
    restored = load_packed_w4_artifact(manifest_path)
    if restored != matrix:
        raise RuntimeError("packed-W4 serialization round trip changed the reference matrix")
    accumulators, output, activation_scale = dequantize_w4a8_matvec(restored, activations)
    for shape in LLAMA2_13B_LINEAR_SHAPES:
        validate_w4a8_linear_shape(*shape, group_size=128)
    return {
        "scope": "kernel-free packed-W4/W4A8 reference; not CUDA or deployment evidence",
        "format": FORMAT_VERSION,
        "matrix": {"out_features": out_features, "in_features": in_features, "group_size": group_size},
        "activation": {"bits": 8, "scale": activation_scale},
        "integer_accumulator_sha256": _digest_accumulators(accumulators),
        "output": {"count": len(output), "minimum": min(output), "maximum": max(output)},
        "artifact_manifest": str(manifest_path),
        "llama2_13b_supported_shapes": [list(shape) for shape in LLAMA2_13B_LINEAR_SHAPES],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--artifact-dir", type=Path, help="Directory for packed payload and manifest")
    args = parser.parse_args()
    artifact_dir = args.artifact_dir or args.output.parent / "packed-w4-reference-artifact"
    result = run(artifact_dir)
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
