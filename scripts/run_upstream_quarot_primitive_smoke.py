#!/usr/bin/env python3
"""Validate the pinned upstream QuaRot pack, quantize, GEMM, and scale path."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import torch

import quarot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("upstream QuaRot primitive smoke requires NVIDIA CUDA")

    torch.manual_seed(20260721)
    device = torch.device("cuda:0")
    rows, outputs, features = 128, 128, 128

    # Construct exactly representable inputs so disagreement identifies a
    # packing/kernel contract error rather than an FP16 rounding ambiguity.
    input_i4 = torch.randint(-7, 8, (rows, features), dtype=torch.int8)
    weight_i4 = torch.randint(-7, 8, (outputs, features), dtype=torch.int8)
    input_scale = torch.full((rows, 1), 2.0**-5, dtype=torch.float16, device=device)
    weight_scale = torch.full((outputs, 1), 2.0**-4, dtype=torch.float16, device=device)
    inputs = input_i4.to(device=device, dtype=torch.float16) * input_scale

    packed_input = quarot.sym_quant(inputs.contiguous(), input_scale.contiguous())
    packed_weight = quarot.functional.pack_i4(weight_i4).to(device).contiguous()
    unpacked_input = quarot.functional.unpack_i4(packed_input.cpu())
    unpacked_weight = quarot.functional.unpack_i4(packed_weight.cpu())

    accumulator = quarot.matmul(packed_input, packed_weight)
    torch.cuda.synchronize()
    accumulator_reference = input_i4.to(torch.int32) @ weight_i4.to(torch.int32).T
    dequantized = quarot.sym_dequant(accumulator, input_scale, weight_scale)
    dequantized_reference = (
        accumulator_reference.to(device=device, dtype=torch.float16)
        * input_scale
        * weight_scale.T
    )
    torch.cuda.synchronize()

    checks = {
        "input_pack_roundtrip_exact": torch.equal(unpacked_input, input_i4.to(torch.int32)),
        "weight_pack_roundtrip_exact": torch.equal(unpacked_weight, weight_i4.to(torch.int32)),
        "int32_accumulator_exact": torch.equal(accumulator.cpu(), accumulator_reference),
        "scaled_output_exact": torch.equal(dequantized, dequantized_reference),
        "scaled_output_finite": bool(torch.isfinite(dequantized).all().item()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "pinned upstream QuaRot CUDA primitive correctness; not layer, KV-cache, model, quality, or performance evidence",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "revisions": {
            "project": _revision(PROJECT_ROOT),
            "upstream_quarot": _revision(PROJECT_ROOT / "QuaRot"),
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "shapes": {"input": [rows, features], "weight": [outputs, features]},
        "checks": checks,
        "maximum_absolute_accumulator": int(accumulator_reference.abs().max().item()),
        "maximum_absolute_scaled_error": float(
            (dequantized - dequantized_reference).abs().max().item()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
