#!/usr/bin/env python3
"""Validate exact W4A8 int32 accumulators for Llama-2-13B linear shapes."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from repro.packed_w4 import LLAMA2_13B_LINEAR_SHAPES, validate_w4a8_linear_shape
from repro.w4a8_cuda import grouped_int32_matmul


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _packed_w4_weights(out_features: int, in_features: int, device: torch.device) -> torch.Tensor:
    """Create deterministic signed W4 values and pack them in the owned layout."""
    values = (torch.arange(out_features * in_features, device=device, dtype=torch.int32) % 16 - 8).to(torch.int8)
    values = values.view(out_features, in_features)
    low = (values[:, 0::2].to(torch.int16) & 0x0F).to(torch.uint8)
    high = (values[:, 1::2].to(torch.int16) & 0x0F).to(torch.uint8)
    return low | (high << 4)


def _a8_activations(tokens: int, in_features: int, device: torch.device) -> torch.Tensor:
    return (torch.arange(tokens * in_features, device=device, dtype=torch.int32) % 255 - 127).to(torch.int8).view(
        tokens, in_features
    )


def _reference_accumulators(
    activations: torch.Tensor, packed_weights: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Independent torch integer reference; it does not call the CUDA extension."""
    tokens, in_features = activations.shape
    out_features = packed_weights.shape[0]
    groups = in_features // group_size
    packed = packed_weights.reshape(out_features, -1)
    low = (packed & 0x0F).to(torch.int16)
    high = (packed >> 4).to(torch.int16)
    weights = torch.empty((out_features, in_features), dtype=torch.int8, device=packed.device)
    weights[:, 0::2] = torch.where(low >= 8, low - 16, low).to(torch.int8)
    weights[:, 1::2] = torch.where(high >= 8, high - 16, high).to(torch.int8)
    product = activations.view(tokens, 1, groups, group_size).to(torch.int32) * weights.view(
        1, out_features, groups, group_size
    ).to(torch.int32)
    return product.sum(dim=-1, dtype=torch.int32).contiguous()


def run(tokens: int, group_size: int, verbose_build: bool) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("shape matrix requires CUDA; CPU or MPS fallback is not permitted")
    device = torch.device("cuda")
    rows = []
    for out_features, in_features in LLAMA2_13B_LINEAR_SHAPES:
        validate_w4a8_linear_shape(out_features, in_features, group_size)
        activations = _a8_activations(tokens, in_features, device)
        packed_weights = _packed_w4_weights(out_features, in_features, device)
        expected = _reference_accumulators(activations, packed_weights, group_size)
        candidate = grouped_int32_matmul(activations, packed_weights, group_size, verbose=verbose_build)
        torch.cuda.synchronize()
        if not torch.equal(candidate, expected):
            mismatch = (candidate != expected).nonzero(as_tuple=False)[0].tolist()
            raise RuntimeError("int32 accumulator mismatch for {}x{} at {}".format(out_features, in_features, mismatch))
        rows.append(
            {
                "out_features": out_features,
                "in_features": in_features,
                "group_size": group_size,
                "tokens": tokens,
                "accumulator_shape": list(candidate.shape),
                "int32_exact_match": True,
            }
        )
        del activations, packed_weights, expected, candidate
        torch.cuda.empty_cache()
    return {
        "scope": "owned W4A8 Llama-shape integer-kernel correctness matrix; not a model or performance result",
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "project_revision": _project_revision(),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    try:
        result = run(args.tokens, args.group_size, args.verbose_build)
    except RuntimeError as error:
        print("W4A8 SHAPE MATRIX FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
