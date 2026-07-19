#!/usr/bin/env python3
"""Compile the owned W4A8 CUDA correctness kernel and compare exact int32 results."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from repro.packed_w4 import dequantize_w4a8_matvec, quantize_a8, quantize_w4_rows
from repro.w4a8_cuda import grouped_int32_matmul


def _nvcc_version() -> str:
    candidates = [shutil.which("nvcc")]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "bin" / "nvcc"))
    if torch.version.cuda:
        candidates.append("/usr/local/cuda-{}/bin/nvcc".format(torch.version.cuda))
    nvcc = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
    if nvcc is None:
        return "unavailable"
    completed = subprocess.run([nvcc, "--version"], text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def run(verbose_build: bool) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("W4A8 CUDA smoke requires CUDA; CPU or MPS fallback is not permitted")
    tokens, out_features, in_features, group_size = 3, 16, 128, 32
    rows = [
        [((row * 31 + column * 17) % 29 - 14) / 5.0 for column in range(in_features)]
        for row in range(out_features)
    ]
    activation_rows = [
        [((token * 13 + column * 11) % 23 - 11) / 4.0 for column in range(in_features)]
        for token in range(tokens)
    ]
    matrix = quantize_w4_rows(rows, group_size=group_size)
    expected_accumulators = []
    expected_outputs = []
    quantized_activations = []
    activation_scales = []
    for activation_row in activation_rows:
        accumulators, output, activation_scale = dequantize_w4a8_matvec(matrix, activation_row)
        quantized, observed_scale = quantize_a8(activation_row)
        if observed_scale != activation_scale:
            raise RuntimeError("activation quantization disagrees with the independent reference")
        expected_accumulators.append(accumulators)
        expected_outputs.append(output)
        quantized_activations.append(quantized)
        activation_scales.append(activation_scale)

    device = torch.device("cuda")
    activations = torch.tensor(quantized_activations, dtype=torch.int8, device=device)
    packed_weights = torch.tensor(list(matrix.data), dtype=torch.uint8, device=device).view(
        matrix.out_features, matrix.in_features // 2
    )
    candidate = grouped_int32_matmul(activations, packed_weights, group_size, verbose=verbose_build)
    torch.cuda.synchronize()
    expected = torch.tensor(expected_accumulators, dtype=torch.int32, device=device)
    if not torch.equal(candidate, expected):
        mismatch = (candidate != expected).nonzero(as_tuple=False)[0].tolist()
        raise RuntimeError("CUDA int32 accumulator mismatch at {}".format(mismatch))

    scales = torch.tensor(matrix.scales, dtype=torch.float32, device=device).view(
        matrix.out_features, matrix.groups_per_row
    )
    a_scales = torch.tensor(activation_scales, dtype=torch.float32, device=device).view(tokens, 1, 1)
    candidate_output = (candidate.float() * scales.unsqueeze(0) * a_scales).sum(dim=-1)
    expected_output = torch.tensor(expected_outputs, dtype=torch.float32, device=device)
    output_error = (candidate_output - expected_output).abs()
    if not torch.allclose(candidate_output, expected_output, atol=1e-5, rtol=1e-6):
        raise RuntimeError("CUDA scaled output exceeds the declared FP32 reference tolerance")
    return {
        "scope": "owned W4A8 CUDA integer-accumulator correctness smoke; not a performance result",
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvcc": _nvcc_version(),
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "project_revision": _project_revision(),
        },
        "matrix": {
            "tokens": tokens,
            "out_features": out_features,
            "in_features": in_features,
            "group_size": group_size,
            "weight_bits": 4,
            "activation_bits": 8,
            "accumulator_dtype": "int32",
        },
        "int32_accumulators": {"exact_match": True, "shape": list(candidate.shape)},
        "scaled_output": {"dtype": "float32", "max_absolute_error": float(output_error.max().item())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.verbose_build)
    except RuntimeError as error:
        print("W4A8 CUDA SMOKE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
