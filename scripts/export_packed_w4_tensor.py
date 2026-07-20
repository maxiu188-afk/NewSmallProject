#!/usr/bin/env python3
"""Export one trusted 2-D PyTorch tensor into the owned packed-W4 format."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.packed_w4 import PackedW4Error, quantize_w4_rows, validate_w4a8_linear_shape, write_packed_w4_artifact


def _load_tensor(path: Path, tensor_key: str):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("export requires PyTorch in the selected environment") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("this exporter requires torch.load(..., weights_only=True)") from error
    tensor = payload if not tensor_key else payload[tensor_key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise RuntimeError("selected payload must be one 2-D torch.Tensor [out_features, in_features]")
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Trusted .pt file containing a tensor or state-dict entry")
    parser.add_argument("--tensor-key", default="", help="State-dict key; omit when the payload is the tensor itself")
    parser.add_argument("--tensor-name", required=True, help="Output tensor name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--source-json", type=Path, help="Optional JSON provenance object merged into the manifest")
    args = parser.parse_args()
    try:
        tensor = _load_tensor(args.input, args.tensor_key)
        out_features, in_features = tensor.shape
        validate_w4a8_linear_shape(out_features, in_features, args.group_size)
        source = {"input_file": str(args.input), "tensor_key": args.tensor_key, "source_dtype": str(tensor.dtype)}
        if args.source_json:
            source.update(json.loads(args.source_json.read_text(encoding="utf-8")))
        matrix = quantize_w4_rows(tensor.tolist(), group_size=args.group_size)
        manifest = write_packed_w4_artifact(args.output_dir, args.tensor_name, matrix, source)
    except (OSError, RuntimeError, ValueError, PackedW4Error) as error:
        print("PACKED-W4 EXPORT FAILED: {}".format(error), file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
