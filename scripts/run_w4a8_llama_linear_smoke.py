#!/usr/bin/env python3
"""Pack a real cached Llama-2-13B q_proj and validate W4A8 module output."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface"))

import torch
from transformers import AutoModelForCausalLM

from repro.w4a8_linear import W4A8Linear, pack_w4_weight, w4a8_reference_linear


MODEL_ID = "meta-llama/Llama-2-13b-hf"
MODEL_REVISION = "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Llama linear smoke requires CUDA")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    linear = model.model.layers[0].self_attn.q_proj
    inputs = torch.randn((1, 2, linear.in_features), device="cuda", dtype=torch.bfloat16)
    packed_weight, scales = pack_w4_weight(linear.weight.detach(), group_size=128)
    candidate_module = W4A8Linear(packed_weight, scales, linear.bias).cuda().eval()
    with torch.inference_mode():
        reference = w4a8_reference_linear(inputs, packed_weight, scales)
        candidate = candidate_module(inputs)
    error = (candidate - reference).abs()
    if not torch.allclose(candidate, reference, atol=1e-4, rtol=1e-5):
        raise RuntimeError("W4A8 module output differs from its packed floating reference")
    result = {
        "scope": "real Llama linear W4A8 module integration smoke; RTN-style packing only, not a GPTQ checkpoint/model PPL result",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "tensor": "model.layers.0.self_attn.q_proj"},
        "runtime": {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "linear": {"in_features": linear.in_features, "out_features": linear.out_features, "group_size": 128},
        "output": {"shape": list(candidate.shape), "max_absolute_error": float(error.max().item())},
    }
    del model, candidate_module
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run()
    except (OSError, RuntimeError) as error:
        print("W4A8 LLAMA LINEAR SMOKE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
