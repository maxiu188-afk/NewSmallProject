#!/usr/bin/env python3
"""Run a tiny CUDA LLaMA GPTQ smoke without downloading a model or dataset."""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from repro.gptq import GPTQSettings, quantize_llama_weights_gptq


def run() -> Dict[str, Any]:
    """Quantize all 14 linears in a deterministic, two-layer CUDA LLaMA."""
    if not torch.cuda.is_available():
        raise RuntimeError("tiny GPTQ smoke requires CUDA; CPU fallback is not permitted")
    torch.manual_seed(17)
    device = torch.device("cuda")
    config = LlamaConfig(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()
    calibration = [torch.randint(0, config.vocab_size, (1, 32), device=device) for _ in range(4)]
    evaluation = torch.randint(0, config.vocab_size, (1, 32), device=device)
    with torch.inference_mode():
        reference = model(input_ids=evaluation, use_cache=False).logits.float()
    summary = quantize_llama_weights_gptq(
        model,
        calibration,
        GPTQSettings(bits=4, group_size=32, block_size=32, damp_percent=0.01, act_order=True),
    )
    with torch.inference_mode():
        candidate = model(input_ids=evaluation, use_cache=False).logits.float()
    error = (reference - candidate).abs()
    if summary["layers"] != 2 or summary["linear_layers"] != 14:
        raise RuntimeError("tiny GPTQ smoke did not quantize the expected 2 layers and 14 linears")
    if not torch.isfinite(candidate).all() or not torch.isfinite(error).all():
        raise RuntimeError("tiny GPTQ smoke produced non-finite logits")
    torch.cuda.synchronize()
    return {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "model": {"layers": 2, "hidden_size": 64, "intermediate_size": 128},
        "calibration": {"sequences": 4, "sequence_length": 32},
        "gptq": summary,
        "logit_error": {"mean_absolute": float(error.mean().item()), "max_absolute": float(error.max().item())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()
    try:
        result = run()
    except RuntimeError as error:
        print("TINY GPTQ SMOKE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
