#!/usr/bin/env python3
"""Run the official QuaRot W4A4KV4 model path on a tiny random Llama."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "QuaRot"))

import quarot  # noqa: E402
from e2e.quantized_llama import modeling_llama  # noqa: E402


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_model(device: torch.device):
    config = modeling_llama.QuarotLlamaConfig(
        vocab_size=320,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_2"
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260721)
            model = modeling_llama.QuarotLlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    generator = torch.Generator(device="cpu").manual_seed(20260721)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.randn(model.model.embed_tokens.weight.shape, generator=generator, dtype=torch.float16) * 0.02
        )
        model.lm_head.weight.copy_(
            torch.randn(model.lm_head.weight.shape, generator=generator, dtype=torch.float16) * 0.02
        )
        for module in model.modules():
            if isinstance(module, quarot.nn.Linear4bit):
                module.weight_scales.fill_(2.0**-5)
    return model.to(device).eval()


@torch.no_grad()
def _run_once(model, input_ids: torch.Tensor, decode_id: torch.Tensor):
    model._expected_max_length = input_ids.shape[1] + 1
    prefill = model(input_ids, use_cache=True)
    decoded = model(decode_id, past_key_values=prefill.past_key_values, use_cache=True)
    torch.cuda.synchronize()
    return prefill.logits.detach().clone(), decoded.logits.detach().clone(), prefill.past_key_values.length


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("upstream QuaRot tiny e2e smoke requires NVIDIA CUDA")
    device = torch.device("cuda:0")
    model = _build_model(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    decode_id = torch.tensor([[5]], dtype=torch.long, device=device)
    prefill_a, decode_a, length_a = _run_once(model, input_ids, decode_id)
    prefill_b, decode_b, length_b = _run_once(model, input_ids, decode_id)
    checks = {
        "prefill_shape": list(prefill_a.shape) == [1, 4, 320],
        "decode_shape": list(decode_a.shape) == [1, 1, 320],
        "prefill_finite": bool(torch.isfinite(prefill_a).all().item()),
        "decode_finite": bool(torch.isfinite(decode_a).all().item()),
        "repeat_prefill_exact": torch.equal(prefill_a, prefill_b),
        "repeat_decode_exact": torch.equal(decode_a, decode_b),
        "kv_length": length_a == 5 and length_b == 5,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "tiny random one-layer execution of the pinned upstream QuaRot W4A4 linear and KV4 model path; not pretrained quality or performance evidence",
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
        "model": {
            "layers": 1,
            "hidden_size": 256,
            "intermediate_size": 512,
            "attention_heads": 2,
            "kv_heads": 2,
            "head_dim": 128,
            "vocab_size": 320,
        },
        "checks": checks,
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
