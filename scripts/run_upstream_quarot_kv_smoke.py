#!/usr/bin/env python3
"""Validate upstream QuaRot int4 KV initialization, append, and decode."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path

import torch

from quarot.transformers.kv_cache import (
    MultiLayerPagedKVCache4Bit,
    asym_quantize_and_pack_i4,
    unpack_i4_and_asym_dequantize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stored_token(cache, token: int, kind: int) -> tuple[torch.Tensor, torch.Tensor]:
    page = token // cache.page_size
    offset = token % cache.page_size
    packed = cache.pages[page, 0, kind, :, offset, :]
    params = cache.scales[page, 0, kind, :, offset, :]
    dequantized = unpack_i4_and_asym_dequantize(
        packed, params[..., :1], params[..., 1:]
    )
    return packed, dequantized


def run() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("upstream QuaRot KV smoke requires NVIDIA CUDA")

    torch.manual_seed(20260721)
    device = torch.device("cuda:0")
    batch, heads, head_dim = 1, 2, 128
    prefill, page_size = 4, 4
    keys = torch.randn(batch, prefill + 1, heads, head_dim, device=device, dtype=torch.float16)
    values = torch.randn_like(keys)
    query = torch.randn(batch, 1, heads, head_dim, device=device, dtype=torch.float16)

    cache = MultiLayerPagedKVCache4Bit(
        batch_size=batch,
        page_size=page_size,
        max_seq_len=8,
        device=device,
        n_layers=1,
        num_heads=heads,
        head_dim=head_dim,
        disable_quant=False,
        hadamard_dtype=None,
    )
    cache.update(keys[:, :prefill], values[:, :prefill], 0, cache_kwargs={})
    decode = cache.update(keys[:, prefill:], values[:, prefill:], 0, cache_kwargs={})
    candidate = decode(query)
    torch.cuda.synchronize()

    stored_keys = []
    stored_values = []
    storage_checks = []
    for token in range(prefill + 1):
        stored_k, dequant_k = _stored_token(cache, token, 0)
        stored_v, dequant_v = _stored_token(cache, token, 1)
        expected_k, expected_k_scale, expected_k_zero = asym_quantize_and_pack_i4(keys[:, token])
        expected_v, expected_v_scale, expected_v_zero = asym_quantize_and_pack_i4(values[:, token])
        page = token // page_size
        offset = token % page_size
        params_k = cache.scales[page, 0, 0, :, offset, :]
        params_v = cache.scales[page, 0, 1, :, offset, :]
        storage_checks.append(
            torch.equal(stored_k, expected_k[0])
            and torch.equal(stored_v, expected_v[0])
            and torch.equal(params_k[..., :1], expected_k_scale[0])
            and torch.equal(params_k[..., 1:], expected_k_zero[0])
            and torch.equal(params_v[..., :1], expected_v_scale[0])
            and torch.equal(params_v[..., 1:], expected_v_zero[0])
        )
        stored_keys.append(dequant_k)
        stored_values.append(dequant_v)

    key_reference = torch.stack(stored_keys, dim=0)
    value_reference = torch.stack(stored_values, dim=0)
    scores = torch.einsum(
        "bhd,thd->bht", query[:, 0].float(), key_reference.float()
    ) / math.sqrt(head_dim)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum(
        "bht,thd->bhd", probabilities, value_reference.float()
    ).unsqueeze(1).to(torch.float16)
    absolute_error = (candidate - reference).abs()
    checks = {
        "prefill_and_append_storage_exact": all(storage_checks),
        "decode_output_finite": bool(torch.isfinite(candidate).all().item()),
        "decode_matches_dequantized_torch_reference": bool(
            torch.allclose(candidate, reference, atol=3e-2, rtol=3e-2)
        ),
        "crossed_page_boundary": cache.length == prefill + 1 and cache.page_cnt_from_length(cache.length) == 2,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "pinned upstream QuaRot int4 KV init/append/decode correctness across one page boundary; not full-layer, full-model, quality, or performance evidence",
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
        "configuration": {
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "prefill_tokens": prefill,
            "appended_tokens": 1,
            "page_size": page_size,
        },
        "checks": checks,
        "maximum_absolute_decode_error": float(absolute_error.max().item()),
        "mean_absolute_decode_error": float(absolute_error.float().mean().item()),
        "tolerance": {"atol": 3e-2, "rtol": 3e-2},
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
