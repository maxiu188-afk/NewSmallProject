#!/usr/bin/env python3
"""Run the local random tiny-LLaMA QuaRot smoke test without any model download."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
cache_home = PROJECT_ROOT / ".cache" / "huggingface"
cache_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(cache_home))

import torch
import transformers

from repro.torch_smoke import run_tiny_llama_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "random tiny LLaMA only; no pretrained model, tokenizer, dataset, or CUDA kernel",
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "errors": run_tiny_llama_smoke(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
