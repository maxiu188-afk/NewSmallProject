#!/usr/bin/env python3
"""Run the independent tiny SpinQuant R1/R2 fake-quantization smoke."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from repro.spinquant.fake_quant import run_tiny_spinquant_fake_quant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "experiment": run_tiny_spinquant_fake_quant(config),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
