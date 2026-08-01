#!/usr/bin/env python3
"""Run the shared full-model service protocol for four W4AFP8 variants."""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_w4a16_llama2_13b_serving import main  # noqa: E402


def _require_accepted_source_gate(arguments: list[str]) -> None:
    if "--config" not in arguments:
        raise ValueError("W4AFP8 serving requires --config")
    index = arguments.index("--config")
    if index + 1 >= len(arguments):
        raise ValueError("W4AFP8 serving --config is missing its path")
    config = json.loads(Path(arguments[index + 1]).read_text(encoding="utf-8"))
    if config["source_gate"].get("status") != "accepted":
        raise RuntimeError("W4AFP8 source gate is still pending")


if __name__ == "__main__":
    _require_accepted_source_gate(sys.argv[1:])
    raise SystemExit(main())
