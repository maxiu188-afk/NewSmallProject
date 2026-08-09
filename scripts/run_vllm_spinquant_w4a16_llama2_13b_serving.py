#!/usr/bin/env python3
"""Run the shared serving protocol for BF16 and packed SpinQuant W4A16."""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_w4a16_llama2_13b_serving import main  # noqa: E402


EXPECTED_MODELS = ["bf16", "spinquant_w4a16"]


def _require_accepted_source_gate(arguments: list[str]) -> None:
    if "--config" not in arguments:
        raise ValueError("SpinQuant W4A16 serving requires --config")
    index = arguments.index("--config")
    if index + 1 >= len(arguments):
        raise ValueError("SpinQuant W4A16 serving --config is missing its path")
    config = json.loads(Path(arguments[index + 1]).read_text(encoding="utf-8"))
    gate = config["source_gate"]
    if gate.get("status") != "accepted":
        raise RuntimeError("SpinQuant W4A16 source gate is not accepted")
    if gate.get("expected_models") != EXPECTED_MODELS:
        raise RuntimeError("SpinQuant W4A16 serving model order changed")
    if config.get("execution", {}).get("kind") != "formal_only":
        raise RuntimeError("SpinQuant W4A16 serving execution policy changed")


if __name__ == "__main__":
    _require_accepted_source_gate(sys.argv[1:])
    raise SystemExit(main())
