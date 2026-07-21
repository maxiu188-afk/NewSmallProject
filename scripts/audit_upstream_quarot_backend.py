#!/usr/bin/env python3
"""Record source-level readiness of the pinned upstream QuaRot backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import subprocess


EXPECTED_REVISION = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
EXPECTED_SUBMODULES = {
    "third-party/cutlass": "ffa34e70756b0bc744e1dfcc115b5a991a68f132",
    "third-party/fast-hadamard-transform": "4ea722e434e3d4f2a14522341959ebdbe62be2de",
    "third-party/nvbench": "d8dced8a64d9ce305add92fa6d274fd49b569b7e",
}


def _run(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def audit(source: Path) -> dict:
    source = source.resolve()
    revision = _run(["git", "rev-parse", "HEAD"], source)
    submodule_lines = _run(["git", "submodule", "status"], source).splitlines()
    submodules = {}
    for line in submodule_lines:
        marker = line[:1]
        fields = line[1:].strip().split()
        if len(fields) >= 2:
            submodules[fields[1]] = {"revision": fields[0], "initialized": marker != "-", "marker": marker}

    setup_text = (source / "setup.py").read_text(encoding="utf-8")
    requirement_text = (source / "requirements.txt").read_text(encoding="utf-8")
    checkpoint_text = (source / "e2e/checkpoint_utils/quantize_llama_checkpoint.py").read_text(encoding="utf-8")
    arch_targets = sorted(set(re.findall(r"code=sm_(\d+)", setup_text)))
    pins = {}
    for name in ("torch", "transformers"):
        match = re.search(rf"^{name}==([^\s]+)$", requirement_text, flags=re.MULTILINE)
        pins[name] = match.group(1) if match else None
    supported_models = sorted(set(re.findall(r"meta-llama/Llama-2-(?:7b|13b|70b)-hf", checkpoint_text)))

    checks = {
        "revision_matches": revision == EXPECTED_REVISION,
        "submodules_present": set(EXPECTED_SUBMODULES).issubset(submodules),
        "submodules_initialized": all(submodules.get(path, {}).get("initialized", False) for path in EXPECTED_SUBMODULES),
        "submodule_revisions_match": all(
            submodules.get(path, {}).get("revision") == expected for path, expected in EXPECTED_SUBMODULES.items()
        ),
        "legacy_dependency_pins_observed": pins == {"torch": "2.2.1", "transformers": "4.38.0"},
        "upstream_arch_targets_observed": arch_targets == ["75", "80", "86"],
    }
    return {
        "status": "ready_for_unmodified_build" if all(checks.values()) else "source_prerequisites_incomplete",
        "scope": "source audit only; not a CUDA build or W4A4 execution result",
        "host": {"system": platform.system(), "machine": platform.machine()},
        "source": str(source),
        "revision": revision,
        "expected_revision": EXPECTED_REVISION,
        "dependency_pins": pins,
        "cuda_arch_targets": arch_targets,
        "supported_checkpoint_models": supported_models,
        "submodules": submodules,
        "checks": checks,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=project_root / "QuaRot")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
