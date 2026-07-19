#!/usr/bin/env python3
"""Capture RunPod host facts before installing/building or benchmarking QuaRot."""

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _command_output(command: List[str]) -> Dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _nvcc_command() -> List[str]:
    """Locate a CUDA compiler even when its versioned install is not on PATH."""
    candidates = [shutil.which("nvcc")]
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "bin" / "nvcc"))
    candidates.append("/usr/local/cuda/bin/nvcc")
    candidates.extend(str(path) for path in sorted(Path("/usr/local").glob("cuda-*/bin/nvcc")))
    executable = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), "nvcc")
    return [executable, "--version"]


def _git_revision(path: Path) -> str:
    result = _command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    return str(result.get("output", "unavailable")) if result.get("returncode") == 0 else "unavailable"


def collect_preflight() -> Dict[str, object]:
    return {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "revisions": {
            "project": _git_revision(PROJECT_ROOT),
            "upstream_quarot": _git_revision(PROJECT_ROOT / "QuaRot"),
        },
        "commands": {
            "nvidia_smi": _command_output([
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]),
            "nvcc": _command_output(_nvcc_command()),
            "git": _command_output(["git", "--version"]),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "datasets", "accelerate", "lm-eval", "flash-attn")
        },
        "upstream_build_note": "The reference setup.py explicitly lists sm_75, sm_80, and sm_86. Preserve this preflight before any architecture patch.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="JSON file to create")
    args = parser.parse_args()
    report = collect_preflight()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("PRELIGHT WRITTEN: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
