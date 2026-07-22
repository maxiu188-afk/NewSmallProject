#!/usr/bin/env python3
"""Check Isambard vLLM preparation without importing CUDA libraries."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


EXPECTED = {
    "torch": "2.11.0+cu129",
    "vllm": "0.25.1",
    "compressed-tensors": "0.17.0",
}


def run_check() -> dict:
    installed = {}
    errors = []
    warnings = []
    for package, expected in EXPECTED.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        installed[package] = actual
        comparable = actual.partition("+")[0] if package == "vllm" and actual else actual
        if comparable != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    pip_check_output = f"{pip_check.stdout}{pip_check.stderr}".strip()
    known_sbsa_warning = (
        platform.machine() == "aarch64"
        and pip_check_output
        == "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform"
    )
    if known_sbsa_warning:
        distribution = metadata.distribution("nvidia-cusparselt-cu12")
        wheel_files = [file for file in distribution.files or [] if file.name == "WHEEL"]
        wheel_text = (
            distribution.locate_file(wheel_files[0]).read_text(encoding="utf-8")
            if len(wheel_files) == 1
            else ""
        )
        if "Tag: py3-none-manylinux2014_sbsa" in wheel_text:
            warnings.append(pip_check_output)
        else:
            errors.append(f"pip check failed without the expected SBSA tag: {pip_check_output}")
    elif pip_check.returncode:
        errors.append(f"pip check failed: {pip_check_output}")
    if platform.machine() != "aarch64":
        errors.append(f"expected aarch64, found {platform.machine()}")

    return {
        "status": "passed" if not errors else "failed",
        "scope": "login-node package and architecture preflight; no GPU execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "installed": installed,
        "pip_check_output": pip_check_output,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_check()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
