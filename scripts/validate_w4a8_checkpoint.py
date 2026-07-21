#!/usr/bin/env python3
"""Validate every checksum and structural entry in a sharded W4A8 checkpoint."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.w4a8_checkpoint import load_checkpoint_manifest, load_linear_artifact, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_checkpoint_manifest(args.manifest)
        for name in sorted(manifest["tensors"]):
            load_linear_artifact(args.manifest, manifest, name)
    except (OSError, RuntimeError, ValueError) as error:
        print("W4A8 CHECKPOINT VALIDATION FAILED: {}".format(error), file=sys.stderr)
        return 1
    result = {
        "passed": True,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "tensor_count": len(manifest["tensors"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
