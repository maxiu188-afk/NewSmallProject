#!/usr/bin/env python3
"""Validate an offline QuaRot experiment config and optionally write its manifest."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.manifest import ConfigValidationError, build_run_manifest, load_config, validate_config, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to a JSON experiment config")
    parser.add_argument("--command", default="not-run", help="Command recorded in an optional manifest")
    parser.add_argument("--write-manifest", type=Path, help="Write an offline manifest after validation")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        validate_config(config)
    except (OSError, ValueError, ConfigValidationError) as error:
        print("CONFIG INVALID: {}".format(error), file=sys.stderr)
        return 1

    print("CONFIG VALID: {}".format(args.config))
    if args.write_manifest:
        write_manifest(args.write_manifest, build_run_manifest(config, args.command))
        print("MANIFEST WRITTEN: {}".format(args.write_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
