#!/usr/bin/env python3
"""Materialize a planned QuaRot matrix into individually auditable JSON configs."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from repro.experiment_matrix import materialize_matrix
from repro.manifest import ConfigValidationError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        with args.template.open(encoding="utf-8") as handle:
            matrix = materialize_matrix(json.load(handle))
    except (OSError, ValueError, ConfigValidationError) as error:
        print("MATRIX INVALID: {}".format(error), file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for experiment_id, config in sorted(matrix.items()):
        output = args.output_dir / "{}.json".format(experiment_id)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("CONFIG WRITTEN: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
