#!/usr/bin/env python3
"""Run a configuration-driven local QuaRot pipeline and write JSON results."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
cache_home = PROJECT_ROOT / ".cache" / "huggingface"
cache_home.mkdir(parents=True, exist_ok=True)
# `HF_HOME` normally places Hub files in an extra `hub/` directory.  Pin the
# actual Hub cache too so this runner agrees with prepare_hf_model.py exactly.
os.environ["HF_HOME"] = str(cache_home)
os.environ["HF_HUB_CACHE"] = str(cache_home)

from repro.quarot_pipeline import PipelineConfigError, load_pipeline_config, run_pipeline, validate_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Portable JSON pipeline configuration")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Result JSON path; required unless validating only")
    args = parser.parse_args()
    try:
        config = load_pipeline_config(args.config)
        validate_pipeline_config(config)
        if args.validate_only:
            print("PIPELINE CONFIG VALID: {}".format(args.config))
            return 0
        if args.output is None:
            raise PipelineConfigError("--output is required unless --validate-only is used")
        result = run_pipeline(config)
    except (OSError, ValueError, RuntimeError, PipelineConfigError) as error:
        print("PIPELINE FAILED: {}".format(error), file=sys.stderr)
        return 1
    result["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
