#!/usr/bin/env python3
"""Resolve a Hugging Face dataset commit and write a pinned pipeline config."""

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi
from repro.quarot_pipeline import PipelineConfigError, load_pipeline_config, validate_pipeline_config


_UNPINNED_SENTINEL = "pin-before-first-download"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and pin a configured Hugging Face dataset revision")
    parser.add_argument("config", type=Path, help="Pipeline configuration with data.source=huggingface_text")
    parser.add_argument("--output", type=Path, required=True, help="Write dataset provenance JSON here")
    parser.add_argument("--write-resolved-config", type=Path, required=True, help="Write the configuration with the resolved dataset revision")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs")
    args = parser.parse_args()

    try:
        config = load_pipeline_config(args.config)
        validate_pipeline_config(config)
        data = config["data"]
        if data["source"] != "huggingface_text":
            raise PipelineConfigError("dataset preparation requires data.source=huggingface_text")
        for target in (args.output, args.write_resolved_config):
            if target.exists() and not args.overwrite:
                raise PipelineConfigError("refusing to overwrite {}; use --overwrite explicitly".format(target))
        requested = data.get("revision")
        if requested in {None, "", _UNPINNED_SENTINEL}:
            requested = None
        info = HfApi().dataset_info(repo_id=data["id"], revision=requested)
        if not isinstance(info.sha, str) or len(info.sha) != 40:
            raise RuntimeError("could not resolve a 40-character dataset commit")
        resolved = copy.deepcopy(config)
        resolved.pop("_config_dir", None)
        resolved["data"]["revision"] = info.sha
        record: Dict[str, Any] = {
            "dataset_id": data["id"],
            "subset": data.get("subset"),
            "split": data.get("split", "validation"),
            "requested_revision": data.get("revision"),
            "resolved_revision": info.sha,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        args.write_resolved_config.parent.mkdir(parents=True, exist_ok=True)
        with args.write_resolved_config.open("w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError, RuntimeError, PipelineConfigError) as error:
        print("DATASET PREPARATION FAILED: {}".format(error), file=sys.stderr)
        return 1

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
