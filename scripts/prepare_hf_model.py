#!/usr/bin/env python3
"""Download a configured Hugging Face model snapshot and pin its revision.

This is deliberately separate from the evaluation runner.  A pipeline config
with ``local_files_only: true`` can therefore never begin a large download by
accident, while this explicit preparation step records the immutable snapshot
commit used by a subsequent local or remote run.
"""

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
cache_home = PROJECT_ROOT / ".cache" / "huggingface"
cache_home.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(cache_home)
os.environ["HF_HUB_CACHE"] = str(cache_home)

from huggingface_hub import snapshot_download
from repro.quarot_pipeline import PipelineConfigError, load_pipeline_config, validate_pipeline_config


_UNPINNED_SENTINEL = "pin-before-first-download"


def _snapshot_manifest(snapshot_path: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(snapshot_path).as_posix(), "bytes": path.stat().st_size})
    return {"path": str(snapshot_path), "file_count": len(files), "files": files}


def _resolved_config(config: Dict[str, Any], revision: str) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved.pop("_config_dir", None)
    resolved["model"]["revision"] = revision
    resolved["model"]["local_files_only"] = True
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and pin a configured Hugging Face model snapshot")
    parser.add_argument("config", type=Path, help="Pipeline configuration containing model.kind=pretrained")
    parser.add_argument("--output", type=Path, required=True, help="Local JSON snapshot manifest")
    parser.add_argument("--write-resolved-config", type=Path, help="Write a copy of the config with the resolved commit")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    args = parser.parse_args()

    try:
        config = load_pipeline_config(args.config)
        validate_pipeline_config(config)
        model = config["model"]
        if model["kind"] != "pretrained":
            raise PipelineConfigError("model preparation requires model.kind=pretrained")
        for target in (args.output, args.write_resolved_config):
            if target is not None and target.exists() and not args.overwrite:
                raise PipelineConfigError("refusing to overwrite {}; use --overwrite explicitly".format(target))

        requested = model.get("revision")
        if requested in {None, "", _UNPINNED_SENTINEL}:
            requested = None
        snapshot = Path(snapshot_download(repo_id=model["id"], revision=requested, cache_dir=str(cache_home)))
        resolved_revision = snapshot.name
        if len(resolved_revision) != 40 or any(char not in "0123456789abcdef" for char in resolved_revision):
            raise RuntimeError("could not derive a 40-character snapshot commit from {}".format(snapshot))

        manifest = {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_id": model["id"],
            "requested_revision": model.get("revision"),
            "resolved_revision": resolved_revision,
            "snapshot": _snapshot_manifest(snapshot),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if args.write_resolved_config is not None:
            args.write_resolved_config.parent.mkdir(parents=True, exist_ok=True)
            with args.write_resolved_config.open("w", encoding="utf-8") as handle:
                json.dump(_resolved_config(config, resolved_revision), handle, indent=2, sort_keys=True)
                handle.write("\n")
    except (OSError, ValueError, RuntimeError, PipelineConfigError) as error:
        print("MODEL PREPARATION FAILED: {}".format(error), file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
