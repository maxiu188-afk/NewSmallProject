#!/usr/bin/env python3
"""Materialize pinned WikiText-2 windows for SpinQuant rotation learning."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from transformers import AutoTokenizer

from repro.spinquant.calibration import (
    sample_token_windows,
    write_calibration_artifact,
)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config_path: Path, model_snapshot: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    calibration = config["calibration"]
    if model_snapshot.name != model["revision"]:
        raise ValueError("model snapshot directory does not match the pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    source = load_dataset(
        calibration["dataset_id"],
        calibration["dataset_subset"],
        split=calibration["dataset_split"],
        revision=calibration["dataset_revision"],
    )
    texts = []
    for row in source:
        value = row.get(calibration["text_field"])
        if isinstance(value, str):
            texts.append(value)
    joined = str(calibration["join_separator"]).join(texts)
    token_ids = tokenizer(
        joined,
        add_special_tokens=bool(calibration["add_special_tokens"]),
    )["input_ids"]
    sequences = sample_token_windows(
        token_ids,
        samples=int(calibration["samples"]),
        sequence_length=int(calibration["sequence_length"]),
        seed=int(calibration["seed"]),
    )
    provenance = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "model": model,
        "model_snapshot": str(model_snapshot),
        "model_config_sha256": _sha256(model_snapshot / "config.json"),
        "dataset": calibration,
        "protocol": (
            "join configured WikiText-2 rows, tokenize once without implicit "
            "revision drift, then sample fixed-length windows with replacement"
        ),
    }
    manifest_path = write_calibration_artifact(
        output_dir,
        sequences,
        provenance=provenance,
    )
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.config.resolve(),
        args.model_snapshot.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
