#!/usr/bin/env python3
"""Materialize one deterministic tokenized calibration set for both 13B exports."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(config: dict) -> Path:
    hub = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    repo_dir = "models--" + config["model"]["id"].replace("/", "--")
    return hub / repo_dir / "snapshots" / config["model"]["revision"]


def prepare(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["calibration"]
    snapshot = _snapshot(config)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing pinned model snapshot: {snapshot}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty calibration directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    source = load_dataset(
        spec["dataset_id"],
        spec["dataset_subset"],
        split=spec["dataset_split"],
        revision=spec["dataset_revision"],
    )
    width = int(spec["sequence_length"])
    wanted = int(spec["samples"])
    token_buffer: list[int] = []
    rows: list[list[int]] = []
    rows_seen = 0
    for row in source:
        if rows_seen >= int(spec["max_rows"]) or len(rows) >= wanted:
            break
        rows_seen += 1
        text = row.get(spec["text_field"])
        if not isinstance(text, str):
            continue
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(token_buffer) >= width and len(rows) < wanted:
            rows.append(token_buffer[:width])
            del token_buffer[:width]
    if len(rows) != wanted:
        raise RuntimeError(f"expected {wanted} calibration rows, produced {len(rows)}")

    token_digest = hashlib.sha256()
    for row in rows:
        for token in row:
            token_digest.update(struct.pack("<I", int(token)))
    dataset = Dataset.from_dict(
        {
            "input_ids": rows,
            "attention_mask": [[1] * width for _ in rows],
        }
    )
    dataset_dir = output_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset.save_to_disk(dataset_dir)
    result = {
        "status": "passed",
        "scope": "deterministic shared GPTQ calibration tokens; no model execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "model": config["model"],
        "dataset": {
            **spec,
            "rows_seen": rows_seen,
            "materialized_samples": len(rows),
            "materialized_tokens": len(rows) * width,
            "token_ids_sha256": token_digest.hexdigest(),
        },
        "dataset_path": str(dataset_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
