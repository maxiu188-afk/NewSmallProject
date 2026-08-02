#!/usr/bin/env python3
"""Materialize the pinned BoolQ validation examples used by deployed evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

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


def _canonical_line(record: dict) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare(config_path: Path, output_dir: Path) -> dict:
    from datasets import load_dataset

    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["dataset"]
    installed_datasets = metadata.version("datasets")
    expected_datasets = spec["preparation_runtime"]["datasets"]
    if installed_datasets != expected_datasets:
        raise RuntimeError(
            f"BoolQ preparation requires datasets=={expected_datasets}, "
            f"found {installed_datasets}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")

    source = load_dataset(
        spec["repository"],
        spec["subset"],
        split=spec["split"],
        revision=spec["revision"],
    )
    if len(source) != int(spec["expected_rows"]):
        raise RuntimeError(f"expected {spec['expected_rows']} BoolQ rows, got {len(source)}")
    if source._fingerprint != spec["expected_fingerprint"]:
        raise RuntimeError(
            f"BoolQ fingerprint changed: {source._fingerprint}"
        )

    records = []
    label_counts: Counter[int] = Counter()
    seen_indices: set[int] = set()
    for row in source:
        record = {
            "idx": int(row["idx"]),
            "label": int(row["label"]),
            "passage": row["passage"],
            "question": row["question"],
        }
        if record["idx"] in seen_indices:
            raise RuntimeError(f"duplicate BoolQ idx: {record['idx']}")
        if record["label"] not in (0, 1):
            raise RuntimeError(f"invalid BoolQ label: {record['label']}")
        if not record["passage"] or not record["question"]:
            raise RuntimeError(f"empty BoolQ text at idx {record['idx']}")
        seen_indices.add(record["idx"])
        label_counts[record["label"]] += 1
        records.append(record)

    normalized_counts = {str(key): value for key, value in sorted(label_counts.items())}
    if normalized_counts != spec["expected_label_counts"]:
        raise RuntimeError(f"BoolQ label counts changed: {normalized_counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "validation.jsonl"
    temporary = examples_path.with_name(f".{examples_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_line(record))
    temporary.replace(examples_path)
    examples_sha256 = _sha256(examples_path)
    if examples_sha256 != spec["expected_examples_sha256"]:
        raise RuntimeError(f"BoolQ examples SHA-256 changed: {examples_sha256}")

    result = {
        "status": "passed",
        "scope": "pinned BoolQ validation examples for zero-shot multiple-choice evaluation; no model execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "dataset": {
            **spec,
            "fingerprint": source._fingerprint,
            "rows": len(records),
            "label_counts": normalized_counts,
        },
        "examples": {
            "path": str(examples_path.resolve()),
            "sha256": examples_sha256,
        },
        "runtime": {"datasets": installed_datasets},
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print("BOOLQ_VALIDATION_PREPARED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
