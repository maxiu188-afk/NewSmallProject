#!/usr/bin/env python3
"""Materialize the frozen WikiText-2 test token stream for deployed PPL."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from datasets import load_dataset
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


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare(
    config_path: Path,
    model_snapshot: Path,
    output_dir: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = config["evaluation"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")
    if not (model_snapshot / "tokenizer.json").is_file():
        raise FileNotFoundError(model_snapshot / "tokenizer.json")

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
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
    sequences: list[list[int]] = []
    rows_seen = 0
    for row in source:
        if rows_seen >= int(spec["max_rows"]) or len(sequences) >= wanted:
            break
        rows_seen += 1
        text = row.get(spec["text_field"])
        if not isinstance(text, str):
            continue
        token_buffer.extend(
            tokenizer(
                text,
                add_special_tokens=bool(spec["add_special_tokens"]),
            )["input_ids"]
        )
        while len(token_buffer) >= width and len(sequences) < wanted:
            sequences.append(token_buffer[:width])
            del token_buffer[:width]
    if len(sequences) != wanted:
        raise RuntimeError(f"expected {wanted} sequences, produced {len(sequences)}")
    if any(len(sequence) != width for sequence in sequences):
        raise RuntimeError("materialized sequence width changed")

    scored_tokens = len(sequences) * (width - 1)
    if scored_tokens != int(spec["expected_scored_tokens"]):
        raise RuntimeError(
            f"expected {spec['expected_scored_tokens']} scored tokens, got {scored_tokens}"
        )
    token_digest = hashlib.sha256()
    for sequence in sequences:
        for token in sequence:
            token_digest.update(struct.pack("<I", int(token)))

    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "token-ids.json"
    _write_json(tokens_path, sequences)
    result = {
        "status": "passed",
        "scope": "frozen WikiText-2 test token stream for matched deployed PPL; no model execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "model_snapshot": str(model_snapshot.resolve()),
        "dataset": {
            **spec,
            "rows_seen": rows_seen,
            "materialized_samples": len(sequences),
            "materialized_tokens": len(sequences) * width,
            "scored_tokens": scored_tokens,
            "token_ids_sha256": token_digest.hexdigest(),
        },
        "tokens": {
            "path": str(tokens_path.resolve()),
            "sha256": _sha256(tokens_path),
        },
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.config.resolve(),
        args.model_snapshot.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("VLLM_13B_PPL_TOKENS_PREPARED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
