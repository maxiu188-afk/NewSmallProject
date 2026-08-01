"""Pinned token-window artifacts for SpinQuant rotation optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct
from typing import Any, Dict, Mapping, Sequence


FORMAT = "newsmallproject-spinquant-calibration-v1"
TOKENS_FILE = "token-ids.json"
MANIFEST_FILE = "calibration-manifest.json"


class CalibrationArtifactError(ValueError):
    """Raised when calibration tokens or provenance fail validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(sequences: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        for token in sequence:
            if token < 0 or token > 0xFFFFFFFF:
                raise CalibrationArtifactError("token IDs must fit unsigned int32")
            digest.update(struct.pack("<I", int(token)))
    return digest.hexdigest()


def sample_token_windows(
    token_ids: Sequence[int],
    *,
    samples: int,
    sequence_length: int,
    seed: int,
) -> list[list[int]]:
    """Sample fixed-length windows with replacement using a local RNG."""

    if samples < 1 or sequence_length < 2:
        raise ValueError("samples must be positive and sequence_length at least two")
    if len(token_ids) <= sequence_length:
        raise ValueError("token stream is too short for the requested windows")
    generator = random.Random(seed)
    maximum_start = len(token_ids) - sequence_length
    result = []
    for _ in range(samples):
        start = generator.randint(0, maximum_start)
        result.append([int(token) for token in token_ids[start : start + sequence_length]])
    return result


def write_calibration_artifact(
    output_dir: Path,
    sequences: Sequence[Sequence[int]],
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Write immutable JSON tokens plus a checksummed provenance manifest."""

    if not sequences:
        raise CalibrationArtifactError("calibration sequences must not be empty")
    width = len(sequences[0])
    if width < 2 or any(len(sequence) != width for sequence in sequences):
        raise CalibrationArtifactError("calibration sequences must have one fixed width")
    token_digest = token_ids_sha256(sequences)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / TOKENS_FILE
    manifest_path = output_dir / MANIFEST_FILE
    if tokens_path.exists() or manifest_path.exists():
        raise CalibrationArtifactError(
            "refusing to overwrite an existing calibration artifact"
        )
    try:
        normalized_provenance = json.loads(
            json.dumps(dict(provenance), sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise CalibrationArtifactError(
            "calibration provenance must be JSON serializable"
        ) from error
    tokens_path.write_text(
        json.dumps([list(sequence) for sequence in sequences], separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "tokens": {
            "name": TOKENS_FILE,
            "sha256": _sha256(tokens_path),
            "token_ids_sha256": token_digest,
            "samples": len(sequences),
            "sequence_length": width,
            "total_tokens": len(sequences) * width,
        },
        "provenance": normalized_provenance,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_calibration_artifact(manifest_path: Path) -> Dict[str, Any]:
    """Load calibration tokens only after validating both recorded digests."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationArtifactError("calibration manifest is unreadable") from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != FORMAT
    ):
        raise CalibrationArtifactError("unsupported calibration artifact format")
    token_info = manifest.get("tokens")
    if not isinstance(token_info, dict) or token_info.get("name") != TOKENS_FILE:
        raise CalibrationArtifactError("calibration token metadata is invalid")
    tokens_path = manifest_path.parent / TOKENS_FILE
    if not tokens_path.is_file() or _sha256(tokens_path) != token_info.get("sha256"):
        raise CalibrationArtifactError("calibration token file checksum mismatch")
    try:
        sequences = json.loads(tokens_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationArtifactError("calibration token file is unreadable") from error
    if not isinstance(sequences, list) or not sequences:
        raise CalibrationArtifactError("calibration token file must contain sequences")
    expected_samples = int(token_info.get("samples", -1))
    expected_width = int(token_info.get("sequence_length", -1))
    if (
        len(sequences) != expected_samples
        or expected_width < 2
        or any(
            not isinstance(sequence, list) or len(sequence) != expected_width
            for sequence in sequences
        )
    ):
        raise CalibrationArtifactError("calibration token shapes changed")
    if token_ids_sha256(sequences) != token_info.get("token_ids_sha256"):
        raise CalibrationArtifactError("calibration token ID digest mismatch")
    return {"manifest": manifest, "sequences": sequences}
