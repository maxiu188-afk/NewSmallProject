#!/usr/bin/env python3
"""Train independent SpinQuant R1/R2 from pinned local artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM

from repro.spinquant.artifacts import save_rotation_artifact
from repro.spinquant.calibration import load_calibration_artifact
from repro.spinquant.training import train_llama_rotations


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


def _validate_calibration_provenance(
    config: dict,
    calibration: dict,
    model_snapshot: Path,
) -> None:
    expected = config["calibration"]
    calibration_config_path = (PROJECT_ROOT / expected["config"]).resolve()
    calibration_config = json.loads(
        calibration_config_path.read_text(encoding="utf-8")
    )
    expected_model = calibration_config["model"]
    expected_dataset = calibration_config["calibration"]
    training_model = config["model"]
    for key in ("id", "revision", "trust_remote_code"):
        if expected_model[key] != training_model[key]:
            raise ValueError(
                "calibration and training model definitions do not match"
            )

    provenance = calibration["manifest"].get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("calibration provenance is missing")
    required = {
        "project_revision": _revision(),
        "config_sha256": _sha256(calibration_config_path),
        "model": expected_model,
        "model_config_sha256": _sha256(model_snapshot / "config.json"),
        "dataset": expected_dataset,
    }
    for key, value in required.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"calibration provenance mismatch for {key}"
            )


def train(
    config_path: Path,
    model_snapshot: Path,
    calibration_manifest: Path,
    output_dir: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal SpinQuant rotation training requires CUDA")
    if model_snapshot.name != config["model"]["revision"]:
        raise ValueError("model snapshot directory does not match pinned revision")
    calibration = load_calibration_artifact(calibration_manifest)
    _validate_calibration_provenance(config, calibration, model_snapshot)
    expected = config["calibration"]
    token_info = calibration["manifest"]["tokens"]
    if (
        int(token_info["samples"]) != int(expected["samples"])
        or int(token_info["sequence_length"]) != int(expected["sequence_length"])
    ):
        raise ValueError("calibration artifact does not match training config")

    dtype_name = config["model"]["dtype"]
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in dtypes:
        raise ValueError("formal SpinQuant model dtype must be bfloat16 or float16")
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtypes[dtype_name],
        device_map="cuda",
        attn_implementation=str(config["runtime"]["attention_implementation"]),
    )
    if model.config.architectures != [config["model"]["expected_architecture"]]:
        raise RuntimeError("loaded model architecture changed")
    if int(model.config.num_hidden_layers) != int(config["model"]["expected_layers"]):
        raise RuntimeError("loaded model layer count changed")

    rotations, training = train_llama_rotations(
        model,
        calibration["sequences"],
        config,
    )
    provenance = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config": str(config_path),
        "model": config["model"],
        "calibration_manifest": str(calibration_manifest),
        "calibration_token_ids_sha256": token_info["token_ids_sha256"],
        "training": training,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    manifest_path = save_rotation_artifact(
        output_dir,
        rotations,
        provenance=provenance,
    )
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train(
        args.config.resolve(),
        args.model_snapshot.resolve(),
        args.calibration_manifest.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
