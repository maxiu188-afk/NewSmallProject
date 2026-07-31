#!/usr/bin/env python3
"""Evaluate paper-aligned SpinQuant no-had GPTQ W4A8KV16 perplexity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spinquant.evaluate_llama_w4a16_ppl import (
    _comparison,
    _evaluate_model,
    _load_sequences,
    _revision,
    _rotation_module,
    _sha256,
    _write_json,
)


EXPECTED_CASES = (
    "bf16",
    "unrotated_w4a8",
    "spinquant_nohad_w4a8",
)


def _validate_config(config: dict[str, Any]) -> None:
    if tuple(config["evaluation"]["cases"]) != EXPECTED_CASES:
        raise ValueError(f"evaluation cases must be {EXPECTED_CASES}")
    quantization = config["quantization"]
    expected = {
        "scheme": "W4A8KV16",
        "weight_algorithm": "GPTQ",
        "weight_bits": 4,
        "activation_bits": 8,
        "activation_symmetric": False,
        "activation_clipping": False,
        "activation_granularity": "per_token_last_axis",
        "key_bits": 16,
        "value_bits": 16,
        "quantize_lm_head": False,
    }
    for name, value in expected.items():
        if quantization.get(name) != value:
            raise ValueError(f"quantization.{name} must be {value!r}")
    if (
        int(quantization["weight_group_size"]) != 128
        or int(quantization["weight_block_size"]) != 128
        or not bool(quantization["weight_act_order"])
        or not bool(quantization["weight_symmetric"])
    ):
        raise ValueError("GPTQ must use symmetric group/block-128 W4 with act order")


def _load_gptq_calibration(
    config: dict[str, Any],
    calibration_dir: Path,
    max_sequences: int,
) -> tuple[list[Any], dict[str, Any]]:
    import torch
    from datasets import load_from_disk

    spec = config["gptq_calibration"]
    manifest_path = calibration_dir / "manifest.json"
    if _sha256(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError("GPTQ calibration manifest SHA256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError("GPTQ calibration manifest did not pass")
    dataset_spec = manifest.get("dataset", {})
    required = {
        "dataset_id": spec["dataset_id"],
        "dataset_subset": spec["dataset_subset"],
        "dataset_split": spec["dataset_split"],
        "dataset_revision": spec["dataset_revision"],
        "samples": int(spec["samples"]),
        "sequence_length": int(spec["sequence_length"]),
        "token_ids_sha256": spec["token_ids_sha256"],
    }
    for name, value in required.items():
        if dataset_spec.get(name) != value:
            raise RuntimeError(f"GPTQ calibration {name} changed")
    if max_sequences < 1 or max_sequences > int(spec["samples"]):
        raise ValueError("GPTQ calibration selection is outside the materialized range")

    dataset = load_from_disk(calibration_dir / "dataset")
    if len(dataset) != int(spec["samples"]):
        raise RuntimeError("GPTQ calibration dataset sample count changed")
    digest = hashlib.sha256()
    selected = []
    width = int(spec["sequence_length"])
    for index, row in enumerate(dataset):
        tokens = row.get("input_ids")
        if (
            not isinstance(tokens, list)
            or len(tokens) != width
            or not all(isinstance(token, int) for token in tokens)
        ):
            raise RuntimeError("GPTQ calibration token shape or type changed")
        for token in tokens:
            digest.update(struct.pack("<I", token))
        if index < max_sequences:
            selected.append(torch.tensor([tokens], dtype=torch.long))
    if digest.hexdigest() != spec["token_ids_sha256"]:
        raise RuntimeError("GPTQ calibration token IDs changed")
    return selected, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "token_ids_sha256": digest.hexdigest(),
        "available_sequences": len(dataset),
        "selected_sequences": len(selected),
        "sequence_length": width,
        "selected_tokens": len(selected) * width,
        "selection": spec["selection"],
    }


def _prepare_case(
    model: Any,
    case: str,
    config: dict[str, Any],
    rotation_manifest_path: Path,
    calibration_dir: Path,
    max_calibration_sequences: int,
) -> tuple[dict[str, Any], Any]:
    from repro.gptq import GPTQSettings, quantize_llama_weights_gptq
    from repro.spinquant.artifacts import load_rotation_artifact
    from repro.spinquant.offline import (
        apply_spinquant_llama_offline,
        fake_quantize_llama_decoder_activations,
    )

    rotation_summary = None
    rotation_errors = None
    if case == "spinquant_nohad_w4a8":
        rotations = _rotation_module(model, int(config["seed"]))
        loaded = load_rotation_artifact(
            rotation_manifest_path,
            target=rotations,
            maximum_orthogonality_error=1e-4,
        )
        rotation_errors = loaded["observed_orthogonality_error"]
        rotation_summary = apply_spinquant_llama_offline(model, rotations)
        del rotations

    if case == "bf16":
        return {
            "rotation": None,
            "rotation_orthogonality_error": None,
            "gptq_calibration": None,
            "gptq": None,
            "activation_quantization": None,
        }, None

    calibration_batches, calibration_summary = _load_gptq_calibration(
        config,
        calibration_dir,
        max_calibration_sequences,
    )
    quantization = config["quantization"]
    gptq = quantize_llama_weights_gptq(
        model,
        calibration_batches,
        GPTQSettings(
            bits=int(quantization["weight_bits"]),
            group_size=int(quantization["weight_group_size"]),
            damp_percent=float(quantization["weight_damp_percent"]),
            block_size=int(quantization["weight_block_size"]),
            act_order=bool(quantization["weight_act_order"]),
            symmetric=bool(quantization["weight_symmetric"]),
        ),
    )
    expected_linears = int(config["model"]["expected_decoder_linears"])
    if gptq["linear_layers"] != expected_linears:
        raise RuntimeError("GPTQ did not quantize every decoder linear")
    activation_context = fake_quantize_llama_decoder_activations(
        model,
        bits=int(quantization["activation_bits"]),
        symmetric=bool(quantization["activation_symmetric"]),
    )
    preparation = {
        "rotation": rotation_summary,
        "rotation_orthogonality_error": rotation_errors,
        "gptq_calibration": calibration_summary,
        "gptq": gptq,
    }
    return preparation, activation_context


def _worker(
    *,
    case: str,
    config: dict[str, Any],
    model_snapshot: Path,
    rotation_manifest_path: Path,
    calibration_dir: Path,
    max_calibration_sequences: int,
    sequences: list[list[int]],
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("formal SpinQuant no-had W4A8 PPL requires CUDA")
    dtype = {"bfloat16": torch.bfloat16}[config["model"]["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map="cuda",
        attn_implementation=config["evaluation"]["attention_implementation"],
    )
    if model.config.architectures != [config["model"]["expected_architecture"]]:
        raise RuntimeError("loaded model architecture changed")
    if int(model.config.num_hidden_layers) != int(config["model"]["expected_layers"]):
        raise RuntimeError("loaded model layer count changed")

    preparation, activation_context = _prepare_case(
        model,
        case,
        config,
        rotation_manifest_path,
        calibration_dir,
        max_calibration_sequences,
    )
    if activation_context is None:
        evaluation = _evaluate_model(
            model,
            sequences,
            batch_size=int(config["evaluation"]["batch_size"]),
            logit_chunk_tokens=int(config["evaluation"]["logit_chunk_tokens"]),
        )
    else:
        with activation_context as activation_summary:
            evaluation = _evaluate_model(
                model,
                sequences,
                batch_size=int(config["evaluation"]["batch_size"]),
                logit_chunk_tokens=int(config["evaluation"]["logit_chunk_tokens"]),
            )
        preparation["activation_quantization"] = activation_summary
    return {
        "name": case,
        "preparation": preparation,
        **evaluation,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }


def _run_parent(
    *,
    config_path: Path,
    tokens_manifest_path: Path,
    model_snapshot: Path,
    rotation_manifest_path: Path,
    calibration_dir: Path,
    max_sequences: int | None,
    max_calibration_sequences: int,
    output_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    sequences, token_manifest = _load_sequences(
        config,
        tokens_manifest_path,
        max_sequences,
    )
    worker_dir = output_path.parent / f"{output_path.stem}-workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in EXPECTED_CASES:
        worker_output = worker_dir / f"{case}.json"
        worker_output.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--tokens-manifest",
            str(tokens_manifest_path),
            "--gptq-calibration-dir",
            str(calibration_dir),
            "--model-snapshot",
            str(model_snapshot),
            "--rotation-manifest",
            str(rotation_manifest_path),
            "--max-calibration-sequences",
            str(max_calibration_sequences),
            "--worker-case",
            case,
            "--output",
            str(worker_output),
        ]
        if max_sequences is not None:
            command.extend(["--max-sequences", str(max_sequences)])
        subprocess.run(command, check=True)
        result = json.loads(worker_output.read_text(encoding="utf-8"))
        if result["name"] != case:
            raise RuntimeError("W4A8 PPL worker returned the wrong case")
        results[case] = result
    runtimes = {json.dumps(value["runtime"], sort_keys=True) for value in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("W4A8 PPL workers used inconsistent runtimes")
    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "config_sha256": _sha256(config_path),
        "tokens_manifest": str(tokens_manifest_path),
        "tokens_manifest_sha256": _sha256(tokens_manifest_path),
        "token_ids_sha256": token_manifest["dataset"]["token_ids_sha256"],
        "rotation_manifest": str(rotation_manifest_path),
        "rotation_manifest_sha256": _sha256(rotation_manifest_path),
        "gptq_calibration_manifest": str(calibration_dir / "manifest.json"),
        "gptq_calibration_manifest_sha256": _sha256(calibration_dir / "manifest.json"),
        "gptq_calibration_sequences": max_calibration_sequences,
        "evaluated_sequences": len(sequences),
        "scored_tokens": len(sequences) * (len(sequences[0]) - 1),
        "models": results,
        "comparisons": {
            f"{case}_vs_bf16": _comparison(results[case], results["bf16"])
            for case in EXPECTED_CASES
            if case != "bf16"
        }
        | {
            "spinquant_vs_unrotated_w4a8": _comparison(
                results["spinquant_nohad_w4a8"],
                results["unrotated_w4a8"],
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-manifest", type=Path, required=True)
    parser.add_argument("--gptq-calibration-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-calibration-sequences", type=int, required=True)
    parser.add_argument("--worker-case", choices=EXPECTED_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    model_snapshot = args.model_snapshot.resolve()
    if model_snapshot.name != config["model"]["revision"]:
        raise ValueError("model snapshot does not match the pinned revision")
    tokens_manifest_path = args.tokens_manifest.resolve()
    sequences, _ = _load_sequences(
        config,
        tokens_manifest_path,
        args.max_sequences,
    )
    rotation_manifest_path = args.rotation_manifest.resolve()
    calibration_dir = args.gptq_calibration_dir.resolve()
    if args.worker_case is not None:
        result = _worker(
            case=args.worker_case,
            config=config,
            model_snapshot=model_snapshot,
            rotation_manifest_path=rotation_manifest_path,
            calibration_dir=calibration_dir,
            max_calibration_sequences=args.max_calibration_sequences,
            sequences=sequences,
        )
        _write_json(args.output.resolve(), result)
        print(f"SPINQUANT_NOHAD_W4A8_PPL_MODEL_PASSED={args.worker_case}")
        return 0

    result = _run_parent(
        config_path=config_path,
        tokens_manifest_path=tokens_manifest_path,
        model_snapshot=model_snapshot,
        rotation_manifest_path=rotation_manifest_path,
        calibration_dir=calibration_dir,
        max_sequences=args.max_sequences,
        max_calibration_sequences=args.max_calibration_sequences,
        output_path=args.output.resolve(),
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
