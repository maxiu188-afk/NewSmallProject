#!/usr/bin/env python3
"""Audit runtime A8 coverage and error for matched no-had W4A8 cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spinquant.evaluate_llama_nohad_w4a8_ppl import (
    _load_gptq_calibration,
    _validate_config,
)
from scripts.spinquant.evaluate_llama_w4a16_ppl import (
    _comparison,
    _evaluate_model,
    _load_sequences,
    _revision,
    _rotation_module,
    _sha256,
    _write_json,
)


AUDIT_CASES = ("unrotated_w4a8", "spinquant_nohad_w4a8")
AUDIT_MODES = ("current_repo", "paper_aligned")


def _prepare_weights(
    model: Any,
    *,
    case: str,
    config: dict[str, Any],
    rotation_manifest_path: Path,
    calibration_dir: Path,
    max_calibration_sequences: int,
) -> dict[str, Any]:
    from repro.gptq import GPTQSettings, quantize_llama_weights_gptq
    from repro.spinquant.artifacts import load_rotation_artifact
    from repro.spinquant.offline import apply_spinquant_llama_offline

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
    expected = int(config["model"]["expected_decoder_linears"])
    if gptq["linear_layers"] != expected:
        raise RuntimeError("GPTQ did not quantize every decoder linear")
    return {
        "rotation": rotation_summary,
        "rotation_orthogonality_error": rotation_errors,
        "gptq_calibration": calibration_summary,
        "gptq": gptq,
    }


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

    from repro.spinquant.activation_audit import audit_llama_decoder_activations

    if not torch.cuda.is_available():
        raise RuntimeError("Llama-2-13B activation audit requires CUDA")
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

    preparation = _prepare_weights(
        model,
        case=case,
        config=config,
        rotation_manifest_path=rotation_manifest_path,
        calibration_dir=calibration_dir,
        max_calibration_sequences=max_calibration_sequences,
    )
    modes = {}
    for mode in AUDIT_MODES:
        with audit_llama_decoder_activations(model, mode=mode) as audit:
            evaluation = _evaluate_model(
                model,
                sequences,
                batch_size=int(config["evaluation"]["batch_size"]),
                logit_chunk_tokens=int(config["evaluation"]["logit_chunk_tokens"]),
            )
        activation_audit = audit.summary()
        expected = int(config["model"]["expected_decoder_linears"])
        if (
            activation_audit["registered_modules"] != expected
            or activation_audit["called_modules"] != expected
            or activation_audit["missing_modules"]
        ):
            raise RuntimeError("runtime A8 audit did not cover every decoder linear")
        modes[mode] = {
            **evaluation,
            "activation_audit": activation_audit,
        }

    return {
        "name": case,
        "preparation": preparation,
        "modes": modes,
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
    max_sequences: int,
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
    cases = {}
    for case in AUDIT_CASES:
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
            "--max-sequences",
            str(max_sequences),
            "--max-calibration-sequences",
            str(max_calibration_sequences),
            "--worker-case",
            case,
            "--output",
            str(worker_output),
        ]
        subprocess.run(command, check=True)
        result = json.loads(worker_output.read_text(encoding="utf-8"))
        if result["name"] != case:
            raise RuntimeError("activation audit worker returned the wrong case")
        cases[case] = result

    runtimes = {json.dumps(case["runtime"], sort_keys=True) for case in cases.values()}
    if len(runtimes) != 1:
        raise RuntimeError("activation audit workers used inconsistent runtimes")
    comparisons = {
        f"spinquant_vs_unrotated_{mode}": _comparison(
            cases["spinquant_nohad_w4a8"]["modes"][mode],
            cases["unrotated_w4a8"]["modes"][mode],
        )
        for mode in AUDIT_MODES
    }
    comparisons.update(
        {
            f"paper_aligned_vs_current_repo_{case}": _comparison(
                cases[case]["modes"]["paper_aligned"],
                cases[case]["modes"]["current_repo"],
            )
            for case in AUDIT_CASES
        }
    )
    return {
        "status": "passed",
        "scope": (
            "two-sequence runtime activation coverage and quantization-error audit; "
            "PPL values are diagnostic smoke evidence, not formal quality results"
        ),
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
        "cases": cases,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-manifest", type=Path, required=True)
    parser.add_argument("--gptq-calibration-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--max-calibration-sequences", type=int, required=True)
    parser.add_argument("--worker-case", choices=AUDIT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    model_snapshot = args.model_snapshot.resolve()
    if model_snapshot.name != config["model"]["revision"]:
        raise ValueError("model snapshot does not match the pinned revision")
    tokens_manifest_path = args.tokens_manifest.resolve()
    sequences, _ = _load_sequences(config, tokens_manifest_path, args.max_sequences)
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
        print(f"SPINQUANT_A8_AUDIT_MODEL_PASSED={args.worker_case}")
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
