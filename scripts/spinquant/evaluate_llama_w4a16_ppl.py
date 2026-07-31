#!/usr/bin/env python3
"""Evaluate matched Llama-2-13B R1/R2 W4A16 fake-quant perplexity."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_CASES = (
    "bf16",
    "unrotated_w4a16",
    "quarot_r1r2_w4a16",
    "spinquant_r1r2_w4a16",
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_sequences(
    config: dict,
    tokens_manifest_path: Path,
    max_sequences: int | None,
) -> tuple[list[list[int]], dict]:
    token_spec = config["tokens"]
    manifest = json.loads(tokens_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise RuntimeError("PPL token manifest did not pass")
    if _sha256(tokens_manifest_path) != token_spec["manifest_sha256"]:
        raise RuntimeError("PPL token manifest SHA256 changed")
    if manifest["dataset"]["token_ids_sha256"] != token_spec["token_ids_sha256"]:
        raise RuntimeError("PPL token ID SHA256 changed")
    token_config_path = (PROJECT_ROOT / token_spec["config"]).resolve()
    if manifest["config_sha256"] != _sha256(token_config_path):
        raise RuntimeError("PPL token manifest used a different frozen config")
    tokens_path = Path(manifest["tokens"]["path"])
    if _sha256(tokens_path) != manifest["tokens"]["sha256"]:
        raise RuntimeError("PPL token file SHA256 changed")
    sequences = json.loads(tokens_path.read_text(encoding="utf-8"))
    if (
        len(sequences) != int(token_spec["samples"])
        or any(
            not isinstance(sequence, list)
            or len(sequence) != int(token_spec["sequence_length"])
            or not all(isinstance(token, int) for token in sequence)
            for sequence in sequences
        )
    ):
        raise RuntimeError("PPL token shape or type changed")
    if max_sequences is not None:
        if max_sequences < 1 or max_sequences > len(sequences):
            raise ValueError("max_sequences is outside the materialized range")
        sequences = sequences[:max_sequences]
    return sequences, manifest


def _rotation_module(model: Any, seed: int):
    import torch

    from repro.spinquant.rotations import SpinQuantRotations

    config = model.config
    return SpinQuantRotations(
        hidden_size=int(config.hidden_size),
        head_dim=int(config.hidden_size) // int(config.num_attention_heads),
        num_layers=int(config.num_hidden_layers),
        seed=seed,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )


def _prepare_case(
    model: Any,
    case: str,
    config: dict,
    rotation_manifest_path: Path,
) -> dict:
    from repro.spinquant.artifacts import load_rotation_artifact
    from repro.spinquant.offline import (
        apply_spinquant_llama_offline,
        fake_quantize_llama_decoder_w4_,
    )

    rotation_summary = None
    rotation_errors = None
    if case in {"quarot_r1r2_w4a16", "spinquant_r1r2_w4a16"}:
        rotations = _rotation_module(model, int(config["seed"]))
        if case == "spinquant_r1r2_w4a16":
            loaded = load_rotation_artifact(
                rotation_manifest_path,
                target=rotations,
                maximum_orthogonality_error=1e-4,
            )
            rotation_errors = loaded["observed_orthogonality_error"]
        else:
            rotation_errors = rotations.errors()
        rotation_summary = apply_spinquant_llama_offline(model, rotations)
        del rotations

    quantization_summary = None
    if case != "bf16":
        spec = config["quantization"]
        quantization_summary = fake_quantize_llama_decoder_w4_(
            model,
            bits=int(spec["weight_bits"]),
            group_size=int(spec["weight_group_size"]),
            symmetric=bool(spec["weight_symmetric"]),
        )
        expected = int(config["model"]["expected_layers"]) * 7
        if quantization_summary["quantized_decoder_linears"] != expected:
            raise RuntimeError("fake quantization did not cover every decoder linear")
    return {
        "rotation": rotation_summary,
        "rotation_orthogonality_error": rotation_errors,
        "quantization": quantization_summary,
    }


def _evaluate_model(
    model: Any,
    sequences: list[list[int]],
    *,
    batch_size: int,
    logit_chunk_tokens: int,
) -> dict:
    import torch
    import torch.nn.functional as functional

    if batch_size < 1 or logit_chunk_tokens < 1:
        raise ValueError("evaluation batch and chunk sizes must be positive")
    device = next(model.parameters()).device
    model.eval()
    model.config.use_cache = False
    total_nll = 0.0
    total_tokens = 0
    batch_metrics = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(sequences), batch_size)):
            selected = sequences[start : start + batch_size]
            input_ids = torch.tensor(selected, dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            batch_nll = 0.0
            for position in range(0, input_ids.shape[1] - 1, logit_chunk_tokens):
                stop = min(
                    position + logit_chunk_tokens,
                    input_ids.shape[1] - 1,
                )
                selected_logits = logits[:, position:stop, :].float()
                labels = input_ids[:, position + 1 : stop + 1]
                nll = functional.cross_entropy(
                    selected_logits.reshape(-1, selected_logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
                batch_nll += float(nll)
            tokens = len(selected) * (input_ids.shape[1] - 1)
            if not math.isfinite(batch_nll):
                raise RuntimeError("fake-quant PPL produced non-finite NLL")
            total_nll += batch_nll
            total_tokens += tokens
            batch_metrics.append(
                {
                    "index": batch_index,
                    "sequences": len(selected),
                    "tokens": tokens,
                    "nll": batch_nll,
                    "mean_nll": batch_nll / tokens,
                }
            )
            del logits, input_ids
    mean_nll = total_nll / total_tokens
    return {
        "evaluated_sequences": len(sequences),
        "tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "batch_metrics": batch_metrics,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _worker(
    *,
    case: str,
    config: dict,
    model_snapshot: Path,
    rotation_manifest_path: Path,
    sequences: list[list[int]],
) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("formal SpinQuant fake-quant PPL requires CUDA")
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
    preparation = _prepare_case(
        model,
        case,
        config,
        rotation_manifest_path,
    )
    evaluation = _evaluate_model(
        model,
        sequences,
        batch_size=int(config["evaluation"]["batch_size"]),
        logit_chunk_tokens=int(config["evaluation"]["logit_chunk_tokens"]),
    )
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


def _comparison(candidate: dict, reference: dict) -> dict[str, float]:
    if candidate["tokens"] != reference["tokens"]:
        raise RuntimeError("PPL comparison uses different token counts")
    return {
        "mean_nll_delta": candidate["mean_nll"] - reference["mean_nll"],
        "perplexity_delta": candidate["perplexity"] - reference["perplexity"],
        "perplexity_ratio": candidate["perplexity"] / reference["perplexity"],
    }


def _run_parent(
    *,
    config_path: Path,
    tokens_manifest_path: Path,
    model_snapshot: Path,
    rotation_manifest_path: Path,
    max_sequences: int | None,
    output_path: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["evaluation"]["cases"]) != EXPECTED_CASES:
        raise ValueError(f"evaluation cases must be {EXPECTED_CASES}")
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
            "--model-snapshot",
            str(model_snapshot),
            "--rotation-manifest",
            str(rotation_manifest_path),
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
            raise RuntimeError("fake-quant PPL worker returned the wrong case")
        results[case] = result
    runtimes = {json.dumps(value["runtime"], sort_keys=True) for value in results.values()}
    if len(runtimes) != 1:
        raise RuntimeError("fake-quant PPL workers used inconsistent runtimes")
    return {
        "status": "passed",
        "scope": config["scope"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "tokens_manifest": str(tokens_manifest_path),
        "tokens_manifest_sha256": _sha256(tokens_manifest_path),
        "token_ids_sha256": token_manifest["dataset"]["token_ids_sha256"],
        "rotation_manifest": str(rotation_manifest_path),
        "rotation_manifest_sha256": _sha256(rotation_manifest_path),
        "evaluated_sequences": len(sequences),
        "scored_tokens": len(sequences) * (len(sequences[0]) - 1),
        "models": results,
        "comparisons": {
            f"{case}_vs_bf16": _comparison(results[case], results["bf16"])
            for case in EXPECTED_CASES
            if case != "bf16"
        }
        | {
            "spinquant_vs_quarot_r1r2": _comparison(
                results["spinquant_r1r2_w4a16"],
                results["quarot_r1r2_w4a16"],
            ),
            "spinquant_vs_unrotated": _comparison(
                results["spinquant_r1r2_w4a16"],
                results["unrotated_w4a16"],
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens-manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--rotation-manifest", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--worker-case", choices=EXPECTED_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokens_manifest_path = args.tokens_manifest.resolve()
    model_snapshot = args.model_snapshot.resolve()
    rotation_manifest_path = args.rotation_manifest.resolve()
    if model_snapshot.name != config["model"]["revision"]:
        raise ValueError("model snapshot does not match the pinned revision")
    sequences, _ = _load_sequences(
        config,
        tokens_manifest_path,
        args.max_sequences,
    )
    if args.worker_case is not None:
        result = _worker(
            case=args.worker_case,
            config=config,
            model_snapshot=model_snapshot,
            rotation_manifest_path=rotation_manifest_path,
            sequences=sequences,
        )
        _write_json(args.output.resolve(), result)
        print(f"SPINQUANT_FAKE_QUANT_PPL_MODEL_PASSED={args.worker_case}")
        return 0
    result = _run_parent(
        config_path=config_path,
        tokens_manifest_path=tokens_manifest_path,
        model_snapshot=model_snapshot,
        rotation_manifest_path=rotation_manifest_path,
        max_sequences=args.max_sequences,
        output_path=args.output.resolve(),
    )
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
