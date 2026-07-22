#!/usr/bin/env python3
"""Create deterministic BF16 and GPTQ W4A16 tiny Llama checkpoints."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from repro.offline_llama_rotation import apply_offline_llama_rotation


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _model(seed: int, tiny: dict) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=tiny["vocab_size"],
        hidden_size=tiny["hidden_size"],
        intermediate_size=tiny["intermediate_size"],
        num_hidden_layers=tiny["num_hidden_layers"],
        num_attention_heads=tiny["num_attention_heads"],
        num_key_value_heads=tiny["num_key_value_heads"],
        max_position_embeddings=tiny["max_position_embeddings"],
        tie_word_embeddings=True,
        attention_bias=False,
    )
    return LlamaForCausalLM(config).float().eval()


def _calibration_dataset(samples: int, sequence_length: int, vocab_size: int) -> Dataset:
    rows = []
    masks = []
    for row in range(samples):
        tokens = ((torch.arange(sequence_length) * (row + 3) + row + 1) % (vocab_size - 1)) + 1
        rows.append(tokens.tolist())
        masks.append([1] * sequence_length)
    return Dataset.from_dict({"input_ids": rows, "attention_mask": masks})


def _processor(vocab_size: int) -> PreTrainedTokenizerFast:
    vocabulary = {f"token_{index}": index for index in range(vocab_size)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="token_0"))
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="token_0",
        pad_token="token_0",
        bos_token="token_1",
        eos_token="token_2",
    )


def _quantize(
    model: LlamaForCausalLM,
    dataset: Dataset,
    output: Path,
    quantization: dict,
    processor: PreTrainedTokenizerFast,
) -> dict:
    if quantization["group_size"] != 128:
        raise ValueError("the frozen W4A16 recipe requires group_size=128")
    recipe = GPTQModifier(
        targets=quantization["targets"],
        scheme=quantization["scheme"],
        ignore=quantization["ignore"],
    )
    oneshot(
        model=model,
        dataset=dataset,
        processor=processor,
        recipe=recipe,
        max_seq_length=len(dataset[0]["input_ids"]),
        num_calibration_samples=len(dataset),
    )
    model.save_pretrained(output, save_compressed=True, safe_serialization=True)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise AssertionError(f"missing quantization_config in {output}")
    groups = list(quantization.get("config_groups", {}).values())
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("format") != "pack-quantized"
        or quantization.get("quantization_status") != "compressed"
        or len(groups) != 1
        or groups[0].get("weights", {}).get("num_bits") != 4
        or groups[0].get("weights", {}).get("group_size") != 128
    ):
        raise AssertionError(f"checkpoint is not packed compressed-tensors W4 group-128: {quantization}")

    packed_weights = []
    for weights_file in output.glob("*.safetensors"):
        with safe_open(weights_file, framework="pt", device="cpu") as tensors:
            packed_weights.extend(key for key in tensors.keys() if key.endswith(".weight_packed"))
    expected_packed_linears = model.config.num_hidden_layers * 7
    if len(packed_weights) != expected_packed_linears:
        raise AssertionError(
            f"expected {expected_packed_linears} packed decoder linears, found {len(packed_weights)}"
        )
    return {
        "quantization_config": quantization,
        "packed_decoder_linear_count": len(packed_weights),
    }


def prepare(output_dir: Path, config: dict) -> dict:
    tiny = config["tiny_smoke"]
    quantization = config["quantization"]
    seed = tiny["seed"]
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _calibration_dataset(
        samples=tiny["calibration_samples"],
        sequence_length=tiny["calibration_sequence_length"],
        vocab_size=tiny["vocab_size"],
    )
    processor = _processor(tiny["vocab_size"])
    baseline = _model(seed, tiny)
    state = copy.deepcopy(baseline.state_dict())

    bf16_dir = output_dir / "bf16"
    baseline.bfloat16().save_pretrained(bf16_dir, safe_serialization=True)

    unrotated = _model(seed, tiny)
    unrotated.load_state_dict(state)
    unrotated_quantization = _quantize(
        unrotated,
        dataset,
        output_dir / "unrotated-w4a16",
        quantization,
        processor,
    )

    rotated = _model(seed, tiny)
    rotated.load_state_dict(state)
    rotation = apply_offline_llama_rotation(rotated)
    rotated_quantization = _quantize(
        rotated,
        dataset,
        output_dir / "rotated-w4a16",
        quantization,
        processor,
    )

    result = {
        "status": "passed",
        "scope": "deterministic tiny checkpoints; no vLLM or GPU execution",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_revision": _revision(),
        "seed": seed,
        "calibration_samples": len(dataset),
        "calibration_sequence_length": len(dataset[0]["input_ids"]),
        "checkpoints": {
            "bf16": str(bf16_dir),
            "unrotated_w4a16": str(output_dir / "unrotated-w4a16"),
            "rotated_w4a16": str(output_dir / "rotated-w4a16"),
        },
        "rotation": rotation,
        "unrotated_quantization_config": unrotated_quantization,
        "rotated_quantization_config": rotated_quantization,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/deployment/vllm_w4a16_isambard.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = prepare(args.output_dir.resolve(), config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
