"""Portable, configuration-driven local pipeline for LLaMA-family QuaRot checks.

The runner never embeds a concrete model name or dataset.  Model, revision,
data source, device policy, rotation features, and fake-quant bit widths belong
to JSON configuration files.  CUDA-specific work is intentionally absent.
"""

import copy
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import nn

_CACHE_HOME = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_HOME))

from repro.fake_quant_smoke import quantize_linear_weights_in_place, quantize_linear_inputs, quantize_value_projection_outputs
from repro.qk_post_rope import install_post_rope_qk
from repro.structured_hadamard import (
    StructuredHadamardInputLinear,
    normalized_structured_hadamard_matrix,
    structured_hadamard,
    supports_structured_hadamard,
)
from repro.torch_smoke import HadamardInputLinear, UnitRMSNorm, normalized_hadamard_matrix
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, __version__ as TRANSFORMERS_VERSION


class PipelineConfigError(ValueError):
    """Raised when a portable pipeline configuration is incomplete or unsafe."""


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise PipelineConfigError("pipeline configuration must be a JSON object")
    config["_config_dir"] = str(path.resolve().parent)
    return config


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise PipelineConfigError("{} must be an object".format(name))
    return value


def validate_pipeline_config(config: Mapping[str, Any]) -> None:
    required = {"pipeline_version", "model", "data", "runtime", "experiment"}
    missing = sorted(required - set(config))
    if missing:
        raise PipelineConfigError("missing top-level fields: {}".format(", ".join(missing)))
    if config["pipeline_version"] != 1:
        raise PipelineConfigError("unsupported pipeline_version")

    model = _mapping(config, "model")
    data = _mapping(config, "data")
    runtime = _mapping(config, "runtime")
    experiment = _mapping(config, "experiment")
    model_kind = model.get("kind")
    if model_kind not in {"random_config", "pretrained"}:
        raise PipelineConfigError("model.kind must be random_config or pretrained")
    if model_kind == "random_config":
        if not isinstance(model.get("architecture"), str) or not isinstance(model.get("config_overrides"), Mapping):
            raise PipelineConfigError("random_config requires architecture and config_overrides")
    if model_kind == "pretrained" and not isinstance(model.get("id"), str):
        raise PipelineConfigError("pretrained model requires model.id")
    if model.get("dtype", "float32") not in {"float32", "float16", "bfloat16"}:
        raise PipelineConfigError("model.dtype must be float32, float16, or bfloat16")

    if data.get("source") not in {"synthetic", "jsonl_text", "huggingface_text"}:
        raise PipelineConfigError("data.source must be synthetic, jsonl_text, or huggingface_text")
    if data.get("source") == "synthetic" and not isinstance(data.get("num_batches"), int):
        raise PipelineConfigError("synthetic data requires num_batches")
    if data.get("source") == "jsonl_text" and not isinstance(data.get("path"), str):
        raise PipelineConfigError("jsonl_text data requires path")
    if data.get("source") == "huggingface_text" and not isinstance(data.get("id"), str):
        raise PipelineConfigError("huggingface_text data requires id")
    for name in ("batch_size", "sequence_length", "max_samples"):
        value = data.get(name)
        if not isinstance(value, int) or value <= 0:
            raise PipelineConfigError("data.{} must be a positive integer".format(name))

    if runtime.get("device") not in {"auto", "cpu", "mps", "cuda"}:
        raise PipelineConfigError("runtime.device must be auto, cpu, mps, or cuda")
    if not isinstance(runtime.get("allow_fallback", False), bool):
        raise PipelineConfigError("runtime.allow_fallback must be boolean")

    rotation = _mapping(experiment, "rotation")
    quantization = _mapping(experiment, "quantization")
    if rotation.get("residual_mode") not in {"none", "hadamard", "random"}:
        raise PipelineConfigError("rotation.residual_mode must be none, hadamard, or random")
    for name in ("w_bits", "a_bits", "k_bits", "v_bits"):
        bits = quantization.get(name)
        if not isinstance(bits, int) or bits < 2 or bits > 16:
            raise PipelineConfigError("quantization.{} must be an integer in [2, 16]".format(name))
    for name in ("k_group_size", "v_group_size"):
        group_size = quantization.get(name, -1)
        if not isinstance(group_size, int) or group_size == 0 or group_size < -1:
            raise PipelineConfigError("quantization.{} must be -1 or a positive integer".format(name))
    for name in ("k_symmetric", "v_symmetric"):
        if not isinstance(quantization.get(name, True), bool):
            raise PipelineConfigError("quantization.{} must be boolean".format(name))
    for name in ("k_clip_ratio", "v_clip_ratio"):
        clip_ratio = quantization.get(name, 1.0)
        if not isinstance(clip_ratio, (int, float)) or not 0.0 < float(clip_ratio) <= 1.0:
            raise PipelineConfigError("quantization.{} must be in (0, 1]".format(name))


def resolve_device(runtime: Mapping[str, Any]) -> torch.device:
    requested = runtime["device"]
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if available[candidate]:
                return torch.device(candidate)
    if available[requested]:
        return torch.device(requested)
    if runtime.get("allow_fallback", False):
        for candidate in ("cuda", "mps", "cpu"):
            if available[candidate]:
                return torch.device(candidate)
    raise RuntimeError("requested device {} is unavailable; fallback is disabled".format(requested))


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _model_dtype_kwargs(dtype: torch.dtype) -> Dict[str, torch.dtype]:
    """Use the Transformers 5 dtype spelling while retaining 4.x support."""
    major_version = int(TRANSFORMERS_VERSION.split(".", 1)[0])
    return {"dtype": dtype} if major_version >= 5 else {"torch_dtype": dtype}


def load_model_and_tokenizer(config: Mapping[str, Any], device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
    model_spec = config["model"]
    dtype = _torch_dtype(model_spec.get("dtype", "float32"))
    if model_spec["kind"] == "random_config":
        model_config = AutoConfig.for_model(model_spec["architecture"], **dict(model_spec["config_overrides"]))
        model = AutoModelForCausalLM.from_config(model_config, **_model_dtype_kwargs(dtype))
        tokenizer = None
    else:
        common_kwargs = {
            "revision": model_spec.get("revision"),
            "trust_remote_code": bool(model_spec.get("trust_remote_code", False)),
            "local_files_only": bool(model_spec.get("local_files_only", False)),
        }
        model = AutoModelForCausalLM.from_pretrained(model_spec["id"], **common_kwargs, **_model_dtype_kwargs(dtype))
        tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], **common_kwargs)
    return model.to(device).eval(), tokenizer


def _resolve_path(config: Mapping[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def _synthetic_batches(data: Mapping[str, Any], vocab_size: int) -> Iterator[torch.Tensor]:
    batch_size, sequence_length = data["batch_size"], data["sequence_length"]
    seed = int(data.get("seed", 0))
    total = batch_size * sequence_length
    for batch_index in range(data["num_batches"]):
        values = torch.arange(total, dtype=torch.long).reshape(batch_size, sequence_length)
        yield (values + seed + batch_index * total) % vocab_size


def _text_batches(texts: Iterable[str], tokenizer: Any, data: Mapping[str, Any]) -> Iterator[torch.Tensor]:
    sequence_length, batch_size, max_samples = data["sequence_length"], data["batch_size"], data["max_samples"]
    token_buffer: List[int] = []
    yielded, seen = 0, 0
    for text in texts:
        if seen >= max_samples:
            break
        seen += 1
        token_buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        while len(token_buffer) >= sequence_length * batch_size:
            values = token_buffer[: sequence_length * batch_size]
            del token_buffer[: sequence_length * batch_size]
            yield torch.tensor(values, dtype=torch.long).reshape(batch_size, sequence_length)
            yielded += 1
            if yielded >= max_samples:
                return


def token_batches(config: Mapping[str, Any], model: nn.Module, tokenizer: Optional[Any]) -> Iterator[torch.Tensor]:
    data = config["data"]
    if data["source"] == "synthetic":
        yield from _synthetic_batches(data, int(model.config.vocab_size))
        return
    if tokenizer is None:
        raise RuntimeError("a tokenizer is required for text data")
    if data["source"] == "jsonl_text":
        path = _resolve_path(config, data["path"])
        field = data.get("text_field", "text")
        def texts() -> Iterator[str]:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if isinstance(row.get(field), str):
                        yield row[field]
        yield from _text_batches(texts(), tokenizer, data)
        return
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("huggingface_text requires the optional datasets package") from error
    dataset = load_dataset(data["id"], data.get("subset"), split=data.get("split", "validation"), revision=data.get("revision"))
    field = data.get("text_field", "text")
    yield from _text_batches((row[field] for row in dataset if isinstance(row.get(field), str)), tokenizer, data)


def _random_orthogonal(size: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((size, size), dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(dtype=dtype, device=device)


def _rotation_matrix(size: int, mode: str, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
    if mode == "hadamard":
        if supports_structured_hadamard(size):
            return normalized_structured_hadamard_matrix(size, dtype, device)
        return normalized_hadamard_matrix(size, dtype, device)
    if mode == "random":
        return _random_orthogonal(size, dtype, device, seed)
    raise ValueError("rotation matrix requested for unsupported mode")


def _untie_llama_output_head_if_needed(model: nn.Module) -> bool:
    """Separate a tied output head before residual-space reparameterization.

    Input embeddings must be right-multiplied by the residual rotation, while
    the output head must additionally absorb the final RMSNorm scale.  Those
    operations are incompatible on one shared Parameter, so models such as
    SmolLM2 need a copied output head for this mathematically exact transform.
    """
    input_weight = model.model.embed_tokens.weight
    output_weight = model.lm_head.weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        return False
    model.lm_head = copy.deepcopy(model.lm_head)
    return True


def _fuse_llama_norms(model: nn.Module) -> None:
    with torch.no_grad():
        for layer in model.model.layers:
            scale = layer.input_layernorm.weight.detach()
            for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                linear.weight.mul_(scale)
            layer.input_layernorm = UnitRMSNorm(layer.input_layernorm.variance_epsilon)
            scale = layer.post_attention_layernorm.weight.detach()
            for linear in (layer.mlp.up_proj, layer.mlp.gate_proj):
                linear.weight.mul_(scale)
            layer.post_attention_layernorm = UnitRMSNorm(layer.post_attention_layernorm.variance_epsilon)
        scale = model.model.norm.weight.detach()
        model.lm_head.weight.mul_(scale)
        model.model.norm = UnitRMSNorm(model.model.norm.variance_epsilon)


def apply_llama_quarot(
    model: nn.Module, rotation: Mapping[str, Any], quantization: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply the portable QuaRot transformations, including post-RoPE Q/K.

    This adapter supports the standard LLaMA module layout, including GQA. It
    MLP online transforms support a power-of-two dimension and the 12 x
    power-of-two structure used by SmolLM2-135M's 1536-wide MLP.  Post-RoPE
    Q/K processing is installed as a narrow wrapper around the Transformers
    helper so keys are QDQ-ed before cache insertion.
    """
    if getattr(model.config, "model_type", None) != "llama":
        raise RuntimeError("portable QuaRot adapter currently supports model_type=llama only")
    mode = rotation["residual_mode"]
    config = model.config
    hidden_size = int(config.hidden_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    if hidden_size % num_q_heads:
        raise RuntimeError("hidden_size must divide num_attention_heads")
    head_dim = hidden_size // num_q_heads
    if head_dim & (head_dim - 1):
        raise RuntimeError("head_dim must be a power of two for V/O Hadamard rotation")
    for name in ("k_group_size", "v_group_size"):
        group_size = int(quantization.get(name, -1))
        if group_size not in {-1, head_dim}:
            raise RuntimeError("{} must be -1 or head_dim ({})".format(name, head_dim))
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    head = normalized_hadamard_matrix(head_dim, dtype, device)
    qk_post_rope = bool(rotation.get("qk_post_rope", False))
    key_bits = int(quantization["k_bits"])
    if mode == "none":
        if qk_post_rope or key_bits < 16:
            install_post_rope_qk(
                model,
                head,
                rotate_qk=qk_post_rope,
                key_bits=key_bits,
                key_group_size=int(quantization.get("k_group_size", -1)),
                key_symmetric=bool(quantization.get("k_symmetric", True)),
                key_clip_ratio=float(quantization.get("k_clip_ratio", 1.0)),
            )
        return {
            "applied": qk_post_rope or key_bits < 16,
            "reason": "residual_mode=none",
            "qk_post_rope": qk_post_rope,
            "k_bits": key_bits,
            "v_bits": int(quantization["v_bits"]),
        }
    residual = _rotation_matrix(hidden_size, mode, dtype, device, int(rotation.get("seed", 0)))
    query_head_block = torch.block_diag(*([head] * num_q_heads))
    kv_head_block = torch.block_diag(*([head] * num_kv_heads))
    mlp_online = bool(rotation.get("mlp_online", False))
    intermediate_size = int(config.intermediate_size)
    intermediate_is_pow2 = not (intermediate_size & (intermediate_size - 1))
    if mlp_online and not (intermediate_is_pow2 or supports_structured_hadamard(intermediate_size)):
        raise RuntimeError("mlp_online requires a power-of-two or a supported structured intermediate_size")

    output_head_was_untied = _untie_llama_output_head_if_needed(model)
    _fuse_llama_norms(model)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight @ residual)
        model.lm_head.weight.copy_(model.lm_head.weight @ residual)
        for layer in model.model.layers:
            attention, mlp = layer.self_attn, layer.mlp
            for linear in (attention.q_proj, attention.k_proj, mlp.up_proj, mlp.gate_proj):
                linear.weight.copy_(linear.weight @ residual)
            if rotation.get("vo_rotation", True):
                attention.v_proj.weight.copy_(kv_head_block @ attention.v_proj.weight @ residual)
                attention.o_proj.weight.copy_(residual.T @ attention.o_proj.weight @ query_head_block)
                if attention.v_proj.bias is not None:
                    attention.v_proj.bias.copy_(kv_head_block @ attention.v_proj.bias)
                if attention.o_proj.bias is not None:
                    attention.o_proj.bias.copy_(residual.T @ attention.o_proj.bias)
            else:
                attention.v_proj.weight.copy_(attention.v_proj.weight @ residual)
                attention.o_proj.weight.copy_(residual.T @ attention.o_proj.weight)
            if mlp_online:
                if mlp.down_proj.bias is not None:
                    mlp.down_proj.bias.copy_(residual.T @ mlp.down_proj.bias)
                if intermediate_is_pow2:
                    intermediate = normalized_hadamard_matrix(intermediate_size, dtype, device)
                    mlp.down_proj.weight.copy_(residual.T @ mlp.down_proj.weight @ intermediate)
                    mlp.down_proj = HadamardInputLinear(mlp.down_proj, intermediate)
                else:
                    mlp.down_proj.weight.copy_(structured_hadamard(residual.T @ mlp.down_proj.weight))
                    mlp.down_proj = StructuredHadamardInputLinear(mlp.down_proj)
            else:
                mlp.down_proj.weight.copy_(residual.T @ mlp.down_proj.weight)
                if mlp.down_proj.bias is not None:
                    mlp.down_proj.bias.copy_(residual.T @ mlp.down_proj.bias)
    if qk_post_rope or key_bits < 16:
        install_post_rope_qk(
            model,
            head,
            rotate_qk=qk_post_rope,
            key_bits=key_bits,
            key_group_size=int(quantization.get("k_group_size", -1)),
            key_symmetric=bool(quantization.get("k_symmetric", True)),
            key_clip_ratio=float(quantization.get("k_clip_ratio", 1.0)),
        )
    return {
        "applied": True,
        "residual_mode": mode,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "vo_rotation": bool(rotation.get("vo_rotation", True)),
        "mlp_online": mlp_online,
        "mlp_hadamard": "power_of_two" if mlp_online and intermediate_is_pow2 else ("structured" if mlp_online else "none"),
        "qk_post_rope": qk_post_rope,
        "k_bits": key_bits,
        "v_bits": int(quantization["v_bits"]),
        "output_head_was_untied": output_head_was_untied,
    }


def _forward_logits(model: nn.Module, input_ids: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
    if not use_kv_cache:
        return model(input_ids=input_ids, use_cache=False).logits.float()
    cached_logits: List[torch.Tensor] = []
    past_key_values = None
    for index in range(input_ids.shape[1]):
        output = model(input_ids=input_ids[:, index : index + 1], past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        cached_logits.append(output.logits.float())
    return torch.cat(cached_logits, dim=1)


def _evaluate(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    activation_bits: int,
    quantization: Mapping[str, Any],
    use_kv_cache: bool,
) -> Dict[str, float]:
    total_nll, total_tokens, first_logits = 0.0, 0, None
    with torch.inference_mode(), quantize_linear_inputs(model, activation_bits), quantize_value_projection_outputs(
        model,
        int(quantization["v_bits"]),
        group_size=int(quantization.get("v_group_size", -1)),
        symmetric=bool(quantization.get("v_symmetric", True)),
        clip_ratio=float(quantization.get("v_clip_ratio", 1.0)),
    ):
        for input_ids in batches:
            input_ids = input_ids.to(device)
            logits = _forward_logits(model, input_ids, use_kv_cache)
            if first_logits is None:
                first_logits = logits.detach().cpu()
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1), reduction="sum"
            )
            total_nll += loss.item()
            total_tokens += input_ids[:, 1:].numel()
    if total_tokens == 0 or first_logits is None:
        raise RuntimeError("data source did not produce a complete token batch")
    mean_nll = total_nll / total_tokens
    return {"mean_nll": mean_nll, "perplexity": math.exp(min(mean_nll, 80.0)), "tokens": float(total_tokens), "first_logits": first_logits}


def run_pipeline(config: Mapping[str, Any]) -> Dict[str, Any]:
    validate_pipeline_config(config)
    device = resolve_device(config["runtime"])
    torch.manual_seed(int(config["experiment"].get("seed", 0)))
    model, tokenizer = load_model_and_tokenizer(config, device)
    batches = list(token_batches(config, model, tokenizer))
    if not batches:
        raise RuntimeError("data source yielded no batches")
    quantization = config["experiment"]["quantization"]
    use_kv_cache = int(quantization["k_bits"]) < 16 or int(quantization["v_bits"]) < 16
    reference_quantization = dict(quantization)
    reference_quantization.update({"v_bits": 16, "k_bits": 16})
    reference = _evaluate(model, batches, device, activation_bits=16, quantization=reference_quantization, use_kv_cache=use_kv_cache)
    no_candidate_change = (
        config["experiment"]["rotation"]["residual_mode"] == "none"
        and all(int(quantization[name]) >= 16 for name in ("w_bits", "a_bits", "k_bits", "v_bits"))
    )
    if no_candidate_change:
        evaluated = dict(reference)
        rotation_summary = {"applied": False, "reason": "full-precision baseline"}
    elif config["model"]["kind"] == "pretrained":
        # A 13B BF16 checkpoint does not fit twice on a 48 GB smoke GPU.  The
        # reference values are already moved to CPU by _evaluate, so release
        # its model before loading the candidate from the pinned local cache.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        candidate, _ = load_model_and_tokenizer(config, device)
        rotation_summary = apply_llama_quarot(candidate, config["experiment"]["rotation"], quantization)
        quantize_linear_weights_in_place(candidate, int(quantization["w_bits"]))
        evaluated = _evaluate(
            candidate,
            batches,
            device,
            activation_bits=int(quantization["a_bits"]),
            quantization=quantization,
            use_kv_cache=use_kv_cache,
        )
    else:
        candidate = copy.deepcopy(model).eval()
        rotation_summary = apply_llama_quarot(candidate, config["experiment"]["rotation"], quantization)
        quantize_linear_weights_in_place(candidate, int(quantization["w_bits"]))
        evaluated = _evaluate(
            candidate,
            batches,
            device,
            activation_bits=int(quantization["a_bits"]),
            quantization=quantization,
            use_kv_cache=use_kv_cache,
        )
    logit_error = (reference.pop("first_logits") - evaluated.pop("first_logits")).abs()
    return {
        "device": str(device),
        "model": {
            "kind": config["model"]["kind"],
            "id": config["model"].get("id"),
            "revision": config["model"].get("revision"),
            "dtype": config["model"].get("dtype", "float32"),
        },
        "data": {
            "source": config["data"]["source"],
            "id": config["data"].get("id"),
            "subset": config["data"].get("subset"),
            "split": config["data"].get("split"),
            "revision": config["data"].get("revision"),
            "sequence_length": config["data"]["sequence_length"],
            "batch_size": config["data"]["batch_size"],
            "max_samples": config["data"]["max_samples"],
            "batches": len(batches),
        },
        "experiment": {"seed": int(config["experiment"].get("seed", 0))},
        "reference": reference,
        "candidate": evaluated,
        "rotation": rotation_summary,
        "quantization": dict(quantization),
        "kv_cache_simulated": use_kv_cache,
        "logit_error": {"mean_absolute": logit_error.mean().item(), "max_absolute": logit_error.max().item()},
    }
