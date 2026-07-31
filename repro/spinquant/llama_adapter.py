"""Training-only SpinQuant adapter for the standard Transformers Llama graph.

The adapter delegates attention, RoPE, cache handling, MLP activations, and the
model forward to the installed Transformers implementation.  It replaces only
embeddings, RMSNorms, and Linear leaves with small mathematical wrappers; no
Transformers model source is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import weakref

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from repro.qdq import qdq_last_axis
from repro.spinquant.rotations import SpinQuantRotations
from repro.torch_smoke import UnitRMSNorm


@dataclass(frozen=True)
class SpinQuantFakeQuantSpec:
    """Quantization simulated while learning R1/R2."""

    weight_bits: int
    activation_bits: int = 16
    weight_group_size: int = -1
    weight_symmetric: bool = True
    activation_symmetric: bool = False
    quantize_lm_head: bool = False

    def validate(self) -> None:
        for name, bits in (
            ("weight_bits", self.weight_bits),
            ("activation_bits", self.activation_bits),
        ):
            if bits < 2 or bits > 16:
                raise ValueError(f"{name} must be in [2, 16]")
        if self.weight_group_size == 0 or self.weight_group_size < -1:
            raise ValueError("weight_group_size must be -1 or positive")


def ste_qdq(
    values: Tensor,
    bits: int,
    *,
    symmetric: bool,
    group_size: int = -1,
) -> Tensor:
    """Group-wise final-axis QDQ with an identity straight-through gradient."""

    if bits >= 16:
        return values
    width = values.shape[-1]
    if group_size == -1:
        grouped = values
    else:
        if width % group_size:
            raise ValueError(
                f"last-axis width {width} is not divisible by group_size {group_size}"
            )
        grouped = values.reshape(*values.shape[:-1], width // group_size, group_size)
    quantized = qdq_last_axis(grouped, bits, symmetric=symmetric)
    quantized = quantized.reshape_as(values)
    return values + (quantized - values).detach()


def _norm_epsilon(norm: nn.Module) -> float:
    for name in ("variance_epsilon", "eps"):
        if hasattr(norm, name):
            return float(getattr(norm, name))
    raise ValueError("unsupported RMSNorm module: missing epsilon")


def _left_apply_head_rotation_transpose(
    weight: Tensor,
    head_rotation: Tensor,
    num_heads: int,
) -> Tensor:
    """Compute block_diag(R.T) @ W without materializing block_diag."""

    head_dim = head_rotation.shape[0]
    if weight.shape[0] != num_heads * head_dim:
        raise ValueError("head rotation does not match linear output width")
    heads = weight.reshape(num_heads, head_dim, weight.shape[1])
    transformed = head_rotation.transpose(-1, -2).unsqueeze(0) @ heads
    return transformed.reshape_as(weight)


def _right_apply_head_rotation(
    weight: Tensor,
    head_rotation: Tensor,
    num_heads: int,
) -> Tensor:
    """Compute W @ block_diag(R) without materializing block_diag."""

    head_dim = head_rotation.shape[0]
    if weight.shape[1] != num_heads * head_dim:
        raise ValueError("head rotation does not match linear input width")
    heads = weight.reshape(weight.shape[0], num_heads, head_dim)
    transformed = heads @ head_rotation
    return transformed.reshape_as(weight)


class _RotationReference:
    """Non-registering reference shared by training-only wrappers."""

    def __init__(self, rotations: SpinQuantRotations) -> None:
        self._reference = weakref.ref(rotations)

    def get(self) -> SpinQuantRotations:
        rotations = self._reference()
        if rotations is None:
            raise RuntimeError("SpinQuant rotations were released before the adapted model")
        return rotations


class SpinQuantEmbedding(nn.Module):
    """Rotate embedding outputs into the learned residual basis during training."""

    def __init__(self, embedding: nn.Embedding, rotations: SpinQuantRotations) -> None:
        super().__init__()
        self.embedding = embedding
        self._rotations = _RotationReference(rotations)
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        values = self.embedding(input_ids)
        rotation = self._rotations.get().r1.to(dtype=values.dtype)
        return values @ rotation


class SpinQuantLinear(nn.Module):
    """Dynamically transform and fake-quantize one frozen Llama Linear weight."""

    _VALID_ROLES = {
        "q_reader",
        "k_reader",
        "v_reader",
        "o_writer",
        "gate_reader",
        "up_reader",
        "down_writer",
        "lm_head",
    }

    def __init__(
        self,
        linear: nn.Linear,
        rotations: SpinQuantRotations,
        *,
        role: str,
        spec: SpinQuantFakeQuantSpec,
        layer_index: Optional[int] = None,
        input_scale: Optional[Tensor] = None,
        num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        if role not in self._VALID_ROLES:
            raise ValueError(f"unsupported SpinQuant linear role: {role}")
        if role in {"v_reader", "o_writer"} and (
            layer_index is None or num_heads is None
        ):
            raise ValueError("V/O roles require a layer index and head count")
        self.linear = linear
        self._rotations = _RotationReference(rotations)
        self.role = role
        self.spec = spec
        self.layer_index = layer_index
        self.num_heads = num_heads
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if input_scale is None:
            self.register_buffer("input_scale", None)
        else:
            if input_scale.shape != (linear.in_features,):
                raise ValueError("RMSNorm scale does not match linear input width")
            self.register_buffer("input_scale", input_scale.detach().clone())

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.linear.bias

    def _transformed_weight(self) -> Tensor:
        rotations = self._rotations.get()
        base = self.linear.weight.float()
        if self.input_scale is not None:
            base = base * self.input_scale.float()
        r1 = rotations.r1

        if self.role in {"q_reader", "k_reader", "gate_reader", "up_reader", "lm_head"}:
            transformed = base @ r1
        elif self.role == "v_reader":
            assert self.layer_index is not None and self.num_heads is not None
            transformed = _left_apply_head_rotation_transpose(
                base,
                rotations.r2[self.layer_index],
                self.num_heads,
            )
            transformed = transformed @ r1
        elif self.role == "o_writer":
            assert self.layer_index is not None and self.num_heads is not None
            transformed = _right_apply_head_rotation(
                base,
                rotations.r2[self.layer_index],
                self.num_heads,
            )
            transformed = r1.transpose(-1, -2) @ transformed
        elif self.role == "down_writer":
            transformed = r1.transpose(-1, -2) @ base
        else:
            raise AssertionError("unreachable SpinQuant role")

        quantize_weight = self.role != "lm_head" or self.spec.quantize_lm_head
        if quantize_weight:
            transformed = ste_qdq(
                transformed,
                self.spec.weight_bits,
                symmetric=self.spec.weight_symmetric,
                group_size=self.spec.weight_group_size,
            )
        return transformed.to(dtype=self.linear.weight.dtype)

    def _transformed_bias(self) -> Optional[Tensor]:
        if self.linear.bias is None:
            return None
        rotations = self._rotations.get()
        bias = self.linear.bias.float()
        if self.role == "v_reader":
            assert self.layer_index is not None and self.num_heads is not None
            head_dim = rotations.head_dim
            heads = bias.reshape(self.num_heads, head_dim, 1)
            bias = (
                rotations.r2[self.layer_index]
                .transpose(-1, -2)
                .unsqueeze(0)
                @ heads
            )
            bias = bias.reshape_as(self.linear.bias)
        elif self.role in {"o_writer", "down_writer"}:
            bias = rotations.r1.transpose(-1, -2) @ bias
        return bias.to(dtype=self.linear.bias.dtype)

    def forward(self, values: Tensor) -> Tensor:
        quantize_activation = self.role != "lm_head"
        if quantize_activation:
            values = ste_qdq(
                values,
                self.spec.activation_bits,
                symmetric=self.spec.activation_symmetric,
            )
        return F.linear(values, self._transformed_weight(), self._transformed_bias())


def _freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def apply_spinquant_llama_training_adapter(
    model: nn.Module,
    rotations: SpinQuantRotations,
    spec: SpinQuantFakeQuantSpec,
) -> Dict[str, Any]:
    """Install differentiable R1/R2 fake-quant wrappers on a standard Llama."""

    spec.validate()
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "llama":
        raise ValueError("SpinQuant training adapter currently supports model_type=llama")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("model does not expose the standard Llama decoder layout")
    if isinstance(model.model.embed_tokens, SpinQuantEmbedding):
        raise ValueError("SpinQuant training adapter is already installed")

    hidden_size = int(config.hidden_size)
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    if int(getattr(config, "pretraining_tp", 1)) != 1:
        raise ValueError(
            "SpinQuant training wrappers require pretraining_tp=1 so Transformers "
            "does not bypass module forward methods by slicing raw weights"
        )
    head_dim = hidden_size // num_q_heads
    if (
        rotations.hidden_size != hidden_size
        or rotations.head_dim != head_dim
        or rotations.num_layers != len(model.model.layers)
    ):
        raise ValueError("rotation shapes do not match the selected Llama model")

    _freeze_model(model)
    tied_embeddings = (
        model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    )
    model.model.embed_tokens = SpinQuantEmbedding(model.model.embed_tokens, rotations)

    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        mlp = layer.mlp
        input_scale = layer.input_layernorm.weight.detach()
        post_scale = layer.post_attention_layernorm.weight.detach()
        layer.input_layernorm = UnitRMSNorm(_norm_epsilon(layer.input_layernorm))
        layer.post_attention_layernorm = UnitRMSNorm(
            _norm_epsilon(layer.post_attention_layernorm)
        )

        attention.q_proj = SpinQuantLinear(
            attention.q_proj,
            rotations,
            role="q_reader",
            spec=spec,
            layer_index=layer_index,
            input_scale=input_scale,
        )
        attention.k_proj = SpinQuantLinear(
            attention.k_proj,
            rotations,
            role="k_reader",
            spec=spec,
            layer_index=layer_index,
            input_scale=input_scale,
        )
        attention.v_proj = SpinQuantLinear(
            attention.v_proj,
            rotations,
            role="v_reader",
            spec=spec,
            layer_index=layer_index,
            input_scale=input_scale,
            num_heads=num_kv_heads,
        )
        attention.o_proj = SpinQuantLinear(
            attention.o_proj,
            rotations,
            role="o_writer",
            spec=spec,
            layer_index=layer_index,
            num_heads=num_q_heads,
        )
        mlp.gate_proj = SpinQuantLinear(
            mlp.gate_proj,
            rotations,
            role="gate_reader",
            spec=spec,
            layer_index=layer_index,
            input_scale=post_scale,
        )
        mlp.up_proj = SpinQuantLinear(
            mlp.up_proj,
            rotations,
            role="up_reader",
            spec=spec,
            layer_index=layer_index,
            input_scale=post_scale,
        )
        mlp.down_proj = SpinQuantLinear(
            mlp.down_proj,
            rotations,
            role="down_writer",
            spec=spec,
            layer_index=layer_index,
        )

    final_scale = model.model.norm.weight.detach()
    model.model.norm = UnitRMSNorm(_norm_epsilon(model.model.norm))
    model.lm_head = SpinQuantLinear(
        model.lm_head,
        rotations,
        role="lm_head",
        spec=spec,
        input_scale=final_scale,
    )
    return {
        "applied": True,
        "model_type": "llama",
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_layers": len(model.model.layers),
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "tied_embeddings_retained_as_shared_frozen_weight": tied_embeddings,
        "weight_bits": spec.weight_bits,
        "activation_bits": spec.activation_bits,
        "weight_group_size": spec.weight_group_size,
        "weight_runtime_form": (
            "floating_qdq_with_ste" if spec.weight_bits < 16 else "unquantized"
        ),
        "activation_runtime_form": (
            "floating_qdq_with_ste"
            if spec.activation_bits < 16
            else "unquantized"
        ),
        "activation_granularity": "per_token_last_axis",
        "online_rotation_scope": "training_only",
        "transformers_model_source_copied": False,
    }
