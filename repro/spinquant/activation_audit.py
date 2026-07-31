"""Independent runtime audit helpers for SpinQuant decoder activation QDQ.

This module is deliberately separate from the production fake-quant path.  It
can reproduce that path for measurement, or apply the activation granularity
used by the SpinQuant reference evaluation, without changing saved weights or
the normal evaluator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator

import torch
from torch import Tensor, nn

from repro.qdq import qdq_last_axis
from repro.spinquant.activation_qdq import spinquant_activation_qdq


AUDIT_MODES = ("current_repo", "paper_aligned")
_LINEAR_ROLES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class _Stats:
    calls: int = 0
    elements: int = 0
    vectors: int = 0
    same_sign_vectors: Tensor | None = None
    all_zero_vectors: Tensor | None = None
    signal_squared_sum: Tensor | None = None
    error_squared_sum: Tensor | None = None
    peak_to_rms_sum: Tensor | None = None
    peak_to_rms_max: Tensor | None = None
    max_abs_input: Tensor | None = None
    max_abs_error: Tensor | None = None

    def observe(self, values: Tensor, quantized: Tensor) -> None:
        source = values.detach().float().reshape(-1, values.shape[-1])
        observed = quantized.detach().float().reshape_as(source)
        error = observed - source
        minimum = source.amin(dim=-1)
        maximum = source.amax(dim=-1)
        all_zero = (minimum == 0) & (maximum == 0)
        same_sign = ((minimum >= 0) | (maximum <= 0)) & ~all_zero
        peak = source.abs().amax(dim=-1)
        rms = torch.sqrt(torch.mean(source.square(), dim=-1))
        peak_to_rms = peak / torch.clamp(rms, min=torch.finfo(source.dtype).tiny)

        self.calls += 1
        self.elements += source.numel()
        self.vectors += source.shape[0]
        self.same_sign_vectors = _add(self.same_sign_vectors, same_sign.sum())
        self.all_zero_vectors = _add(self.all_zero_vectors, all_zero.sum())
        self.signal_squared_sum = _add(self.signal_squared_sum, source.square().sum())
        self.error_squared_sum = _add(self.error_squared_sum, error.square().sum())
        self.peak_to_rms_sum = _add(self.peak_to_rms_sum, peak_to_rms.sum())
        self.peak_to_rms_max = _maximum(self.peak_to_rms_max, peak_to_rms.amax())
        self.max_abs_input = _maximum(self.max_abs_input, peak.amax())
        self.max_abs_error = _maximum(self.max_abs_error, error.abs().amax())


def _add(current: Tensor | None, value: Tensor | None) -> Tensor | None:
    if value is None:
        return current
    value = value.detach()
    return value if current is None else current + value


def _maximum(current: Tensor | None, value: Tensor | None) -> Tensor | None:
    if value is None:
        return current
    value = value.detach()
    return value if current is None else torch.maximum(current, value)


def _number(value: Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.item())


def _summary(stats: _Stats) -> Dict[str, Any]:
    signal = _number(stats.signal_squared_sum)
    error = _number(stats.error_squared_sum)
    if error == 0:
        sqnr_db = math.inf
    elif signal == 0:
        sqnr_db = -math.inf
    else:
        sqnr_db = 10.0 * math.log10(signal / error)
    same_sign = int(round(_number(stats.same_sign_vectors)))
    all_zero = int(round(_number(stats.all_zero_vectors)))
    return {
        "calls": stats.calls,
        "elements": stats.elements,
        "vectors": stats.vectors,
        "same_sign_vectors": same_sign,
        "same_sign_fraction": same_sign / stats.vectors if stats.vectors else 0.0,
        "all_zero_vectors": all_zero,
        "signal_squared_sum": signal,
        "error_squared_sum": error,
        "sqnr_db": sqnr_db,
        "mean_peak_to_rms": (
            _number(stats.peak_to_rms_sum) / stats.vectors if stats.vectors else 0.0
        ),
        "max_peak_to_rms": _number(stats.peak_to_rms_max),
        "max_abs_input": _number(stats.max_abs_input),
        "max_abs_error": _number(stats.max_abs_error),
    }


def _merge(target: _Stats, source: _Stats) -> None:
    target.calls += source.calls
    target.elements += source.elements
    target.vectors += source.vectors
    for field in (
        "same_sign_vectors",
        "all_zero_vectors",
        "signal_squared_sum",
        "error_squared_sum",
        "peak_to_rms_sum",
    ):
        setattr(target, field, _add(getattr(target, field), getattr(source, field)))
    for field in ("peak_to_rms_max", "max_abs_input", "max_abs_error"):
        setattr(target, field, _maximum(getattr(target, field), getattr(source, field)))


def _named_decoder_linears(model: nn.Module) -> list[tuple[str, str, nn.Linear]]:
    linears = []
    for layer_index, layer in enumerate(model.model.layers):
        modules = (
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
            ("o_proj", layer.self_attn.o_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
            ("down_proj", layer.mlp.down_proj),
        )
        for role, module in modules:
            if not isinstance(module, nn.Linear):
                raise ValueError("activation audit requires standard Llama linears")
            prefix = "self_attn" if role in _LINEAR_ROLES[:4] else "mlp"
            linears.append((f"model.layers.{layer_index}.{prefix}.{role}", role, module))
    return linears


class ActivationAuditCollector:
    """Quantize decoder linear inputs while collecting runtime coverage/error."""

    def __init__(self, model: nn.Module, *, mode: str, bits: int = 8) -> None:
        if mode not in AUDIT_MODES:
            raise ValueError(f"mode must be one of {AUDIT_MODES}")
        if bits != 8:
            raise ValueError("this audit is intentionally restricted to A8")
        self.model = model
        self.mode = mode
        self.bits = bits
        self.head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
        self.linears = _named_decoder_linears(model)
        self.stats = {name: _Stats() for name, _role, _module in self.linears}
        self.roles = {name: role for name, role, _module in self.linears}
        self._handles: list[Any] = []

    def _quantize(self, values: Tensor, role: str) -> Tensor:
        if self.mode == "current_repo":
            return qdq_last_axis(values, self.bits, symmetric=False)
        group_size = self.head_dim if role == "o_proj" else -1
        return spinquant_activation_qdq(
            values,
            self.bits,
            symmetric=False,
            group_size=group_size,
        )

    def __enter__(self) -> "ActivationAuditCollector":
        if self._handles:
            raise RuntimeError("activation audit collector is already active")
        for name, role, module in self.linears:
            def hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                module_name: str = name,
                module_role: str = role,
            ) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise ValueError("decoder linear did not receive a tensor input")
                quantized = self._quantize(inputs[0], module_role)
                self.stats[module_name].observe(inputs[0], quantized)
                return (quantized,) + inputs[1:]

            self._handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> Dict[str, Any]:
        missing = [name for name, stats in self.stats.items() if stats.calls == 0]
        per_role = {role: _Stats() for role in _LINEAR_ROLES}
        global_stats = _Stats()
        for name, stats in self.stats.items():
            _merge(per_role[self.roles[name]], stats)
            _merge(global_stats, stats)
        return {
            "mode": self.mode,
            "runtime_form": "floating_qdq_forward_pre_hook_with_audit",
            "activation_bits": self.bits,
            "activation_symmetric": False,
            "paper_o_proj_group_size": self.head_dim if self.mode == "paper_aligned" else None,
            "registered_modules": len(self.linears),
            "called_modules": len(self.linears) - len(missing),
            "missing_modules": missing,
            "role_module_counts": {
                role: sum(1 for value in self.roles.values() if value == role)
                for role in _LINEAR_ROLES
            },
            "global": _summary(global_stats),
            "per_role": {role: _summary(stats) for role, stats in per_role.items()},
            "per_module": {name: _summary(stats) for name, stats in self.stats.items()},
            "quantized_lm_head": False,
            "kv_bits": 16,
        }


@contextmanager
def audit_llama_decoder_activations(
    model: nn.Module,
    *,
    mode: str,
    bits: int = 8,
) -> Iterator[ActivationAuditCollector]:
    collector = ActivationAuditCollector(model, mode=mode, bits=bits)
    with collector:
        yield collector
