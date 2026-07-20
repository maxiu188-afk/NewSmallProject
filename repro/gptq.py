"""Small, explicit GPTQ implementation for the portable LLaMA pipeline.

This module intentionally keeps GPTQ as a floating-point, in-place weight
reparameterization.  It does not claim to provide packed weights or a CUDA
kernel.  The LLaMA runner quantizes one decoder layer at a time so the
calibration activations and Hessians, rather than a second 13B checkpoint, are
the main additional GPU memory consumers.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn


class GPTQError(RuntimeError):
    """Raised when a GPTQ calibration or factorization is not usable."""


@dataclass
class GPTQSettings:
    bits: int
    group_size: int = 128
    damp_percent: float = 0.01
    block_size: int = 128
    act_order: bool = True
    symmetric: bool = True


@dataclass
class GPTQPackedWeight:
    """Exact W4 grouping metadata emitted by one GPTQ linear quantization."""

    packed_weight: torch.Tensor
    scales: torch.Tensor
    input_permutation: Optional[torch.Tensor]


class _SymmetricGroupQuantizer:
    def __init__(self, bits: int) -> None:
        if bits < 2 or bits > 16:
            raise GPTQError("GPTQ weight bits must be in [2, 16]")
        self.bits = bits
        self.maxq = 2 ** (bits - 1) - 1
        self.scale: torch.Tensor | None = None

    def find_params(self, values: torch.Tensor) -> None:
        bound = values.abs().amax(dim=1, keepdim=True)
        self.scale = torch.where(bound == 0, torch.ones_like(bound), bound / self.maxq)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            raise GPTQError("quantizer parameters have not been initialized")
        return torch.clamp(torch.round(values / self.scale), -self.maxq - 1, self.maxq) * self.scale


class GPTQLinear:
    """Accumulate an input Hessian and quantize one ``nn.Linear`` with GPTQ."""

    def __init__(self, layer: nn.Linear) -> None:
        self.layer = layer
        self.columns = int(layer.weight.shape[1])
        self.hessian = torch.zeros((self.columns, self.columns), device=layer.weight.device, dtype=torch.float32)
        self.samples = 0
        self.packed_weight: Optional[GPTQPackedWeight] = None

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        values = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if values.shape[1] != self.columns:
            raise GPTQError("calibration input width does not match linear weight")
        self.hessian.addmm_(values.T, values)
        self.samples += int(values.shape[0])

    @torch.no_grad()
    def quantize(self, settings: GPTQSettings, capture_packed_weight: bool = False) -> Dict[str, float]:
        if not settings.symmetric:
            raise GPTQError("portable GPTQ currently supports symmetric weight quantization only")
        if self.samples == 0:
            raise GPTQError("cannot run GPTQ without calibration activations")
        if settings.group_size == 0 or settings.group_size < -1:
            raise GPTQError("GPTQ group_size must be -1 or a positive integer")
        if not 0.0 < settings.damp_percent:
            raise GPTQError("GPTQ damp_percent must be positive")
        if settings.block_size <= 0:
            raise GPTQError("GPTQ block_size must be positive")

        weight = self.layer.weight.detach().float().clone()
        hessian = self.hessian.div(float(self.samples))
        self.hessian = torch.empty(0, device=weight.device)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        if settings.act_order:
            permutation = torch.argsort(torch.diag(hessian), descending=True)
            inverse_permutation = torch.argsort(permutation)
            weight = weight[:, permutation]
            hessian = hessian[permutation][:, permutation]
        else:
            permutation = inverse_permutation = None

        diagonal = torch.arange(self.columns, device=weight.device)
        hessian[diagonal, diagonal] += settings.damp_percent * torch.diag(hessian).mean()
        try:
            factor = torch.linalg.cholesky(hessian)
            inverse = torch.cholesky_inverse(factor)
            inverse_factor = torch.linalg.cholesky(inverse, upper=True)
        except torch.linalg.LinAlgError as error:
            raise GPTQError("calibration Hessian is not positive definite after damping") from error

        quantized = torch.zeros_like(weight)
        integer_quantized = torch.empty_like(weight, dtype=torch.int8) if capture_packed_weight else None
        losses = torch.zeros_like(weight)
        quantizer = _SymmetricGroupQuantizer(settings.bits)
        group_scales: List[torch.Tensor] = []
        for block_start in range(0, self.columns, settings.block_size):
            block_end = min(block_start + settings.block_size, self.columns)
            block = weight[:, block_start:block_end].clone()
            errors = torch.zeros_like(block)
            block_inverse = inverse_factor[block_start:block_end, block_start:block_end]
            for offset in range(block_end - block_start):
                column = block[:, offset]
                absolute_index = block_start + offset
                if settings.group_size == -1 or absolute_index % settings.group_size == 0:
                    group_end = self.columns if settings.group_size == -1 else min(absolute_index + settings.group_size, self.columns)
                    quantizer.find_params(weight[:, absolute_index:group_end])
                    if capture_packed_weight:
                        group_scales.append(quantizer.scale.squeeze(1).clone())
                q_column = quantizer.quantize(column.unsqueeze(1)).flatten()
                quantized[:, absolute_index] = q_column
                if integer_quantized is not None:
                    integer_quantized[:, absolute_index] = torch.round(q_column / quantizer.scale.squeeze(1)).to(torch.int8)
                diagonal_value = block_inverse[offset, offset]
                losses[:, absolute_index] = (column - q_column).square() / diagonal_value.square()
                error = (column - q_column) / diagonal_value
                block[:, offset:] -= error.unsqueeze(1) * block_inverse[offset, offset:].unsqueeze(0)
                errors[:, offset] = error
            if block_end < self.columns:
                weight[:, block_end:] -= errors @ inverse_factor[block_start:block_end, block_end:]

        if integer_quantized is not None:
            if settings.bits != 4:
                raise GPTQError("packed W4 export requires bits=4")
            low = (integer_quantized[:, 0::2].to(torch.int16) & 0x0F).to(torch.uint8)
            high = (integer_quantized[:, 1::2].to(torch.int16) & 0x0F).to(torch.uint8)
            self.packed_weight = GPTQPackedWeight(
                packed_weight=(low | (high << 4)).contiguous(),
                scales=torch.stack(group_scales, dim=1).float().contiguous(),
                input_permutation=None if permutation is None else permutation.contiguous(),
            )
        if inverse_permutation is not None:
            quantized = quantized[:, inverse_permutation]
        if not torch.isfinite(quantized).all():
            raise GPTQError("GPTQ produced non-finite weights")
        self.layer.weight.copy_(quantized.to(dtype=self.layer.weight.dtype))
        return {
            "calibration_tokens": float(self.samples),
            "mean_estimated_loss": float(losses.mean().item() / 2.0),
        }


def _linear_groups(layer: nn.Module) -> List[List[Tuple[str, nn.Linear]]]:
    named = dict((name, module) for name, module in layer.named_modules() if isinstance(module, nn.Linear))
    used: set[str] = set()
    groups: List[List[Tuple[str, nn.Linear]]] = []
    for suffixes in (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",)):
        group = [(name, module) for name, module in named.items() if any(name.endswith(suffix) for suffix in suffixes)]
        if group:
            groups.append(group)
            used.update(name for name, _ in group)
    leftovers = [(name, module) for name, module in named.items() if name not in used]
    if leftovers:
        groups.append(leftovers)
    return groups


def _hidden_states(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "hidden_states", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    raise GPTQError("LLaMA decoder layer returned an unsupported output")


@torch.no_grad()
def quantize_llama_weights_gptq(
    model: nn.Module,
    calibration_batches: Sequence[torch.Tensor],
    settings: GPTQSettings,
    capture_packed_linears: Iterable[str] = (),
    captured_packed_weights: Optional[MutableMapping[str, GPTQPackedWeight]] = None,
) -> Dict[str, Any]:
    """Apply sequential GPTQ to a standard LLaMA decoder stack.

    Calibration batches must be fixed-size, batch-one token tensors.  Keeping
    this explicit avoids silently treating variable-length text rows as GPTQ
    samples and makes the recorded calibration token count reproducible.
    """
    if getattr(model.config, "model_type", None) != "llama":
        raise GPTQError("portable GPTQ currently supports model_type=llama only")
    if not calibration_batches:
        raise GPTQError("GPTQ requires at least one calibration batch")
    if any(batch.ndim != 2 or batch.shape[0] != 1 for batch in calibration_batches):
        raise GPTQError("GPTQ calibration requires fixed-length batch-one token tensors")
    sequence_length = int(calibration_batches[0].shape[1])
    if any(batch.shape[1] != sequence_length for batch in calibration_batches):
        raise GPTQError("GPTQ calibration batches must share one sequence length")

    device = next(model.parameters()).device
    layers = model.model.layers
    dtype = next(model.parameters()).dtype
    samples = len(calibration_batches)
    capture_names = set(capture_packed_linears)
    if capture_names and captured_packed_weights is None:
        raise GPTQError("captured_packed_weights is required when capture_packed_linears is set")
    inputs = torch.empty((samples, sequence_length, model.config.hidden_size), dtype=dtype, device=device)
    captured_kwargs: Dict[str, Any] = {}
    captured = 0

    class CaptureFirstLayer(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            nonlocal captured
            if captured >= samples:
                raise GPTQError("received more calibration batches than requested")
            inputs[captured].copy_(hidden_states[0])
            if not captured_kwargs:
                captured_kwargs.update(kwargs)
            captured += 1
            raise _CaptureComplete()

    original_first = layers[0]
    previous_cache = bool(model.config.use_cache)
    model.config.use_cache = False
    layers[0] = CaptureFirstLayer(original_first)
    try:
        for batch in calibration_batches:
            try:
                model(input_ids=batch.to(device), use_cache=False)
            except _CaptureComplete:
                pass
    finally:
        layers[0] = original_first
    if captured != samples:
        raise GPTQError("failed to capture every requested calibration batch")

    outputs = torch.empty_like(inputs)
    layer_results: Dict[str, Dict[str, float]] = {}
    try:
        for layer_index, layer in enumerate(layers):
            for group in _linear_groups(layer):
                collectors = {name: GPTQLinear(linear) for name, linear in group}
                handles = []
                for name, linear in group:
                    def collect(_module: nn.Module, module_inputs: Tuple[torch.Tensor, ...], _output: torch.Tensor, *, name: str = name) -> None:
                        collectors[name].add_batch(module_inputs[0])
                    handles.append(linear.register_forward_hook(collect))
                try:
                    for sample_index in range(samples):
                        _hidden_states(layer(inputs[sample_index : sample_index + 1], **captured_kwargs))
                finally:
                    for handle in handles:
                        handle.remove()
                for name, collector in collectors.items():
                    full_name = "model.layers.{}.{}".format(layer_index, name)
                    layer_results[full_name] = collector.quantize(settings, capture_packed_weight=full_name in capture_names)
                    if full_name in capture_names:
                        if collector.packed_weight is None:
                            raise GPTQError("failed to capture packed GPTQ weight for {}".format(full_name))
                        captured_packed_weights[full_name] = collector.packed_weight
            for sample_index in range(samples):
                outputs[sample_index] = _hidden_states(layer(inputs[sample_index : sample_index + 1], **captured_kwargs))[0]
            inputs, outputs = outputs, inputs
    finally:
        model.config.use_cache = previous_cache
    return {
        "method": "gptq",
        "layers": len(layers),
        "linear_layers": len(layer_results),
        "calibration_sequences": samples,
        "calibration_sequence_length": sequence_length,
        "per_linear": layer_results,
    }


class _CaptureComplete(Exception):
    """Private control-flow exception used to stop the first-layer capture."""
