"""Cayley updates for square orthogonal SpinQuant rotation matrices.

This is an independent implementation of Equations 3--4 in the SpinQuant
paper.  The public API accepts the ordinary Euclidean gradient and constructs
the negative-gradient search direction explicitly, so a positive learning
rate performs descent.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


def _validate_square_matrices(values: Tensor, name: str) -> None:
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError(f"{name} must contain square matrices")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def orthogonality_error(rotation: Tensor) -> Tensor:
    """Return the maximum absolute entry of ``R.T @ R - I`` per matrix."""

    _validate_square_matrices(rotation, "rotation")
    size = rotation.shape[-1]
    identity = torch.eye(size, dtype=rotation.dtype, device=rotation.device)
    gram_error = rotation.transpose(-1, -2) @ rotation - identity
    return gram_error.abs().amax(dim=(-2, -1))


def cayley_generator(rotation: Tensor, gradient: Tensor) -> Tensor:
    """Construct the skew-symmetric descent generator from a Euclidean gradient.

    The paper defines its generator from a search direction.  Optimizers expose
    the positive Euclidean gradient, so this function first uses ``-gradient``.
    Retaining the full projected expression makes the convention auditable even
    though SpinQuant currently uses square orthogonal matrices.
    """

    _validate_square_matrices(rotation, "rotation")
    _validate_square_matrices(gradient, "gradient")
    if rotation.shape != gradient.shape:
        raise ValueError("rotation and gradient shapes must match")

    search = -gradient
    rotation_t = rotation.transpose(-1, -2)
    projected = search @ rotation_t
    projected = projected - 0.5 * (
        rotation @ rotation_t @ search @ rotation_t
    )
    return projected - projected.transpose(-1, -2)


def _effective_step(
    generator: Tensor,
    learning_rate: float,
    *,
    cap_fixed_point_step: bool,
    contraction_margin: float,
) -> Tensor:
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if not 0.0 < contraction_margin < 1.0:
        raise ValueError("contraction_margin must be in (0, 1)")

    batch_shape = generator.shape[:-2]
    requested = torch.full(
        batch_shape + (1, 1),
        learning_rate,
        dtype=generator.dtype,
        device=generator.device,
    )
    if not cap_fixed_point_step:
        return requested

    # The fixed-point coefficient is alpha * Y / 2.  Keeping its induced
    # one-norm below one provides a simple sufficient contraction condition.
    generator_norm = torch.linalg.matrix_norm(generator, ord=1).unsqueeze(-1).unsqueeze(-1)
    maximum = (2.0 * contraction_margin) / torch.clamp(
        generator_norm,
        min=torch.finfo(generator.dtype).eps,
    )
    return torch.minimum(requested, maximum)


def cayley_retraction(
    rotation: Tensor,
    gradient: Tensor,
    learning_rate: float,
    *,
    method: str = "fixed_point",
    fixed_point_steps: int = 5,
    contraction_margin: float = 0.95,
) -> Tensor:
    """Apply one orthogonality-preserving Cayley descent step.

    ``method="exact"`` evaluates the paper's linear solve directly and is
    intended for tests and tiny matrices.  ``method="fixed_point"`` avoids a
    matrix inverse and is the scalable path intended for model-sized rotations.
    """

    if method not in {"exact", "fixed_point"}:
        raise ValueError("method must be exact or fixed_point")
    if fixed_point_steps < 1:
        raise ValueError("fixed_point_steps must be positive")

    generator = cayley_generator(rotation, gradient)
    step = _effective_step(
        generator,
        learning_rate,
        cap_fixed_point_step=method == "fixed_point",
        contraction_margin=contraction_margin,
    )
    half_step_generator = 0.5 * step * generator

    if method == "exact":
        size = rotation.shape[-1]
        identity = torch.eye(size, dtype=rotation.dtype, device=rotation.device)
        left = identity - half_step_generator
        right = (identity + half_step_generator) @ rotation
        return torch.linalg.solve(left, right)

    # Rearranging (I - aY/2)R' = (I + aY/2)R gives the fixed-point map below.
    updated = rotation + step * (generator @ rotation)
    for _ in range(fixed_point_steps):
        updated = rotation + half_step_generator @ (rotation + updated)
    return updated


class CayleySGD(Optimizer):
    """Minimal PyTorch optimizer for SpinQuant's square rotation matrices."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        *,
        method: str = "fixed_point",
        fixed_point_steps: int = 5,
        contraction_margin: float = 0.95,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if method not in {"exact", "fixed_point"}:
            raise ValueError("method must be exact or fixed_point")
        defaults = {
            "lr": lr,
            "method": method,
            "fixed_point_steps": fixed_point_steps,
            "contraction_margin": contraction_margin,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                updated = cayley_retraction(
                    parameter,
                    parameter.grad,
                    float(group["lr"]),
                    method=str(group["method"]),
                    fixed_point_steps=int(group["fixed_point_steps"]),
                    contraction_margin=float(group["contraction_margin"]),
                )
                parameter.copy_(updated)
        return loss
