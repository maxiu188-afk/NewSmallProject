"""A framework-free, one-token LLaMA-style QuaRot equivalence oracle.

This is intentionally a mathematical smoke test, not a language model or an
alternative implementation of the upstream project.  It exercises the
transformation identities that later must hold in PyTorch and CUDA execution.
"""

import math
from typing import Dict, List, Sequence, Tuple

from repro.primitives import dot, hadamard_matrix, matvec, normalized_hadamard, right_multiply


Vector = List[float]
Matrix = List[List[float]]


def _identity(size: int) -> Matrix:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def _transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("incompatible matrix shapes")
    right_transposed = _transpose(right)
    return [[dot(row, column) for column in right_transposed] for row in left]


def _diag(values: Sequence[float]) -> Matrix:
    return [[value if row == col else 0.0 for col, _ in enumerate(values)] for row, value in enumerate(values)]


def _add(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(left, right)]


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _rms_norm(values: Sequence[float], weight: Sequence[float]) -> Vector:
    variance = sum(value * value for value in values) / len(values)
    inv_rms = 1.0 / math.sqrt(variance + 1e-5)
    return [value * inv_rms * scale for value, scale in zip(values, weight)]


def _rmsn(values: Sequence[float]) -> Vector:
    return _rms_norm(values, [1.0] * len(values))


def _block_diagonal(repetitions: int, block: Sequence[Sequence[float]]) -> Matrix:
    block_size = len(block)
    size = repetitions * block_size
    result = [[0.0] * size for _ in range(size)]
    for copy_index in range(repetitions):
        offset = copy_index * block_size
        for row in range(block_size):
            for column in range(block_size):
                result[offset + row][offset + column] = block[row][column]
    return result


def _deterministic_matrix(rows: int, columns: int, offset: int) -> Matrix:
    """Create small bounded, non-symmetric weights without a random dependency."""
    return [
        [((row * 13 + column * 7 + offset) % 19 - 9) / 23.0 for column in range(columns)]
        for row in range(rows)
    ]


def _deterministic_vector(size: int, offset: int) -> Vector:
    return [((index * 11 + offset) % 17 - 8) / 19.0 for index in range(size)]


def _linear(weight: Matrix, values: Vector) -> Vector:
    return matvec(weight, values)


def _baseline_forward(parameters: Dict[str, object], values: Vector) -> Dict[str, Vector]:
    input_norm = _rms_norm(values, parameters["input_norm"])  # type: ignore[arg-type]
    q = _linear(parameters["q_proj"], input_norm)  # type: ignore[arg-type]
    k = _linear(parameters["k_proj"], input_norm)  # type: ignore[arg-type]
    v = _linear(parameters["v_proj"], input_norm)  # type: ignore[arg-type]

    # This one-token attention has softmax(score) = 1. The score is retained as
    # an explicit output so Q/K rotation is still independently checked.
    attention_score = [dot(q[:4], k[:4]), dot(q[4:], k[4:])]
    attention_output = _linear(parameters["o_proj"], v)  # type: ignore[arg-type]
    post_attention = _add(values, attention_output)

    mlp_norm = _rms_norm(post_attention, parameters["post_attention_norm"])  # type: ignore[arg-type]
    up = _linear(parameters["up_proj"], mlp_norm)  # type: ignore[arg-type]
    gate = _linear(parameters["gate_proj"], mlp_norm)  # type: ignore[arg-type]
    intermediate = [up_value * _silu(gate_value) for up_value, gate_value in zip(up, gate)]
    mlp_output = _linear(parameters["down_proj"], intermediate)  # type: ignore[arg-type]
    hidden = _add(post_attention, mlp_output)
    logits = _linear(parameters["lm_head"], _rms_norm(hidden, parameters["final_norm"]))  # type: ignore[arg-type]
    return {
        "q": q,
        "k": k,
        "attention_score": attention_score,
        "hidden": hidden,
        "logits": logits,
    }


def _rotated_parameters(parameters: Dict[str, object]) -> Dict[str, object]:
    hidden_rotation = hadamard_matrix(8)
    head_rotation = _block_diagonal(2, hadamard_matrix(4))
    intermediate_rotation = hadamard_matrix(16)
    input_scale = _diag(parameters["input_norm"])  # type: ignore[arg-type]
    mlp_scale = _diag(parameters["post_attention_norm"])  # type: ignore[arg-type]
    final_scale = _diag(parameters["final_norm"])  # type: ignore[arg-type]

    # For column-vector notation, the residual becomes Hx. Linear inputs gain
    # H on their right; V/O and MLP down-projection include online rotations.
    return {
        "hidden_rotation": hidden_rotation,
        "head_rotation": head_rotation,
        "input_q": _matmul(_matmul(parameters["q_proj"], input_scale), hidden_rotation),  # type: ignore[arg-type]
        "input_k": _matmul(_matmul(parameters["k_proj"], input_scale), hidden_rotation),  # type: ignore[arg-type]
        "input_v": _matmul(_matmul(head_rotation, parameters["v_proj"]), _matmul(input_scale, hidden_rotation)),  # type: ignore[arg-type]
        "o_proj": _matmul(_matmul(hidden_rotation, parameters["o_proj"]), head_rotation),  # type: ignore[arg-type]
        "up_proj": _matmul(_matmul(parameters["up_proj"], mlp_scale), hidden_rotation),  # type: ignore[arg-type]
        "gate_proj": _matmul(_matmul(parameters["gate_proj"], mlp_scale), hidden_rotation),  # type: ignore[arg-type]
        "down_proj": _matmul(_matmul(hidden_rotation, parameters["down_proj"]), intermediate_rotation),  # type: ignore[arg-type]
        "intermediate_rotation": intermediate_rotation,
        "lm_head": _matmul(_matmul(parameters["lm_head"], final_scale), hidden_rotation),  # type: ignore[arg-type]
    }


def _rotated_forward(parameters: Dict[str, object], values: Vector) -> Dict[str, Vector]:
    rotated = _rotated_parameters(parameters)
    hidden_rotation = rotated["hidden_rotation"]  # type: ignore[assignment]
    head_rotation = rotated["head_rotation"]  # type: ignore[assignment]
    rotated_values = _linear(hidden_rotation, values)  # type: ignore[arg-type]
    input_norm = _rmsn(rotated_values)
    q_unrotated = _linear(rotated["input_q"], input_norm)  # type: ignore[arg-type]
    k_unrotated = _linear(rotated["input_k"], input_norm)  # type: ignore[arg-type]
    v = _linear(rotated["input_v"], input_norm)  # type: ignore[arg-type]
    q = _linear(head_rotation, q_unrotated)  # type: ignore[arg-type]
    k = _linear(head_rotation, k_unrotated)  # type: ignore[arg-type]
    attention_score = [dot(q[:4], k[:4]), dot(q[4:], k[4:])]
    attention_output = _linear(rotated["o_proj"], v)  # type: ignore[arg-type]
    post_attention = _add(rotated_values, attention_output)

    mlp_norm = _rmsn(post_attention)
    up = _linear(rotated["up_proj"], mlp_norm)  # type: ignore[arg-type]
    gate = _linear(rotated["gate_proj"], mlp_norm)  # type: ignore[arg-type]
    intermediate = [up_value * _silu(gate_value) for up_value, gate_value in zip(up, gate)]
    rotated_intermediate = _linear(rotated["intermediate_rotation"], intermediate)  # type: ignore[arg-type]
    mlp_output = _linear(rotated["down_proj"], rotated_intermediate)  # type: ignore[arg-type]
    hidden = _add(post_attention, mlp_output)
    logits = _linear(rotated["lm_head"], _rmsn(hidden))  # type: ignore[arg-type]
    return {
        "q": q,
        "k": k,
        "attention_score": attention_score,
        "hidden": hidden,
        "logits": logits,
    }


def run_toy_equivalence() -> Dict[str, float]:
    """Return maximum errors for a complete, deterministic QuaRot toy block."""
    parameters: Dict[str, object] = {
        "input_norm": _deterministic_vector(8, 1),
        "post_attention_norm": _deterministic_vector(8, 2),
        "final_norm": _deterministic_vector(8, 3),
        "q_proj": _deterministic_matrix(8, 8, 4),
        "k_proj": _deterministic_matrix(8, 8, 5),
        "v_proj": _deterministic_matrix(8, 8, 6),
        "o_proj": _deterministic_matrix(8, 8, 7),
        "up_proj": _deterministic_matrix(16, 8, 8),
        "gate_proj": _deterministic_matrix(16, 8, 9),
        "down_proj": _deterministic_matrix(8, 16, 10),
        "lm_head": _deterministic_matrix(12, 8, 11),
    }
    values = _deterministic_vector(8, 12)
    baseline = _baseline_forward(parameters, values)
    rotated = _rotated_forward(parameters, values)
    hidden_rotation = hadamard_matrix(8)
    recovered_hidden = _linear(hidden_rotation, rotated["hidden"])
    return {
        "max_logit_error": max(abs(a - b) for a, b in zip(baseline["logits"], rotated["logits"])),
        "max_hidden_error": max(abs(a - b) for a, b in zip(baseline["hidden"], recovered_hidden)),
        "max_attention_score_error": max(abs(a - b) for a, b in zip(baseline["attention_score"], rotated["attention_score"])),
    }
