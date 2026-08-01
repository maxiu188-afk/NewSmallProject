"""Independent SpinQuant-style fake-quantization building blocks.

The modules in this package are derived from the equations and experimental
protocol in the SpinQuant paper.  They do not import or copy code from the
locally retained upstream reference checkout.
"""

from repro.spinquant.artifacts import (
    RotationArtifactError,
    load_rotation_artifact,
    save_rotation_artifact,
)
from repro.spinquant.llama_adapter import (
    SpinQuantFakeQuantSpec,
    apply_spinquant_llama_training_adapter,
)
from repro.spinquant.rotations import SpinQuantRotations, random_signed_hadamard
from repro.spinquant.stiefel import CayleySGD, cayley_retraction, orthogonality_error

__all__ = [
    "CayleySGD",
    "RotationArtifactError",
    "SpinQuantFakeQuantSpec",
    "SpinQuantRotations",
    "apply_spinquant_llama_training_adapter",
    "cayley_retraction",
    "load_rotation_artifact",
    "orthogonality_error",
    "random_signed_hadamard",
    "save_rotation_artifact",
]
