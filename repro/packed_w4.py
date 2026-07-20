"""Independent packed-W4 reference for the owned deployment track.

This module deliberately uses Python sequences and standard-library files.  It
is a deterministic correctness oracle for the later CUDA extension, not a
serving implementation.  The layout is documented in
``docs/PACKED_W4_FORMAT.md`` and is independent of the ignored upstream tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence, Tuple


FORMAT_VERSION = "packed-w4-row-major-le-nibble-v1"
SCALE_DTYPE = "float32-le"
LLAMA2_13B_LINEAR_SHAPES: Tuple[Tuple[int, int], ...] = (
    (5120, 5120),
    (13824, 5120),
    (5120, 13824),
)


class PackedW4Error(ValueError):
    """Raised when a packed-W4 tensor would violate the format contract."""


def _round_and_clamp(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32(value: float) -> float:
    """Normalize a scale to the format's declared on-disk FP32 precision."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _validate_shape(out_features: int, in_features: int, group_size: int) -> None:
    if out_features <= 0 or in_features <= 0:
        raise PackedW4Error("out_features and in_features must be positive")
    if in_features % 2:
        raise PackedW4Error("in_features must be even because each byte stores two int4 values")
    if group_size <= 0 or group_size % 2:
        raise PackedW4Error("group_size must be a positive even integer")
    if in_features % group_size:
        raise PackedW4Error("in_features must be divisible by group_size")


def validate_w4a8_linear_shape(out_features: int, in_features: int, group_size: int = 128) -> None:
    """Validate a Phase-1 W4A8 GEMM shape without allocating its tensors."""
    _validate_shape(out_features, in_features, group_size)
    if in_features % 32:
        raise PackedW4Error("W4A8 GEMM requires K (in_features) to be divisible by 32")


def pack_signed_int4(values: Sequence[int]) -> bytes:
    """Pack signed int4 values with the first value in each byte's low nibble."""
    if len(values) % 2:
        raise PackedW4Error("signed int4 packing requires an even number of values")
    packed = bytearray()
    for index in range(0, len(values), 2):
        low, high = values[index], values[index + 1]
        if not -8 <= low <= 7 or not -8 <= high <= 7:
            raise PackedW4Error("signed int4 values must lie in [-8, 7]")
        packed.append((low & 0x0F) | ((high & 0x0F) << 4))
    return bytes(packed)


def unpack_signed_int4(packed: bytes) -> Tuple[int, ...]:
    """Invert :func:`pack_signed_int4` using the format's little-nibble order."""
    values = []
    for byte in packed:
        for nibble in (byte & 0x0F, byte >> 4):
            values.append(nibble - 16 if nibble >= 8 else nibble)
    return tuple(values)


@dataclass(frozen=True)
class PackedW4Matrix:
    """Row-major groupwise W4 matrix plus one FP32 scale per output/group."""

    out_features: int
    in_features: int
    group_size: int
    data: bytes
    scales: Tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_shape(self.out_features, self.in_features, self.group_size)
        if len(self.data) != self.out_features * self.in_features // 2:
            raise PackedW4Error("packed data length does not match matrix shape")
        expected_scales = self.out_features * self.groups_per_row
        if len(self.scales) != expected_scales:
            raise PackedW4Error("scale count does not match matrix shape and group_size")
        if any(scale <= 0.0 for scale in self.scales):
            raise PackedW4Error("all W4 scales must be positive")

    @property
    def groups_per_row(self) -> int:
        return self.in_features // self.group_size

    def quantized_row(self, row: int) -> Tuple[int, ...]:
        if not 0 <= row < self.out_features:
            raise PackedW4Error("row index out of range")
        offset = row * self.in_features // 2
        return unpack_signed_int4(self.data[offset : offset + self.in_features // 2])

    def scale(self, row: int, group: int) -> float:
        if not 0 <= row < self.out_features or not 0 <= group < self.groups_per_row:
            raise PackedW4Error("scale index out of range")
        return self.scales[row * self.groups_per_row + group]


def quantize_w4_rows(rows: Sequence[Sequence[float]], group_size: int = 128) -> PackedW4Matrix:
    """Quantize a dense row-major matrix to the owned symmetric groupwise W4 format."""
    if not rows or not rows[0]:
        raise PackedW4Error("rows must be a non-empty rectangular matrix")
    out_features = len(rows)
    in_features = len(rows[0])
    _validate_shape(out_features, in_features, group_size)
    if any(len(row) != in_features for row in rows):
        raise PackedW4Error("rows must be rectangular")

    packed = bytearray()
    scales = []
    for row in rows:
        quantized = []
        for start in range(0, in_features, group_size):
            group = row[start : start + group_size]
            max_abs = max(abs(float(value)) for value in group)
            scale = _float32(max_abs / 7.0 if max_abs else 1.0)
            scales.append(scale)
            quantized.extend(_round_and_clamp(float(value) / scale, -8, 7) for value in group)
        packed.extend(pack_signed_int4(quantized))
    return PackedW4Matrix(out_features, in_features, group_size, bytes(packed), tuple(scales))


def quantize_a8(values: Sequence[float]) -> Tuple[Tuple[int, ...], float]:
    """Return per-token symmetric int8 activations and one positive scale."""
    if not values:
        raise PackedW4Error("activation vector must not be empty")
    max_abs = max(abs(float(value)) for value in values)
    scale = max_abs / 127.0 if max_abs else 1.0
    return tuple(_round_and_clamp(float(value) / scale, -128, 127) for value in values), scale


def int32_matvec_by_group(matrix: PackedW4Matrix, activations: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Compute the independent W4A8 integer accumulator for every output/group."""
    if len(activations) != matrix.in_features:
        raise PackedW4Error("activation length does not match in_features")
    if any(value < -128 or value > 127 for value in activations):
        raise PackedW4Error("A8 activations must lie in [-128, 127]")
    output = []
    for row in range(matrix.out_features):
        quantized = matrix.quantized_row(row)
        group_sums = []
        for group in range(matrix.groups_per_row):
            start = group * matrix.group_size
            end = start + matrix.group_size
            group_sums.append(sum(weight * activation for weight, activation in zip(quantized[start:end], activations[start:end])))
        output.append(tuple(group_sums))
    return tuple(output)


def dequantize_w4a8_matvec(
    matrix: PackedW4Matrix, activations: Sequence[float]
) -> Tuple[Tuple[int, ...], Tuple[float, ...], float]:
    """Return integer group accumulators, scaled output, and the A8 scale."""
    if len(activations) != matrix.in_features:
        raise PackedW4Error("activation length does not match in_features")
    quantized_activations, activation_scale = quantize_a8(activations)
    accumulators = int32_matvec_by_group(matrix, quantized_activations)
    output = tuple(
        activation_scale * sum(value * matrix.scale(row, group) for group, value in enumerate(row_accumulators))
        for row, row_accumulators in enumerate(accumulators)
    )
    return accumulators, output, activation_scale


def write_packed_w4_artifact(
    output_dir: Path, tensor_name: str, matrix: PackedW4Matrix, source: Mapping[str, Any]
) -> Path:
    """Write packed bytes, FP32 scales, and a checksummed JSON manifest."""
    if not tensor_name or Path(tensor_name).name != tensor_name:
        raise PackedW4Error("tensor_name must be a non-empty file name, not a path")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / (tensor_name + ".w4.bin")
    scales_path = output_dir / (tensor_name + ".scales.f32")
    manifest_path = output_dir / (tensor_name + ".packed-w4.json")
    data_path.write_bytes(matrix.data)
    scales_path.write_bytes(struct.pack("<{}f".format(len(matrix.scales)), *matrix.scales))
    manifest = {
        "schema_version": 1,
        "format": FORMAT_VERSION,
        "tensor_name": tensor_name,
        "shape": [matrix.out_features, matrix.in_features],
        "quantization": {
            "weight_bits": 4,
            "symmetric": True,
            "group_axis": "input_features",
            "group_size": matrix.group_size,
            "scale_dtype": SCALE_DTYPE,
            "packing": "two signed int4 values per uint8; even input index in low nibble",
        },
        "files": {
            "data": {"name": data_path.name, "sha256": _sha256(data_path)},
            "scales": {"name": scales_path.name, "sha256": _sha256(scales_path)},
        },
        "source": dict(source),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_packed_w4_artifact(manifest_path: Path) -> PackedW4Matrix:
    """Load a packed artifact and reject any checksum or format mismatch."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("format") != FORMAT_VERSION:
        raise PackedW4Error("unsupported packed-W4 manifest format")
    shape = manifest.get("shape")
    quantization = manifest.get("quantization", {})
    if not isinstance(shape, list) or len(shape) != 2:
        raise PackedW4Error("manifest shape must be a two-element list")
    data_info = manifest.get("files", {}).get("data", {})
    scales_info = manifest.get("files", {}).get("scales", {})
    data_path = manifest_path.parent / data_info.get("name", "")
    scales_path = manifest_path.parent / scales_info.get("name", "")
    if _sha256(data_path) != data_info.get("sha256") or _sha256(scales_path) != scales_info.get("sha256"):
        raise PackedW4Error("packed-W4 artifact checksum mismatch")
    data = data_path.read_bytes()
    scale_bytes = scales_path.read_bytes()
    if len(scale_bytes) % 4:
        raise PackedW4Error("scale payload is not float32 aligned")
    scales = struct.unpack("<{}f".format(len(scale_bytes) // 4), scale_bytes)
    return PackedW4Matrix(
        int(shape[0]), int(shape[1]), int(quantization.get("group_size", 0)), data, tuple(scales)
    )
