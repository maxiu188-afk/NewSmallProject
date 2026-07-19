import json
from pathlib import Path
import tempfile
import unittest

from repro.packed_w4 import (
    FORMAT_VERSION,
    LLAMA2_13B_LINEAR_SHAPES,
    PackedW4Error,
    dequantize_w4a8_matvec,
    int32_matvec_by_group,
    load_packed_w4_artifact,
    pack_signed_int4,
    quantize_a8,
    quantize_w4_rows,
    unpack_signed_int4,
    validate_w4a8_linear_shape,
    write_packed_w4_artifact,
)


class PackedW4Tests(unittest.TestCase):
    def test_signed_int4_format_round_trip_and_nibble_order(self):
        packed = pack_signed_int4([-8, 7, -1, 0])
        self.assertEqual(packed, bytes([0x78, 0x0F]))
        self.assertEqual(unpack_signed_int4(packed), (-8, 7, -1, 0))

    def test_groupwise_w4a8_reference_exposes_integer_accumulators(self):
        matrix = quantize_w4_rows(
            [[-7.0, -3.0, 2.0, 6.0, 4.0, -4.0, 1.0, -1.0]], group_size=4
        )
        activations = [-1.5, 0.5, 2.5, -3.5, 1.0, -2.0, 0.5, 4.0]
        a8, activation_scale = quantize_a8(activations)
        expected = int32_matvec_by_group(matrix, a8)
        accumulators, output, observed_scale = dequantize_w4a8_matvec(matrix, activations)
        self.assertEqual(accumulators, expected)
        self.assertEqual(observed_scale, activation_scale)
        expected_output = activation_scale * sum(
            expected[0][group] * matrix.scale(0, group) for group in range(matrix.groups_per_row)
        )
        self.assertAlmostEqual(output[0], expected_output, places=12)

    def test_artifact_round_trip_includes_checksums_and_provenance(self):
        matrix = quantize_w4_rows([[0.0, 1.0, -2.0, 3.0], [4.0, -5.0, 6.0, -7.0]], group_size=2)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_packed_w4_artifact(
                Path(directory), "layer0_q_proj", matrix, {"model_revision": "pinned", "seed": 0}
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], FORMAT_VERSION)
            self.assertEqual(manifest["source"]["model_revision"], "pinned")
            self.assertEqual(load_packed_w4_artifact(manifest_path), matrix)

    def test_corrupted_artifact_is_rejected(self):
        matrix = quantize_w4_rows([[1.0, 2.0, 3.0, 4.0]], group_size=2)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_packed_w4_artifact(Path(directory), "weight", matrix, {"seed": 0})
            (Path(directory) / "weight.w4.bin").write_bytes(b"corrupt")
            with self.assertRaises(PackedW4Error):
                load_packed_w4_artifact(manifest_path)

    def test_llama_shapes_pass_without_allocating_large_tensors(self):
        for out_features, in_features in LLAMA2_13B_LINEAR_SHAPES:
            validate_w4a8_linear_shape(out_features, in_features)
        with self.assertRaises(PackedW4Error):
            validate_w4a8_linear_shape(32, 48)

    def test_invalid_group_layout_is_rejected(self):
        with self.assertRaises(PackedW4Error):
            quantize_w4_rows([[1.0, 2.0, 3.0, 4.0]], group_size=3)


if __name__ == "__main__":
    unittest.main()
