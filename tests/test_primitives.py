import math
import unittest

from repro.primitives import (
    asymmetric_dequantize,
    asymmetric_quantize,
    dot,
    hadamard_matrix,
    matvec,
    normalized_hadamard,
    pack_signed_int4,
    right_multiply,
    select_hadamard_plan,
    symmetric_dequantize,
    symmetric_quantize,
    unpack_signed_int4,
)


class PrimitiveTests(unittest.TestCase):
    def test_int4_round_trip_for_full_signed_range(self):
        values = [-8, -7, -1, 0, 1, 6, 7, -8]
        self.assertEqual(unpack_signed_int4(pack_signed_int4(values)), values)

    def test_symmetric_quantization_respects_signed_int4_range(self):
        values = [-4.0, -0.25, 0.0, 0.5, 3.5]
        quantized, scale = symmetric_quantize(values, bits=4)
        self.assertTrue(all(-8 <= value <= 7 for value in quantized))
        restored = symmetric_dequantize(quantized, scale)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(values, restored)), scale / 2 + 1e-12)

    def test_asymmetric_quantization_respects_unsigned_int4_range(self):
        values = [-3.0, -0.2, 0.0, 1.1, 4.0]
        quantized, scale, zero = asymmetric_quantize(values, bits=4)
        self.assertTrue(all(0 <= value <= 15 for value in quantized))
        restored = asymmetric_dequantize(quantized, scale, zero)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(values, restored)), scale / 2 + 1e-12)

    def test_hadamard_is_its_own_inverse(self):
        values = [0.25, -1.0, 2.5, 3.0, -0.5, 1.75, 0.0, -2.25]
        restored = normalized_hadamard(normalized_hadamard(values))
        self.assertLess(max(abs(a - b) for a, b in zip(values, restored)), 1e-12)

    def test_hadamard_preserves_dot_product(self):
        left = [1.0, -2.0, 3.5, 0.5, -1.0, 2.0, 0.0, 4.0]
        right = [-0.25, 1.0, 2.0, -3.0, 0.5, 1.0, -2.0, 0.0]
        self.assertAlmostEqual(dot(left, right), dot(normalized_hadamard(left), normalized_hadamard(right)), places=12)

    def test_llama2_relevant_dimension_plans(self):
        self.assertEqual(select_hadamard_plan(128), (1, 128))
        self.assertEqual(select_hadamard_plan(4096), (1, 4096))
        self.assertEqual(select_hadamard_plan(11008), (172, 64))
        self.assertEqual(select_hadamard_plan(13824), (108, 128))

    def test_llama2_hidden_width_hadamard_round_trip(self):
        values = [float((index % 17) - 8) / 8.0 for index in range(4096)]
        restored = normalized_hadamard(normalized_hadamard(values))
        self.assertLess(max(abs(a - b) for a, b in zip(values, restored)), 1e-11)

    def test_residual_rotation_preserves_a_linear_output(self):
        vector = [0.5, -1.0, 2.0, 3.0]
        weight = [[1.0, 2.0, -1.0, 0.5], [-2.0, 0.0, 1.5, 1.0]]
        rotation = hadamard_matrix(4)
        rotated_vector = normalized_hadamard(vector)
        rotated_weight = right_multiply(weight, rotation)
        original_output = matvec(weight, vector)
        rotated_output = matvec(rotated_weight, rotated_vector)
        self.assertLess(max(abs(a - b) for a, b in zip(original_output, rotated_output)), 1e-12)

    def test_invalid_int4_value_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_signed_int4([0, 8])


if __name__ == "__main__":
    unittest.main()
