import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class PackedW4A8ReferenceLinearTests(unittest.TestCase):
    def test_reference_module_preserves_packed_a8_and_input_permutation(self):
        import torch

        from repro.w4a8_linear import PackedW4A8ReferenceLinear, pack_w4_weight, w4a8_reference_linear

        torch.manual_seed(23)
        weight = torch.randn(5, 16)
        packed_weight, scales = pack_w4_weight(weight, group_size=8)
        permutation = torch.randperm(16)
        bias = torch.randn(5)
        inputs = torch.randn(2, 3, 16)
        module = PackedW4A8ReferenceLinear(packed_weight, scales, bias, permutation).eval()

        actual = module(inputs)
        expected = w4a8_reference_linear(inputs, packed_weight, scales, permutation) + bias
        self.assertEqual(actual.shape, (2, 3, 5))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        bf16_module = PackedW4A8ReferenceLinear(
            packed_weight, scales, bias, permutation, output_dtype=torch.bfloat16
        ).eval()
        self.assertEqual(bf16_module(inputs).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
