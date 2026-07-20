import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class GPTQLinearTests(unittest.TestCase):
    def test_gptq_linear_quantizes_with_finite_calibration_statistics(self):
        import torch
        from torch import nn

        from repro.gptq import GPTQLinear, GPTQSettings

        torch.manual_seed(7)
        linear = nn.Linear(16, 12, bias=False).float().eval()
        original = linear.weight.detach().clone()
        collector = GPTQLinear(linear)
        for _ in range(4):
            collector.add_batch(torch.randn(2, 8, 16))
        summary = collector.quantize(GPTQSettings(bits=4, group_size=8, act_order=True), capture_packed_weight=True)

        self.assertTrue(torch.isfinite(linear.weight).all())
        self.assertGreater(summary["calibration_tokens"], 0.0)
        self.assertGreater((linear.weight - original).abs().max().item(), 0.0)
        self.assertGreaterEqual(summary["mean_estimated_loss"], 0.0)
        self.assertIsNotNone(collector.packed_weight)
        self.assertEqual(collector.packed_weight.packed_weight.shape, (12, 8))
        self.assertEqual(collector.packed_weight.scales.shape, (12, 2))
        self.assertEqual(collector.packed_weight.input_permutation.shape, (16,))
        self.assertEqual(collector.packed_weight.packed_weight.dtype, torch.uint8)

        from repro.w4a8_linear import quantize_a8, unpack_w4_weight, w4a8_reference_linear

        packed = collector.packed_weight
        integers = unpack_w4_weight(packed.packed_weight)
        dequantized_ordered = integers.reshape(12, 2, 8).float() * packed.scales.unsqueeze(-1)
        reconstructed = torch.empty_like(dequantized_ordered.reshape(12, 16))
        reconstructed[:, packed.input_permutation] = dequantized_ordered.reshape(12, 16)
        self.assertTrue(torch.allclose(reconstructed, linear.weight, atol=1e-6, rtol=1e-6))

        inputs = torch.randn(3, 16)
        activations, activation_scales = quantize_a8(inputs)
        expected = torch.nn.functional.linear(activations.float() * activation_scales, reconstructed)
        reference = w4a8_reference_linear(inputs, packed.packed_weight, packed.scales, packed.input_permutation)
        self.assertTrue(torch.allclose(reference, expected, atol=1e-5, rtol=1e-5))
