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

    def test_gptq_linear_capture_can_stream_through_callback(self):
        import torch
        from torch import nn

        from repro.gptq import GPTQSettings, quantize_llama_weights_gptq

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8, bias=False)

            def forward(self, hidden_states, **_kwargs):
                return (self.proj(hidden_states),)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(32, 8)
                self.layers = nn.ModuleList([Layer()])

        class Config:
            model_type = "llama"
            hidden_size = 8
            use_cache = False

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Config()
                self.model = Backbone()

            def forward(self, input_ids, use_cache=False):
                hidden = self.model.embed_tokens(input_ids)
                return self.model.layers[0](hidden, use_cache=use_cache)

        model = Model().float().eval()
        streamed = []
        summary = quantize_llama_weights_gptq(
            model,
            [torch.tensor([[1, 2, 3, 4]])],
            GPTQSettings(bits=4, group_size=8),
            capture_packed_linears=["model.layers.0.proj"],
            packed_weight_callback=lambda name, packed: streamed.append((name, packed)),
        )
        self.assertEqual(summary["linear_layers"], 1)
        self.assertEqual(len(streamed), 1)
        self.assertEqual(streamed[0][0], "model.layers.0.proj")
        self.assertEqual(streamed[0][1].packed_weight.shape, (8, 4))
