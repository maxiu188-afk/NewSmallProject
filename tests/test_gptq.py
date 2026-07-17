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
        summary = collector.quantize(GPTQSettings(bits=4, group_size=8, act_order=True))

        self.assertTrue(torch.isfinite(linear.weight).all())
        self.assertGreater(summary["calibration_tokens"], 0.0)
        self.assertGreater((linear.weight - original).abs().max().item(), 0.0)
        self.assertGreaterEqual(summary["mean_estimated_loss"], 0.0)
