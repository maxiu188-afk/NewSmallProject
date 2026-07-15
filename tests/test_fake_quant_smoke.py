import importlib.util
import math
import unittest


HAS_TORCH_SMOKE_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH_SMOKE_DEPS, "requires the isolated local smoke environment")
class FakeQuantSmokeTests(unittest.TestCase):
    def test_tiny_fake_quant_ablation_runs_all_controlled_cases(self):
        from repro.fake_quant_smoke import run_tiny_fake_quant_ablation

        results = run_tiny_fake_quant_ablation()
        self.assertEqual(set(results), {"F0-fp32", "F1-naive-w4", "F2-quarot-w4", "F3-naive-w4a4", "F4-quarot-w4a4"})
        self.assertEqual(results["F0-fp32"]["max_absolute_logit_error"], 0.0)
        for result in results.values():
            self.assertTrue(math.isfinite(result["mean_absolute_logit_error"]))
            self.assertTrue(math.isfinite(result["max_absolute_logit_error"]))
        self.assertGreater(results["F1-naive-w4"]["max_absolute_logit_error"], 0.0)
        self.assertGreater(results["F3-naive-w4a4"]["max_absolute_logit_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
