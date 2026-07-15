import importlib.util
import unittest


HAS_TORCH_SMOKE_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TORCH_SMOKE_DEPS, "requires the isolated local smoke environment")
class TorchSmokeTests(unittest.TestCase):
    def test_random_tiny_llama_rotation_preserves_logits_and_hidden_states(self):
        from repro.torch_smoke import run_tiny_llama_smoke

        errors = run_tiny_llama_smoke()
        self.assertLess(errors["max_logit_error"], 2e-5)
        self.assertLess(errors["max_hidden_error"], 2e-5)
        self.assertEqual(errors["num_hidden_state_tensors"], 3.0)


if __name__ == "__main__":
    unittest.main()
