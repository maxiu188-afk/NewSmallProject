import importlib.util
import json
import unittest
from pathlib import Path


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(HAS_DEPS, "requires torch and transformers")
class TinyLlamaSpinQuantSmokeTests(unittest.TestCase):
    def test_w4_smoke_reduces_quantized_logit_loss(self):
        from repro.spinquant.llama_smoke import (
            run_tiny_llama_spinquant_fake_quant,
        )

        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/tiny_llama_w4a16_fake_quant_smoke.json"
            ).read_text(encoding="utf-8")
        )
        result = run_tiny_llama_spinquant_fake_quant(config)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["adapter"]["transformers_model_source_copied"])
        self.assertEqual(result["adapter"]["num_key_value_heads"], 2)
        self.assertLess(result["loss_ratio"], 0.99)
        self.assertLess(result["orthogonality_error"]["r1"], 1e-5)
        self.assertLess(result["orthogonality_error"]["r2"], 1e-5)


if __name__ == "__main__":
    unittest.main()
