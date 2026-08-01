import importlib.util
import json
import unittest
from pathlib import Path


HAS_TORCH = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(HAS_TORCH, "requires torch")
class SpinQuantFakeQuantTests(unittest.TestCase):
    def test_tiny_w4_smoke_learns_rotations_without_losing_orthogonality(self):
        from repro.spinquant.fake_quant import run_tiny_spinquant_fake_quant

        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/tiny_w4a16_fake_quant_smoke.json"
            ).read_text(encoding="utf-8")
        )
        result = run_tiny_spinquant_fake_quant(config)
        self.assertEqual(result["status"], "passed")
        self.assertLess(result["loss_ratio"], 0.95)
        self.assertLess(result["orthogonality_error"]["r1"], 1e-5)
        self.assertLess(result["orthogonality_error"]["r2"], 1e-5)


if __name__ == "__main__":
    unittest.main()
