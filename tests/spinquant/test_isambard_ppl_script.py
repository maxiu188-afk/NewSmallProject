from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_fake_quant_ppl.sbatch"
)


class SpinQuantIsambardPplScriptTests(unittest.TestCase):
    def test_ppl_job_is_training_and_smoke_gated(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('training_job="5842047"', text)
        self.assertIn("SPINQUANT_13B_PPL_TRAINING_GATE_ACCEPTED", text)
        self.assertIn("SPINQUANT_13B_PPL_FORMAL_REQUIRES_AFTEROK_DEPENDENCY", text)
        self.assertIn("SPINQUANT_13B_PPL_SMOKE_DEPENDENCY_ACCEPTED", text)
        self.assertIn("SPINQUANT_13B_PPL_DIRTY_CHECKOUT", text)
        self.assertIn("fake-quant-ppl-smoke.json", text)
        self.assertIn("fake-quant-ppl-formal.json", text)
        self.assertIn('quantization["quantized_decoder_linears"] == 280', text)
        self.assertIn(
            "ISAMBARD_SPINQUANT_LLAMA2_13B_FAKE_QUANT_PPL_FORMAL_PASSED",
            text,
        )


if __name__ == "__main__":
    unittest.main()
