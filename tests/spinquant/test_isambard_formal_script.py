from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_100step.sbatch"
)


class SpinQuantIsambardFormalScriptTests(unittest.TestCase):
    def test_formal_job_is_smoke_gated_and_fail_closed(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('smoke_job="5841874"', text)
        self.assertIn("sha256sum --check --status", text)
        self.assertIn("merge-base", text)
        self.assertIn("SPINQUANT_13B_FORMAL_SMOKE_GATE_ACCEPTED", text)
        self.assertIn("SPINQUANT_13B_FORMAL_DIRTY_CHECKOUT", text)
        self.assertIn("newsmallproject-spinquant", text)
        self.assertNotIn("newsmallproject-vllm/llama2-13b-w4a16", text)
        self.assertIn('training["optimizer_steps"] == 100', text)
        self.assertIn('len(training["gradient_maxima"]) == 100', text)
        self.assertIn("maximum_orthogonality_error=1e-4", text)
        self.assertIn(
            "ISAMBARD_SPINQUANT_LLAMA2_13B_100STEP_PASSED",
            text,
        )


if __name__ == "__main__":
    unittest.main()
