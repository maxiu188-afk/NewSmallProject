from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_1step_smoke.sbatch"
)


class SpinQuantIsambardSmokeScriptTests(unittest.TestCase):
    def test_smoke_is_isolated_and_fail_closed(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('project_root="${SLURM_SUBMIT_DIR:-', text)
        self.assertIn("newsmallproject-spinquant", text)
        self.assertNotIn("newsmallproject-vllm/llama2-13b-w4a16", text)
        self.assertIn("SPINQUANT_13B_SMOKE_DIRTY_CHECKOUT", text)
        self.assertIn("HF_DATASETS_OFFLINE=1", text)
        self.assertIn("SPINQUANT_13B_SMOKE_PARTIAL_CALIBRATION", text)
        self.assertIn("maximum_orthogonality_error=1e-4", text)
        self.assertIn(
            "ISAMBARD_SPINQUANT_LLAMA2_13B_1STEP_SMOKE_PASSED",
            text,
        )


if __name__ == "__main__":
    unittest.main()
