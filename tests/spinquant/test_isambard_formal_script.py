from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_100step.sbatch"
)
PAPER_ALIGNED_SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_w16a8_100step.sbatch"
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

    def test_paper_aligned_formal_is_smoke_gated_and_fail_closed(self):
        text = PAPER_ALIGNED_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('smoke_job="${1:-}"', text)
        self.assertIn('"${SLURM_JOB_DEPENDENCY:-}" != afterok:*', text)
        self.assertIn("sha256sum --check --status", text)
        self.assertIn("merge-base", text)
        self.assertIn("SPINQUANT_13B_W16A8_FORMAL_SMOKE_GATE_ACCEPTED", text)
        self.assertIn("SPINQUANT_13B_W16A8_FORMAL_DIRTY_CHECKOUT", text)
        self.assertIn("llama2_13b_w16a8_rotation_train_100.json", text)
        self.assertIn("llama2-13b-paper-nohad-w16a8", text)
        self.assertNotIn("newsmallproject-vllm/llama2-13b-w4a16", text)
        self.assertIn('training["rotation_objective"] == "activation_qdq"', text)
        self.assertIn('training["adapter"]["weight_bits"] == 16', text)
        self.assertIn('training["adapter"]["activation_bits"] == 8', text)
        self.assertIn(
            'training["adapter"]["activation_o_proj_group_size"] == 128',
            text,
        )
        self.assertIn(
            'training["adapter"]["activation_ungrouped_include_zero"] is True',
            text,
        )
        self.assertIn('training["optimizer_steps"] == 100', text)
        self.assertIn('len(training["gradient_maxima"]) == 100', text)
        self.assertIn("maximum_orthogonality_error=1e-4", text)
        self.assertIn(
            "ISAMBARD_SPINQUANT_LLAMA2_13B_W16A8_100STEP_PASSED",
            text,
        )


if __name__ == "__main__":
    unittest.main()
