from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_1step_smoke.sbatch"
)
PAPER_ALIGNED_SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_w16a8_1step_smoke.sbatch"
)
FP8_SCRIPT = (
    PROJECT_ROOT / "scripts/run_isambard_spinquant_llama2_13b_fp8.sbatch"
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

    def test_paper_aligned_smoke_is_activation_only_and_fail_closed(self):
        text = PAPER_ALIGNED_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('project_root="${SLURM_SUBMIT_DIR:-', text)
        self.assertIn("llama2_13b_w16a8_rotation_train_1step_smoke.json", text)
        self.assertIn("llama2-13b-paper-nohad-w16a8", text)
        self.assertNotIn("newsmallproject-vllm/llama2-13b-w4a16", text)
        self.assertIn("SPINQUANT_13B_W16A8_SMOKE_DIRTY_CHECKOUT", text)
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
        self.assertIn("maximum_orthogonality_error=1e-4", text)
        self.assertIn(
            "ISAMBARD_SPINQUANT_LLAMA2_13B_W16A8_1STEP_SMOKE_PASSED",
            text,
        )

    def test_fp8_smoke_freezes_vllm_quantizer_contract(self):
        text = FP8_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("llama2_13b_w16afp8_rotation_train_1step_smoke.json", text)
        self.assertIn("llama2-13b-fp8-targeted", text)
        self.assertIn('training["rotation_objective"] == "fp8_activation_qdq"', text)
        self.assertIn('training["adapter"]["activation_dtype"] == "float8_e4m3fn"', text)
        self.assertIn("vllm_dynamic_per_token_fp8_e4m3fn_qdq_with_ste", text)
        self.assertIn('training["adapter"]["activation_o_proj_group_size"] == -1', text)
        self.assertIn("SPINQUANT_13B_FP8_STAGE=train", text)


if __name__ == "__main__":
    unittest.main()
