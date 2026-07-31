from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PROJECT_ROOT
    / "scripts/run_isambard_spinquant_llama2_13b_nohad_w4a8_ppl.sbatch"
)


class SpinQuantIsambardW4A8PplScriptTests(unittest.TestCase):
    def test_script_gates_smoke_formal_and_runtime_forms(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('mode="${1:-}"', text)
        self.assertIn('"${SLURM_JOB_DEPENDENCY:-}" != afterok:*', text)
        self.assertIn('smoke["project_revision"] == sys.argv[2]', text)
        self.assertIn('max_calibration_sequences=8', text)
        self.assertIn('max_calibration_sequences=128', text)
        self.assertIn('preparation["gptq"]["linear_layers"] == 280', text)
        self.assertIn('activation["activation_bits"] == 8', text)
        self.assertIn('activation["activation_symmetric"] is False', text)
        self.assertIn('activation["activation_o_proj_group_size"] == 128', text)
        self.assertIn(
            'activation["activation_ungrouped_include_zero"] is True', text
        )
        self.assertIn('preparation["rotation"]["online_rotation_modules"] == 0', text)
        self.assertIn('torch.cuda.get_device_name(0) == "NVIDIA GH200 120GB"', text)
        self.assertIn("SLURM_SUBMIT_DIR", text)


if __name__ == "__main__":
    unittest.main()
