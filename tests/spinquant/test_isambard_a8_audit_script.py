from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/run_isambard_spinquant_llama2_13b_a8_audit.sbatch"


class SpinQuantIsambardA8AuditScriptTests(unittest.TestCase):
    def test_script_gates_runtime_coverage_and_provenance(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('project_root="${SLURM_SUBMIT_DIR:-', text)
        self.assertIn("max_sequences=2", text)
        self.assertIn("max_calibration_sequences=8", text)
        self.assertIn('audit["registered_modules"] == 280', text)
        self.assertIn('audit["called_modules"] == 280', text)
        self.assertIn('audit["missing_modules"] == []', text)
        self.assertIn('set(audit["role_module_counts"].values()) == {40}', text)
        self.assertIn("NVIDIA GH200 120GB", text)
        self.assertIn("ISAMBARD_SPINQUANT_LLAMA2_13B_A8_AUDIT_PASSED", text)


if __name__ == "__main__":
    unittest.main()
