import importlib.util
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/vllm_quarot_w4a16_llama2_13b_boolq_isambard.json"
)
REFERENCE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/vllm_spinquant_w4a16_llama2_13b_boolq_isambard.json"
)
SBATCH_PATH = (
    PROJECT_ROOT
    / "scripts/run_isambard_vllm_quarot_w4a16_llama2_13b_boolq.sbatch"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module(
    "run_vllm_quarot_w4a16_llama2_13b_boolq_test",
    PROJECT_ROOT / "scripts/run_vllm_quarot_w4a16_llama2_13b_boolq.py",
)


def _model(correct_values):
    rows = [
        {"idx": index, "correct": correct}
        for index, correct in enumerate(correct_values)
    ]
    return {
        "evaluated_examples": len(rows),
        "correct": sum(correct_values),
        "accuracy": sum(correct_values) / len(rows),
        "example_metrics": rows,
    }


class VllmQuarotW4A16BoolQTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.reference = json.loads(
            REFERENCE_CONFIG_PATH.read_text(encoding="utf-8")
        )

    def test_reuses_the_complete_frozen_boolq_protocol(self):
        for key in ("protocol", "dataset", "engine", "formal", "runtime"):
            self.assertEqual(self.config[key], self.reference[key])
        self.assertEqual(self.config["formal"]["examples"], 3270)
        self.assertEqual(self.config["formal"]["expected_requests"], 6540)

    def test_binds_the_accepted_rotated_w4a16_checkpoint(self):
        gate = self.config["source_gate"]
        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["job_id"], "5769503")
        self.assertEqual(gate["expected_model"], "rotated_w4a16")
        self.assertEqual(
            tuple(gate["expected_models"]), ("bf16", "quarot_w4a16")
        )
        self.assertEqual(self.config["serving_reference"]["job_id"], "5960180")

    def test_runner_uses_fresh_process_model_order(self):
        self.assertEqual(RUNNER.EXPECTED_MODELS, ("bf16", "quarot_w4a16"))
        with self.assertRaisesRegex(ValueError, "model order"):
            RUNNER._parse_models([])

    def test_accuracy_and_paired_audit_are_retained(self):
        bf16 = _model([True, True, False, True])
        quarot = _model([False, True, True, True])
        comparison = RUNNER._accuracy_comparison(quarot, bf16)
        self.assertEqual(comparison["correct_delta"], 0)
        paired = RUNNER._paired_correctness_audit(quarot, bf16)
        self.assertEqual(paired["bf16_only_correct"], 1)
        self.assertEqual(paired["quarot_w4a16_only_correct"], 1)
        self.assertEqual(paired["prediction_disagreements"], 2)
        self.assertEqual(paired["mcnemar_exact_two_sided_p"], 1.0)

    def test_batch_is_formal_only_and_does_not_submit_other_jobs(self):
        text = SBATCH_PATH.read_text(encoding="utf-8")
        self.assertIn('mode="${1:-}"', text)
        self.assertIn("--max-examples 3270", text)
        self.assertIn('--model "bf16=${model_snapshot}"', text)
        self.assertIn('--model "quarot_w4a16=${rotated_dir}"', text)
        self.assertIn("validate_sglang_vllm_quarot_w4a16_inputs.py", text)
        self.assertIn("serving_reference_job=5960180", text)
        self.assertNotIn("afterok", text)
        executable_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(line.startswith("sbatch ") for line in executable_lines))


if __name__ == "__main__":
    unittest.main()
