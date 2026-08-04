import importlib.util
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/deployment/sglang_vllm_quarot_w4a16_boolq_smoke_isambard.json"
SPEC = importlib.util.spec_from_file_location(
    "run_sglang_vllm_quarot_w4a16_boolq_test",
    PROJECT_ROOT / "scripts/run_sglang_vllm_quarot_w4a16_boolq.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SglangVllmQuarotW4A16BoolQTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_freezes_quarot_only_boolq_smoke(self):
        MODULE._validate_config(self.config)
        self.assertEqual(self.config["model"], "quarot_w4a16")
        self.assertEqual(self.config["source_gate"]["job_id"], "5769503")
        self.assertEqual(self.config["protocol"]["choices"], ["no", "yes"])
        self.assertEqual(self.config["smoke"]["examples"], 32)
        self.assertEqual(self.config["sglang"]["attention_backend"], "flashinfer")
        self.assertIsNone(self.config["sglang"]["offline_quantization_argument"])

    def test_sglang_loglikelihood_requires_exact_continuation_ids(self):
        meta = {"input_token_logprobs": [[-0.25, 12], [-0.75, 13]]}
        self.assertAlmostEqual(
            MODULE._sglang_loglikelihood(meta, [10, 11, 12, 13], 2), -1.0
        )
        with self.assertRaisesRegex(RuntimeError, "different order"):
            MODULE._sglang_loglikelihood(
                {"input_token_logprobs": [[-0.25, 99], [-0.75, 13]]},
                [10, 11, 12, 13],
                2,
            )

    def test_shared_server_command_selects_flashinfer_without_requantizing(self):
        from scripts.run_sglang_vllm_llama2_13b_smoke import _server_command

        command = _server_command(
            backend="sglang",
            executable=Path("/sglang/bin/python"),
            model_path=Path("/models/rotated-w4a16"),
            served_name="quarot-w4a16",
            config=self.config,
        )
        self.assertEqual(
            command[command.index("--attention-backend") + 1], "flashinfer"
        )
        self.assertNotIn("--quantization", command)

    def test_comparison_records_prediction_and_score_differences(self):
        cases = {
            "vllm": {
                "status": "passed",
                "accuracy": 1.0,
                "example_metrics": [
                    {"idx": 7, "prediction": 1, "choice_loglikelihoods": [-2.0, -1.0]}
                ],
            },
            "sglang": {
                "status": "passed",
                "accuracy": 0.0,
                "example_metrics": [
                    {"idx": 7, "prediction": 0, "choice_loglikelihoods": [-1.5, -1.6]}
                ],
            },
        }
        result = MODULE._comparison(cases)
        self.assertEqual(result["prediction_disagreement_count"], 1)
        self.assertEqual(result["prediction_disagreement_indices"], [7])
        self.assertAlmostEqual(result["accuracy_delta_sglang_minus_vllm"], -1.0)
        self.assertAlmostEqual(result["max_absolute_choice_loglikelihood_delta"], 0.6)

    def test_batch_script_has_no_formal_submission(self):
        text = (
            PROJECT_ROOT
            / "scripts/run_isambard_sglang_vllm_quarot_w4a16_boolq_smoke.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("formal_job=not_submitted", text)
        self.assertNotIn("--dependency=afterok", text)


if __name__ == "__main__":
    unittest.main()
