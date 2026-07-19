import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_llama2_13b_gptq_study", PROJECT_ROOT / "scripts" / "run_llama2_13b_gptq_study.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LlamaGptqStudyTests(unittest.TestCase):
    def test_full_study_is_ordered_and_reuses_128_calibration_endpoints(self):
        rows = MODULE.select_rows([])
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0].identifier, "naive-f3-cal128")
        self.assertEqual(rows[4].identifier, "quarot-f4-cal128")
        self.assertEqual([row.calibration_sequences for row in rows], [128, 128, 128, 128, 128, 32, 32, 64, 64])

    def test_selection_preserves_study_order(self):
        rows = MODULE.select_rows(["quarot-f4-cal64", "naive-f3-cal32"])
        self.assertEqual([row.identifier for row in rows], ["naive-f3-cal32", "quarot-f4-cal64"])

    def test_selection_rejects_unknown_or_repeated_rows(self):
        with self.assertRaises(MODULE.StudyError):
            MODULE.select_rows(["unknown"])
        with self.assertRaises(MODULE.StudyError):
            MODULE.select_rows(["naive-f3-cal128", "naive-f3-cal128"])

    def test_resume_requires_the_full_13b_gptq_completion_record(self):
        row = MODULE.STUDY_ROWS[0]
        result = {
            "candidate": {"perplexity": 6.0, "mean_nll": 1.8, "tokens": 331614.0},
            "calibration": {"batches": 128, "tokens": 128 * 2048},
            "gptq": {"layers": 40, "linear_layers": 280, "calibration_sequences": 128, "calibration_sequence_length": 2048},
            "logit_error": {"mean_absolute": 0.1, "max_absolute": 1.0},
            "weight_quantization": {"method": "gptq"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertTrue(MODULE._result_is_complete(path, row))
            result["gptq"]["linear_layers"] = 279
            path.write_text(json.dumps(result), encoding="utf-8")
            self.assertFalse(MODULE._result_is_complete(path, row))


if __name__ == "__main__":
    unittest.main()
