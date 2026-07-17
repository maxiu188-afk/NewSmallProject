import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_rtn_w4a4", PROJECT_ROOT / "scripts" / "compare_rtn_w4a4.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(rotation):
    return {
        "model": {"kind": "pretrained", "id": "org/model", "revision": "a" * 40, "dtype": "bfloat16"},
        "data": {
            "source": "huggingface_text", "id": "org/data", "subset": "subset", "split": "test", "revision": "b" * 40,
            "sequence_length": 2048, "batch_size": 1, "max_samples": 10, "batches": 3,
        },
        "experiment": {"seed": 0},
        "quantization": {"w_bits": 4, "a_bits": 4, "k_bits": 16, "v_bits": 16},
        "rotation": rotation,
        "kv_cache_simulated": False,
        "candidate": {"perplexity": 6.0, "mean_nll": 1.8, "tokens": 12.0},
        "logit_error": {"mean_absolute": 0.1, "max_absolute": 1.0},
    }


class CompareRtnW4A4Tests(unittest.TestCase):
    def test_accepts_matched_naive_and_quarot_results(self):
        naive = _result({"applied": False, "reason": "residual_mode=none"})
        quarot = _result({"applied": True, "residual_mode": "hadamard"})
        quarot["candidate"]["perplexity"] = 5.0
        summary = MODULE.compare(naive, quarot)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["delta_quarot_minus_naive"]["perplexity"], -1.0)

    def test_rejects_unmatched_dataset_revision(self):
        naive = _result({"applied": False, "reason": "residual_mode=none"})
        quarot = _result({"applied": True, "residual_mode": "hadamard"})
        quarot["data"]["revision"] = "c" * 40
        with self.assertRaises(MODULE.ComparisonError):
            MODULE.compare(naive, quarot)
