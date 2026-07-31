import json
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SpinQuantPplTests(unittest.TestCase):
    def test_config_pins_matched_quarot_and_spinquant_cases(self):
        from scripts.spinquant.evaluate_llama_w4a16_ppl import EXPECTED_CASES

        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/llama2_13b_w4a16_fake_quant_ppl_isambard.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(config["evaluation"]["cases"]), EXPECTED_CASES)
        self.assertEqual(config["model"]["id"], "meta-llama/Llama-2-13b-hf")
        self.assertEqual(config["quantization"]["scheme"], "W4A16")
        self.assertEqual(config["quantization"]["weight_group_size"], 128)
        self.assertEqual(config["tokens"]["samples"], 162)
        self.assertEqual(config["tokens"]["scored_tokens"], 331614)

    def test_comparison_uses_matched_token_count(self):
        from scripts.spinquant.evaluate_llama_w4a16_ppl import _comparison

        reference = {"tokens": 10, "mean_nll": 1.0, "perplexity": 2.0}
        candidate = {"tokens": 10, "mean_nll": 1.2, "perplexity": 2.5}
        self.assertEqual(
            _comparison(candidate, reference),
            {
                "mean_nll_delta": 0.19999999999999996,
                "perplexity_delta": 0.5,
                "perplexity_ratio": 1.25,
            },
        )
        candidate["tokens"] = 9
        with self.assertRaises(RuntimeError):
            _comparison(candidate, reference)

    def test_token_loader_rejects_manifest_hash_drift(self):
        from scripts.spinquant.evaluate_llama_w4a16_ppl import _load_sequences

        config = {
            "tokens": {
                "config": "configs/deployment/vllm_w4a16_llama2_13b_ppl_isambard.json",
                "manifest_sha256": "0" * 64,
                "token_ids_sha256": "1" * 64,
                "samples": 1,
                "sequence_length": 4,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest SHA256"):
                _load_sequences(config, manifest, None)


if __name__ == "__main__":
    unittest.main()
