import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4a16_llama2_13b_ppl_isambard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_vllm_w4a16_llama2_13b_ppl",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_ppl.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VllmW4A16LlamaPplTests(unittest.TestCase):
    def test_frozen_protocol_matches_formal_fake_quant_token_count(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        evaluation = config["evaluation"]
        self.assertEqual(evaluation["dataset_split"], "test")
        self.assertEqual(evaluation["samples"], 162)
        self.assertEqual(evaluation["sequence_length"], 2048)
        self.assertEqual(evaluation["expected_scored_tokens"], 162 * 2047)
        self.assertFalse(evaluation["add_special_tokens"])
        self.assertFalse(config["engine"]["enforce_eager"])
        self.assertEqual(config["engine"]["max_model_len"], 2049)
        self.assertEqual(config["engine"]["prompt_logprobs"], 1)

    def test_sequence_nll_uses_each_observed_prompt_token(self):
        tokens = [10, 11, 12]
        output = types.SimpleNamespace(
            prompt_token_ids=tokens,
            prompt_logprobs=[
                None,
                {11: types.SimpleNamespace(logprob=-1.25)},
                {
                    99: types.SimpleNamespace(logprob=-0.1),
                    12: types.SimpleNamespace(logprob=-0.75),
                },
            ],
        )
        result = MODULE._sequence_nll(output, tokens)
        self.assertEqual(result["tokens"], 2)
        self.assertAlmostEqual(result["nll"], 2.0)
        self.assertAlmostEqual(result["mean_nll"], 1.0)

    def test_sequence_nll_rejects_missing_observed_token(self):
        output = types.SimpleNamespace(
            prompt_token_ids=[10, 11],
            prompt_logprobs=[None, {99: types.SimpleNamespace(logprob=-0.1)}],
        )
        with self.assertRaisesRegex(RuntimeError, "missing target token"):
            MODULE._sequence_nll(output, [10, 11])

    def test_comparison_reports_delta_and_ratio(self):
        reference = {"tokens": 10, "mean_nll": 1.0, "perplexity": math.e}
        candidate = {"tokens": 10, "mean_nll": 1.2, "perplexity": math.exp(1.2)}
        result = MODULE._comparison(candidate, reference)
        self.assertAlmostEqual(result["mean_nll_delta"], 0.2)
        self.assertAlmostEqual(result["perplexity_ratio"], math.exp(0.2))

    def test_parent_launches_models_in_fresh_workers(self):
        runtime = {
            "vllm": "0.25.1+cu129",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.json"
            config_path = root / "config.json"
            models = {
                name: root / name
                for name in MODULE.EXPECTED_MODELS
            }

            def fake_worker(**kwargs):
                name = kwargs["name"]
                return {
                    "name": name,
                    "tokens": 2,
                    "mean_nll": 1.0,
                    "perplexity": math.e,
                    "runtime": runtime,
                }

            with (
                mock.patch.object(
                    MODULE,
                    "_load_sequences",
                    return_value=([[1, 2, 3]], {"dataset": {"token_ids_sha256": "x"}}),
                ),
                mock.patch.object(MODULE, "_run_worker", side_effect=fake_worker) as worker,
                mock.patch.object(MODULE, "_revision", return_value="revision"),
                mock.patch.object(MODULE, "_sha256", return_value="digest"),
            ):
                result = MODULE.run(
                    config={"engine": {}},
                    config_path=config_path,
                    manifest_path=manifest,
                    models=models,
                    max_sequences=1,
                    worker_output_dir=root / "workers",
                )
        self.assertEqual(worker.call_count, 3)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(set(result["models"]), set(MODULE.EXPECTED_MODELS))


if __name__ == "__main__":
    unittest.main()
