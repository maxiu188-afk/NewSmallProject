import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/vllm_w4afp8_llama2_13b_boolq_isambard.json"
)
FP8_TARGETED_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/vllm_w4afp8_fp8_spinquant_llama2_13b_boolq_isambard.json"
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BOOLQ = _load_script(
    "run_vllm_w4afp8_llama2_13b_boolq_test",
    PROJECT_ROOT / "scripts/run_vllm_w4afp8_llama2_13b_boolq.py",
)
PREPARE = _load_script(
    "prepare_boolq_validation_test",
    PROJECT_ROOT / "scripts/prepare_boolq_validation.py",
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


class VllmW4AFP8BoolQTests(unittest.TestCase):
    def test_config_freezes_quarot_lm_eval_boolq_protocol(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol = config["protocol"]
        dataset = config["dataset"]
        self.assertEqual(
            protocol["lm_eval_commit"],
            "9b0b15b1ccace3534ffbd13298c569869ce8eaf3",
        )
        self.assertEqual(protocol["num_fewshot"], 0)
        self.assertEqual(protocol["choices"], ["no", "yes"])
        self.assertEqual(protocol["target_delimiter"], " ")
        self.assertEqual(dataset["split"], "validation")
        self.assertEqual(dataset["expected_rows"], 3270)
        self.assertEqual(len(dataset["expected_examples_sha256"]), 64)
        self.assertEqual(config["paper_reference"]["paper_precision"], "W4A8KV16")
        self.assertIn("not a reproduction", config["scope"])

    def test_encode_pair_matches_pinned_harness_boundary(self):
        whole, continuation_start = BOOLQ._encode_pair(
            FakeTokenizer(),
            "passage\nQuestion: q?\nAnswer:",
            " yes",
            2049,
        )
        self.assertEqual(bytes(whole).decode(), "passage\nQuestion: q?\nAnswer: yes")
        self.assertEqual(bytes(whole[continuation_start:]).decode(), " yes")

    def test_fp8_targeted_config_freezes_accepted_source_gate(self):
        config = json.loads(FP8_TARGETED_CONFIG_PATH.read_text(encoding="utf-8"))
        gate = config["source_gate"]
        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["job_id"], "5881273")
        self.assertEqual(
            gate["result_sha256"],
            "c292e5ec7abfcfff762e5b1f59139fe0cb423b64cf24ba370969adf222e96a55",
        )
        self.assertEqual(
            gate["capability_result_sha256"],
            "495bcc458681f5473e4b1ad50db96b249c82c1ef88cd0f7085e505e5dbdc8d62",
        )
        self.assertEqual(config["dataset"]["expected_rows"], 3270)
        self.assertEqual(
            config["dataset"]["expected_manifest_config_sha256"],
            "fa44d2c61317022bac346ff4a14824d8da90f192a00c1e6204493ed8d80f818f",
        )
        self.assertIn("FP8-targeted SpinQuant", config["scope"])

    def test_load_examples_accepts_explicit_materialized_config_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            examples_path = root / "validation.jsonl"
            examples_path.write_text(
                '{"idx":0,"label":1,"passage":"p","question":"q"}\n',
                encoding="utf-8",
            )
            examples_sha256 = BOOLQ._sha256(examples_path)
            manifest_path = root / "manifest.json"
            manifest = {
                "status": "passed",
                "config_sha256": "prepared-config-digest",
                "dataset": {"revision": "revision", "fingerprint": "fingerprint"},
                "examples": {
                    "path": str(examples_path),
                    "sha256": examples_sha256,
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            config = {
                "_config_path": str(root / "new-config.json"),
                "dataset": {
                    "revision": "revision",
                    "expected_fingerprint": "fingerprint",
                    "expected_examples_sha256": examples_sha256,
                    "expected_rows": 1,
                    "expected_manifest_config_sha256": "prepared-config-digest",
                },
            }
            examples, loaded_manifest = BOOLQ._load_examples(
                config, manifest_path, max_examples=1
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(loaded_manifest, manifest)

            manifest["config_sha256"] = "wrong-digest"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different config"):
                BOOLQ._load_examples(config, manifest_path, max_examples=1)

    def test_choice_loglikelihood_scores_only_continuation(self):
        tokens = [10, 11, 12, 13]
        output = types.SimpleNamespace(
            prompt_token_ids=tokens,
            prompt_logprobs=[
                None,
                {11: types.SimpleNamespace(logprob=-8.0)},
                {12: types.SimpleNamespace(logprob=-0.25)},
                {99: types.SimpleNamespace(logprob=-0.01), 13: types.SimpleNamespace(logprob=-0.75)},
            ],
        )
        self.assertAlmostEqual(
            BOOLQ._choice_loglikelihood(output, tokens, continuation_start=2),
            -1.0,
        )

    def test_canonical_dataset_line_has_stable_key_order(self):
        line = PREPARE._canonical_line(
            {"question": "q", "passage": "p", "label": 1, "idx": 2}
        )
        self.assertEqual(
            line,
            '{"idx":2,"label":1,"passage":"p","question":"q"}\n',
        )

    def test_parent_runs_all_models_in_fresh_workers(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["_config_path"] = str(CONFIG_PATH)
        runtime = {
            "vllm": "0.25.1+cu129",
            "torch": "2.11.0+cu129",
            "transformers": "5.14.1",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
            "tokenizer_class": "LlamaTokenizerFast",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = {name: root / name for name in BOOLQ.EXPECTED_VARIANTS}

            def fake_worker(**kwargs):
                name = kwargs["name"]
                return {
                    "name": name,
                    "evaluated_examples": 1,
                    "evaluated_requests": 2,
                    "correct": 1,
                    "accuracy": 1.0,
                    "runtime": runtime,
                    "example_metrics": [{"idx": 7}],
                }

            with (
                mock.patch.object(
                    BOOLQ,
                    "_load_examples",
                    return_value=([{"idx": 7}], {"examples": {"sha256": "x"}}),
                ),
                mock.patch.object(
                    BOOLQ, "_run_worker", side_effect=fake_worker
                ) as worker,
                mock.patch.object(BOOLQ, "_revision", return_value="revision"),
                mock.patch.object(BOOLQ, "_sha256", return_value="digest"),
            ):
                result = BOOLQ.run(
                    config=config,
                    config_path=CONFIG_PATH,
                    manifest_path=root / "manifest.json",
                    tokenizer_path=root / "tokenizer",
                    models=models,
                    max_examples=1,
                    worker_output_dir=root / "workers",
                )
        self.assertEqual(worker.call_count, 4)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["evaluated_examples"], 1)
        self.assertEqual(set(result["models"]), set(BOOLQ.EXPECTED_VARIANTS))

    def test_batch_script_requires_reviewed_smoke_dependency(self):
        text = (
            PROJECT_ROOT
            / "scripts/run_isambard_vllm_w4afp8_llama2_13b_boolq.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("VLLM_W4AFP8_BOOLQ_FORMAL_REQUIRES_AFTEROK", text)
        self.assertIn("VLLM_W4AFP8_BOOLQ_SMOKE_ACCEPTED", text)
        self.assertIn('max_examples=32', text)
        self.assertIn('max_examples=3270', text)
        self.assertIn("spinquant-w4afp8-int8-transfer", text)
        self.assertIn("VLLM_W4AFP8_BOOLQ_CONFIG", text)
        self.assertIn("VLLM_W4AFP8_SPINQUANT_CHECKPOINT_DIR", text)
        self.assertIn("VLLM_W4AFP8_RESULTS_DIR", text)


if __name__ == "__main__":
    unittest.main()
