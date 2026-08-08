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
    / "configs/deployment/vllm_spinquant_w4a16_llama2_13b_boolq_isambard.json"
)
BASE_BOOLQ_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/vllm_w4afp8_llama2_13b_boolq_isambard.json"
)
SBATCH_PATH = (
    PROJECT_ROOT
    / "scripts/run_isambard_vllm_spinquant_w4a16_llama2_13b_boolq.sbatch"
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BOOLQ = _load_script(
    "run_vllm_spinquant_w4a16_llama2_13b_boolq_test",
    PROJECT_ROOT / "scripts/run_vllm_spinquant_w4a16_llama2_13b_boolq.py",
)
VALIDATOR = _load_script(
    "validate_vllm_spinquant_w4a16_boolq_inputs_test",
    PROJECT_ROOT / "scripts/validate_vllm_spinquant_w4a16_boolq_inputs.py",
)


def _valid_quantization_config():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                    "actorder": "static",
                    "observer": "memoryless_minmax",
                },
            }
        },
    }


class VllmSpinQuantW4A16BoolQTests(unittest.TestCase):
    def test_config_reuses_frozen_w4afp8_boolq_protocol_and_dataset(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        base = json.loads(BASE_BOOLQ_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["protocol"], base["protocol"])
        for field in (
            "revision",
            "expected_rows",
            "expected_fingerprint",
            "expected_examples_sha256",
            "expected_label_counts",
        ):
            self.assertEqual(config["dataset"][field], base["dataset"][field])
        self.assertEqual(
            config["dataset"]["expected_manifest_config_sha256"],
            "fa44d2c61317022bac346ff4a14824d8da90f192a00c1e6204493ed8d80f818f",
        )
        self.assertEqual(config["execution"]["kind"], "formal_only")
        self.assertIn("not SpinQuant W4A8KV16 reproduction", config["scope"])

    def test_config_binds_accepted_export_and_load_gate(self):
        gate = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["source_gate"]
        self.assertEqual(gate["job_id"], "5945162")
        self.assertEqual(
            gate["export_result_sha256"],
            "b21ca19f34bf24470fdec797f90821b46edb77bfc1717c23724b989aa4ff05cf",
        )
        self.assertEqual(
            gate["inference_result_sha256"],
            "fa533a64f7b8345b547f45b6fa666ca8e6181d3369dc487dc74c5cf171f995c4",
        )
        self.assertEqual(
            gate["checkpoint_tree_sha256"],
            "41a79153d2ad7e0fb598819adc538ce65ba7c1a05629459f77cb816d226ce2da",
        )
        self.assertEqual(gate["expected_models"], list(BOOLQ.EXPECTED_MODELS))

    def test_runner_reuses_accepted_prompt_boundary_and_scoring(self):
        self.assertIs(BOOLQ._load_examples, BOOLQ.common._load_examples)
        self.assertIs(BOOLQ._encode_pair, BOOLQ.common._encode_pair)
        self.assertIs(BOOLQ._choice_loglikelihood, BOOLQ.common._choice_loglikelihood)

    def test_parent_runs_both_models_in_fresh_workers(self):
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
            models = {name: root / name for name in BOOLQ.EXPECTED_MODELS}

            def fake_worker(**kwargs):
                name = kwargs["name"]
                return {
                    "name": name,
                    "evaluated_examples": 1,
                    "evaluated_requests": 2,
                    "correct": int(name == "bf16"),
                    "accuracy": float(name == "bf16"),
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

        self.assertEqual(worker.call_count, 2)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(tuple(result["models"]), BOOLQ.EXPECTED_MODELS)
        self.assertEqual(
            result["comparisons"]["spinquant_w4a16_vs_bf16"]["correct_delta"],
            -1,
        )

    def test_source_validator_binds_paths_tree_and_ancestor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bf16 = root / "bf16"
            spinquant = root / "spinquant"
            bf16.mkdir()
            spinquant.mkdir()
            (bf16 / "config.json").write_text("{}", encoding="utf-8")
            (spinquant / "config.json").write_text("{}", encoding="utf-8")

            source_revision = "source-revision"
            export_path = root / "export.json"
            inference_path = root / "inference.json"
            manifest_path = root / "source-manifest.txt"
            stdout_path = root / "job.out"
            export_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "mode": "spinquant",
                        "project_revision": source_revision,
                        "checkpoint": str(spinquant),
                        "checkpoint_tree_sha256": "tree-digest",
                        "packed_decoder_linear_count": 280,
                        "quantization_config": _valid_quantization_config(),
                    }
                ),
                encoding="utf-8",
            )
            inference_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "project_revision": source_revision,
                        "models": {
                            "bf16": {"model_path": str(bf16)},
                            "spinquant_w4a16": {"model_path": str(spinquant)},
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                f"git_revision={source_revision}\n", encoding="utf-8"
            )
            stdout_path.write_text(
                "Using MacheteLinearKernel for CompressedTensorsWNA16\n"
                "GATE_PASSED\n",
                encoding="utf-8",
            )
            config = {
                "source_gate": {
                    "status": "accepted",
                    "job_id": "job",
                    "required_marker": "GATE_PASSED",
                    "required_kernel_evidence": (
                        "Using MacheteLinearKernel for CompressedTensorsWNA16"
                    ),
                    "export_result_sha256": VALIDATOR._sha256(export_path),
                    "inference_result_sha256": VALIDATOR._sha256(inference_path),
                    "source_manifest_sha256": VALIDATOR._sha256(manifest_path),
                    "stdout_sha256": VALIDATOR._sha256(stdout_path),
                    "checkpoint_tree_sha256": "tree-digest",
                    "expected_models": list(VALIDATOR.EXPECTED_MODELS),
                }
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def fake_git(_project_root, *arguments, **_kwargs):
                if arguments[:2] == ("rev-parse", "HEAD"):
                    return types.SimpleNamespace(stdout="current-revision\n", returncode=0)
                if arguments[:2] == ("merge-base", "--is-ancestor"):
                    return types.SimpleNamespace(stdout="", returncode=0)
                if arguments[:2] == ("diff", "--name-only"):
                    return types.SimpleNamespace(stdout="", returncode=0)
                raise AssertionError(arguments)

            with (
                mock.patch.object(VALIDATOR, "_git", side_effect=fake_git),
                mock.patch.object(
                    VALIDATOR,
                    "checkpoint_tree_sha256",
                    return_value="tree-digest",
                ),
                mock.patch.object(
                    VALIDATOR,
                    "_checkpoint_metadata",
                    return_value={
                        "packed_decoder_linear_count": 280,
                        "quantization_config": _valid_quantization_config(),
                    },
                ),
            ):
                result = VALIDATOR.validate(
                    project_root=root,
                    config_path=config_path,
                    export_result_path=export_path,
                    inference_result_path=inference_path,
                    source_manifest_path=manifest_path,
                    stdout_path=stdout_path,
                    models={"bf16": bf16, "spinquant_w4a16": spinquant},
                )

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["source_revision"], source_revision)
        self.assertEqual(result["checkpoint_tree_sha256"], "tree-digest")

    def test_batch_is_formal_only_and_checks_full_result(self):
        text = SBATCH_PATH.read_text(encoding="utf-8")
        self.assertIn('mode="${1:-}"', text)
        self.assertIn('if [[ "${mode}" != "formal" ]]', text)
        self.assertIn("--max-examples 3270", text)
        self.assertIn("VLLM_SPINQUANT_W4A16_BOOLQ_INPUTS_ACCEPTED", (
            PROJECT_ROOT
            / "scripts/validate_vllm_spinquant_w4a16_boolq_inputs.py"
        ).read_text(encoding="utf-8"))
        self.assertIn(
            "ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_BOOLQ_PASSED",
            text,
        )
        self.assertNotIn("afterok", text)


if __name__ == "__main__":
    unittest.main()
