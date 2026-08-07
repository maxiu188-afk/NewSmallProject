import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_vllm_w4a16_llama2_13b_gate",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VllmW4A16LlamaGateTests(unittest.TestCase):
    def test_worker_launch_uses_current_interpreter_and_fresh_output(self):
        runtime = {
            "vllm": "0.25.1",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "model.json"
            output.write_text("stale", encoding="utf-8")

            def fake_run(command, check):
                self.assertTrue(check)
                self.assertEqual(command[0], sys.executable)
                self.assertIn("--worker-name", command)
                self.assertIn("--worker-model", command)
                self.assertFalse(output.exists())
                output.write_text(
                    json.dumps(
                        {
                            "name": "bf16",
                            "generated_token_ids": [1],
                            "first_token_logprobs": {"1": -0.5},
                            "runtime": runtime,
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch.object(MODULE.subprocess, "run", side_effect=fake_run):
                result = MODULE._run_model_in_subprocess(
                    root / "config.json",
                    "bf16",
                    root / "model",
                    [1, 2],
                    output,
                )
            self.assertEqual(result["name"], "bf16")

    def test_parent_runs_all_three_models_through_separate_workers(self):
        config = {"inference": {"prompt": "The capital of France is"}}
        runtime = {
            "vllm": "0.25.1",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        tokenizer = mock.Mock()
        tokenizer.return_value = {"input_ids": [1, 2, 3]}
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer

        def fake_worker(_config, name, model_path, prompt, output):
            self.assertEqual(prompt, [1, 2, 3])
            return {
                "name": name,
                "model_path": str(model_path),
                "generated_token_ids": [4],
                "first_token_logprobs": {"4": -0.25},
                "runtime": runtime,
                "worker_output": str(output),
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                mock.patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}),
                mock.patch.dict(sys.modules, {"transformers": transformers}),
                mock.patch.object(
                    MODULE,
                    "_run_model_in_subprocess",
                    side_effect=fake_worker,
                ) as launch,
                mock.patch.object(MODULE, "_revision", return_value="revision"),
            ):
                result = MODULE.run(
                    config,
                    root / "config.json",
                    root / "bf16",
                    root / "unrotated",
                    root / "rotated",
                    root / "workers",
                )

        self.assertEqual(launch.call_count, 3)
        self.assertEqual(
            [call.args[1] for call in launch.call_args_list],
            ["bf16", "unrotated_w4a16", "rotated_w4a16"],
        )
        self.assertEqual(
            [call.args[4].name for call in launch.call_args_list],
            ["bf16.json", "unrotated_w4a16.json", "rotated_w4a16.json"],
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(set(result["models"]), {"bf16", "unrotated_w4a16", "rotated_w4a16"})

    def test_spinquant_parent_runs_bf16_and_packed_endpoint_in_fresh_workers(self):
        config = {"inference": {"prompt": "The capital of France is"}}
        runtime = {
            "vllm": "0.25.1",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        tokenizer = mock.Mock()
        tokenizer.return_value = {"input_ids": [1, 2, 3]}
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer

        def fake_worker(_config, name, model_path, prompt, output):
            return {
                "name": name,
                "model_path": str(model_path),
                "generated_token_ids": [4],
                "first_token_logprobs": {"4": -0.25},
                "runtime": runtime,
                "worker_output": str(output),
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original = root / "bf16"
            spinquant = root / "spinquant"
            with (
                mock.patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}),
                mock.patch.dict(sys.modules, {"transformers": transformers}),
                mock.patch.object(
                    MODULE,
                    "_run_model_in_subprocess",
                    side_effect=fake_worker,
                ) as launch,
                mock.patch.object(MODULE, "_revision", return_value="revision"),
            ):
                result = MODULE.run_model_set(
                    config,
                    root / "config.json",
                    original,
                    [("bf16", original), ("spinquant_w4a16", spinquant)],
                    root / "workers",
                )

        self.assertEqual(launch.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in launch.call_args_list],
            ["bf16", "spinquant_w4a16"],
        )
        self.assertEqual(set(result["models"]), {"bf16", "spinquant_w4a16"})
        self.assertTrue(
            result["comparisons"]["spinquant_w4a16_vs_bf16"][
                "generated_tokens_equal"
            ]
        )


if __name__ == "__main__":
    unittest.main()
