import importlib.util
import inspect
import json
import sys
import unittest
from unittest import mock
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
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_sglang_vllm_quarot_w4a16_inputs_test",
    PROJECT_ROOT / "scripts/validate_sglang_vllm_quarot_w4a16_inputs.py",
)
assert VALIDATOR_SPEC is not None and VALIDATOR_SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = VALIDATOR
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


class SglangVllmQuarotW4A16BoolQTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_freezes_quarot_only_boolq_smoke(self):
        MODULE._validate_config(self.config)
        self.assertEqual(self.config["model"], "quarot_w4a16")
        self.assertEqual(self.config["source_gate"]["job_id"], "5769503")
        self.assertEqual(
            self.config["source_gate"]["checkpoint_tree_sha256"],
            "2f22f56a5edb32e037416c78be49e617bcee796abca26822704a6ef825ff8e99",
        )
        self.assertEqual(self.config["protocol"]["choices"], ["no", "yes"])
        self.assertEqual(self.config["smoke"]["examples"], 32)
        self.assertEqual(self.config["sglang"]["attention_backend"], "flashinfer")
        self.assertIsNone(self.config["sglang"]["offline_quantization_argument"])
        self.assertEqual(
            self.config["sglang"]["toolchain"],
            {
                "compiler_module": "gcc-native/14.2",
                "expected_compiler_major": 14,
                "expected_flashinfer_version": "0.6.14",
                "expected_ninja_version": "1.13.0",
                "expected_nvcc_release": "13.3",
                "expected_sglang_kernel_version": "0.4.5+cu129",
                "expected_transformers_version": "5.12.1",
                "expected_tvm_ffi_version": "0.1.11",
            },
        )

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

    def test_sglang_logprob_start_len_scores_first_continuation_token(self):
        self.assertEqual(MODULE._sglang_logprob_start_len(11), 10)
        with self.assertRaisesRegex(ValueError, "second token"):
            MODULE._sglang_logprob_start_len(0)

    def test_requests_apply_sglang_logprob_offset(self):
        examples = [{"idx": 0, "label": 1}]
        with mock.patch.object(MODULE, "_context", return_value="prompt"), mock.patch.object(
            MODULE, "_encode_pair", return_value=([1, 2, 3], 2)
        ):
            requests = MODULE._requests(object(), examples, self.config)
        self.assertEqual(len(requests), 2)
        self.assertEqual([item["continuation_start"] for item in requests], [2, 2])
        self.assertEqual([item["logprob_start_len"] for item in requests], [1, 1])

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

    def test_parent_preserves_sglang_virtualenv_executable(self):
        source = inspect.getsource(MODULE.main)
        self.assertIn(
            "sglang_python = _absolute_executable(args.sglang_python)", source
        )
        self.assertNotIn("args.sglang_python.resolve()", source)

    def test_validator_accepts_only_exact_w4a16_metadata(self):
        quantization = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                        "dynamic": False,
                        "actorder": "static",
                    },
                }
            },
        }
        VALIDATOR._validate_quantization_config(quantization)
        quantization["config_groups"]["group_0"]["weights"]["group_size"] = 64
        with self.assertRaisesRegex(RuntimeError, "group_size"):
            VALIDATOR._validate_quantization_config(quantization)

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
        self.assertIn('export PATH="${sglang_env}/bin:${PATH}"', text)
        self.assertIn("module load gcc-native/14.2", text)
        self.assertIn('export CXX="$(command -v g++)"', text)
        self.assertIn('export NVCC_CCBIN="${CXX}"', text)
        self.assertIn('"${sglang_env}/bin/ninja"', text)
        self.assertIn("preflight_sglang_wna16_runtime.py", text)
        self.assertLess(
            text.index("SGLANG_VLLM_W4A16_BOOLQ_STAGE=runtime_preflight"),
            text.index("SGLANG_VLLM_W4A16_BOOLQ_STAGE=source_gate"),
        )
        self.assertIn(
            'assert result["comparison_status"] == "both_backends_scored"', text
        )
        self.assertIn("formal_job=not_submitted", text)
        self.assertNotIn("--dependency=afterok", text)

    def test_runtime_preflight_compiles_and_executes_exact_jit(self):
        text = (
            PROJECT_ROOT / "scripts/preflight_sglang_wna16_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_jit_gptq_marlin_repack_module()", text)
        self.assertIn("repacked = gptq_marlin_repack(", text)
        self.assertIn("GenerateReqInput(", text)
        self.assertIn("request.normalize_batch_and_arguments()", text)

    def test_environment_setup_pins_build_toolchain(self):
        text = (
            PROJECT_ROOT / "scripts/setup_isambard_sglang_env.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("module load gcc-native/14.2", text)
        self.assertIn('"ninja==1.13.0"', text)
        self.assertIn('export NVCC_CCBIN="${CXX}"', text)


if __name__ == "__main__":
    unittest.main()
