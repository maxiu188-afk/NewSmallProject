import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(HAS_DEPS, "requires torch and transformers")
class SpinQuantTrainingTests(unittest.TestCase):
    def test_one_step_and_formal_configs_differ_only_in_planned_scale(self):
        smoke = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/llama2_13b_w4a16_rotation_train_1step_smoke.json"
            ).read_text(encoding="utf-8")
        )
        formal = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/llama2_13b_w4a16_rotation_train_100.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(smoke["model"], formal["model"])
        self.assertEqual(smoke["quantization"], formal["quantization"])
        self.assertEqual(smoke["runtime"], formal["runtime"])
        self.assertEqual(smoke["calibration"]["sequence_length"], 2048)
        self.assertEqual(formal["calibration"]["sequence_length"], 2048)
        self.assertEqual(smoke["optimization"]["steps"], 1)
        self.assertEqual(formal["optimization"]["steps"], 100)
        self.assertEqual(smoke["calibration"]["samples"], 8)
        self.assertEqual(formal["calibration"]["samples"], 800)

    def test_tiny_training_consumes_exact_protocol_and_updates_rotations(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from repro.spinquant.training import train_llama_rotations

        torch.manual_seed(53)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                tie_word_embeddings=False,
                use_cache=False,
            )
        ).float()
        sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
        ]
        config = {
            "seed": 0,
            "quantization": {
                "weight_bits": 4,
                "activation_bits": 16,
                "weight_group_size": 8,
                "weight_symmetric": True,
                "activation_symmetric": False,
                "quantize_lm_head": False,
            },
            "optimization": {
                "steps": 1,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1.5,
                "cayley_method": "fixed_point",
                "fixed_point_steps": 5,
                "gradient_checkpointing": False,
            },
        }
        rotations, result = train_llama_rotations(model, sequences, config)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["calibration_sequences"], 2)
        self.assertGreater(result["gradient_maxima"][0], 0.0)
        self.assertLess(result["orthogonality_error"]["r1"], 1e-5)
        self.assertLess(result["orthogonality_error"]["r2"], 1e-5)
        self.assertTrue(all(parameter.requires_grad for parameter in rotations.parameters()))

    def test_training_rejects_implicit_sequence_reuse(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from repro.spinquant.training import train_llama_rotations

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )
        config = {
            "seed": 0,
            "quantization": {
                "weight_bits": 4,
                "activation_bits": 16,
                "weight_group_size": 8,
                "weight_symmetric": True,
                "activation_symmetric": False,
                "quantize_lm_head": False,
            },
            "optimization": {
                "steps": 2,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1.5,
                "cayley_method": "fixed_point",
                "fixed_point_steps": 5,
            },
        }
        with self.assertRaises(ValueError):
            train_llama_rotations(model, [[1, 2, 3, 4]], config)

    def test_runner_requires_exact_calibration_provenance(self):
        from scripts.spinquant import train_llama_rotations as runner

        training_config = json.loads(
            (
                PROJECT_ROOT
                / "configs/spinquant/llama2_13b_w4a16_rotation_train_1step_smoke.json"
            ).read_text(encoding="utf-8")
        )
        calibration_path = (
            PROJECT_ROOT / training_config["calibration"]["config"]
        )
        calibration_config = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        config_digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory)
                / training_config["model"]["revision"]
            )
            snapshot.mkdir()
            model_config = snapshot / "config.json"
            model_config.write_text('{"model_type":"llama"}\n', encoding="utf-8")
            model_digest = hashlib.sha256(model_config.read_bytes()).hexdigest()
            calibration = {
                "manifest": {
                    "provenance": {
                        "project_revision": "fixed-revision",
                        "config_sha256": config_digest,
                        "model": calibration_config["model"],
                        "model_config_sha256": model_digest,
                        "dataset": calibration_config["calibration"],
                    }
                }
            }
            with mock.patch.object(
                runner,
                "_revision",
                return_value="fixed-revision",
            ):
                runner._validate_calibration_provenance(
                    training_config,
                    calibration,
                    snapshot,
                )
                calibration["manifest"]["provenance"]["config_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "config_sha256",
                ):
                    runner._validate_calibration_provenance(
                        training_config,
                        calibration,
                        snapshot,
                    )


if __name__ == "__main__":
    unittest.main()
