import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_spinquant_w4a16_llama2_13b_isambard.json"
)
EXPORTER = PROJECT_ROOT / "scripts" / "export_vllm_w4a16_llama2_13b.py"
SBATCH = (
    PROJECT_ROOT
    / "scripts"
    / "run_isambard_vllm_spinquant_w4a16_llama2_13b_gate.sbatch"
)

from repro.vllm_w4a16 import (  # noqa: E402
    checkpoint_tree_sha256,
    validate_checkpoint_quantization_config,
    validate_spinquant_export_config,
)


def _valid_checkpoint_config():
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


class VllmSpinQuantW4A16ExportTests(unittest.TestCase):
    def test_export_config_freezes_w4a16_training_artifact(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_spinquant_export_config(config)
        rotation = config["rotation"]
        self.assertEqual(rotation["spinquant_training_job_id"], "5842047")
        self.assertEqual(
            rotation["spinquant_training_project_revision"],
            "ee5e754f80f29925ce0531c5357bee3b835de5d0",
        )
        self.assertEqual(
            rotation["spinquant_rotation_manifest_sha256"],
            "825701f474aa38d9b3c063775429813c9ce7673f229282df7ed6b7d54f156ad5",
        )
        self.assertEqual(
            rotation["spinquant_rotation_safetensors_sha256"],
            "303c614f425ea5d37e138643dd52a2410a4747fda7e3038587c93a2db05a57fe",
        )
        self.assertEqual(
            config["runtime"]["quantizer_compressed_tensors_version"],
            "0.17.1",
        )

    def test_checkpoint_contract_is_weight_only_static_actorder(self):
        metadata = _valid_checkpoint_config()
        validate_checkpoint_quantization_config(metadata)
        metadata["config_groups"]["group_0"]["input_activations"] = {
            "num_bits": 8
        }
        with self.assertRaisesRegex(ValueError, "must not quantize input"):
            validate_checkpoint_quantization_config(metadata)
        metadata = _valid_checkpoint_config()
        metadata["config_groups"]["group_0"]["weights"]["actorder"] = None
        with self.assertRaisesRegex(ValueError, "actorder"):
            validate_checkpoint_quantization_config(metadata)

    def test_exporter_adds_learned_rotation_without_changing_old_modes(self):
        text = EXPORTER.read_text(encoding="utf-8")
        self.assertIn('choices=("unrotated", "rotated", "spinquant")', text)
        self.assertIn("load_rotation_artifact", text)
        self.assertIn("apply_spinquant_llama_offline", text)
        self.assertIn("SpinQuant W4A16 export requires --rotation-manifest", text)
        self.assertIn("validate_checkpoint_quantization_config", text)
        self.assertIn(
            '"checkpoint_tree_sha256": checkpoint_tree_sha256(output_dir)',
            text,
        )

    def test_checkpoint_tree_hash_binds_paths_and_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "config.json").write_text("one", encoding="utf-8")
            shard = root / "weights"
            shard.mkdir()
            (shard / "model.safetensors").write_bytes(b"two")
            first = checkpoint_tree_sha256(root)
            self.assertEqual(first, checkpoint_tree_sha256(root))
            (shard / "model.safetensors").write_bytes(b"three")
            self.assertNotEqual(first, checkpoint_tree_sha256(root))

    def test_isambard_gate_is_isolated_and_result_gated(self):
        text = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=02:00:00", text)
        self.assertIn("VLLM_SPINQUANT_W4A16_ARTIFACT_ROOT", text)
        self.assertIn("SPINQUANT_W4A16_ROTATION_MANIFEST", text)
        self.assertIn("SPINQUANT_W4A16_TRAINING_RESULT", text)
        self.assertIn("--mode spinquant", text)
        self.assertIn("--spinquant", text)
        self.assertIn("VLLM_SPINQUANT_W4A16_REFUSES_EXISTING_OUTPUT", text)
        self.assertIn(
            "ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_GATE_PASSED",
            text,
        )
        self.assertNotIn("sbatch ", text)


if __name__ == "__main__":
    unittest.main()
