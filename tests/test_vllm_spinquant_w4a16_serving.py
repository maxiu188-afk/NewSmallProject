import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_spinquant_w4a16_llama2_13b_serving_isambard.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVING = _load_module(
    "run_vllm_w4a16_llama2_13b_serving_for_spinquant_test",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_serving.py",
)
WRAPPER = _load_module(
    "run_vllm_spinquant_w4a16_llama2_13b_serving_for_test",
    PROJECT_ROOT / "scripts" / "run_vllm_spinquant_w4a16_llama2_13b_serving.py",
)


class VllmSpinQuantW4A16ServingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_reuses_the_formal_matched_w4a16_protocol(self):
        SERVING._validate_config(self.config)
        self.assertEqual(
            SERVING._expected_models(self.config),
            ("bf16", "spinquant_w4a16"),
        )
        self.assertEqual(self.config["execution"]["kind"], "formal_only")
        self.assertEqual(self.config["benchmark"]["num_prompts"], 64)
        self.assertEqual(self.config["benchmark"]["num_warmups"], 4)
        self.assertEqual(
            [case["max_concurrency"] for case in self.config["benchmark"]["cases"]],
            [1, 8],
        )
        self.assertEqual(self.config["server"]["kv_cache_memory_bytes"], 8 * 1024**3)

    def test_config_requires_machete_for_only_the_spinquant_checkpoint(self):
        self.assertEqual(
            self.config["kernel_gate"]["quantized_models"],
            ["spinquant_w4a16"],
        )
        self.assertIn(
            "MacheteLinearKernel",
            self.config["kernel_gate"]["required_quantized_log_pattern"],
        )

    def test_wrapper_requires_the_accepted_formal_only_source_gate(self):
        WRAPPER._require_accepted_source_gate(["--config", str(CONFIG_PATH)])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            changed = json.loads(json.dumps(self.config))
            changed["source_gate"]["status"] = "pending"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                WRAPPER._require_accepted_source_gate(["--config", str(path)])
            changed["source_gate"]["status"] = "accepted"
            changed["execution"]["kind"] = "smoke_then_formal"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                WRAPPER._require_accepted_source_gate(["--config", str(path)])

    def test_batch_is_formal_only_and_has_no_smoke_dependency(self):
        batch = (
            PROJECT_ROOT
            / "scripts"
            / "run_isambard_vllm_spinquant_w4a16_llama2_13b_serving.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ "${mode}" != "formal" ]]', batch)
        self.assertIn("execution_kind=formal_only", batch)
        self.assertNotIn("--dependency", batch)
        self.assertNotIn("afterok:", batch)
        self.assertNotIn("--mode smoke", batch)
        self.assertIn("--mode benchmark", batch)
        self.assertIn(
            "ISAMBARD_VLLM_SPINQUANT_W4A16_LLAMA2_13B_SERVING_FORMAL_PASSED",
            batch,
        )


if __name__ == "__main__":
    unittest.main()
