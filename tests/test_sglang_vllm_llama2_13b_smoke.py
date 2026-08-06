import importlib.util
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "sglang_vllm_llama2_13b_smoke_isambard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_sglang_vllm_llama2_13b_smoke",
    PROJECT_ROOT / "scripts" / "run_sglang_vllm_llama2_13b_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SglangVllmLlamaSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_freezes_exact_checkpoint_smoke(self):
        MODULE._validate_config(self.config)
        self.assertEqual(self.config["source_gate"]["job_id"], "5881273")
        self.assertEqual(self.config["server"]["kv_cache_memory_bytes"], 8 * 1024**3)
        self.assertEqual(self.config["server"]["max_total_tokens"], 10485)
        self.assertIsNone(self.config["sglang"]["offline_quantization_argument"])
        self.assertFalse(self.config["sglang"]["enable_jit_deep_gemm"])
        self.assertEqual(
            self.config["sglang"]["cuda_home_source"],
            "sglang_environment_nvidia_cu13",
        )

    def test_vllm_command_preserves_accepted_resource_flags(self):
        command = MODULE._server_command(
            backend="vllm",
            executable=Path("/vllm/bin/vllm"),
            model_path=Path("/models/bf16"),
            served_name="bf16",
            config=self.config,
        )
        self.assertEqual(command[:3], ["/vllm/bin/vllm", "serve", "/models/bf16"])
        self.assertEqual(command[command.index("--kv-cache-memory-bytes") + 1], str(8 * 1024**3))
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "8")

    def test_sglang_command_uses_derived_token_pool_without_requantizing(self):
        command = MODULE._server_command(
            backend="sglang",
            executable=Path("/sglang/bin/python"),
            model_path=Path("/models/w4afp8"),
            served_name="w4afp8",
            config=self.config,
        )
        self.assertEqual(command[:3], ["/sglang/bin/python", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--max-total-tokens") + 1], "10485")
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "bfloat16")
        self.assertIn("--disable-radix-cache", command)
        self.assertEqual(command[command.index("--chunked-prefill-size") + 1], "-1")
        self.assertNotIn("--quantization", command)

    def test_virtualenv_python_symlink_is_not_resolved(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            base_python = root / "base-python"
            base_python.write_text("", encoding="utf-8")
            environment_python = root / "environment" / "bin" / "python"
            environment_python.parent.mkdir(parents=True)
            environment_python.symlink_to(base_python)
            self.assertEqual(
                MODULE._absolute_executable(environment_python),
                environment_python,
            )
            self.assertNotEqual(environment_python.resolve(), environment_python)

    def test_compatibility_status_distinguishes_bf16_only(self):
        cases = {
            "vllm:bf16": {"status": "passed"},
            "sglang:bf16": {"status": "passed"},
            "vllm:fp8_targeted_w4afp8": {"status": "passed"},
            "sglang:fp8_targeted_w4afp8": {"status": "failed"},
        }
        self.assertEqual(
            MODULE._compatibility_status(cases),
            "bf16_passed_w4afp8_unresolved",
        )

    def test_compatibility_status_gates_cross_backend_text_mismatch(self):
        cases = {
            f"{backend}:{model}": {"status": "passed"}
            for backend in ("vllm", "sglang")
            for model in ("bf16", "fp8_targeted_w4afp8")
        }
        self.assertEqual(
            MODULE._compatibility_status(
                cases,
                {"bf16": True, "fp8_targeted_w4afp8": False},
                require_equal_text=True,
            ),
            "checkpoint_generation_mismatch",
        )

    def test_log_excerpt_is_case_insensitive(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("start\nUsing W4AFP8 kernel\nend\n", encoding="utf-8")
            self.assertEqual(
                MODULE._log_excerpt(path, ["w4afp8"]),
                ["Using W4AFP8 kernel"],
            )


if __name__ == "__main__":
    unittest.main()
