import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4a16_llama2_13b_serving_isambard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_vllm_w4a16_llama2_13b_serving",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_serving.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VllmW4A16LlamaServingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_protocol_is_valid_and_uses_fixed_kv_memory(self):
        MODULE._validate_config(self.config)
        self.assertEqual(self.config["server"]["kv_cache_memory_bytes"], 8 * 1024**3)
        self.assertFalse(self.config["server"]["enforce_eager"])
        self.assertEqual(
            [
                (case["name"], case["max_concurrency"])
                for case in self.config["benchmark"]["cases"]
            ],
            [("latency_c1", 1), ("throughput_c8", 8)],
        )
        self.assertEqual(self.config["benchmark"]["num_prompts"], 64)
        self.assertEqual(self.config["benchmark"]["num_warmups"], 4)

    def test_server_command_keeps_all_models_on_identical_flags(self):
        command = MODULE._server_command(
            Path("/venv/bin/vllm"),
            Path("/models/example"),
            "llama2-13b-example",
            self.config,
        )
        self.assertEqual(command[:3], ["/venv/bin/vllm", "serve", "/models/example"])
        self.assertIn("--kv-cache-memory-bytes", command)
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1],
            str(8 * 1024**3),
        )
        self.assertIn("--disable-log-stats", command)
        self.assertNotIn("--enforce-eager", command)

    def test_benchmark_command_freezes_workload_and_provenance(self):
        case = self.config["benchmark"]["cases"][1]
        with mock.patch.object(MODULE, "_revision", return_value="revision"):
            command = MODULE._benchmark_command(
                vllm_executable=Path("/venv/bin/vllm"),
                tokenizer=Path("/models/tokenizer"),
                served_name="llama2-13b-bf16",
                base_url="http://127.0.0.1:18000",
                config=self.config,
                case=case,
                raw_result_dir=Path("/results/raw"),
                raw_result_name="result.json",
                model_name="bf16",
            )
        self.assertEqual(command[:3], ["/venv/bin/vllm", "bench", "serve"])
        self.assertEqual(command[command.index("--input-len") + 1], "256")
        self.assertEqual(command[command.index("--output-len") + 1], "64")
        self.assertEqual(command[command.index("--num-prompts") + 1], "64")
        self.assertEqual(command[command.index("--max-concurrency") + 1], "8")
        self.assertIn("project_revision=revision", command)
        self.assertIn("model_variant=bf16", command)
        self.assertIn("case=throughput_c8", command)

    def test_raw_benchmark_requires_complete_finite_metrics(self):
        raw = {
            "completed": 64,
            "failed": 0,
            "request_throughput": 2.0,
            "output_throughput": 120.0,
            "total_token_throughput": 600.0,
            "mean_ttft_ms": 10.0,
            "median_ttft_ms": 9.0,
            "mean_tpot_ms": 5.0,
            "median_tpot_ms": 4.0,
            "mean_e2el_ms": 330.0,
            "median_e2el_ms": 320.0,
        }
        for metric in ("ttft", "tpot", "e2el"):
            for percentile in (50, 90, 99):
                raw[f"p{percentile}_{metric}_ms"] = 10.0
        metrics = MODULE._validate_raw_benchmark(raw, self.config["benchmark"])
        self.assertEqual(metrics["request_throughput"], 2.0)
        raw["failed"] = 1
        with self.assertRaises(RuntimeError):
            MODULE._validate_raw_benchmark(raw, self.config["benchmark"])

    def test_model_parser_requires_the_frozen_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            values = []
            for name in MODULE.EXPECTED_MODELS:
                model = root / name
                model.mkdir()
                (model / "config.json").write_text("{}", encoding="utf-8")
                values.append(f"{name}={model}")
            parsed = MODULE._parse_models(values)
            self.assertEqual(tuple(parsed), MODULE.EXPECTED_MODELS)
            with self.assertRaises(ValueError):
                MODULE._parse_models(list(reversed(values)))

    def test_memory_sampler_selects_the_slurm_visible_gpu(self):
        completed = mock.Mock()
        completed.stdout = (
            "0, GPU-zero, 10\n"
            "1, GPU-one, 20\n"
        )
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-one"}),
            mock.patch.object(MODULE.subprocess, "run", return_value=completed),
        ):
            self.assertEqual(MODULE._gpu_memory_used_mib(), 20)


if __name__ == "__main__":
    unittest.main()
