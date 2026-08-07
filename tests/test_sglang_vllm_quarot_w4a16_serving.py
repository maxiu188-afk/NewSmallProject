import importlib.util
import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_sglang_vllm_quarot_w4a16_serving_test",
    PROJECT_ROOT / "scripts/run_sglang_vllm_quarot_w4a16_serving.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SglangVllmQuarotW4A16ServingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_is_a_bounded_result_gated_smoke(self):
        MODULE._validate_config(self.config)
        self.assertEqual(self.config["model"], "quarot_w4a16")
        self.assertEqual(self.config["source_gate"]["job_id"], "5769503")
        self.assertEqual(self.config["compatibility_gate"]["job_id"], "5941763")
        self.assertEqual(self.config["benchmark"]["measured_requests"], 8)
        self.assertEqual(
            self.config["benchmark"]["cases"],
            [{"name": "smoke_c1", "max_concurrency": 1}],
        )
        self.assertIn("not formal", self.config["scope"])

    def test_request_corpus_is_precommitted_by_three_hashes(self):
        corpus = self.config["request_corpus"]
        self.assertEqual(corpus["requests"], 64)
        self.assertEqual(corpus["input_tokens_with_special"], 256)
        self.assertEqual(corpus["output_tokens"], 64)
        self.assertEqual(
            corpus["file_sha256"],
            "1a0d120959122836499220a5b65538e7548c86f30f30224530e8f7385d2b65e1",
        )
        self.assertEqual(len(corpus["prompt_list_sha256"]), 64)
        self.assertEqual(len(corpus["token_id_list_sha256"]), 64)
        self.assertEqual(
            corpus["legacy_vllm_evidence"]["job_id"], "5780631"
        )

    def test_server_commands_align_cache_and_scheduler_controls(self):
        from scripts.run_sglang_vllm_llama2_13b_smoke import _server_command

        vllm = _server_command(
            backend="vllm",
            executable=Path("/vllm/bin/vllm"),
            model_path=Path("/models/rotated-w4a16"),
            served_name="quarot-w4a16",
            config=self.config,
        )
        self.assertIn("--no-enable-prefix-caching", vllm)
        self.assertIn("--no-enable-chunked-prefill", vllm)
        self.assertEqual(
            vllm[vllm.index("--kv-cache-memory-bytes") + 1], str(8 * 1024**3)
        )
        self.assertEqual(
            vllm[vllm.index("--kv-cache-dtype") + 1], "bfloat16"
        )

        sglang = _server_command(
            backend="sglang",
            executable=Path("/sglang/bin/python"),
            model_path=Path("/models/rotated-w4a16"),
            served_name="quarot-w4a16",
            config=self.config,
        )
        self.assertIn("--disable-radix-cache", sglang)
        self.assertEqual(
            sglang[sglang.index("--chunked-prefill-size") + 1], "-1"
        )
        self.assertEqual(
            sglang[sglang.index("--served-model-name") + 1], "quarot-w4a16"
        )
        self.assertNotIn("--quantization", sglang)

    def test_summary_reports_all_required_throughputs_and_percentiles(self):
        measured = []
        for index in range(8):
            measured.append(
                {
                    "status": "passed",
                    "prompt_tokens": 256,
                    "completion_tokens": 64,
                    "ttft_seconds": 0.01 + index * 0.001,
                    "tpot_seconds": 0.005 + index * 0.0001,
                    "e2e_seconds": 0.4 + index * 0.01,
                    "itl_seconds": [0.005, 0.006],
                }
            )
        summary = MODULE._summarize_workload(
            {"measured": measured, "measurement_duration_seconds": 4.0},
            self.config,
        )
        self.assertEqual(summary["completed"], 8)
        self.assertEqual(summary["input_tokens"], 8 * 256)
        self.assertEqual(summary["output_tokens"], 8 * 64)
        self.assertEqual(summary["request_throughput_per_second"], 2.0)
        for name in ("ttft", "tpot", "e2e"):
            self.assertIn("p50_ms", summary[name])
            self.assertIn("p95_ms", summary[name])
            self.assertIn("p99_ms", summary[name])

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(MODULE._percentile([1.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(MODULE._percentile([0.0, 10.0], 95), 9.5)
        with self.assertRaises(ValueError):
            MODULE._percentile([], 50)

    def test_legacy_output_mismatch_is_diagnostic_not_corpus_identity(self):
        measured = [{"generated_text": str(index)} for index in range(8)]
        result = MODULE._legacy_vllm_output_diagnostic(measured, self.config)
        self.assertEqual(result["status"], "mismatched")
        self.assertEqual(
            result["policy"], "diagnostic_only_not_corpus_identity"
        )
        self.assertEqual(result["accepted_job"]["job_id"], "5780631")

    def test_batch_script_submits_no_formal_work(self):
        text = (
            PROJECT_ROOT
            / "scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_smoke.sbatch"
        ).read_text(encoding="utf-8")
        self.assertNotIn("afterok", text)
        self.assertNotIn("sbatch ", text.replace("#   sbatch ", ""))
        self.assertIn("formal_job=not_submitted", text)
        self.assertNotIn(
            'legacy_vllm_corpus_link"]["status"] == "matched"', text
        )
        self.assertIn("module load cuda/12.6", text)
        self.assertIn("module load gcc-native/13.2", text)
        self.assertIn("SGLANG_VLLM_W4A16_SERVING_EXTERNAL_SOURCES_ACCEPTED", text)


if __name__ == "__main__":
    unittest.main()
