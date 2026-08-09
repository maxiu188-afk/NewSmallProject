import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_SPEC_PATH = (
    PROJECT_ROOT
    / "configs/deployment/sglang_vllm_quarot_w4a16_serving_formal_isambard.json"
)
BASE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module(
    "validate_sglang_vllm_quarot_w4a16_serving_formal_inputs_test",
    PROJECT_ROOT
    / "scripts/validate_sglang_vllm_quarot_w4a16_serving_formal_inputs.py",
)
RUNNER = _load_module(
    "run_sglang_vllm_quarot_w4a16_serving_formal_test",
    PROJECT_ROOT / "scripts/run_sglang_vllm_quarot_w4a16_serving_formal.py",
)


def _case(backend: str, case_name: str, scale: float) -> dict:
    metrics = {
        "completed": 64,
        "failed": 0,
        "request_throughput_per_second": 2.0 * scale,
        "input_token_throughput_per_second": 512.0 * scale,
        "output_token_throughput_per_second": 128.0 * scale,
        "total_token_throughput_per_second": 640.0 * scale,
    }
    for name, base in (("ttft", 20.0), ("tpot", 5.0), ("e2e", 350.0)):
        metrics[name] = {
            "p50_ms": base * scale,
            "p95_ms": base * 1.1 * scale,
            "p99_ms": base * 1.2 * scale,
        }
    metrics["stream_event_itl"] = {
        "p50_ms": 5.0 * scale,
        "p95_ms": 6.0 * scale,
        "p99_ms": 7.0 * scale,
    }
    return {
        "status": "passed",
        "backend": backend,
        "case": {"name": case_name},
        "metrics": metrics,
        "startup_seconds": 10.0 * scale,
        "ready_gpu_memory_used_mib": 16000.0 * scale,
        "peak_gpu_memory_used_mib": 17000.0 * scale,
        "released_gpu_memory_used_mib": 100.0 * scale,
    }


class SglangVllmQuarotW4A16ServingFormalTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(FORMAL_SPEC_PATH.read_text(encoding="utf-8"))
        self.base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_protocol_is_the_predeclared_formal_matrix(self):
        VALIDATOR._validate_protocol(self.spec)
        benchmark = self.spec["benchmark"]
        self.assertEqual(benchmark["measured_requests"], 64)
        self.assertEqual(benchmark["repetitions"], 3)
        self.assertEqual(benchmark["cases"], VALIDATOR.EXPECTED_CASES)
        self.assertEqual(
            benchmark["backend_orders"], VALIDATOR.EXPECTED_BACKEND_ORDERS
        )
        self.assertEqual(self.spec["serving_smoke_gate"]["job_id"], "5952554")

    def test_effective_config_keeps_all_frozen_smoke_sections(self):
        effective = dict(self.base)
        effective["status"] = self.spec["status"]
        effective["serving_smoke_gate"] = self.spec["serving_smoke_gate"]
        effective["benchmark"] = self.spec["benchmark"]
        effective["scope"] = self.spec["scope"]
        RUNNER._validate_formal_config(effective)
        for name in (
            "source_gate",
            "compatibility_gate",
            "model",
            "backends",
            "server",
            "request_corpus",
            "vllm",
            "sglang",
            "runtime",
        ):
            self.assertEqual(effective[name], self.base[name])

    def test_aggregate_keeps_every_repetition_and_reports_spread(self):
        repetitions = []
        for index, factor in enumerate((1.0, 2.0, 3.0), start=1):
            cases = {}
            for case in RUNNER.EXPECTED_CASES:
                name = case["name"]
                cases[f"vllm:{name}"] = _case("vllm", name, factor)
                cases[f"sglang:{name}"] = _case("sglang", name, factor * 2.0)
            repetitions.append(
                {
                    "repetition": index,
                    "cases": cases,
                    "comparison": RUNNER._paired_comparison(cases),
                }
            )
        aggregate = RUNNER._aggregate(repetitions)
        self.assertEqual(aggregate["status"], "formal_matrix_complete")
        latency = aggregate["cases"]["latency_c1"]
        throughput = latency["backends"]["vllm"]["metrics"][
            "request_throughput_per_second"
        ]
        self.assertEqual(throughput["values"], [2.0, 4.0, 6.0])
        self.assertEqual(throughput["median"], 4.0)
        self.assertEqual(throughput["range"], 4.0)
        ratio = latency["ratios_sglang_over_vllm"][
            "request_throughput_per_second"
        ]
        self.assertEqual(ratio["values"], [2.0, 2.0, 2.0])
        self.assertEqual(ratio["median"], 2.0)

    def test_formal_batch_is_result_gated_without_resubmitting_smoke(self):
        text = (
            PROJECT_ROOT
            / "scripts/run_isambard_sglang_vllm_quarot_w4a16_serving_formal.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("serving-client-smoke-5952554.json", text)
        self.assertIn("accepted_smoke_gate", text)
        self.assertIn("--time=06:00:00", text)
        self.assertNotIn("afterok", text)
        self.assertNotIn("sbatch ", text.replace("#   sbatch ", ""))
        self.assertIn("formal_repetitions=3", text)
        self.assertIn("formal_concurrency=1,8", text)

    def test_smoke_producers_are_unchanged_from_accepted_revision(self):
        producer_paths = list(
            self.spec["serving_smoke_gate"]["producer_sha256"]
        )
        changed = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "diff",
                "--name-only",
                "752a01d950e3886989ad4541f901c9a31d6ba195..HEAD",
                "--",
                *producer_paths,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(changed, "")


if __name__ == "__main__":
    unittest.main()
