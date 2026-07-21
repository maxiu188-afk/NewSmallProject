import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class W4A8BenchmarkTests(unittest.TestCase):
    def test_checked_configs_define_matched_bf16_and_w4a8_protocols(self):
        from repro.w4a8_benchmark import validate_benchmark_config

        for name in ("llama2_13b_w4a8_benchmark_smoke.json", "llama2_13b_w4a8_benchmark.json"):
            config = json.loads((PROJECT_ROOT / "configs" / "deployment" / name).read_text(encoding="utf-8"))
            validate_benchmark_config(config)
            self.assertEqual(config["modes"], ["bf16", "w4a8"])
            self.assertEqual(len(config["linear"]["targets"]), 3)
            self.assertEqual(config["generation"]["new_tokens"], 4 if "smoke" in name else 32)

    def test_sample_summary_keeps_raw_values_and_uses_cuda_time_for_throughput(self):
        from repro.w4a8_benchmark import summarize_samples

        samples = [
            {"cuda_ms": 2.0, "wall_ms": 2.4, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200},
            {"cuda_ms": 4.0, "wall_ms": 4.5, "peak_allocated_bytes": 120, "peak_reserved_bytes": 240},
        ]
        result = summarize_samples(samples, work_items=6)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["samples"], samples)
        self.assertAlmostEqual(result["cuda_ms"]["mean"], 3.0)
        self.assertAlmostEqual(result["work_items_per_second"], 2000.0)
        self.assertEqual(result["peak_allocated_bytes"], 120)
        self.assertEqual(result["peak_reserved_bytes"], 240)

    def test_mode_comparison_reports_speedup_and_memory_ratio(self):
        from repro.w4a8_benchmark import compare_modes

        baseline_case = {"cuda_ms": {"mean": 4.0}, "peak_allocated_bytes": 200}
        candidate_case = {"cuda_ms": {"mean": 2.0}, "peak_allocated_bytes": 100}
        baseline = {section: {"case": baseline_case} for section in ("linear", "prefill", "decode", "generation")}
        candidate = {section: {"case": candidate_case} for section in ("linear", "prefill", "decode", "generation")}
        result = compare_modes(baseline, candidate)
        self.assertEqual(result["linear"]["case"]["w4a8_speedup_over_bf16"], 2.0)
        self.assertEqual(result["linear"]["case"]["w4a8_peak_allocated_ratio_to_bf16"], 0.5)

    def test_rejects_unmatched_mode_order(self):
        from repro.w4a8_benchmark import W4A8BenchmarkError, validate_benchmark_config

        config = json.loads(
            (PROJECT_ROOT / "configs/deployment/llama2_13b_w4a8_benchmark_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        config["modes"] = ["w4a8"]
        with self.assertRaises(W4A8BenchmarkError):
            validate_benchmark_config(config)


if __name__ == "__main__":
    unittest.main()
