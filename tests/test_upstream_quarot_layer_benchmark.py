import copy
import inspect
import unittest
from pathlib import Path

from repro.upstream_quarot_layer_benchmark import (
    OfficialLayerBenchmarkError,
    compare_case,
    load_plan,
    select_groups,
    summarize_samples,
    validate_plan,
)
from scripts.run_upstream_quarot_layer_benchmark import _build_cache, _completion_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "configs/deployment/quarot_official_single_block_rtx6000ada.json"


class OfficialQuaRotLayerBenchmarkTests(unittest.TestCase):
    def test_cache_builder_accepts_benchmark_positional_shape_arguments(self):
        parameters = list(inspect.signature(_build_cache).parameters.values())
        self.assertEqual(parameters[2].name, "batch_size")
        self.assertEqual(parameters[2].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters[3].name, "length")
        self.assertEqual(parameters[3].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_completion_status_retains_oom_as_a_completed_measurement(self):
        self.assertEqual(
            _completion_status(
                {
                    "case": {
                        "modes": {"w4a4kv4": {}},
                        "oom_by_mode": {"fp16": {"metric": "prefill"}},
                    }
                },
                ["w4a4kv4", "fp16"],
            ),
            ("completed_with_oom", 1),
        )
        self.assertEqual(
            _completion_status(
                {"case": {"modes": {"w4a4kv4": {}, "fp16": {}}}},
                ["w4a4kv4", "fp16"],
            ),
            ("passed", 0),
        )
        with self.assertRaises(OfficialLayerBenchmarkError):
            _completion_status(
                {"case": {"modes": {"w4a4kv4": {}}}},
                ["w4a4kv4", "fp16"],
            )

    def test_plan_freezes_paper_prefill_and_decode_memory_grids(self):
        plan = load_plan(PLAN_PATH)
        groups = {group["name"]: group for group in plan["groups"]}
        self.assertEqual(
            [(case["batch_size"], case["prefill_tokens"]) for case in groups["paper_prefill"]["cases"]],
            [(1, 2048), (4, 2048), (16, 2048), (64, 2048)],
        )
        self.assertEqual(
            [
                (case["batch_size"], case["prefill_tokens"], case["decode_tokens"])
                for case in groups["paper_decode_memory"]["cases"]
            ],
            [(batch, length, 50) for batch in (1, 16) for length in (256, 512, 1024, 2048, 4096)],
        )
        self.assertEqual(plan["protocols"]["official"], {"warmup_steps": 3, "bench_steps": 10, "repetitions": 10})

    def test_group_selection_preserves_requested_order_and_rejects_unknown(self):
        plan = load_plan(PLAN_PATH)
        selected = select_groups(plan, ["paper_decode_memory", "paper_prefill"])
        self.assertEqual([group["name"] for group in selected], ["paper_decode_memory", "paper_prefill"])
        with self.assertRaises(OfficialLayerBenchmarkError):
            select_groups(plan, ["unknown"])

    def test_plan_rejects_scope_drift(self):
        plan = load_plan(PLAN_PATH)
        changed = copy.deepcopy(plan)
        changed["groups"][1]["cases"][-1]["batch_size"] = 32
        with self.assertRaises(OfficialLayerBenchmarkError):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["source"]["required_compatibility_patches"].append(
            "patches/quarot-gptq-v-proj.patch"
        )
        with self.assertRaises(OfficialLayerBenchmarkError):
            validate_plan(changed)

    def test_sample_summary_retains_raw_values(self):
        summary = summarize_samples([2.0, 4.0], [100, 120], work_items=8)
        self.assertEqual(summary["elapsed_ms_samples"], [2.0, 4.0])
        self.assertEqual(summary["peak_allocated_bytes_samples"], [100, 120])
        self.assertEqual(summary["elapsed_ms"]["mean"], 3.0)
        self.assertEqual(summary["peak_allocated_bytes"]["maximum"], 120)
        self.assertAlmostEqual(summary["work_items_per_second_from_mean"], 8000.0 / 3.0)

    def test_mode_comparison_uses_official_mean_and_retains_median(self):
        fp16 = {
            "prefill": {
                "elapsed_ms": {"mean": 4.0, "median": 3.0},
                "peak_allocated_bytes": {"mean": 200.0},
            }
        }
        w4 = {
            "prefill": {
                "elapsed_ms": {"mean": 2.0, "median": 2.0},
                "peak_allocated_bytes": {"mean": 50.0},
            }
        }
        comparison = compare_case(fp16, w4)["prefill"]
        self.assertEqual(comparison["speedup_fp16_over_w4a4kv4_mean"], 2.0)
        self.assertEqual(comparison["speedup_fp16_over_w4a4kv4_median"], 1.5)
        self.assertEqual(comparison["memory_saving_fp16_over_w4a4kv4_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
