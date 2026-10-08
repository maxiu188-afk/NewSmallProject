import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    PROJECT_ROOT
    / "configs/deployment"
    / "sglang_quarot_w4a16_disable_overlap_serving_formal_isambard.json"
)
BASE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/deployment/sglang_vllm_quarot_w4a16_serving_smoke_isambard.json"
)
SBATCH_PATH = (
    PROJECT_ROOT
    / "scripts/run_isambard_sglang_quarot_w4a16_disable_overlap_serving.sbatch"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module(
    "validate_sglang_quarot_w4a16_disable_overlap_inputs_test",
    PROJECT_ROOT / "scripts/validate_sglang_quarot_w4a16_disable_overlap_inputs.py",
)
RUNNER = _load_module(
    "run_sglang_quarot_w4a16_disable_overlap_serving_test",
    PROJECT_ROOT / "scripts/run_sglang_quarot_w4a16_disable_overlap_serving.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(scale: float = 1.0) -> dict:
    metrics = {
        "completed": 64,
        "failed": 0,
        "input_tokens": 64 * 256,
        "output_tokens": 64 * 64,
        "request_throughput_per_second": 2.0 * scale,
        "input_token_throughput_per_second": 512.0 * scale,
        "output_token_throughput_per_second": 128.0 * scale,
        "total_token_throughput_per_second": 640.0 * scale,
    }
    for name, base in ("ttft", 20.0), ("tpot", 5.0), ("e2e", 350.0):
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
        "metrics": metrics,
        "startup_seconds": 10.0 * scale,
        "ready_gpu_memory_used_mib": 16000.0 * scale,
        "peak_gpu_memory_used_mib": 17000.0 * scale,
        "released_gpu_memory_used_mib": 4.0 * scale,
    }


class SglangQuarotW4A16DisableOverlapServingTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))

    def _effective(self) -> dict:
        effective = json.loads(json.dumps(self.base))
        for name in (
            "status",
            "reference_formal_job",
            "experiment",
            "benchmark",
            "scope",
        ):
            effective[name] = self.spec[name]
        return effective

    def test_spec_is_one_protocol_matched_sglang_third_arm(self):
        VALIDATOR._validate_spec(self.spec)
        self.assertEqual(self.spec["experiment"]["backend"], "sglang")
        self.assertEqual(
            self.spec["experiment"]["server_argument"],
            "--disable-overlap-schedule",
        )
        self.assertEqual(self.spec["reference_formal_job"]["job_id"], "5960180")
        self.assertEqual(self.spec["benchmark"]["repetitions"], 3)
        self.assertEqual(self.spec["benchmark"]["cases"], VALIDATOR.EXPECTED_CASES)

    def test_base_config_is_hash_bound_and_runtime_fields_are_unchanged(self):
        self.assertEqual(_sha256(BASE_CONFIG_PATH), self.spec["base_config_sha256"])
        VALIDATOR._validate_base_config(self.base)
        effective = self._effective()
        RUNNER._validate_config(effective)
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

    def test_server_command_adds_only_disable_overlap_once(self):
        config = self._effective()
        kwargs = {
            "backend": "sglang",
            "executable": Path("/sglang/bin/python"),
            "model_path": Path("/models/rotated-w4a16"),
            "served_name": "/models/rotated-w4a16",
            "config": config,
        }
        baseline = RUNNER._ORIGINAL_SERVER_COMMAND(**kwargs)
        variant = RUNNER._server_command_with_disabled_overlap(**kwargs)
        self.assertEqual(variant[:-1], baseline)
        self.assertEqual(variant[-1], "--disable-overlap-schedule")
        self.assertEqual(variant.count("--disable-overlap-schedule"), 1)

    def test_command_patch_is_scoped_and_restored(self):
        original_smoke = RUNNER.smoke_runtime._server_command
        original_serving = RUNNER.serving._server_command
        with RUNNER._disabled_overlap_command_scope():
            self.assertIs(
                RUNNER.smoke_runtime._server_command,
                RUNNER._server_command_with_disabled_overlap,
            )
            self.assertIs(
                RUNNER.serving._server_command,
                RUNNER._server_command_with_disabled_overlap,
            )
        self.assertIs(RUNNER.smoke_runtime._server_command, original_smoke)
        self.assertIs(RUNNER.serving._server_command, original_serving)

    def test_aggregate_retains_three_repetitions_without_paired_claim(self):
        repetitions = []
        for index, scale in enumerate((1.0, 2.0, 3.0), start=1):
            repetitions.append(
                {
                    "repetition": index,
                    "cases": {
                        "latency_c1": _case(scale),
                        "throughput_c8": _case(scale * 2.0),
                    },
                }
            )
        aggregate = RUNNER._aggregate(repetitions)
        self.assertEqual(aggregate["status"], "third_arm_complete")
        self.assertEqual(
            aggregate["pairing_boundary"],
            "not_paired_with_reference_job_5960180",
        )
        throughput = aggregate["cases"]["latency_c1"]["metrics"][
            "request_throughput_per_second"
        ]
        self.assertEqual(throughput["values"], [2.0, 4.0, 6.0])
        self.assertEqual(throughput["median"], 4.0)

    def test_reference_validator_requires_all_twelve_accepted_cells(self):
        benchmark = json.loads(json.dumps(self.spec["benchmark"]))
        benchmark["backend_orders"] = [
            ["vllm", "sglang"],
            ["sglang", "vllm"],
            ["vllm", "sglang"],
        ]
        repetitions = []
        for index in range(1, 4):
            repetitions.append(
                {
                    "repetition": index,
                    "cases": {
                        name: _case()
                        for name in VALIDATOR.EXPECTED_REFERENCE_CASES
                    },
                }
            )
        reference = self.spec["reference_formal_job"]
        result = {
            "status": "passed",
            "comparison_status": "formal_matrix_complete",
            "project_revision": reference["project_revision"],
            "config_sha256": reference["effective_config_sha256"],
            "model": "quarot_w4a16",
            "request_corpus": {"sha256": reference["request_corpus_sha256"]},
            "benchmark": benchmark,
            "repetitions": repetitions,
        }
        VALIDATOR._validate_reference_result(result, reference)
        repetitions[0]["cases"]["sglang:latency_c1"]["metrics"]["failed"] = 1
        with self.assertRaisesRegex(RuntimeError, "request counts drifted"):
            VALIDATOR._validate_reference_result(result, reference)

    def test_historical_sources_are_rehashed_from_recorded_git_revision(self):
        reference = self.spec["reference_formal_job"]
        relative_path = Path(
            "configs/deployment/"
            "sglang_vllm_quarot_w4a16_serving_formal_isambard.json"
        )
        historical_path = PROJECT_ROOT / relative_path
        self.assertFalse(historical_path.exists())
        missing_expected = (
            "35e6d4e3fc9861ccf3782ec60ff7ee019faa9361099a98a4b908dfb4a114e49e"
        )
        changed_path = BASE_CONFIG_PATH
        changed_expected = (
            "b87d2debd15b320c0a4b2f65096a25dc2f145b6af5bb4b29898178fbca641eee"
        )
        self.assertNotEqual(_sha256(changed_path), changed_expected)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "source-manifest.txt"
            manifest_path.write_text(
                "\n".join(
                    (
                        f"git_revision={reference['project_revision']}",
                        "git_status=clean",
                        f"{missing_expected}  {historical_path}",
                        f"{changed_expected}  {changed_path}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            entries, git_blob_entries = VALIDATOR._validate_manifest(
                manifest_path,
                project_root=PROJECT_ROOT,
                expected_revision=reference["project_revision"],
                minimum_entries=2,
            )
        self.assertEqual(
            entries[str(historical_path.resolve())], missing_expected
        )
        self.assertEqual(entries[str(changed_path.resolve())], changed_expected)
        self.assertEqual(git_blob_entries, 2)

    def test_missing_external_manifest_artifact_is_not_replaced_by_git(self):
        reference = self.spec["reference_formal_job"]
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "source-manifest.txt"
            missing = Path(directory) / "missing-result.json"
            manifest_path.write_text(
                "\n".join(
                    (
                        f"git_revision={reference['project_revision']}",
                        "git_status=clean",
                        f"{'0' * 64}  {missing}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "artifact is missing"):
                VALIDATOR._validate_manifest(
                    manifest_path,
                    project_root=PROJECT_ROOT,
                    expected_revision=reference["project_revision"],
                    minimum_entries=1,
                )

    def test_batch_runs_only_the_new_sglang_configuration(self):
        text = SBATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("--disable-overlap-schedule", text)
        self.assertIn("reference_formal_job=5960180", text)
        self.assertIn("formal_repetitions=3", text)
        self.assertIn("formal_concurrency=1,8", text)
        self.assertIn("--sglang-python", text)
        self.assertNotIn("--vllm-executable", text)
        self.assertNotIn("afterok", text)
        self.assertNotIn("sbatch ", text.replace("#   sbatch ", ""))


if __name__ == "__main__":
    unittest.main()
