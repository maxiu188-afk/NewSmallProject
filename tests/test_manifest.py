import json
import tempfile
import unittest
from pathlib import Path

from repro.manifest import ConfigValidationError, build_run_manifest, load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_local_smoke_config_is_valid(self):
        config = load_config(PROJECT_ROOT / "configs/smoke/algorithm_tiny_llama_fake_quant.json")
        validate_config(config)

    def test_runpod_smoke_config_is_valid(self):
        config = load_config(PROJECT_ROOT / "configs/smoke/runpod_llama2_w4a4.json")
        validate_config(config)

    def test_algorithm_config_cannot_claim_cuda_execution(self):
        config = load_config(PROJECT_ROOT / "configs/smoke/algorithm_tiny_llama_fake_quant.json")
        config["quantization"]["execution_mode"] = "int4_gemm"
        with self.assertRaises(ConfigValidationError):
            validate_config(config)

    def test_manifest_includes_observed_upstream_revision(self):
        config = load_config(PROJECT_ROOT / "configs/smoke/algorithm_tiny_llama_fake_quant.json")
        manifest = build_run_manifest(config, "python3 -m unittest")
        self.assertEqual(manifest["config"]["experiment_id"], config["experiment_id"])
        self.assertEqual(manifest["host"]["upstream_revision_observed"], config["provenance"]["upstream_commit"])


if __name__ == "__main__":
    unittest.main()
