import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/deployment/vllm_w4afp8_llama2_13b_mse_diagnostic_isambard.json"
)
SBATCH = (
    PROJECT_ROOT
    / "scripts/run_isambard_vllm_w4afp8_llama2_13b_mse_diagnostic.sbatch"
)

from repro.vllm_w4afp8_mse_diagnostic import (
    PPL_MODELS,
    validate_config,
    validate_minmax_checkpoint_quantization_config,
    validate_mse_checkpoint_quantization_config,
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PPL = _load_script(
    "run_vllm_w4afp8_llama2_13b_mse_ppl_test",
    PROJECT_ROOT / "scripts/run_vllm_w4afp8_llama2_13b_mse_ppl.py",
)


def _metadata(observer: str) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                    "actorder": None,
                    "observer": observer,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }


class VllmW4AFP8MseDiagnosticTests(unittest.TestCase):
    def test_config_freezes_one_variable_observer_diagnostic(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config)
        self.assertEqual(config["quantization"]["weight_observer"], "mse")
        self.assertTrue(config["quantization"]["weight_clipping"])
        self.assertIsNone(config["quantization"]["weight_actorder"])
        self.assertEqual(
            config["quantization"]["weight_observer_kwargs"],
            {"maxshrink": 0.8, "grid": 100.0, "norm": 2.4, "patience": 100},
        )

    def test_checkpoint_observer_boundaries(self):
        self.assertEqual(
            validate_mse_checkpoint_quantization_config(_metadata("memoryless_mse")),
            "memoryless_mse",
        )
        self.assertEqual(
            validate_minmax_checkpoint_quantization_config(
                _metadata("memoryless_minmax")
            ),
            "memoryless_minmax",
        )
        with self.assertRaisesRegex(ValueError, "unexpected weight observer"):
            validate_mse_checkpoint_quantization_config(
                _metadata("memoryless_minmax")
            )

    def test_smoke_and_formal_require_afterok_dependencies(self):
        script = SBATCH.read_text(encoding="utf-8")
        self.assertIn("VLLM_W4AFP8_MSE_SMOKE_REQUIRES_AFTEROK", script)
        self.assertIn("VLLM_W4AFP8_MSE_FORMAL_REQUIRES_AFTEROK", script)
        self.assertIn("PPL_SMOKE_ACCEPTED", script)

    def test_parent_runs_five_fresh_workers(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        runtime = {
            "vllm": "0.25.1+cu129",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = {name: root / name for name in PPL_MODELS}

            def fake_worker(**kwargs):
                return {
                    "name": kwargs["name"],
                    "tokens": 2,
                    "mean_nll": 1.0,
                    "perplexity": math.e,
                    "runtime": runtime,
                }

            with (
                mock.patch.object(
                    PPL,
                    "_load_sequences",
                    return_value=([[1, 2, 3]], {"dataset": {"token_ids_sha256": "x"}}),
                ),
                mock.patch.object(PPL, "_run_worker", side_effect=fake_worker) as worker,
                mock.patch.object(PPL, "_revision", return_value="revision"),
                mock.patch.object(PPL, "_sha256", return_value="digest"),
            ):
                result = PPL.run(
                    config=config,
                    config_path=CONFIG,
                    manifest_path=root / "manifest.json",
                    models=models,
                    max_sequences=1,
                    worker_output_dir=root / "workers",
                )
        self.assertEqual(worker.call_count, 5)
        self.assertEqual(set(result["models"]), set(PPL_MODELS))
        self.assertEqual(result["scored_tokens"], 2)
        self.assertIn("mse_quarot_vs_unrotated", result["comparisons"])


if __name__ == "__main__":
    unittest.main()
