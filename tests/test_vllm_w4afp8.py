import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4afp8_llama2_13b_isambard.json"
)
PPL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4afp8_llama2_13b_ppl_isambard.json"
)
SERVING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4afp8_llama2_13b_serving_isambard.json"
)

from repro.vllm_w4afp8 import (  # noqa: E402
    EXPECTED_VARIANTS,
    QUAROT_VARIANTS,
    SPINQUANT_VARIANTS,
    validate_checkpoint_quantization_config,
    validate_export_config,
    validate_kernel_shapes,
    validate_no_runtime_g_idx,
)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PPL = _load_script(
    "run_vllm_w4afp8_llama2_13b_ppl_test",
    PROJECT_ROOT / "scripts" / "run_vllm_w4afp8_llama2_13b_ppl.py",
)
GATE = _load_script(
    "run_vllm_w4afp8_llama2_13b_gate_test",
    PROJECT_ROOT / "scripts" / "run_vllm_w4afp8_llama2_13b_gate.py",
)
SERVING = _load_script(
    "run_vllm_w4a16_llama2_13b_serving_for_w4afp8_test",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_serving.py",
)


def _valid_quantization_config():
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
                    "actorder": "static",
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


class VllmW4AFP8Tests(unittest.TestCase):
    def test_quarot_gate_accepts_only_bf16_control_and_quarot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            values = []
            for name in QUAROT_VARIANTS:
                model = root / name
                model.mkdir()
                (model / "config.json").write_text("{}", encoding="utf-8")
                values.append(f"{name}={model}")
            parsed = GATE._parse_models(values, QUAROT_VARIANTS)
        self.assertEqual(tuple(parsed), QUAROT_VARIANTS)

    def test_spinquant_gate_accepts_only_bf16_control_and_spinquant(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            values = []
            for name in SPINQUANT_VARIANTS:
                model = root / name
                model.mkdir()
                (model / "config.json").write_text("{}", encoding="utf-8")
                values.append(f"{name}={model}")
            parsed = GATE._parse_models(values, SPINQUANT_VARIANTS)
        self.assertEqual(tuple(parsed), SPINQUANT_VARIANTS)

    def test_isambard_gate_has_isolated_quarot_and_spinquant_modes(self):
        text = (
            PROJECT_ROOT
            / "scripts/run_isambard_vllm_w4afp8_llama2_13b_gate.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('variant_set="${1:-}"', text)
        self.assertIn("{quarot|spinquant|joint}", text)
        self.assertIn("--variant-set", text)
        self.assertIn("ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_QUAROT_GATE_PASSED", text)
        self.assertIn(
            "ISAMBARD_VLLM_W4AFP8_LLAMA2_13B_SPINQUANT_TRANSFER_GATE_PASSED",
            text,
        )

    def test_export_config_freezes_hopper_numerical_contract(self):
        config = json.loads(EXPORT_CONFIG.read_text(encoding="utf-8"))
        validate_export_config(config)
        self.assertEqual(tuple(config["variants"]), EXPECTED_VARIANTS[1:])
        self.assertEqual(config["quantization"]["scheme"], "W4AFP8")
        self.assertEqual(config["quantization"]["weight_actorder"], "static")
        self.assertFalse(config["kernel"]["runtime_g_idx"])

    def test_checkpoint_metadata_requires_fp8_tokens_and_no_group_actorder(self):
        metadata = _valid_quantization_config()
        validate_checkpoint_quantization_config(metadata)
        metadata["config_groups"]["group_0"]["weights"]["actorder"] = "group"
        with self.assertRaisesRegex(ValueError, "runtime g_idx"):
            validate_checkpoint_quantization_config(metadata)

    def test_kernel_shape_and_tensor_gates_reject_incompatible_inputs(self):
        validate_kernel_shapes([(5120, 5120), (5120, 13824), (13824, 5120)])
        with self.assertRaisesRegex(ValueError, "divide 128"):
            validate_kernel_shapes([(5120, 13825)])
        validate_no_runtime_g_idx(["model.layers.0.self_attn.q_proj.weight_packed"])
        with self.assertRaisesRegex(ValueError, "g_idx tensors"):
            validate_no_runtime_g_idx(
                ["model.layers.0.self_attn.q_proj.weight_g_idx"]
            )

    def test_serving_protocol_accepts_four_models_and_requires_kernel_log(self):
        config = json.loads(SERVING_CONFIG.read_text(encoding="utf-8"))
        SERVING._validate_config(config)
        self.assertEqual(SERVING._expected_models(config), EXPECTED_VARIANTS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            values = []
            for name in EXPECTED_VARIANTS:
                model = root / name
                model.mkdir()
                (model / "config.json").write_text("{}", encoding="utf-8")
                values.append(f"{name}={model}")
            parsed = SERVING._parse_models(values, EXPECTED_VARIANTS)
            self.assertEqual(tuple(parsed), EXPECTED_VARIANTS)

            log = root / "server.log"
            pattern = config["kernel_gate"]["required_quantized_log_pattern"]
            log.write_text(f"INFO {pattern}\n", encoding="utf-8")
            server = types.SimpleNamespace(
                startup_seconds=1.0,
                baseline_memory_mib=10,
                ready_memory_mib=20,
                log_path=log,
            )
            record = SERVING._server_record(
                server,
                config,
                "unrotated_w4afp8",
            )
            self.assertTrue(record["kernel_evidence"]["matched"])
            log.write_text("no selected kernel\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "required kernel"):
                SERVING._server_record(server, config, "unrotated_w4afp8")

    def test_ppl_parent_runs_four_fresh_workers_and_preserves_token_count(self):
        config = json.loads(PPL_CONFIG.read_text(encoding="utf-8"))
        config["source_gate"]["status"] = "accepted"
        runtime = {
            "vllm": "0.25.1+cu129",
            "torch": "2.11.0+cu129",
            "cuda_runtime": "12.9",
            "gpu": "NVIDIA GH200 120GB",
            "compute_capability": [9, 0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models = {name: root / name for name in EXPECTED_VARIANTS}

            def fake_worker(**kwargs):
                name = kwargs["name"]
                return {
                    "name": name,
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
                    config_path=PPL_CONFIG,
                    manifest_path=root / "manifest.json",
                    models=models,
                    max_sequences=1,
                    worker_output_dir=root / "workers",
                )
        self.assertEqual(worker.call_count, 4)
        self.assertEqual(set(result["models"]), set(EXPECTED_VARIANTS))
        self.assertEqual(result["scored_tokens"], 2)

    def test_pending_source_gate_blocks_quality_execution(self):
        config = json.loads(PPL_CONFIG.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(RuntimeError, "still pending"):
            PPL.run(
                config=config,
                config_path=PPL_CONFIG,
                manifest_path=Path("missing-manifest.json"),
                models={name: Path(name) for name in EXPECTED_VARIANTS},
                max_sequences=1,
                worker_output_dir=Path("workers"),
            )


if __name__ == "__main__":
    unittest.main()
