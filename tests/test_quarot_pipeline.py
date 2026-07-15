import importlib.util
import unittest
from pathlib import Path


HAS_TORCH_SMOKE_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH_SMOKE_DEPS, "requires the isolated local smoke environment")
class PortablePipelineTests(unittest.TestCase):
    def test_synthetic_llama_pipeline_is_configuration_driven_and_equivalent(self):
        from repro.quarot_pipeline import load_pipeline_config, run_pipeline

        result = run_pipeline(load_pipeline_config(PROJECT_ROOT / "configs/pipeline/synthetic_llama_smoke.json"))
        self.assertEqual(result["device"], "cpu")
        self.assertTrue(result["rotation"]["applied"])
        self.assertLess(result["logit_error"]["max_absolute"], 2e-5)
        self.assertEqual(result["reference"]["tokens"], 124.0)

    def test_synthetic_pipeline_applies_configured_w4a4_qdq(self):
        from repro.quarot_pipeline import load_pipeline_config, run_pipeline

        result = run_pipeline(load_pipeline_config(PROJECT_ROOT / "configs/pipeline/synthetic_llama_w4a4_smoke.json"))
        self.assertEqual(result["quantization"], {"w_bits": 4, "a_bits": 4})
        self.assertGreater(result["logit_error"]["max_absolute"], 0.0)

    def test_tied_embedding_llama_is_untied_before_exact_rotation(self):
        from repro.quarot_pipeline import run_pipeline

        config = {
            "pipeline_version": 1,
            "model": {
                "kind": "random_config",
                "architecture": "llama",
                "dtype": "float32",
                "config_overrides": {
                    "vocab_size": 127,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 64,
                    "tie_word_embeddings": True,
                },
            },
            "data": {"source": "synthetic", "num_batches": 1, "batch_size": 1, "sequence_length": 16, "max_samples": 1, "seed": 0},
            "runtime": {"device": "cpu", "allow_fallback": False},
            "experiment": {
                "seed": 0,
                "rotation": {"residual_mode": "random", "seed": 3, "vo_rotation": True, "mlp_online": True, "qk_post_rope": False},
                "quantization": {"w_bits": 16, "a_bits": 16},
            },
        }
        result = run_pipeline(config)
        self.assertTrue(result["rotation"]["output_head_was_untied"])
        self.assertLess(result["logit_error"]["max_absolute"], 2e-5)
