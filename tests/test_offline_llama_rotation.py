import importlib.util
import unittest


HAS_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_DEPS, "requires the isolated Torch/Transformers smoke environment")
class OfflineLlamaRotationTests(unittest.TestCase):
    def test_standard_layout_survives_save_and_reload(self):
        from scripts.run_offline_llama_rotation_smoke import run_smoke

        result = run_smoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["rotation"]["module_layout"], "standard_transformers_llama")
        self.assertTrue(result["rotation"]["output_head_was_untied"])
        self.assertFalse(result["rotation"]["online_mlp_rotation"])
        self.assertFalse(result["rotation"]["post_rope_qk_rotation"])

    def test_structured_residual_size_preserves_logits(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from repro.offline_llama_rotation import (
            apply_offline_llama_rotation,
            assert_standard_llama_layout,
        )

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=127,
            hidden_size=96,
            intermediate_size=192,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=3,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            attention_bias=False,
        )
        model = LlamaForCausalLM(config).float().eval()
        input_ids = torch.arange(16, dtype=torch.long).view(1, -1)
        with torch.inference_mode():
            baseline = model(input_ids=input_ids).logits
            rotation = apply_offline_llama_rotation(model)
            rotated = model(input_ids=input_ids).logits
        assert_standard_llama_layout(model)
        self.assertEqual(rotation["residual_hadamard_kind"], "structured")
        self.assertLessEqual(float((baseline - rotated).abs().max()), 2e-5)


if __name__ == "__main__":
    unittest.main()
