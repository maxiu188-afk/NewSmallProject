import importlib.util
import unittest


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_DEPS, "requires torch and transformers")
class SpinQuantOfflineTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(71)
        return LlamaForCausalLM(
            LlamaConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                tie_word_embeddings=True,
                use_cache=False,
            )
        ).float()

    def test_offline_non_symmetric_r1_r2_preserves_logits(self):
        import torch

        from repro.spinquant.offline import apply_spinquant_llama_offline
        from repro.spinquant.rotations import SpinQuantRotations

        model = self._model().eval()
        rotated = self._model().eval()
        rotated.load_state_dict(model.state_dict())
        rotations = SpinQuantRotations(
            hidden_size=32,
            head_dim=8,
            num_layers=2,
            seed=0,
        )
        generator = torch.Generator().manual_seed(73)
        with torch.no_grad():
            rotations.r1.copy_(
                torch.linalg.qr(torch.randn(32, 32, generator=generator)).Q
            )
            for index in range(2):
                rotations.r2[index].copy_(
                    torch.linalg.qr(
                        torch.randn(8, 8, generator=generator)
                    ).Q
                )
        input_ids = torch.tensor([[1, 4, 9, 16, 25, 36, 49, 64]])
        with torch.inference_mode():
            expected = model(input_ids).logits
        summary = apply_spinquant_llama_offline(rotated, rotations)
        with torch.inference_mode():
            observed = rotated(input_ids).logits
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)
        self.assertTrue(summary["output_head_was_untied"])
        self.assertEqual(summary["online_rotation_modules"], 0)
        self.assertFalse(rotated.config.tie_word_embeddings)
        for layer in rotated.model.layers:
            self.assertTrue(
                torch.equal(
                    layer.input_layernorm.weight,
                    torch.ones_like(layer.input_layernorm.weight),
                )
            )
            self.assertTrue(
                torch.equal(
                    layer.post_attention_layernorm.weight,
                    torch.ones_like(layer.post_attention_layernorm.weight),
                )
            )

    def test_groupwise_w4_quantizes_exactly_seven_linears_per_layer(self):
        from repro.spinquant.offline import fake_quantize_llama_decoder_w4_

        model = self._model()
        result = fake_quantize_llama_decoder_w4_(
            model,
            bits=4,
            group_size=8,
            symmetric=True,
        )
        self.assertEqual(result["quantized_decoder_linears"], 14)
        self.assertFalse(result["quantized_lm_head"])


if __name__ == "__main__":
    unittest.main()
