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

    def test_asymmetric_a8_quantizes_decoder_inputs_only(self):
        import torch

        from repro.spinquant.activation_qdq import spinquant_activation_qdq
        from repro.spinquant.offline import (
            fake_quantize_llama_decoder_activations,
        )

        model = self._model().eval()
        input_ids = torch.tensor([[1, 4, 9, 16, 25, 36, 49, 64]])
        baseline_inputs = []
        observed_inputs = []
        decoder = model.model.layers[0].self_attn.q_proj
        baseline_handle = decoder.register_forward_pre_hook(
            lambda _module, inputs: baseline_inputs.append(inputs[0].detach().clone())
        )
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False)
        finally:
            baseline_handle.remove()

        with fake_quantize_llama_decoder_activations(
            model,
            bits=8,
            symmetric=False,
            o_proj_group_size=8,
        ) as summary:
            self.assertEqual(len(decoder._forward_pre_hooks), 1)
            self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)
            observed_handle = decoder.register_forward_pre_hook(
                lambda _module, inputs: observed_inputs.append(inputs[0].detach().clone())
            )
            try:
                with torch.inference_mode():
                    model(input_ids=input_ids, use_cache=False)
            finally:
                observed_handle.remove()

        self.assertEqual(summary["quantized_decoder_linears"], 14)
        self.assertEqual(summary["activation_bits"], 8)
        self.assertFalse(summary["activation_symmetric"])
        self.assertEqual(summary["activation_o_proj_group_size"], 8)
        self.assertTrue(summary["activation_ungrouped_include_zero"])
        self.assertEqual(
            summary["activation_granularity"],
            "per_token_last_axis_o_proj_grouped",
        )
        self.assertFalse(summary["quantized_lm_head"])
        self.assertEqual(len(baseline_inputs), 1)
        self.assertEqual(len(observed_inputs), 1)
        expected = spinquant_activation_qdq(
            baseline_inputs[0], 8, symmetric=False
        )
        torch.testing.assert_close(observed_inputs[0], expected)

        # The context must remove every temporary hook after evaluation.
        self.assertEqual(len(decoder._forward_pre_hooks), 0)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
