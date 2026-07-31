import importlib.util
import unittest


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_DEPS, "requires torch and transformers")
class ActivationAuditTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(101)
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

    def test_paper_operator_keeps_same_sign_endpoints(self):
        import torch

        from repro.qdq import qdq_last_axis
        from repro.spinquant.activation_qdq import spinquant_activation_qdq

        values = torch.tensor([[1.0, 2.0], [-2.0, -1.0]])
        current = qdq_last_axis(values, 8, symmetric=False)
        observed = spinquant_activation_qdq(values, 8, symmetric=False)

        self.assertTrue(torch.equal(current, torch.tensor([[1.0, 1.0], [-1.0, -1.0]])))
        self.assertFalse(torch.equal(observed[0, 0], observed[0, 1]))
        self.assertFalse(torch.equal(observed[1, 0], observed[1, 1]))
        self.assertLess(
            (observed - values).abs().amax().item(),
            (current - values).abs().amax().item(),
        )

    def test_paper_o_proj_groups_isolate_an_outlier(self):
        import torch

        from repro.spinquant.activation_qdq import spinquant_activation_qdq

        values = torch.linspace(-0.05, 0.05, 256).reshape(1, 1, 256)
        values[..., 0] = 100.0
        full = spinquant_activation_qdq(
            values, 8, symmetric=False, group_size=-1
        )
        grouped = spinquant_activation_qdq(
            values, 8, symmetric=False, group_size=128
        )
        full_error = (full[..., 128:] - values[..., 128:]).square().sum()
        grouped_error = (grouped[..., 128:] - values[..., 128:]).square().sum()
        self.assertLess(grouped_error.item(), full_error.item())

    def test_collector_observes_all_seven_linears_per_layer(self):
        import torch

        from repro.spinquant.activation_audit import audit_llama_decoder_activations

        model = self._model().eval()
        input_ids = torch.tensor([[1, 4, 9, 16, 25, 36, 49, 64]])
        with audit_llama_decoder_activations(model, mode="paper_aligned") as audit:
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False)
        summary = audit.summary()

        self.assertEqual(summary["registered_modules"], 14)
        self.assertEqual(summary["called_modules"], 14)
        self.assertEqual(summary["missing_modules"], [])
        self.assertEqual(set(summary["role_module_counts"].values()), {2})
        self.assertEqual(summary["global"]["calls"], 14)
        self.assertGreater(summary["global"]["vectors"], 0)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)
        for layer in model.model.layers:
            self.assertEqual(len(layer.self_attn.q_proj._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
