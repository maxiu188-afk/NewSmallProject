import copy
import importlib.util
import unittest


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_DEPS, "requires torch and transformers")
class SpinQuantLlamaAdapterTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(17)
        config = LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=True,
            use_cache=False,
        )
        return LlamaForCausalLM(config).float().eval()

    def test_w16_adapter_preserves_gqa_tied_llama_logits_and_hidden_states(self):
        import torch

        from repro.spinquant.llama_adapter import (
            SpinQuantFakeQuantSpec,
            apply_spinquant_llama_training_adapter,
        )
        from repro.spinquant.rotations import SpinQuantRotations

        reference = self._model()
        candidate = copy.deepcopy(reference)
        rotations = SpinQuantRotations(
            hidden_size=32,
            head_dim=8,
            num_layers=2,
            seed=23,
        )
        torch.manual_seed(29)
        with torch.no_grad():
            r1, _ = torch.linalg.qr(torch.randn(32, 32))
            rotations.r1.copy_(r1)
            for layer in range(2):
                r2, _ = torch.linalg.qr(torch.randn(8, 8))
                rotations.r2[layer].copy_(r2)

        summary = apply_spinquant_llama_training_adapter(
            candidate,
            rotations,
            SpinQuantFakeQuantSpec(weight_bits=16),
        )
        input_ids = torch.tensor(
            [[1, 7, 11, 17, 23], [2, 3, 5, 7, 11]],
            dtype=torch.long,
        )
        with torch.no_grad():
            expected = reference(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            observed = candidate(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        self.assertTrue(
            summary["tied_embeddings_retained_as_shared_frozen_weight"]
        )
        self.assertEqual(summary["num_attention_heads"], 4)
        self.assertEqual(summary["num_key_value_heads"], 2)
        self.assertFalse(summary["transformers_model_source_copied"])
        self.assertLess(
            float((expected.logits - observed.logits).abs().max()),
            3e-5,
        )
        for original, rotated in zip(
            expected.hidden_states,
            observed.hidden_states,
        ):
            restored = rotated @ rotations.r1.transpose(-1, -2)
            self.assertLess(
                float((original - restored).abs().max().detach()),
                3e-5,
            )
        self.assertTrue(all(not parameter.requires_grad for parameter in candidate.parameters()))
        self.assertTrue(rotations.r1.requires_grad)
        self.assertTrue(rotations.r2.requires_grad)

    def test_w4_adapter_backpropagates_only_to_rotations(self):
        import torch

        from repro.spinquant.llama_adapter import (
            SpinQuantFakeQuantSpec,
            apply_spinquant_llama_training_adapter,
        )
        from repro.spinquant.rotations import SpinQuantRotations

        model = self._model()
        rotations = SpinQuantRotations(
            hidden_size=32,
            head_dim=8,
            num_layers=2,
            seed=31,
        )
        apply_spinquant_llama_training_adapter(
            model,
            rotations,
            SpinQuantFakeQuantSpec(
                weight_bits=4,
                activation_bits=16,
                weight_group_size=8,
            ),
        )
        input_ids = torch.arange(16, dtype=torch.long).view(2, 8)
        loss = model(input_ids=input_ids, use_cache=False).logits.float().square().mean()
        loss.backward()

        self.assertIsNotNone(rotations.r1.grad)
        self.assertIsNotNone(rotations.r2.grad)
        self.assertGreater(float(rotations.r1.grad.abs().max()), 0.0)
        self.assertGreater(float(rotations.r2.grad.abs().max()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_headwise_weight_transforms_match_dense_block_diagonal(self):
        import torch

        from repro.spinquant.llama_adapter import (
            _left_apply_head_rotation_transpose,
            _right_apply_head_rotation,
        )

        torch.manual_seed(37)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        block = torch.block_diag(rotation, rotation, rotation, rotation)
        value_weight = torch.randn(32, 24, dtype=torch.float64)
        output_weight = torch.randn(24, 32, dtype=torch.float64)
        left = _left_apply_head_rotation_transpose(value_weight, rotation, 4)
        right = _right_apply_head_rotation(output_weight, rotation, 4)
        self.assertLess(
            float((left - block.transpose(-1, -2) @ value_weight).abs().max()),
            1e-12,
        )
        self.assertLess(
            float((right - output_weight @ block).abs().max()),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
