import importlib.util
from pathlib import Path
import tempfile
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class PackedW4A8CheckpointTests(unittest.TestCase):
    def _model(self):
        import torch
        from torch import nn

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 6, bias=True)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer(), Layer()])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

        torch.manual_seed(11)
        return Model().eval()

    def test_streamed_checkpoint_round_trip_and_module_install(self):
        import torch

        from repro.gptq import GPTQPackedWeight
        from repro.w4a8_checkpoint import (
            PackedW4A8CheckpointWriter,
            convert_reference_linears_to_cuda,
            decoder_linear_names,
            install_checkpoint_linears,
            load_checkpoint_manifest,
            load_linear_artifact,
        )
        from repro.w4a8_linear import PackedW4A8ReferenceLinear, W4A8Linear, pack_w4_weight

        model = self._model()
        names = decoder_linear_names(model)
        self.assertEqual(names, ["model.layers.0.proj", "model.layers.1.proj"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PackedW4A8CheckpointWriter(root, names, {"model": {"id": "tiny"}})
            for name in names:
                linear = model.get_submodule(name)
                packed, scales = pack_w4_weight(linear.weight, group_size=8)
                permutation = torch.randperm(8)
                writer.write(name, GPTQPackedWeight(packed, scales, permutation), linear.bias)
            manifest_path = writer.finalize({"decoder_linears": 2})
            manifest = load_checkpoint_manifest(manifest_path)
            self.assertEqual(manifest["tensor_count"], 2)
            artifact = load_linear_artifact(manifest_path, manifest, names[0])
            self.assertEqual(artifact["tensor"], names[0])

            installed = install_checkpoint_linears(model, manifest_path, implementation="reference")
            self.assertEqual(installed, names)
            self.assertIsInstance(model.get_submodule(names[0]), PackedW4A8ReferenceLinear)
            convert_reference_linears_to_cuda(model, names)
            self.assertIsInstance(model.get_submodule(names[0]), W4A8Linear)

    def test_manifest_rejects_tampered_shard(self):
        from repro.gptq import GPTQPackedWeight
        from repro.w4a8_checkpoint import PackedW4A8CheckpointWriter, decoder_linear_names, load_checkpoint_manifest
        from repro.w4a8_linear import pack_w4_weight

        model = self._model()
        names = decoder_linear_names(model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PackedW4A8CheckpointWriter(root, names, {})
            for name in names:
                linear = model.get_submodule(name)
                packed, scales = pack_w4_weight(linear.weight, group_size=8)
                writer.write(name, GPTQPackedWeight(packed, scales, None), linear.bias)
            manifest_path = writer.finalize({})
            shard = next(root.glob("*.pt"))
            with shard.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                load_checkpoint_manifest(manifest_path)

    def test_rotated_tiny_llama_exposes_exactly_seven_linears_per_layer(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        from repro.quarot_pipeline import apply_llama_quarot
        from repro.w4a8_checkpoint import decoder_linear_names

        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=97,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_bias=False,
                mlp_bias=False,
                tie_word_embeddings=False,
            )
        ).to(dtype=torch.float32).eval()
        apply_llama_quarot(
            model,
            {
                "residual_mode": "hadamard",
                "seed": 0,
                "vo_rotation": True,
                "mlp_online": True,
                "qk_post_rope": True,
            },
            {"w_bits": 4, "a_bits": 4, "k_bits": 16, "v_bits": 16},
        )
        names = decoder_linear_names(model)
        self.assertEqual(len(names), 14)
        self.assertIn("model.layers.0.self_attn.attention.q_proj", names)
        self.assertIn("model.layers.0.mlp.down_proj.linear", names)
        self.assertNotIn("lm_head", names)

        from repro.gptq import GPTQPackedWeight
        from repro.w4a8_checkpoint import PackedW4A8CheckpointWriter, install_checkpoint_linears
        from repro.w4a8_linear import pack_w4_weight

        with tempfile.TemporaryDirectory() as directory:
            writer = PackedW4A8CheckpointWriter(Path(directory), names, {"model": {"id": "tiny-llama"}})
            for name in names:
                linear = model.get_submodule(name)
                packed, scales = pack_w4_weight(linear.weight, group_size=64)
                writer.write(name, GPTQPackedWeight(packed, scales, None), linear.bias)
            manifest_path = writer.finalize({"decoder_linears": 14})
            install_checkpoint_linears(model, manifest_path, implementation="reference")
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits
            self.assertEqual(logits.shape, (1, 4, 97))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
