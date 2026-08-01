import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class SpinQuantRotationTests(unittest.TestCase):
    def test_r1_r2_initialization_is_reproducible_and_orthogonal(self):
        from repro.spinquant.rotations import SpinQuantRotations

        first = SpinQuantRotations(
            hidden_size=8,
            head_dim=4,
            num_layers=2,
            seed=7,
        )
        second = SpinQuantRotations(
            hidden_size=8,
            head_dim=4,
            num_layers=2,
            seed=7,
        )
        self.assertTrue(first.r1.equal(second.r1))
        self.assertTrue(first.r2.equal(second.r2))
        self.assertLess(first.errors()["r1"], 1e-6)
        self.assertLess(first.errors()["r2"], 1e-6)

    def test_structured_hadamard_initialization_supports_llama_13b_factor(self):
        from repro.spinquant.rotations import random_signed_hadamard
        from repro.spinquant.stiefel import orthogonality_error

        rotation = random_signed_hadamard(40, seed=5)
        self.assertEqual(tuple(rotation.shape), (40, 40))
        self.assertLess(float(orthogonality_error(rotation)), 1e-6)

    def test_general_r2_value_output_fusion_preserves_full_precision(self):
        import torch
        import torch.nn.functional as F

        from repro.spinquant.rotations import fuse_value_output_pair

        torch.manual_seed(11)
        hidden_size = 8
        r1, _ = torch.linalg.qr(torch.randn(hidden_size, hidden_size, dtype=torch.float64))
        per_head, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
        r2 = torch.block_diag(per_head, per_head)
        value_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
        output_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float64)
        inputs = torch.randn(5, hidden_size, dtype=torch.float64)

        reference = F.linear(F.linear(inputs, value_weight), output_weight)
        value_rotated, output_rotated = fuse_value_output_pair(
            value_weight,
            output_weight,
            r1,
            r2,
        )
        candidate_rotated = F.linear(
            F.linear(inputs @ r1, value_rotated),
            output_rotated,
        )
        candidate = candidate_rotated @ r1.transpose(-1, -2)
        self.assertLess(float((reference - candidate).abs().max()), 1e-11)


if __name__ == "__main__":
    unittest.main()
