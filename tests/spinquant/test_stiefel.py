import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class CayleyRetractionTests(unittest.TestCase):
    def test_exact_step_preserves_orthogonality_and_descends(self):
        import torch

        from repro.spinquant.stiefel import cayley_retraction, orthogonality_error

        rotation = torch.eye(4, dtype=torch.float64, requires_grad=True)
        target, _ = torch.linalg.qr(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, -1.0, 0.5, 1.0],
                    [0.0, 1.0, -2.0, 3.0],
                    [3.0, 0.5, 1.0, -1.0],
                ],
                dtype=torch.float64,
            )
        )
        before = ((rotation - target) ** 2).sum()
        before.backward()
        updated = cayley_retraction(
            rotation.detach(),
            rotation.grad.detach(),
            0.1,
            method="exact",
        )
        after = ((updated - target) ** 2).sum()
        self.assertLess(float(after.detach()), float(before.detach()))
        self.assertLess(float(orthogonality_error(updated)), 1e-12)

    def test_fixed_point_matches_exact_for_a_small_step(self):
        import torch

        from repro.spinquant.stiefel import cayley_retraction

        torch.manual_seed(3)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        gradient = torch.randn(8, 8, dtype=torch.float64)
        exact = cayley_retraction(rotation, gradient, 0.01, method="exact")
        approximate = cayley_retraction(
            rotation,
            gradient,
            0.01,
            method="fixed_point",
            fixed_point_steps=8,
        )
        self.assertLess(float((exact - approximate).abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
