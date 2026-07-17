import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires an environment with PyTorch")
class StructuredHadamardTests(unittest.TestCase):
    def test_h12_is_orthogonal(self):
        import torch
        from repro.structured_hadamard import normalized_h12

        matrix = normalized_h12(torch.float64, torch.device("cpu"))
        self.assertTrue(torch.allclose(matrix @ matrix.T, torch.eye(12, dtype=torch.float64), atol=1e-12, rtol=0.0))

    def test_12x_power2_matches_dense_kron_and_transpose_is_inverse(self):
        import torch
        from repro.structured_hadamard import normalized_h12, structured_hadamard_12x_power2
        from repro.torch_smoke import normalized_hadamard_matrix

        values = torch.randn(3, 24, dtype=torch.float64)
        h12 = normalized_h12(torch.float64, torch.device("cpu"))
        h2 = normalized_hadamard_matrix(2, torch.float64, torch.device("cpu"))
        dense = torch.kron(h12, h2)
        transformed = structured_hadamard_12x_power2(values)
        self.assertTrue(torch.allclose(transformed, values @ dense.T, atol=1e-12, rtol=0.0))
        self.assertTrue(torch.allclose(structured_hadamard_12x_power2(transformed, transpose=True), values, atol=1e-12, rtol=0.0))

    def test_supported_dimensions(self):
        from repro.structured_hadamard import supports_structured_hadamard

        self.assertTrue(supports_structured_hadamard(1536))
        self.assertTrue(supports_structured_hadamard(5120))
        self.assertTrue(supports_structured_hadamard(13824))
        self.assertFalse(supports_structured_hadamard(576))

    def test_108x_power2_is_orthogonal_and_matches_dense_kron(self):
        import torch

        from repro.structured_hadamard import normalized_h108, structured_hadamard
        from repro.torch_smoke import normalized_hadamard_matrix

        h108 = normalized_h108(torch.float64, torch.device("cpu"))
        self.assertTrue(torch.allclose(h108 @ h108.T, torch.eye(108, dtype=torch.float64), atol=1e-12, rtol=0.0))
        values = torch.randn(2, 216, dtype=torch.float64)
        dense = torch.kron(h108, normalized_hadamard_matrix(2, torch.float64, torch.device("cpu")))
        self.assertTrue(torch.allclose(structured_hadamard(values), values @ dense.T, atol=1e-12, rtol=0.0))

    def test_40x_power2_is_orthogonal_and_matches_dense_kron(self):
        import torch

        from repro.structured_hadamard import normalized_h40, normalized_structured_hadamard_matrix, structured_hadamard
        from repro.torch_smoke import normalized_hadamard_matrix

        h40 = normalized_h40(torch.float64, torch.device("cpu"))
        self.assertTrue(torch.allclose(h40 @ h40.T, torch.eye(40, dtype=torch.float64), atol=1e-12, rtol=0.0))
        values = torch.randn(2, 80, dtype=torch.float64)
        dense = torch.kron(h40, normalized_hadamard_matrix(2, torch.float64, torch.device("cpu")))
        self.assertTrue(torch.allclose(structured_hadamard(values), values @ dense.T, atol=1e-12, rtol=0.0))
        self.assertTrue(torch.allclose(normalized_structured_hadamard_matrix(80, torch.float64, torch.device("cpu")), dense, atol=0.0, rtol=0.0))
