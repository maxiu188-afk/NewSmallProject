import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires an NVIDIA CUDA runtime")
class W4A8CudaTests(unittest.TestCase):
    def test_cuda_kernel_matches_independent_int32_reference(self):
        from scripts.run_w4a8_cuda_smoke import run

        result = run(verbose_build=False)
        self.assertTrue(result["int32_accumulators"]["exact_match"])
        self.assertLessEqual(result["scaled_output"]["max_absolute_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
