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


if __name__ == "__main__":
    unittest.main()
