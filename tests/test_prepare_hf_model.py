import unittest
import importlib.util


HAS_HF_HUB = importlib.util.find_spec("huggingface_hub") is not None


@unittest.skipUnless(HAS_HF_HUB, "requires the isolated local model environment")
class PrepareHfModelTests(unittest.TestCase):
    def test_resolved_config_pins_revision_and_stays_offline(self):
        from scripts.prepare_hf_model import _resolved_config

        config = {
            "_config_dir": "/temporary/configs",
            "model": {"kind": "pretrained", "id": "org/model", "revision": "pin-before-first-download"},
        }
        resolved = _resolved_config(config, "a" * 40)
        self.assertNotIn("_config_dir", resolved)
        self.assertEqual(resolved["model"]["revision"], "a" * 40)
        self.assertTrue(resolved["model"]["local_files_only"])
        self.assertEqual(config["model"]["revision"], "pin-before-first-download")


if __name__ == "__main__":
    unittest.main()
