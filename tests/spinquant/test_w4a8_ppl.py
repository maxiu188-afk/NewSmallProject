import copy
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/spinquant/llama2_13b_nohad_w4a8_fake_quant_ppl_isambard.json"
)


class SpinQuantW4A8PplTests(unittest.TestCase):
    def test_config_pins_paper_aligned_nohad_protocol(self):
        from scripts.spinquant.evaluate_llama_nohad_w4a8_ppl import (
            EXPECTED_CASES,
            _validate_config,
        )

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        _validate_config(config)
        self.assertEqual(tuple(config["evaluation"]["cases"]), EXPECTED_CASES)
        self.assertEqual(config["learned_rotation"]["training_job_id"], "5848547")
        self.assertEqual(config["gptq_calibration"]["samples"], 128)
        self.assertEqual(config["gptq_calibration"]["sequence_length"], 2048)
        self.assertEqual(config["quantization"]["scheme"], "W4A8KV16")
        self.assertEqual(config["quantization"]["weight_algorithm"], "GPTQ")
        self.assertEqual(config["quantization"]["weight_group_size"], 128)
        self.assertEqual(config["quantization"]["activation_bits"], 8)
        self.assertFalse(config["quantization"]["activation_symmetric"])
        self.assertFalse(config["quantization"]["activation_clipping"])
        self.assertTrue(
            config["quantization"]["activation_ungrouped_include_zero"]
        )
        self.assertEqual(
            config["quantization"]["activation_o_proj_group_size"], 128
        )
        self.assertEqual(config["quantization"]["key_bits"], 16)
        self.assertEqual(config["quantization"]["value_bits"], 16)

    def test_config_rejects_symmetric_activation_drift(self):
        from scripts.spinquant.evaluate_llama_nohad_w4a8_ppl import _validate_config

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["quantization"]["activation_symmetric"] = True
        with self.assertRaisesRegex(ValueError, "activation_symmetric"):
            _validate_config(changed)


if __name__ == "__main__":
    unittest.main()
