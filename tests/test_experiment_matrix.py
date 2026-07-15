import json
import unittest
from pathlib import Path

from repro.experiment_matrix import materialize_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentMatrixTests(unittest.TestCase):
    def test_fake_quant_matrix_has_controlled_f0_to_f5_variants(self):
        template_path = PROJECT_ROOT / "configs/templates/llama2_fake_quant_matrix.json"
        with template_path.open(encoding="utf-8") as handle:
            matrix = materialize_matrix(json.load(handle))
        self.assertEqual(set(matrix), {"F0-bf16", "F1-naive-w4", "F2-quarot-w4", "F3-naive-w4a4", "F4-quarot-w4a4", "F5-quarot-w4a4kv4"})
        self.assertEqual(matrix["F0-bf16"]["quantization"]["execution_mode"], "bf16")
        self.assertEqual(matrix["F1-naive-w4"]["quantization"]["w_bits"], 4)
        self.assertEqual(matrix["F2-quarot-w4"]["quantization"]["rotation"], "hadamard")
        self.assertEqual(matrix["F4-quarot-w4a4"]["quantization"]["a_bits"], 4)
        self.assertEqual(matrix["F5-quarot-w4a4kv4"]["quantization"]["k_bits"], 4)
        self.assertEqual(matrix["F5-quarot-w4a4kv4"]["quantization"]["v_bits"], 4)


if __name__ == "__main__":
    unittest.main()
