import unittest

from repro.toy_llama import run_toy_equivalence


class ToyLlamaEquivalenceTests(unittest.TestCase):
    def test_complete_rotation_path_preserves_outputs(self):
        errors = run_toy_equivalence()
        self.assertLess(errors["max_logit_error"], 1e-11)
        self.assertLess(errors["max_hidden_error"], 1e-11)
        self.assertLess(errors["max_attention_score_error"], 1e-11)


if __name__ == "__main__":
    unittest.main()
