import importlib.util
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deployment"
    / "vllm_w4a16_llama2_13b_single_block_isambard.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_vllm_w4a16_llama2_13b_single_block",
    PROJECT_ROOT / "scripts" / "run_vllm_w4a16_llama2_13b_single_block.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Generated:
    def __init__(self, token_ids):
        self.outputs = [type("Candidate", (), {"token_ids": token_ids})()]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self, events):
        self.events = events
        self.apply_callable = None

    def generate(self, prompts, params, use_tqdm):
        assert use_tqdm is False
        count = params.kwargs["max_tokens"]
        return [_Generated(list(range(count))) for _ in prompts]

    def apply_model(self, func):
        self.apply_callable = func
        return [self.events]


class VllmW4A16LlamaSingleBlockTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_protocol_is_valid_and_is_explicitly_diagnostic(self):
        MODULE._validate_config(self.config)
        engine = self.config["engine"]
        self.assertEqual(engine["block_index"], 0)
        self.assertTrue(engine["enforce_eager"])
        self.assertFalse(engine["enable_prefix_caching"])
        self.assertEqual(engine["kv_cache_memory_bytes"], 16 * 1024**3)
        self.assertEqual(
            [case["name"] for case in self.config["formal"]["cases"]],
            [
                "prefill-b1-s256",
                "prefill-b1-s2048",
                "prefill-b8-s256",
                "prefill-b8-s2048",
                "decode-b1-s256-d16",
                "decode-b1-s2048-d16",
                "decode-b8-s256-d16",
                "decode-b8-s2048-d16",
            ],
        )
        self.assertIn("diagnostic", self.config["scope"].lower())
        self.assertIn("full-model", self.config["scope"].lower())

    def test_prompt_batch_is_deterministic_and_exact_length(self):
        first = MODULE._prompt_batch(
            batch_size=3,
            prompt_tokens=17,
            vocab_size=32000,
            seed=7,
        )
        second = MODULE._prompt_batch(
            batch_size=3,
            prompt_tokens=17,
            vocab_size=32000,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual([len(item["prompt_token_ids"]) for item in first], [17] * 3)
        self.assertTrue(all(item["prompt_token_ids"][0] == 1 for item in first))
        self.assertEqual(len({tuple(item["prompt_token_ids"]) for item in first}), 3)

    def test_event_validation_separates_prefill_and_decode(self):
        case = {
            "name": "decode",
            "phase": "decode",
            "batch_size": 8,
            "prompt_tokens": 256,
            "decode_steps": 2,
        }
        events = [
            {"token_rows": 2048},
            {"token_rows": 8},
            {"token_rows": 8},
        ]
        prefill, decode = MODULE._validate_events(events, case)
        self.assertEqual(len(prefill), 1)
        self.assertEqual(len(decode), 2)
        with self.assertRaises(RuntimeError):
            MODULE._validate_events(events[:-1], case)

    def test_request_uses_single_argument_apply_model_api(self):
        case = {
            "name": "decode",
            "phase": "decode",
            "batch_size": 1,
            "prompt_tokens": 8,
            "decode_steps": 2,
        }
        events = [
            {"token_rows": 8},
            {"token_rows": 1},
            {"token_rows": 1},
        ]
        llm = _FakeLLM(events)
        observed, _, generated = MODULE._run_request(
            llm,
            _FakeSamplingParams,
            prompts=[{"prompt_token_ids": list(range(8))}],
            case=case,
            block_index=0,
        )
        self.assertEqual(observed, events)
        self.assertEqual(len(generated[0]), 3)
        self.assertIsNotNone(llm.apply_callable)

    def test_summary_and_comparisons_use_model_level_parameter_bytes(self):
        summary = MODULE._summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["p90"], 4.0)

        case_name = self.config["formal"]["cases"][0]["name"]
        models = {}
        for name, elapsed, parameter_bytes in (
            ("bf16", 2.0, 100),
            ("unrotated_w4a16", 1.0, 40),
            ("rotated_w4a16", 1.25, 40),
        ):
            models[name] = {
                "layer_inspection": {"parameter_bytes": parameter_bytes},
                "cases": {
                    case_name: {
                        "layer_call_elapsed_ms": {"median": elapsed},
                        "work_items_per_second": {"median": 1000.0 / elapsed},
                    }
                },
            }
        minimal_config = {
            "formal": {"cases": [{"name": case_name}]},
        }
        comparisons = MODULE._comparisons(models, minimal_config)
        unrotated = comparisons[case_name]["unrotated_w4a16_vs_bf16"]
        self.assertEqual(unrotated["median_layer_speedup_bf16_over_candidate"], 2.0)
        self.assertEqual(unrotated["layer_parameter_bytes_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
