import json
from pathlib import Path
import tempfile
import unittest


class SpinQuantCalibrationTests(unittest.TestCase):
    def test_window_sampling_is_local_deterministic_and_fixed_width(self):
        from repro.spinquant.calibration import sample_token_windows

        tokens = list(range(100))
        first = sample_token_windows(
            tokens,
            samples=4,
            sequence_length=8,
            seed=7,
        )
        second = sample_token_windows(
            tokens,
            samples=4,
            sequence_length=8,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual([len(sequence) for sequence in first], [8] * 4)
        self.assertNotEqual(first[0], first[1])

    def test_token_artifact_round_trip_and_corruption_gate(self):
        from repro.spinquant.calibration import (
            CalibrationArtifactError,
            load_calibration_artifact,
            write_calibration_artifact,
        )

        sequences = [[1, 2, 3, 4], [4, 3, 2, 1]]
        provenance = {
            "dataset": {"id": "synthetic", "revision": "fixed"},
            "seed": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_calibration_artifact(
                root,
                sequences,
                provenance=provenance,
            )
            loaded = load_calibration_artifact(manifest_path)
            self.assertEqual(loaded["sequences"], sequences)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokens"]["total_tokens"], 8)
            tokens_path = root / "token-ids.json"
            tokens_path.write_text("[[1,2,3,5],[4,3,2,1]]\n", encoding="utf-8")
            with self.assertRaises(CalibrationArtifactError):
                load_calibration_artifact(manifest_path)


if __name__ == "__main__":
    unittest.main()
