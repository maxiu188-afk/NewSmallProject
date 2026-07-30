import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
)


@unittest.skipUnless(HAS_DEPS, "requires torch and safetensors")
class SpinQuantArtifactTests(unittest.TestCase):
    def test_safe_rotation_round_trip_with_provenance(self):
        import torch

        from repro.spinquant.artifacts import (
            RotationArtifactError,
            load_rotation_artifact,
            save_rotation_artifact,
        )
        from repro.spinquant.rotations import SpinQuantRotations

        source = SpinQuantRotations(
            hidden_size=8,
            head_dim=4,
            num_layers=2,
            seed=41,
        )
        target = SpinQuantRotations(
            hidden_size=8,
            head_dim=4,
            num_layers=2,
            seed=99,
        )
        provenance = {
            "model": {"id": "random-tiny-llama", "revision": "config-only"},
            "data": {"source": "synthetic", "seed": 0},
            "implementation": "independent-paper-derived",
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = save_rotation_artifact(
                Path(directory),
                source,
                provenance=provenance,
            )
            self.assertFalse((Path(directory) / "R.bin").exists())
            result = load_rotation_artifact(manifest_path, target=target)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with self.assertRaises(RotationArtifactError):
                save_rotation_artifact(
                    Path(directory),
                    source,
                    provenance=provenance,
                )

        self.assertEqual(manifest["format"], "newsmallproject-spinquant-rotations-v1")
        self.assertEqual(manifest["provenance"], provenance)
        self.assertTrue(torch.equal(source.r1, target.r1))
        self.assertTrue(torch.equal(source.r2, target.r2))
        self.assertLess(result["observed_orthogonality_error"]["r1"], 1e-6)

    def test_corrupted_safe_tensor_is_rejected_before_loading(self):
        from repro.spinquant.artifacts import (
            RotationArtifactError,
            load_rotation_artifact,
            save_rotation_artifact,
        )
        from repro.spinquant.rotations import SpinQuantRotations

        rotations = SpinQuantRotations(
            hidden_size=8,
            head_dim=4,
            num_layers=1,
            seed=43,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = save_rotation_artifact(
                root,
                rotations,
                provenance={"test": True},
            )
            tensor_path = root / "rotations.safetensors"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
            with self.assertRaises(RotationArtifactError):
                load_rotation_artifact(manifest_path)


if __name__ == "__main__":
    unittest.main()
