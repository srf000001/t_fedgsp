from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_model_basis.py"
SPEC = importlib.util.spec_from_file_location("prepare_model_basis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ModelBasisTest(unittest.TestCase):
    def test_deterministic_aggregation_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n, source_bins, concepts, modalities = 3, 4, 5, 2
            x = sparse.csr_matrix(
                (
                    np.array([1, 2, 1, 1], dtype=np.float32),
                    (
                        np.array([0, 0, 1, 2]),
                        np.array([0, concepts + 1, 2 * concepts + 2, 3 * concepts + 4]),
                    ),
                ),
                shape=(n, source_bins * concepts),
            )
            m = sparse.csr_matrix(
                (
                    np.ones(4, dtype=np.float32),
                    (
                        np.array([0, 0, 1, 2]),
                        np.array([0, modalities + 1, 2 * modalities, 3 * modalities + 1]),
                    ),
                ),
                shape=(n, source_bins * modalities),
            )
            graph = sparse.eye(concepts, format="csr", dtype=np.float32)
            split = np.array([0, 1, 2], dtype=np.uint8)
            first = MODULE.run_prepare(
                x, m, graph, split, root / "a", 60, 120, concepts, modalities, 3, 1, 7, 0
            )
            second = MODULE.run_prepare(
                x, m, graph, split, root / "b", 60, 120, concepts, modalities, 3, 1, 7, 0
            )
            a = np.load(root / "a" / first["basis"]["path"])
            b = np.load(root / "b" / second["basis"]["path"])
            self.assertEqual(list(a.shape), [3, 2, 2, 3])
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(first["observations"]["zero_active_stays"], 0)


if __name__ == "__main__":
    unittest.main()

