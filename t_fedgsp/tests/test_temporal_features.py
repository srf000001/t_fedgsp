from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_temporal_features.py"
SPEC = importlib.util.spec_from_file_location("build_temporal_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TemporalFeatureTest(unittest.TestCase):
    def test_small_build_removes_future_diagnosis_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            vocab = root / "vocab"
            raw.mkdir()
            vocab.mkdir()
            split = pd.DataFrame(
                {
                    "patientunitstayid": [1, 2, 3],
                    "uniquepid": ["p1", "p2", "p3"],
                    "hospitalid": [10, 10, 20],
                    "split": ["train", "val", "test"],
                }
            )
            split.to_csv(root / "split.csv", index=False)
            target_data = {
                "patientunitstayid": [1, 2, 3],
                "split": ["train", "val", "test"],
                "hospitalid": [10, 10, 20],
            }
            for index in range(50):
                target_data[f"label__L{index}"] = [1 if index == 0 else 0, 0, 0]
            pd.DataFrame(target_data).to_csv(root / "targets.csv", index=False)
            (vocab / "diagnosis_vocab.json").write_text(json.dumps(["L0", "X"]), encoding="utf-8")
            (vocab / "medication_vocab.json").write_text(json.dumps(["drug"]), encoding="utf-8")
            (vocab / "treatment_vocab.json").write_text(json.dumps(["care"]), encoding="utf-8")
            (vocab / "lab_thresholds.json").write_text(
                json.dumps({"lab": {"low": 1.0, "high": 3.0}}), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 1, 2],
                    "diagnosisoffset": [60, 61, 120],
                    "icd9code": ["L0", "X", "X"],
                }
            ).to_csv(raw / "diagnosis.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 2],
                    "drugstartoffset": [None, 120],
                    "drugorderoffset": [60, 100],
                    "drugname": ["drug", "drug"],
                }
            ).to_csv(raw / "medication.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 2, 3],
                    "labresultoffset": [60, 60, 60],
                    "labname": ["lab", "lab", "lab"],
                    "labresult": [0.0, 2.0, 4.0],
                }
            ).to_csv(raw / "lab.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 3],
                    "treatmentoffset": [60, 60],
                    "treatmentstring": ["group|care", "care"],
                }
            ).to_csv(raw / "treatment.csv.gz", index=False, compression="gzip")
            config = {
                "dataset": {
                    "name": "synthetic-test-only",
                    "require_full_eicu": False,
                    "split_path": str((root / "split.csv").relative_to(root)),
                    "target_path": str((root / "targets.csv").relative_to(root)),
                    "vocab_root": str(vocab.relative_to(root)),
                    "input_window_minutes": 1440,
                    "base_bin_minutes": 60,
                    "chunk_size": 2,
                },
                "concept_graph": {
                    "source_split": "train",
                    "min_cooccurrence": 1,
                    "top_k_neighbors": 3,
                    "add_self_loops": True,
                },
                "outputs": {"root": "out"},
            }
            manifest = MODULE.run_build(config, root, raw)
            self.assertEqual(manifest["removed_input_target_diagnosis_events"], 1)
            self.assertEqual(manifest["target_matrix"]["shape"], [3, 50])
            self.assertGreater(manifest["feature_matrix"]["nnz"], 0)
            self.assertEqual(manifest["observation_matrix"]["zero_observation_stays"], 0)
            self.assertEqual(manifest["concept_graph"]["source_split"], "train")
            self.assertTrue(manifest["demo_or_synthetic_data_used"])
            self.assertFalse(manifest["main_results_eligible"])
            with MODULE.np.load(root / "out" / "dataset_arrays.npz") as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {"y", "split_code", "client_code", "patient_cluster_code"},
                )
            self.assertTrue((root / "out" / "M_time_modality_counts.npz").exists())


if __name__ == "__main__":
    unittest.main()
