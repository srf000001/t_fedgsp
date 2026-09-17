from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_temporal_eicu.py"
SPEC = importlib.util.spec_from_file_location("profile_temporal_eicu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TemporalProfileTest(unittest.TestCase):
    def test_small_aggregate_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()
            split = pd.DataFrame(
                {
                    "patientunitstayid": [1, 2, 3],
                    "uniquepid": ["p1", "p2", "p3"],
                    "hospitalid": [10, 10, 20],
                    "split": ["train", "val", "test"],
                }
            )
            split_path = root / "split.csv"
            split.to_csv(split_path, index=False)
            targets = pd.DataFrame(
                {
                    "patientunitstayid": [1, 2, 3],
                    "split": ["train", "val", "test"],
                    "hospitalid": [10, 10, 20],
                    "label__a": [1, 0, 1],
                    "label__b": [0, 1, 0],
                }
            )
            target_path = root / "targets.csv"
            targets.to_csv(target_path, index=False)
            pd.DataFrame(
                {"patientunitstayid": [1, 1, 2, 99], "diagnosisoffset": [0, 1440, -1, 60]}
            ).to_csv(raw / "diagnosis.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 2, 3],
                    "drugstartoffset": [None, 1500, 120],
                    "drugorderoffset": [60, 100, 100],
                }
            ).to_csv(raw / "medication.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {"patientunitstayid": [1, 2, 3], "labresultoffset": [30, None, 720]}
            ).to_csv(raw / "lab.csv.gz", index=False, compression="gzip")
            pd.DataFrame(
                {"patientunitstayid": [1, 2, 3], "treatmentoffset": [360, 361, 2000]}
            ).to_csv(raw / "treatment.csv.gz", index=False, compression="gzip")

            report = MODULE.run_profile(
                raw,
                split_path,
                target_path,
                out,
                "synthetic-test-only",
                1440,
                60,
                [60, 120, 240, 360],
                2,
            )
            self.assertEqual(report["cohort"]["num_stays"], 3)
            self.assertFalse(report["cohort"]["patient_split_leakage"])
            self.assertEqual(report["targets"]["num_labels"], 2)
            self.assertEqual(report["modalities"]["diagnosis"]["window_event_rows"], 2)
            self.assertEqual(report["modalities"]["medication"]["window_event_rows"], 2)
            self.assertEqual(report["modalities"]["lab"]["missing_offset_rows"], 1)
            self.assertEqual(report["modalities"]["treatment"]["post_window_rows"], 1)
            self.assertTrue((out / "temporal_eda.md").exists())
            self.assertTrue((out / "event_time_distribution.csv").exists())

    def test_patient_split_leakage_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            pd.DataFrame(
                {
                    "patientunitstayid": [1, 2],
                    "uniquepid": ["same", "same"],
                    "hospitalid": [10, 10],
                    "split": ["train", "test"],
                }
            ).to_csv(path, index=False)
            _, info = MODULE.read_assignments(path)
            self.assertTrue(info["patient_split_leakage"])
            self.assertEqual(info["max_splits_per_patient"], 2)


if __name__ == "__main__":
    unittest.main()

