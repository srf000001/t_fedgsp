from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_temporal_features import (
    find_column,
    load_locked_task,
    load_vocabulary,
    normalize_codes,
    normalize_text,
    resolve_offsets,
    sha256,
    window_mask,
)
from profile_temporal_eicu import locate_stays


MODALITIES = ["diagnosis", "medication", "lab", "treatment"]


def concatenate(parts: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def scan_modality(
    raw_dir: Path,
    modality: str,
    modality_index: int,
    val_stays: np.ndarray,
    y_val: np.ndarray,
    label_names: list[str],
    concept_maps: dict[str, dict[str, int]],
    lab_thresholds: dict[str, dict[str, float]],
    input_window: int,
    chunk_size: int,
    event_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    specs = {
        "diagnosis": ("diagnosis.csv.gz", ["diagnosisoffset"], ["icd9code"]),
        "medication": (
            "medication.csv.gz",
            ["drugstartoffset", "drugorderoffset"],
            ["drugname"],
        ),
        "lab": ("lab.csv.gz", ["labresultoffset"], ["labname", "labresult"]),
        "treatment": ("treatment.csv.gz", ["treatmentoffset"], ["treatmentstring"]),
    }
    filename, offset_candidates, value_candidates = specs[modality]
    path = raw_dir / filename
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    stay_col = find_column(columns, ["patientunitstayid", "patientUnitStayID"])
    primary = find_column(columns, [offset_candidates[0]])
    fallback = (
        find_column(columns, [offset_candidates[1]], required=False)
        if len(offset_candidates) > 1
        else None
    )
    value_columns = [find_column(columns, [candidate]) for candidate in value_candidates]
    usecols = [stay_col, primary] + ([fallback] if fallback else []) + value_columns

    observation_stay_parts: list[np.ndarray] = []
    observation_offset_parts: list[np.ndarray] = []
    concept_event_parts: list[np.ndarray] = []
    concept_id_parts: list[np.ndarray] = []
    next_event = int(event_offset)
    removed_overlap = 0
    skipped_normal_labs = 0
    label_index = {label.upper(): index for index, label in enumerate(label_names)}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        positions = locate_stays(val_stays, chunk[stay_col])
        offsets = resolve_offsets(chunk, primary, fallback)
        keep, kept_offsets = window_mask(positions, offsets, input_window)
        if not keep.any():
            continue
        source_rows = np.flatnonzero(keep)
        stay_rows = positions[keep].astype(np.int32, copy=False)
        event_ids = np.arange(next_event, next_event + source_rows.size, dtype=np.int64)
        next_event += int(source_rows.size)
        observation_stay_parts.append(stay_rows)
        observation_offset_parts.append(kept_offsets.astype(np.float32, copy=False))

        if modality == "diagnosis":
            output_events: list[int] = []
            output_concepts: list[int] = []
            code_col = value_columns[0]
            for source, stay_row, event_id in zip(source_rows, stay_rows, event_ids):
                for code in normalize_codes(chunk.iloc[source][code_col]):
                    concept = concept_maps["diagnosis"].get(code)
                    if concept is None:
                        continue
                    target_index = label_index.get(code)
                    if target_index is not None and y_val[stay_row, target_index] == 1:
                        removed_overlap += 1
                        continue
                    output_events.append(int(event_id))
                    output_concepts.append(int(concept))
            if output_events:
                concept_event_parts.append(np.asarray(output_events, dtype=np.int64))
                concept_id_parts.append(np.asarray(output_concepts, dtype=np.int16))
            continue

        if modality == "medication":
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            concepts = np.asarray(
                [concept_maps["medication"].get(name, -1) for name in names],
                dtype=np.int64,
            )
        elif modality == "treatment":
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            concepts = np.asarray(
                [
                    concept_maps["treatment"].get(
                        name,
                        concept_maps["treatment"].get(
                            normalize_text(name.split("|")[-1]), -1
                        ),
                    )
                    for name in names
                ],
                dtype=np.int64,
            )
        else:
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            values = pd.to_numeric(
                chunk.iloc[source_rows][value_columns[1]], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
            concept_values = []
            for name, value in zip(names, values):
                threshold = lab_thresholds.get(name)
                if threshold is None or not np.isfinite(value):
                    concept_values.append(-1)
                elif value < threshold["low"]:
                    concept_values.append(concept_maps["lab"][f"{name}:low"])
                elif value > threshold["high"]:
                    concept_values.append(concept_maps["lab"][f"{name}:high"])
                else:
                    concept_values.append(-1)
                    skipped_normal_labs += 1
            concepts = np.asarray(concept_values, dtype=np.int64)
        matched = concepts >= 0
        if matched.any():
            concept_event_parts.append(event_ids[matched])
            concept_id_parts.append(concepts[matched].astype(np.int16, copy=False))

    observation_stay = concatenate(observation_stay_parts, np.dtype(np.int32))
    observation_offset = concatenate(observation_offset_parts, np.dtype(np.float32))
    observation_modality = np.full(
        observation_stay.size, modality_index, dtype=np.int8
    )
    concept_event_id = concatenate(concept_event_parts, np.dtype(np.int64))
    concept_id = concatenate(concept_id_parts, np.dtype(np.int16))
    arrays = {
        "observation_stay": observation_stay,
        "observation_offset": observation_offset,
        "observation_modality": observation_modality,
        "concept_event_id": concept_event_id,
        "concept_id": concept_id,
    }
    stats = {
        "source_file": filename,
        "source_sha256": sha256(path),
        "observation_events": int(observation_stay.size),
        "matched_concept_events": int(concept_id.size),
        "removed_input_target_diagnosis_events": int(removed_overlap),
        "skipped_normal_lab_rows": int(skipped_normal_labs),
        "first_event_id": int(event_offset),
        "last_event_id_exclusive": int(next_event),
    }
    return arrays, stats


def collapsed_matrices(
    observation_stay: np.ndarray,
    observation_offset: np.ndarray,
    observation_modality: np.ndarray,
    concept_event_id: np.ndarray,
    concept_id: np.ndarray,
    stays: int,
    concepts: int,
    bins: int,
    bin_minutes: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    observation_bin = np.clip(
        np.floor(observation_offset / bin_minutes).astype(np.int64), 0, bins - 1
    )
    observations = sparse.coo_matrix(
        (
            np.ones(observation_stay.size, dtype=np.float32),
            (
                observation_stay,
                observation_bin * len(MODALITIES) + observation_modality,
            ),
        ),
        shape=(stays, bins * len(MODALITIES)),
        dtype=np.float32,
    ).tocsr()
    observations.sum_duplicates()
    concept_stay = observation_stay[concept_event_id]
    concept_bin = observation_bin[concept_event_id]
    concept_matrix = sparse.coo_matrix(
        (
            np.ones(concept_id.size, dtype=np.float32),
            (concept_stay, concept_bin * concepts + concept_id),
        ),
        shape=(stays, bins * concepts),
        dtype=np.float32,
    ).tocsr()
    concept_matrix.sum_duplicates()
    return concept_matrix, observations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an anonymous event-level split cache for exact perturbations"
    )
    parser.add_argument(
        "--config",
        default="t_fedgsp/configs/preprocessing/temporal_features.yaml",
    )
    parser.add_argument("--raw-dir")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    raw_value = args.raw_dir or os.environ.get(dataset["raw_dir_env"])
    if not raw_value:
        raise SystemExit(f"set {dataset['raw_dir_env']} or pass --raw-dir")
    raw_dir = Path(raw_value).resolve()
    split_path = (PROJECT_ROOT / dataset["split_path"]).resolve()
    target_path = (PROJECT_ROOT / dataset["target_path"]).resolve()
    vocab_root = (PROJECT_ROOT / dataset["vocab_root"]).resolve()
    meta, y, label_names = load_locked_task(split_path, target_path)
    selected_global = np.flatnonzero(meta["split"].to_numpy() == args.split)
    selected_stays = meta.iloc[selected_global]["patientunitstayid"].to_numpy(
        dtype=np.int64
    )
    y_selected = y[selected_global]
    concepts, concept_maps, lab_thresholds = load_vocabulary(vocab_root)

    observation_stay_parts = []
    observation_offset_parts = []
    observation_modality_parts = []
    concept_event_parts = []
    concept_id_parts = []
    modality_stats = {}
    event_offset = 0
    for modality_index, modality in enumerate(MODALITIES):
        arrays, stats = scan_modality(
            raw_dir,
            modality,
            modality_index,
            selected_stays,
            y_selected,
            label_names,
            concept_maps,
            lab_thresholds,
            int(dataset["input_window_minutes"]),
            int(dataset["chunk_size"]),
            event_offset,
        )
        observation_stay_parts.append(arrays["observation_stay"])
        observation_offset_parts.append(arrays["observation_offset"])
        observation_modality_parts.append(arrays["observation_modality"])
        concept_event_parts.append(arrays["concept_event_id"])
        concept_id_parts.append(arrays["concept_id"])
        event_offset = stats["last_event_id_exclusive"]
        modality_stats[modality] = stats

    observation_stay = concatenate(observation_stay_parts, np.dtype(np.int32))
    observation_offset = concatenate(observation_offset_parts, np.dtype(np.float32))
    observation_modality = concatenate(observation_modality_parts, np.dtype(np.int8))
    concept_event_id = concatenate(concept_event_parts, np.dtype(np.int64))
    concept_id = concatenate(concept_id_parts, np.dtype(np.int16))
    if not np.array_equal(np.arange(observation_stay.size), np.arange(event_offset)):
        raise RuntimeError("event IDs are not contiguous")
    if concept_event_id.size and concept_event_id.max() >= observation_stay.size:
        raise RuntimeError("concept-to-observation event alignment failed")

    rebuilt_x, rebuilt_m = collapsed_matrices(
        observation_stay,
        observation_offset,
        observation_modality,
        concept_event_id,
        concept_id,
        selected_stays.size,
        len(concepts),
        24,
        60,
    )
    cache_root = PROJECT_ROOT / "t_fedgsp/data/temporal_v1"
    original_x = sparse.load_npz(cache_root / "X_time_concept_counts.npz").tocsr()[
        selected_global
    ]
    original_m = sparse.load_npz(cache_root / "M_time_modality_counts.npz").tocsr()[
        selected_global
    ]
    x_difference_nnz = int((rebuilt_x != original_x).nnz)
    m_difference_nnz = int((rebuilt_m != original_m).nnz)
    if x_difference_nnz or m_difference_nnz:
        raise RuntimeError(
            f"raw {args.split} event cache does not reconstruct the locked feature cache: "
            f"X difference nnz={x_difference_nnz}, M difference nnz={m_difference_nnz}"
        )

    output_value = args.output_root or (
        "t_fedgsp/data/validation_event_cache"
        if args.split == "val"
        else "t_fedgsp/data/test_event_cache"
    )
    output_root = (PROJECT_ROOT / output_value).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "events.npz",
        observation_stay=observation_stay,
        observation_offset=observation_offset,
        observation_modality=observation_modality,
        concept_event_id=concept_event_id,
        concept_id=concept_id,
    )
    manifest = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"locked {args.split} partition only",
        "split": args.split,
        "stays": int(selected_stays.size),
        "observation_events": int(observation_stay.size),
        "concept_events": int(concept_id.size),
        "num_concepts": len(concepts),
        "event_ids_contiguous": True,
        "contains_patient_or_stay_ids": False,
        "reconstructs_locked_X_exactly": x_difference_nnz == 0,
        "reconstructs_locked_M_exactly": m_difference_nnz == 0,
        "modality_stats": modality_stats,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
