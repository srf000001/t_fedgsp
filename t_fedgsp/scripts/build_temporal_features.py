from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import yaml

from profile_temporal_eicu import find_column, locate_stays


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def normalize_codes(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [
        part.strip().upper()
        for part in re.split(r"[,;/\s]+", str(value).strip())
        if part.strip() and part.strip().upper() not in {"NAN", "NONE", "NULL"}
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_task(split_path: Path, target_path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    split = pd.read_csv(split_path, low_memory=False)
    required = ["patientunitstayid", "uniquepid", "hospitalid", "split"]
    missing = set(required) - set(split.columns)
    if missing:
        raise ValueError(f"split file missing {sorted(missing)}")
    split = split[required].copy()
    split["patientunitstayid"] = pd.to_numeric(split["patientunitstayid"], errors="raise").astype(np.int64)
    if split["patientunitstayid"].duplicated().any():
        raise ValueError("duplicate stay ID in split file")
    if split.groupby("uniquepid", dropna=False)["split"].nunique().max() > 1:
        raise ValueError("patient-level split leakage detected")
    targets = pd.read_csv(target_path, low_memory=False)
    target_stay = find_column(targets.columns.tolist(), ["patientunitstayid", "patientUnitStayID"])
    label_columns = [column for column in targets.columns if column.startswith("label__")]
    if len(label_columns) != 50:
        raise ValueError(f"expected 50 labels, found {len(label_columns)}")
    target_frame = targets[[target_stay] + label_columns].copy().rename(columns={target_stay: "patientunitstayid"})
    target_frame["patientunitstayid"] = pd.to_numeric(
        target_frame["patientunitstayid"], errors="raise"
    ).astype(np.int64)
    merged = split.merge(target_frame, on="patientunitstayid", how="inner", validate="one_to_one")
    if merged.shape[0] != split.shape[0]:
        raise ValueError(f"target alignment failed: {merged.shape[0]}/{split.shape[0]}")
    merged = merged.sort_values("patientunitstayid", kind="stable").reset_index(drop=True)
    y = merged[label_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.uint8)
    return merged[required], y, [column.removeprefix("label__") for column in label_columns]


def load_vocabulary(vocab_root: Path) -> tuple[list[str], dict[str, dict[str, int]], dict[str, Any]]:
    diagnosis = [str(value).upper() for value in load_json(vocab_root / "diagnosis_vocab.json")]
    medication = [normalize_text(value) for value in load_json(vocab_root / "medication_vocab.json")]
    treatment = [normalize_text(value) for value in load_json(vocab_root / "treatment_vocab.json")]
    thresholds = load_json(vocab_root / "lab_thresholds.json")
    thresholds = {
        normalize_text(name): {"low": float(values["low"]), "high": float(values["high"])}
        for name, values in thresholds.items()
    }
    concepts: list[str] = []
    maps: dict[str, dict[str, int]] = {}
    for modality, values in [
        ("diagnosis", diagnosis),
        ("medication", medication),
    ]:
        maps[modality] = {}
        for value in values:
            maps[modality][value] = len(concepts)
            concepts.append(f"{modality}:{value}")
    maps["lab"] = {}
    for name in thresholds:
        for bucket in ["low", "high"]:
            key = f"{name}:{bucket}"
            maps["lab"][key] = len(concepts)
            concepts.append(f"lab:{key}")
    maps["treatment"] = {}
    for value in treatment:
        maps["treatment"][value] = len(concepts)
        concepts.append(f"treatment:{value}")
    if len(concepts) != len(set(concepts)):
        raise ValueError("concept vocabulary contains duplicate tokens")
    return concepts, maps, thresholds


def resolve_offsets(chunk: pd.DataFrame, primary: str, fallback: str | None = None) -> np.ndarray:
    offsets = pd.to_numeric(chunk[primary], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    if fallback:
        fallback_values = pd.to_numeric(chunk[fallback], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        offsets = np.where(np.isfinite(offsets), offsets, fallback_values)
    return offsets


def window_mask(
    positions: np.ndarray, offsets: np.ndarray, input_window: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = (positions >= 0) & np.isfinite(offsets) & (offsets >= 0) & (offsets <= input_window)
    return keep, np.clip(offsets[keep], 0, input_window)


def make_sparse(
    rows_parts: list[np.ndarray], cols_parts: list[np.ndarray], shape: tuple[int, int]
) -> sparse.csr_matrix:
    if not rows_parts:
        return sparse.csr_matrix(shape, dtype=np.float32)
    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    data = np.ones(rows.size, dtype=np.float32)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_modality_matrix(
    raw_dir: Path,
    modality: str,
    sorted_stays: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    concept_maps: dict[str, dict[str, int]],
    lab_thresholds: dict[str, dict[str, float]],
    n_concepts: int,
    input_window: int,
    base_bin: int,
    chunk_size: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    specs = {
        "diagnosis": ("diagnosis.csv.gz", ["diagnosisoffset"], ["icd9code"]),
        "medication": ("medication.csv.gz", ["drugstartoffset", "drugorderoffset"], ["drugname"]),
        "lab": ("lab.csv.gz", ["labresultoffset"], ["labname", "labresult"]),
        "treatment": ("treatment.csv.gz", ["treatmentoffset"], ["treatmentstring"]),
    }
    filename, offset_candidates, value_candidates = specs[modality]
    path = raw_dir / filename
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    stay_col = find_column(columns, ["patientunitstayid", "patientUnitStayID"])
    primary = find_column(columns, [offset_candidates[0]])
    fallback = find_column(columns, [offset_candidates[1]], required=False) if len(offset_candidates) > 1 else None
    value_columns = [find_column(columns, [candidate]) for candidate in value_candidates]
    usecols = [stay_col, primary] + ([fallback] if fallback else []) + value_columns
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    observation_bins: list[np.ndarray] = []
    window_rows = 0
    matched_events = 0
    removed_overlap = 0
    skipped_normal_labs = 0
    num_bins = int(np.ceil(input_window / base_bin))
    label_index = {label.upper(): index for index, label in enumerate(label_names)}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        positions = locate_stays(sorted_stays, chunk[stay_col])
        offsets = resolve_offsets(chunk, primary, fallback)
        keep, kept_offsets = window_mask(positions, offsets, input_window)
        if not keep.any():
            continue
        source_rows = np.flatnonzero(keep)
        stay_rows = positions[keep]
        bins = np.clip(np.floor(kept_offsets / base_bin).astype(np.int64), 0, num_bins - 1)
        window_rows += int(source_rows.size)
        observation_rows.append(stay_rows.astype(np.int64, copy=False))
        observation_bins.append(bins.astype(np.int64, copy=False))

        if modality == "diagnosis":
            output_rows: list[int] = []
            output_cols: list[int] = []
            code_col = value_columns[0]
            for source, stay_row, time_bin in zip(source_rows, stay_rows, bins):
                for code in normalize_codes(chunk.iloc[source][code_col]):
                    concept = concept_maps["diagnosis"].get(code)
                    if concept is None:
                        continue
                    target_index = label_index.get(code)
                    if target_index is not None and y[stay_row, target_index] == 1:
                        removed_overlap += 1
                        continue
                    output_rows.append(int(stay_row))
                    output_cols.append(int(time_bin * n_concepts + concept))
            if output_rows:
                rows_parts.append(np.asarray(output_rows, dtype=np.int64))
                cols_parts.append(np.asarray(output_cols, dtype=np.int64))
                matched_events += len(output_rows)
            continue

        if modality == "medication":
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            concepts = np.asarray([concept_maps["medication"].get(name, -1) for name in names], dtype=np.int64)
        elif modality == "treatment":
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            concepts = np.asarray(
                [
                    concept_maps["treatment"].get(
                        name, concept_maps["treatment"].get(normalize_text(name.split("|")[-1]), -1)
                    )
                    for name in names
                ],
                dtype=np.int64,
            )
        else:
            names = chunk.iloc[source_rows][value_columns[0]].map(normalize_text).tolist()
            values = pd.to_numeric(chunk.iloc[source_rows][value_columns[1]], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            concept_values: list[int] = []
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
            rows_parts.append(stay_rows[matched].astype(np.int64, copy=False))
            cols_parts.append((bins[matched] * n_concepts + concepts[matched]).astype(np.int64, copy=False))
            matched_events += int(matched.sum())

    matrix = make_sparse(rows_parts, cols_parts, (sorted_stays.size, num_bins * n_concepts))
    observations = make_sparse(observation_rows, observation_bins, (sorted_stays.size, num_bins))
    return matrix, observations, {
        "source_file": filename,
        "source_sha256": sha256(path),
        "window_rows": window_rows,
        "matched_events_before_collapse": matched_events,
        "sparse_nonzeros_after_collapse": int(matrix.nnz),
        "observation_bin_nonzeros": int(observations.nnz),
        "removed_input_target_diagnosis_events": removed_overlap,
        "input_target_overlap_policy_applied": modality == "diagnosis",
        "skipped_normal_lab_rows": skipped_normal_labs,
    }


def build_concept_graph(
    x: sparse.csr_matrix,
    train_mask: np.ndarray,
    num_bins: int,
    num_concepts: int,
    min_cooccurrence: int,
    top_k: int,
    add_self_loops: bool,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    coo = x.tocoo()
    bin_rows = coo.row.astype(np.int64) * num_bins + (coo.col // num_concepts)
    concept_cols = coo.col % num_concepts
    binary = sparse.coo_matrix(
        (np.ones(coo.nnz, dtype=np.float32), (bin_rows, concept_cols)),
        shape=(x.shape[0] * num_bins, num_concepts),
    ).tocsr()
    binary.data[:] = 1.0
    binary.eliminate_zeros()
    train_bin_mask = np.repeat(train_mask, num_bins)
    train_binary = binary[train_bin_mask]
    frequencies = np.asarray(train_binary.sum(axis=0)).ravel().astype(np.float64)
    cooccurrence = (train_binary.T @ train_binary).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    coo_c = cooccurrence.tocoo()
    keep = coo_c.data >= min_cooccurrence
    rows = coo_c.row[keep]
    cols = coo_c.col[keep]
    counts = coo_c.data[keep].astype(np.float64)
    denom = np.sqrt(frequencies[rows] * frequencies[cols])
    weights = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
    weighted = sparse.csr_matrix((weights.astype(np.float32), (rows, cols)), shape=cooccurrence.shape)
    selected_rows: list[np.ndarray] = []
    selected_cols: list[np.ndarray] = []
    selected_data: list[np.ndarray] = []
    for row in range(num_concepts):
        start, end = weighted.indptr[row], weighted.indptr[row + 1]
        if start == end:
            continue
        row_data = weighted.data[start:end]
        row_cols = weighted.indices[start:end]
        if row_data.size > top_k:
            chosen = np.argpartition(row_data, -top_k)[-top_k:]
            row_data = row_data[chosen]
            row_cols = row_cols[chosen]
        selected_rows.append(np.full(row_cols.size, row, dtype=np.int64))
        selected_cols.append(row_cols.astype(np.int64, copy=False))
        selected_data.append(row_data.astype(np.float32, copy=False))
    adjacency = sparse.csr_matrix(
        (
            np.concatenate(selected_data) if selected_data else np.array([], dtype=np.float32),
            (
                np.concatenate(selected_rows) if selected_rows else np.array([], dtype=np.int64),
                np.concatenate(selected_cols) if selected_cols else np.array([], dtype=np.int64),
            ),
        ),
        shape=(num_concepts, num_concepts),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    if add_self_loops:
        adjacency = adjacency + sparse.eye(num_concepts, format="csr", dtype=np.float32)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    normalized = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    normalized = normalized.tocsr().astype(np.float32)
    off_diagonal_edges = int((adjacency.nnz - (num_concepts if add_self_loops else 0)) // 2)
    components = scipy.sparse.csgraph.connected_components(adjacency, directed=False, return_labels=False)
    return normalized, {
        "source_split": "train",
        "train_stays": int(train_mask.sum()),
        "train_bin_rows": int(train_binary.shape[0]),
        "active_concepts_in_train": int((frequencies > 0).sum()),
        "min_cooccurrence": int(min_cooccurrence),
        "top_k_neighbors": int(top_k),
        "undirected_edges_excluding_self": off_diagonal_edges,
        "connected_components": int(components),
        "normalized_adjacency_nnz": int(normalized.nnz),
    }


def render_readiness(manifest: dict[str, Any]) -> str:
    checks = manifest["readiness_checks"]
    lines = [
        "# Temporal feature readiness report",
        "",
        f"- Generated: {manifest['generated_at_utc']}",
        f"- Dataset: {manifest['dataset']}",
        f"- Shape: {manifest['feature_matrix']['shape']}",
        f"- Sparse nonzeros: {manifest['feature_matrix']['nnz']:,}",
        f"- Modality-mask shape: {manifest['observation_matrix']['shape']}",
        f"- Stays without any raw observation: {manifest['observation_matrix']['zero_observation_stays']}",
        f"- Concepts: {manifest['feature_matrix']['num_concepts']}",
        f"- One-hour bins: {manifest['feature_matrix']['num_bins']}",
        f"- Train-only concept graph edges: {manifest['concept_graph']['undirected_edges_excluding_self']:,}",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Leakage and privacy boundary",
            "",
            f"- Input diagnosis events duplicating a future target were checked and removed; observed removals: "
            f"{manifest['removed_input_target_diagnosis_events']:,}.",
            "- Split and label alignment are inherited from the locked full-eICU task and rechecked here.",
            "- No stay ID, patient ID, or raw row is saved in the new cache.",
            "- `dataset_arrays.npz` contains only aligned labels, split codes, anonymous client codes, and anonymous patient-cluster codes for local training/bootstrap inference.",
            "",
            "## Intended use",
            "",
            "This cache is a method-neutral one-hour event tensor. Candidate models may aggregate adjacent bins, "
            "but must report the resulting discrete resolution and may not call it continuous time.",
            "",
        ]
    )
    return "\n".join(lines)


def run_build(config: dict[str, Any], project_root: Path, raw_dir: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    split_path = (project_root / dataset["split_path"]).resolve()
    target_path = (project_root / dataset["target_path"]).resolve()
    vocab_root = (project_root / dataset["vocab_root"]).resolve()
    output_root = (project_root / config["outputs"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    meta, y, labels = load_locked_task(split_path, target_path)
    sorted_stays = meta["patientunitstayid"].to_numpy(dtype=np.int64)
    concepts, maps, thresholds = load_vocabulary(vocab_root)
    input_window = int(dataset["input_window_minutes"])
    base_bin = int(dataset["base_bin_minutes"])
    num_bins = int(np.ceil(input_window / base_bin))
    matrices: list[sparse.csr_matrix] = []
    observation_matrices: list[sparse.csr_matrix] = []
    modality_stats: dict[str, Any] = {}
    for modality in ["diagnosis", "medication", "lab", "treatment"]:
        matrix, observations, stats = build_modality_matrix(
            raw_dir,
            modality,
            sorted_stays,
            y,
            labels,
            maps,
            thresholds,
            len(concepts),
            input_window,
            base_bin,
            int(dataset["chunk_size"]),
        )
        matrices.append(matrix)
        observation_matrices.append(observations)
        modality_stats[modality] = stats
    x = sum(matrices[1:], matrices[0]).tocsr().astype(np.float32)
    x.sum_duplicates()
    observation_rows: list[np.ndarray] = []
    observation_cols: list[np.ndarray] = []
    observation_data: list[np.ndarray] = []
    for modality_index, observations in enumerate(observation_matrices):
        coo_observation = observations.tocoo()
        observation_rows.append(coo_observation.row.astype(np.int64, copy=False))
        observation_cols.append(
            (coo_observation.col * len(observation_matrices) + modality_index).astype(np.int64)
        )
        observation_data.append(coo_observation.data.astype(np.float32, copy=False))
    m = sparse.coo_matrix(
        (
            np.concatenate(observation_data),
            (np.concatenate(observation_rows), np.concatenate(observation_cols)),
        ),
        shape=(x.shape[0], num_bins * len(observation_matrices)),
        dtype=np.float32,
    ).tocsr()
    m.sum_duplicates()
    split_code_map = {"train": 0, "val": 1, "test": 2}
    split_codes = meta["split"].map(split_code_map)
    if split_codes.isna().any():
        raise ValueError("unexpected split value")
    split_codes_np = split_codes.to_numpy(dtype=np.uint8)
    client_codes, _ = pd.factorize(meta["hospitalid"], sort=True)
    patient_cluster_codes, _ = pd.factorize(meta["uniquepid"], sort=True)
    graph_cfg = config["concept_graph"]
    graph, graph_stats = build_concept_graph(
        x,
        split_codes_np == split_code_map[graph_cfg["source_split"]],
        num_bins,
        len(concepts),
        int(graph_cfg["min_cooccurrence"]),
        int(graph_cfg["top_k_neighbors"]),
        bool(graph_cfg["add_self_loops"]),
    )
    sparse.save_npz(output_root / "X_time_concept_counts.npz", x, compressed=True)
    sparse.save_npz(output_root / "M_time_modality_counts.npz", m, compressed=True)
    sparse.save_npz(output_root / "concept_graph_normalized.npz", graph, compressed=True)
    np.savez_compressed(
        output_root / "dataset_arrays.npz",
        y=y,
        split_code=split_codes_np,
        client_code=client_codes.astype(np.int16),
        patient_cluster_code=patient_cluster_codes.astype(np.int32),
    )
    (output_root / "concepts.json").write_text(
        json.dumps(concepts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    removed = int(sum(item["removed_input_target_diagnosis_events"] for item in modality_stats.values()))
    zero_stays = int((np.asarray(x.sum(axis=1)).ravel() == 0).sum())
    zero_observation_stays = int((np.asarray(m.sum(axis=1)).ravel() == 0).sum())
    full_eicu_evidence = (
        str(dataset["name"]) == "eICU-CRD v2.0"
        and x.shape[0] == 107306
        and y.shape[1] == 50
        and np.unique(client_codes).size == 79
    )
    require_full_eicu = bool(dataset.get("require_full_eicu", True))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset["name"],
        "main_results_eligible": bool(full_eicu_evidence),
        "demo_or_synthetic_data_used": bool(not full_eicu_evidence),
        "feature_matrix": {
            "shape": [int(x.shape[0]), int(x.shape[1])],
            "nnz": int(x.nnz),
            "density": float(x.nnz / (x.shape[0] * x.shape[1])),
            "num_bins": num_bins,
            "base_bin_minutes": base_bin,
            "num_concepts": len(concepts),
            "zero_feature_stays": zero_stays,
        },
        "observation_matrix": {
            "shape": [int(m.shape[0]), int(m.shape[1])],
            "nnz": int(m.nnz),
            "num_bins": num_bins,
            "modality_order_within_each_bin": ["diagnosis", "medication", "lab", "treatment"],
            "zero_observation_stays": zero_observation_stays,
        },
        "target_matrix": {
            "shape": [int(y.shape[0]), int(y.shape[1])],
            "positive_pairs": int(y.sum()),
            "micro_prevalence": float(y.mean()),
        },
        "clients": int(np.unique(client_codes).size),
        "anonymous_patient_clusters": int(np.unique(patient_cluster_codes).size),
        "splits": {name: int((split_codes_np == code).sum()) for name, code in split_code_map.items()},
        "modality_stats": modality_stats,
        "removed_input_target_diagnosis_events": removed,
        "concept_graph": graph_stats,
        "source_hashes": {
            "split": sha256(split_path),
            "targets": sha256(target_path),
            **{
                path.name: sha256(path)
                for path in sorted(vocab_root.glob("*.json"))
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "privacy": {
            "raw_rows_written": False,
            "patient_or_stay_ids_written": False,
            "training_cache_contains": [
                "sparse features",
                "labels",
                "split codes",
                "anonymous client codes",
                "anonymous patient-cluster codes",
            ],
        },
    }
    checks = {
        "full_eicu_not_demo": bool(full_eicu_evidence or not require_full_eicu),
        "107306_locked_stays": bool(x.shape[0] == 107306 or not require_full_eicu),
        "50_future_labels": y.shape[1] == 50,
        "79_hospital_clients": bool(manifest["clients"] == 79 or not require_full_eicu),
        "84696_anonymous_patient_clusters": bool(
            manifest["anonymous_patient_clusters"] == 84696 or not require_full_eicu
        ),
        "feature_target_alignment": x.shape[0] == y.shape[0],
        "observation_target_alignment": m.shape[0] == y.shape[0],
        "finite_sparse_values": bool(np.isfinite(x.data).all()),
        "nonempty_feature_matrix": x.nnz > 0,
        "finite_observation_values": bool(np.isfinite(m.data).all()),
        "no_empty_observation_stays": zero_observation_stays == 0,
        "train_only_concept_graph": graph_stats["source_split"] == "train",
        "nonempty_concept_graph": graph_stats["undirected_edges_excluding_self"] > 0,
        "input_target_diagnosis_policy_applied": bool(
            modality_stats["diagnosis"]["input_target_overlap_policy_applied"]
        ),
        "no_patient_or_stay_ids_saved": True,
    }
    manifest["readiness_checks"] = checks
    manifest["readiness_status"] = "PASS" if all(checks.values()) else "FAIL"
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "readiness_report.md").write_text(render_readiness(manifest), encoding="utf-8")
    if manifest["readiness_status"] != "PASS":
        raise RuntimeError("temporal feature readiness failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled sparse temporal eICU features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_value = args.raw_dir or os.environ.get(config["dataset"]["raw_dir_env"])
    if not raw_value:
        raise SystemExit(f"set {config['dataset']['raw_dir_env']} or pass --raw-dir")
    manifest = run_build(config, project_root, Path(raw_value).expanduser().resolve())
    print(
        json.dumps(
            {
                "status": manifest["readiness_status"],
                "shape": manifest["feature_matrix"]["shape"],
                "nnz": manifest["feature_matrix"]["nnz"],
                "concept_edges": manifest["concept_graph"]["undirected_edges_excluding_self"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
