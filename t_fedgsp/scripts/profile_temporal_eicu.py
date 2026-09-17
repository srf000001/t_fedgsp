from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TABLE_SPECS: dict[str, dict[str, Any]] = {
    "diagnosis": {
        "file": "diagnosis.csv.gz",
        "stay": ["patientunitstayid", "patientUnitStayID"],
        "offset": ["diagnosisoffset", "diagnosisOffset"],
    },
    "medication": {
        "file": "medication.csv.gz",
        "stay": ["patientunitstayid", "patientUnitStayID"],
        "offset": ["drugstartoffset", "drugStartOffset"],
        "fallback_offset": ["drugorderoffset", "drugOrderOffset"],
    },
    "lab": {
        "file": "lab.csv.gz",
        "stay": ["patientunitstayid", "patientUnitStayID"],
        "offset": ["labresultoffset", "labResultOffset"],
    },
    "treatment": {
        "file": "treatment.csv.gz",
        "stay": ["patientunitstayid", "patientUnitStayID"],
        "offset": ["treatmentoffset", "treatmentOffset"],
    },
}


def find_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if required:
        raise KeyError(f"missing column; expected one of {candidates}, got {columns}")
    return None


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: 0.0 for key in ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"]}
    probs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    keys = ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    vals = np.quantile(values, probs)
    return {key: float(value) for key, value in zip(keys, vals)}


def safe_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values)) if values.size else 0.0
    return float(np.std(values) / mean) if mean > 0 else 0.0


def locate_stays(sorted_stays: np.ndarray, raw_stays: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(raw_stays, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    positions = np.full(numeric.shape[0], -1, dtype=np.int64)
    finite = np.isfinite(numeric)
    if not finite.any():
        return positions
    integer_stays = numeric[finite].astype(np.int64)
    candidates = np.searchsorted(sorted_stays, integer_stays)
    within = candidates < sorted_stays.size
    matches = np.zeros(candidates.shape[0], dtype=bool)
    matches[within] = sorted_stays[candidates[within]] == integer_stays[within]
    finite_rows = np.flatnonzero(finite)
    positions[finite_rows[matches]] = candidates[matches]
    return positions


def read_assignments(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"patientunitstayid", "uniquepid", "hospitalid", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"split file is missing columns: {sorted(missing)}")
    frame = frame[list(required)].copy()
    frame["patientunitstayid"] = pd.to_numeric(frame["patientunitstayid"], errors="raise").astype(np.int64)
    if frame["patientunitstayid"].duplicated().any():
        raise ValueError("split file contains duplicate stay IDs")
    frame = frame.sort_values("patientunitstayid", kind="stable").reset_index(drop=True)
    leakage = frame.groupby("uniquepid", dropna=False)["split"].nunique().max()
    split_counts = frame["split"].value_counts().sort_index().to_dict()
    info = {
        "num_stays": int(frame.shape[0]),
        "num_unique_patients": int(frame["uniquepid"].nunique(dropna=True)),
        "num_clients": int(frame["hospitalid"].nunique(dropna=True)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "max_splits_per_patient": int(leakage),
        "patient_split_leakage": bool(leakage > 1),
    }
    return frame, info


def analyze_targets(path: Path, sorted_stays: np.ndarray) -> dict[str, Any]:
    targets = pd.read_csv(path, low_memory=False)
    stay_col = find_column(targets.columns.tolist(), ["patientunitstayid", "patientUnitStayID"])
    label_cols = [column for column in targets.columns if column.startswith("label__")]
    if not label_cols:
        raise ValueError("target file contains no label__ columns")
    target_stays = pd.to_numeric(targets[stay_col], errors="coerce").dropna().astype(np.int64).to_numpy()
    aligned = int(np.intersect1d(sorted_stays, target_stays, assume_unique=False).size)
    label_values = targets[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    positives_per_stay = label_values.sum(axis=1)
    prevalence = label_values.mean(axis=0)
    return {
        "num_target_rows": int(targets.shape[0]),
        "num_labels": int(len(label_cols)),
        "aligned_stays": aligned,
        "alignment_fraction": float(aligned / max(1, sorted_stays.size)),
        "positive_labels_per_stay": quantiles(positives_per_stay),
        "micro_prevalence": float(label_values.mean()),
        "label_prevalence": quantiles(prevalence),
    }


def profile_event_table(
    raw_dir: Path,
    modality: str,
    sorted_stays: np.ndarray,
    input_window: int,
    base_bin: int,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = TABLE_SPECS[modality]
    path = raw_dir / spec["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    stay_col = find_column(columns, spec["stay"])
    offset_col = find_column(columns, spec["offset"])
    fallback_col = find_column(columns, spec.get("fallback_offset", []), required=False)
    usecols = [stay_col, offset_col] + ([fallback_col] if fallback_col else [])
    num_bins = int(math.ceil(input_window / base_bin))
    counts = np.zeros((sorted_stays.size, num_bins), dtype=np.int32)
    eligible_rows = 0
    missing_offsets = 0
    before_window = 0
    after_window = 0
    window_rows = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        positions = locate_stays(sorted_stays, chunk[stay_col])
        eligible = positions >= 0
        eligible_rows += int(eligible.sum())
        offsets = pd.to_numeric(chunk[offset_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        if fallback_col:
            fallback = pd.to_numeric(chunk[fallback_col], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            offsets = np.where(np.isfinite(offsets), offsets, fallback)
        finite = np.isfinite(offsets)
        missing_offsets += int((eligible & ~finite).sum())
        before_window += int((eligible & finite & (offsets < 0)).sum())
        after_window += int((eligible & finite & (offsets > input_window)).sum())
        in_window = eligible & finite & (offsets >= 0) & (offsets <= input_window)
        if not in_window.any():
            continue
        rows = positions[in_window]
        bins = np.floor(offsets[in_window] / base_bin).astype(np.int64)
        bins = np.clip(bins, 0, num_bins - 1)
        np.add.at(counts, (rows, bins), 1)
        window_rows += int(rows.size)

    stay_counts = counts.sum(axis=1)
    stats = {
        "file": spec["file"],
        "file_size_bytes": int(path.stat().st_size),
        "eligible_raw_rows": eligible_rows,
        "missing_offset_rows": missing_offsets,
        "negative_offset_rows": before_window,
        "post_window_rows": after_window,
        "window_event_rows": window_rows,
        "stays_with_window_event": int((stay_counts > 0).sum()),
        "stay_coverage": float((stay_counts > 0).mean()),
        "events_per_stay": quantiles(stay_counts),
        "active_base_bins_per_stay": quantiles((counts > 0).sum(axis=1)),
    }
    return counts, stats


def candidate_bin_statistics(
    total_counts: np.ndarray, input_window: int, base_bin: int, candidates: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in candidates:
        if width % base_bin != 0 or input_window % width != 0:
            raise ValueError(f"candidate width {width} must divide {input_window} and be a multiple of {base_bin}")
        factor = width // base_bin
        bins = input_window // width
        aggregated = total_counts.reshape(total_counts.shape[0], bins, factor).sum(axis=2)
        active = aggregated > 0
        active_counts = active.sum(axis=1)
        nonzero_values = aggregated[active]
        rows.append(
            {
                "bin_minutes": int(width),
                "num_bins": int(bins),
                "nonempty_cell_fraction": float(active.mean()),
                "empty_cell_fraction": float(1.0 - active.mean()),
                "stays_with_any_event_fraction": float((active_counts > 0).mean()),
                "active_bins_per_stay": quantiles(active_counts),
                "events_per_active_bin": quantiles(nonzero_values),
            }
        )
    return rows


def client_heterogeneity(
    assignment: pd.DataFrame, modality_counts: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    client_codes, client_values = pd.factorize(assignment["hospitalid"], sort=True)
    num_clients = len(client_values)
    stays_per_client = np.bincount(client_codes, minlength=num_clients).astype(np.float64)
    rows: list[dict[str, Any]] = []
    for modality, counts in modality_counts.items():
        events = counts.sum(axis=1).astype(np.float64)
        events_per_client = np.bincount(client_codes, weights=events, minlength=num_clients)
        rates = np.divide(events_per_client, stays_per_client, out=np.zeros_like(events_per_client), where=stays_per_client > 0)
        item = {"modality": modality, "num_clients": num_clients, "client_rate_cv": safe_cv(rates)}
        item.update({f"events_per_stay_{key}": value for key, value in quantiles(rates).items()})
        rows.append(item)
    return rows


def build_hourly_rows(modality_counts: dict[str, np.ndarray], base_bin: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality, counts in modality_counts.items():
        for index in range(counts.shape[1]):
            rows.append(
                {
                    "modality": modality,
                    "bin_start_minutes": int(index * base_bin),
                    "bin_end_minutes": int((index + 1) * base_bin),
                    "event_rows": int(counts[:, index].sum()),
                    "active_stays": int((counts[:, index] > 0).sum()),
                }
            )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    cohort = report["cohort"]
    target = report["targets"]
    lines = [
        "# Full eICU temporal-event EDA report",
        "",
        f"- Generated: {report['generated_at_utc']}",
        f"- Dataset: {report['dataset_name']}",
        "- Format: compressed CSV tables, processed in chunks",
        f"- Input window: 0–{report['input_window_minutes']} minutes (inclusive)",
        "- Privacy: aggregate-only output; no raw row or patient/stay identifier written",
        "",
        "## Basic information",
        "",
        f"- Locked stays: {cohort['num_stays']:,}",
        f"- Unique patients: {cohort['num_unique_patients']:,}",
        f"- Hospital clients: {cohort['num_clients']}",
        f"- Split counts: {cohort['split_counts']}",
        f"- Patient split leakage: {'FAIL' if cohort['patient_split_leakage'] else 'PASS'}",
        f"- Target alignment: {target['aligned_stays']:,}/{cohort['num_stays']:,}",
        f"- Labels: {target['num_labels']}; micro prevalence: {target['micro_prevalence']:.6f}",
        "",
        "## File type details",
        "",
        "The source consists of gzip-compressed relational CSV tables from eICU-CRD v2.0. "
        "Only stay ID and event-offset columns are loaded for this timing audit. Medication uses "
        "`drugstartoffset` with `drugorderoffset` as fallback, matching the inherited preprocessing policy.",
        "",
        "## Data analysis",
        "",
        "### Event modality coverage",
        "",
        "| Modality | Eligible rows | 0–24 h rows | Stay coverage | Median events/stay | P95 events/stay | Missing offsets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for modality, item in report["modalities"].items():
        lines.append(
            f"| {modality} | {item['eligible_raw_rows']:,} | {item['window_event_rows']:,} | "
            f"{item['stay_coverage']:.3f} | {item['events_per_stay']['median']:.1f} | "
            f"{item['events_per_stay']['p95']:.1f} | {item['missing_offset_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "### Candidate temporal resolutions",
            "",
            "| Bin width | Bins | Nonempty cells | Median active bins/stay | P95 events/active bin |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["candidate_bins"]:
        lines.append(
            f"| {row['bin_minutes']} min | {row['num_bins']} | {row['nonempty_cell_fraction']:.3f} | "
            f"{row['active_bins_per_stay']['median']:.1f} | {row['events_per_active_bin']['p95']:.1f} |"
        )
    total = report["modalities"]["total"]
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"- At least one 0–24 h event is present for {total['stay_coverage']:.1%} of locked stays.",
            f"- The median stay has {total['events_per_stay']['median']:.1f} event rows; the P95 is "
            f"{total['events_per_stay']['p95']:.1f}, confirming strong sampling-density variation.",
            f"- Across-client event-rate CV is {report['total_client_rate_cv']:.3f}; hospital sampling "
            "heterogeneity must be reported rather than assumed away.",
            f"- The patient-level split gate is {'not satisfied' if cohort['patient_split_leakage'] else 'satisfied'}.",
            "",
            "## Recommendations",
            "",
            "1. Choose the finest candidate width whose sparsity is computationally manageable; do not describe binned inputs as continuous time.",
            "2. Preserve event masks/counts so any sampling-aware normalization uses observed density rather than imputed zeros.",
            "3. Construct vocabularies and any concept graph from training stays only; remove per-stay input diagnoses that duplicate future targets.",
            "4. Evaluate at 6 h, 12 h, and 24 h observation windows and under controlled event deletion and timestamp jitter.",
            "5. Report multi-seed aggregate metrics and across-hospital dispersion; never export patient-level predictions in paper artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def run_profile(
    raw_dir: Path,
    split_path: Path,
    target_path: Path,
    output_root: Path,
    dataset_name: str,
    input_window: int,
    base_bin: int,
    candidates: list[int],
    chunk_size: int,
) -> dict[str, Any]:
    assignment, cohort_info = read_assignments(split_path)
    sorted_stays = assignment["patientunitstayid"].to_numpy(dtype=np.int64)
    target_info = analyze_targets(target_path, sorted_stays)
    modality_counts: dict[str, np.ndarray] = {}
    modality_stats: dict[str, dict[str, Any]] = {}
    for modality in TABLE_SPECS:
        counts, stats = profile_event_table(
            raw_dir, modality, sorted_stays, input_window, base_bin, chunk_size
        )
        modality_counts[modality] = counts
        modality_stats[modality] = stats
    total_counts = np.zeros_like(next(iter(modality_counts.values())))
    for counts in modality_counts.values():
        total_counts += counts
    modality_counts["total"] = total_counts
    total_stay_counts = total_counts.sum(axis=1)
    modality_stats["total"] = {
        "file": "aggregate of four event tables",
        "file_size_bytes": int(sum(item["file_size_bytes"] for item in modality_stats.values())),
        "eligible_raw_rows": int(sum(item["eligible_raw_rows"] for item in modality_stats.values())),
        "missing_offset_rows": int(sum(item["missing_offset_rows"] for item in modality_stats.values())),
        "negative_offset_rows": int(sum(item["negative_offset_rows"] for item in modality_stats.values())),
        "post_window_rows": int(sum(item["post_window_rows"] for item in modality_stats.values())),
        "window_event_rows": int(total_counts.sum()),
        "stays_with_window_event": int((total_stay_counts > 0).sum()),
        "stay_coverage": float((total_stay_counts > 0).mean()),
        "events_per_stay": quantiles(total_stay_counts),
        "active_base_bins_per_stay": quantiles((total_counts > 0).sum(axis=1)),
    }
    bin_stats = candidate_bin_statistics(total_counts, input_window, base_bin, candidates)
    heterogeneity = client_heterogeneity(assignment, modality_counts)
    total_client_cv = next(row["client_rate_cv"] for row in heterogeneity if row["modality"] == "total")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "raw_tables_location": str(raw_dir.resolve()),
        "input_window_minutes": input_window,
        "base_bin_minutes": base_bin,
        "candidate_bin_minutes": candidates,
        "chunk_size": chunk_size,
        "privacy": {
            "aggregate_only": True,
            "raw_rows_written": False,
            "patient_or_stay_ids_written": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "cohort": cohort_info,
        "targets": target_info,
        "modalities": modality_stats,
        "candidate_bins": bin_stats,
        "client_heterogeneity": heterogeneity,
        "total_client_rate_cv": total_client_cv,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "temporal_eda.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "temporal_eda.md").write_text(render_markdown(report), encoding="utf-8")
    pd.DataFrame(bin_stats).to_json(
        output_root / "candidate_bin_statistics.json", orient="records", indent=2
    )
    pd.DataFrame(heterogeneity).to_csv(output_root / "client_heterogeneity_summary.csv", index=False)
    pd.DataFrame(build_hourly_rows(modality_counts, base_bin)).to_csv(
        output_root / "event_time_distribution.csv", index=False
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate-only timing profile for the locked full-eICU cohort")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir", default=None, help="Override the raw directory environment variable")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path.cwd().resolve()
    dataset = config["dataset"]
    raw_value = args.raw_dir or os.environ.get(dataset["raw_dir_env"])
    if not raw_value:
        raise SystemExit(f"set {dataset['raw_dir_env']} or pass --raw-dir")
    raw_dir = Path(raw_value).expanduser().resolve()
    report = run_profile(
        raw_dir=raw_dir,
        split_path=(project_root / dataset["split_path"]).resolve(),
        target_path=(project_root / dataset["target_path"]).resolve(),
        output_root=(project_root / config["outputs"]["root"]).resolve(),
        dataset_name=str(dataset["name"]),
        input_window=int(dataset["input_window_minutes"]),
        base_bin=int(dataset["base_bin_minutes"]),
        candidates=[int(value) for value in dataset["candidate_bin_minutes"]],
        chunk_size=int(dataset["chunk_size"]),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stays": report["cohort"]["num_stays"],
                "clients": report["cohort"]["num_clients"],
                "window_events": report["modalities"]["total"]["window_event_rows"],
                "output": str((project_root / config["outputs"]["root"]).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

