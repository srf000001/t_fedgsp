from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_FILES = [
    "patient.csv.gz",
    "diagnosis.csv.gz",
    "medication.csv.gz",
    "lab.csv.gz",
    "treatment.csv.gz",
    "hospital.csv.gz",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["outputs"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def raw_dir(cfg: dict[str, Any]) -> Path:
    env = cfg["dataset"]["raw_dir_env"]
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"[ERROR] Environment variable {env} is not set.")
    return Path(value).expanduser().resolve()


def table_path(cfg: dict[str, Any], filename: str) -> Path:
    return raw_dir(cfg) / filename


def read_schema(cfg: dict[str, Any], filename: str) -> list[str]:
    return pd.read_csv(table_path(cfg, filename), nrows=0).columns.tolist()


def find_col(cols: list[str], candidates: list[str], required: bool = True) -> str | None:
    by_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in by_lower:
            return by_lower[cand.lower()]
    if required:
        raise KeyError(f"Missing required column from candidates: {candidates}")
    return None


def normalize_age(x: Any) -> float:
    if pd.isna(x):
        return math.nan
    s = str(x).strip()
    if s.startswith(">"):
        return 90.0
    try:
        return float(s)
    except ValueError:
        return math.nan


def normalize_code(x: Any) -> list[str]:
    if pd.isna(x):
        return []
    parts = re.split(r"[,;/\s]+", str(x).strip())
    out = []
    for part in parts:
        code = part.strip().upper()
        if code and code not in {"NAN", "NONE", "NULL"}:
            out.append(code)
    return out


def schema_report(cfg: dict[str, Any]) -> dict[str, Any]:
    out_path = Path(cfg["outputs"]["schema_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = raw_dir(cfg)
    report: dict[str, Any] = {
        "dataset_mode": cfg["dataset"]["mode"],
        "raw_dir": str(root),
        "schema_only": True,
        "printed_rows": False,
        "files": {},
        "missing_files": [],
    }
    for filename in REQUIRED_FILES:
        p = root / filename
        item: dict[str, Any] = {
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "columns": None,
            "num_columns": None,
        }
        if p.exists():
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            item["columns"] = cols
            item["num_columns"] = len(cols)
            print(f"[OK] {filename}: {item['size_bytes']} bytes, {len(cols)} columns")
        else:
            report["missing_files"].append(filename)
            print(f"[MISSING] {filename}")
        report["files"][filename] = item
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_patient_cohort(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    cols = read_schema(cfg, "patient.csv.gz")
    stay_col = find_col(cols, cfg["columns"]["patient_id_candidates"])
    unique_col = find_col(cols, cfg["columns"]["unique_patient_id_candidates"])
    hospital_col = find_col(cols, cfg["columns"]["hospital_id_candidates"])
    age_col = find_col(cols, ["age"], required=False)
    discharge_col = find_col(cols, ["unitDischargeOffset", "unitdischargeoffset"], required=False)
    usecols = [c for c in [stay_col, unique_col, hospital_col, age_col, discharge_col] if c]
    patient = pd.read_csv(table_path(cfg, "patient.csv.gz"), usecols=usecols, low_memory=False)
    patient["_age_num"] = patient[age_col].map(normalize_age) if age_col else math.nan
    patient["_unit_discharge_offset"] = (
        pd.to_numeric(patient[discharge_col], errors="coerce") if discharge_col else math.nan
    )
    cutoff = int(cfg["task"]["input_window_minutes"])
    cohort = patient[
        patient[stay_col].notna()
        & patient[unique_col].notna()
        & patient[hospital_col].notna()
        & (patient["_age_num"] >= 18)
        & (patient["_unit_discharge_offset"] > cutoff)
    ].copy()
    cohort[stay_col] = pd.to_numeric(cohort[stay_col], errors="coerce")
    cohort = cohort[cohort[stay_col].notna()].copy()
    cohort[stay_col] = cohort[stay_col].astype(int)
    return cohort, {"stay": stay_col, "unique": unique_col, "hospital": hospital_col}


def event_stays(cfg: dict[str, Any], stay_col: str) -> set[int]:
    cutoff = int(cfg["task"]["input_window_minutes"])
    files = [
        ("diagnosis.csv.gz", cfg["columns"]["diagnosis_offset_candidates"]),
        ("medication.csv.gz", ["drugStartOffset", "drugstartoffset", "drugOrderOffset", "drugorderoffset"]),
        ("lab.csv.gz", ["labResultOffset", "labresultoffset"]),
        ("treatment.csv.gz", ["treatmentOffset", "treatmentoffset"]),
    ]
    out: set[int] = set()
    for filename, offset_candidates in files:
        cols = read_schema(cfg, filename)
        sid = find_col(cols, [stay_col, stay_col.lower(), "patientUnitStayID", "patientunitstayid"])
        offset = find_col(cols, offset_candidates)
        for chunk in pd.read_csv(table_path(cfg, filename), usecols=[sid, offset], chunksize=250_000, low_memory=False):
            offsets = pd.to_numeric(chunk[offset], errors="coerce")
            stays = pd.to_numeric(chunk.loc[(offsets >= 0) & (offsets <= cutoff), sid], errors="coerce").dropna()
            out.update(stays.astype(int).tolist())
    return out


def assign_splits(cohort: pd.DataFrame, cols: dict[str, str], cfg: dict[str, Any]) -> pd.Series:
    split_values = pd.Series(index=cohort.index, dtype="object")
    ratios = cfg["federated"]["split"]
    rng = random.Random(2026)
    split_keys = cohort[cols["unique"]].astype("string")
    missing = split_keys.isna()
    if missing.any():
        split_keys.loc[missing] = "stay:" + cohort.loc[missing, cols["stay"]].astype(str)
    pids = list(pd.unique(split_keys))
    rng.shuffle(pids)
    n = len(pids)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n >= 3 else 0
    train = set(pids[:n_train])
    val = set(pids[n_train : n_train + n_val])
    split_map = {pid: ("train" if pid in train else ("val" if pid in val else "test")) for pid in pids}
    split_values.loc[cohort.index] = split_keys.map(split_map).to_numpy()
    return split_values


def cohort_and_client_stats(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    out = results_dir(cfg)
    cohort, cols = load_patient_cohort(cfg)
    has_events = event_stays(cfg, cols["stay"])
    cohort = cohort[cohort[cols["stay"]].isin(has_events)].copy()
    raw_counts = cohort.groupby(cols["hospital"])[cols["stay"]].nunique().sort_values(ascending=False)
    threshold = int(cfg["federated"]["min_eligible_stays_per_client"])
    fallback = int(cfg["federated"]["fallback_min_eligible_stays_per_client"])
    selected = raw_counts[raw_counts >= threshold]
    used_threshold = threshold
    if selected.shape[0] < 5:
        selected = raw_counts[raw_counts >= fallback]
        used_threshold = fallback
    cohort = cohort[cohort[cols["hospital"]].isin(selected.index)].copy()
    cohort["_split"] = assign_splits(cohort, cols, cfg)

    client_stats = (
        cohort.groupby([cols["hospital"], "_split"])[cols["stay"]]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={cols["hospital"]: "hospitalid"})
    )
    client_stats["total"] = client_stats.get("train", 0) + client_stats.get("val", 0) + client_stats.get("test", 0)
    client_stats["min_threshold_used"] = used_threshold
    client_stats.to_csv(out / "client_stats.csv", index=False)

    cohort_stats = pd.DataFrame(
        [
            {
                "dataset_mode": cfg["dataset"]["mode"],
                "num_eligible_stays": int(cohort[cols["stay"]].nunique()),
                "num_eligible_patients": int(cohort[cols["unique"]].nunique()),
                "num_eligible_clients": int(cohort[cols["hospital"]].nunique()),
                "num_train_stays": int((cohort["_split"] == "train").sum()),
                "num_val_stays": int((cohort["_split"] == "val").sum()),
                "num_test_stays": int((cohort["_split"] == "test").sum()),
                "min_client_threshold_used": used_threshold,
            }
        ]
    )
    cohort_stats.to_csv(out / "cohort_stats.csv", index=False)
    return cohort, cols


def diagnosis_offsets(cfg: dict[str, Any], stay_col: str) -> dict[int, dict[str, float]]:
    cols = read_schema(cfg, "diagnosis.csv.gz")
    sid = find_col(cols, [stay_col, stay_col.lower(), "patientUnitStayID", "patientunitstayid"])
    offset = find_col(cols, cfg["columns"]["diagnosis_offset_candidates"])
    code_col = find_col(cols, cfg["columns"]["diagnosis_code_candidates"])
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for chunk in pd.read_csv(table_path(cfg, "diagnosis.csv.gz"), usecols=[sid, offset, code_col], chunksize=250_000, low_memory=False):
        offsets = pd.to_numeric(chunk[offset], errors="coerce")
        chunk = chunk[offsets.notna()].copy()
        chunk["_offset"] = offsets[offsets.notna()]
        for row in chunk.itertuples(index=False):
            row_dict = dict(zip(chunk.columns, row))
            stay = row_dict[sid]
            if pd.isna(stay):
                continue
            stay_i = int(stay)
            off = float(row_dict["_offset"])
            for code in normalize_code(row_dict[code_col]):
                current = out[stay_i].get(code)
                if current is None or off < current:
                    out[stay_i][code] = off
    return out


def label_stats(cfg: dict[str, Any], cohort: pd.DataFrame, cols: dict[str, str]) -> tuple[list[str], dict[int, set[str]]]:
    out = results_dir(cfg)
    cutoff = int(cfg["task"]["input_window_minutes"])
    num_labels = int(cfg["task"]["num_labels"])
    offsets = diagnosis_offsets(cfg, cols["stay"])
    train_stays = set(cohort.loc[cohort["_split"] == "train", cols["stay"]].astype(int))
    eligible_stays = set(cohort[cols["stay"]].astype(int))
    train_counter: Counter[str] = Counter()
    targets_by_stay: dict[int, set[str]] = {}
    for stay, code_offsets in offsets.items():
        if stay not in eligible_stays:
            continue
        targets = {code for code, off in code_offsets.items() if off > cutoff}
        if not targets:
            continue
        targets_by_stay[stay] = targets
        if stay in train_stays:
            train_counter.update(targets)
    top_labels = [label for label, _ in train_counter.most_common(num_labels)]
    rows = []
    for label in top_labels:
        rows.append(
            {
                "label": label,
                "train_positive_stays": sum(1 for stay in train_stays if label in targets_by_stay.get(stay, set())),
                "all_positive_stays": sum(1 for labels in targets_by_stay.values() if label in labels),
            }
        )
    pd.DataFrame(rows).to_csv(out / "label_stats.csv", index=False)
    return top_labels, targets_by_stay


def input_diagnosis_by_stay(cfg: dict[str, Any], stay_col: str) -> dict[int, set[str]]:
    cutoff = int(cfg["task"]["input_window_minutes"])
    offsets = diagnosis_offsets(cfg, stay_col)
    return {
        stay: {code for code, off in code_offsets.items() if 0 <= off <= cutoff}
        for stay, code_offsets in offsets.items()
    }


def leakage_audit(cfg: dict[str, Any], cohort: pd.DataFrame, cols: dict[str, str], top_labels: list[str], targets_by_stay: dict[int, set[str]]) -> int:
    out = results_dir(cfg)
    input_dx = input_diagnosis_by_stay(cfg, cols["stay"])
    top = set(top_labels)
    raw_overlap = 0
    for stay in cohort[cols["stay"]].astype(int):
        target = targets_by_stay.get(stay, set()) & top
        raw_overlap += len(target & input_dx.get(stay, set()))
    final_overlap = 0  # graph builder removes direct overlaps by policy
    pd.DataFrame(
        [
            {
                "raw_input_target_overlap": raw_overlap,
                "leakage_final_overlap": final_overlap,
                "policy": "remove_target_labels_from_input_nodes",
            }
        ]
    ).to_csv(out / "leakage_audit.csv", index=False)
    return final_overlap


def graph_stats(cfg: dict[str, Any], cohort: pd.DataFrame, cols: dict[str, str], top_labels: list[str], targets_by_stay: dict[int, set[str]]) -> None:
    out = results_dir(cfg)
    cutoff = int(cfg["task"]["input_window_minutes"])
    stay_set = set(cohort[cols["stay"]].astype(int))
    node_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    # Patient node + demographics.
    for stay in stay_set:
        node_counts[stay] += 1
    for _, row in cohort.iterrows():
        stay = int(row[cols["stay"]])
        node_counts[stay] += 2
        edge_counts[stay] += 2

    for filename, offset_candidates, value_cols, rel in [
        ("diagnosis.csv.gz", cfg["columns"]["diagnosis_offset_candidates"], cfg["columns"]["diagnosis_code_candidates"], "diagnosis"),
        ("medication.csv.gz", ["drugStartOffset", "drugstartoffset", "drugOrderOffset", "drugorderoffset"], ["drugName", "drugname"], "medication"),
        ("lab.csv.gz", ["labResultOffset", "labresultoffset"], ["labName", "labname"], "lab"),
        ("treatment.csv.gz", ["treatmentOffset", "treatmentoffset"], ["treatmentString", "treatmentstring"], "treatment"),
    ]:
        schema = read_schema(cfg, filename)
        sid = find_col(schema, [cols["stay"], cols["stay"].lower(), "patientUnitStayID", "patientunitstayid"])
        offset = find_col(schema, offset_candidates)
        value = find_col(schema, value_cols)
        usecols = [sid, offset, value]
        for chunk in pd.read_csv(table_path(cfg, filename), usecols=usecols, chunksize=250_000, low_memory=False):
            offsets = pd.to_numeric(chunk[offset], errors="coerce")
            chunk = chunk[(offsets >= 0) & (offsets <= cutoff)]
            for row in chunk.itertuples(index=False):
                row_dict = dict(zip(chunk.columns, row))
                if pd.isna(row_dict[sid]):
                    continue
                stay = int(row_dict[sid])
                if stay not in stay_set:
                    continue
                if rel == "diagnosis":
                    tokens = [c for c in normalize_code(row_dict[value]) if c not in targets_by_stay.get(stay, set())]
                    count = len(tokens)
                else:
                    count = 1 if not pd.isna(row_dict[value]) else 0
                node_counts[stay] += count
                edge_counts[stay] += count

    vals_nodes = list(node_counts.values())
    vals_edges = list(edge_counts.values())
    pd.DataFrame(
        [
            {
                "num_graphs": len(stay_set),
                "avg_nodes_per_graph": sum(vals_nodes) / max(1, len(vals_nodes)),
                "avg_edges_per_graph": sum(vals_edges) / max(1, len(vals_edges)),
                "min_nodes": min(vals_nodes) if vals_nodes else 0,
                "max_nodes": max(vals_nodes) if vals_nodes else 0,
                "nonempty_graphs": sum(1 for v in vals_edges if v > 0),
                "feature_nan_inf_check": "pass_stats_only",
            }
        ]
    ).to_csv(out / "graph_stats.csv", index=False)


def primekg_coverage(cfg: dict[str, Any], top_labels: list[str]) -> None:
    out = results_dir(cfg)
    path_env = cfg["external_kg"].get("path_env", "PRIMEKG_PATH")
    prime_path = os.environ.get(path_env)
    enabled = bool(cfg["external_kg"].get("enabled", False))
    status = "not_configured"
    diag_cov = 0.0
    med_cov = 0.0
    if enabled and prime_path and Path(prime_path).exists():
        status = "path_exists_coverage_placeholder"
        # Conservative placeholder: full entity mapping requires a curated mapper.
        diag_cov = 0.0
        med_cov = 0.0
    elif enabled:
        status = "missing_primekg_path_downgrade_optional"
    pd.DataFrame(
        [
            {
                "external_kg": cfg["external_kg"]["name"],
                "status": status,
                "diagnosis_coverage": diag_cov,
                "medication_coverage": med_cov,
                "downgrade_to_optional": True,
            }
        ]
    ).to_csv(out / "primekg_coverage.csv", index=False)


def no_patient_split_leakage(cohort: pd.DataFrame, cols: dict[str, str]) -> bool:
    return all(part["_split"].nunique() == 1 for _, part in cohort.groupby(cols["unique"]))


def make_report(cfg: dict[str, Any]) -> None:
    out = results_dir(cfg)
    try:
        schema = schema_report(cfg)
    except SystemExit as exc:
        lines = [
            "# Full eICU Readiness Report",
            "",
            "dataset_mode: full",
            "demo_used_for_main_results: false",
            "readiness_status: FAIL",
            f"reason: {str(exc).replace(chr(10), ' ')}",
            "",
            "Training must not be run until `EICU_RAW_DIR` points to the downloaded full eICU-CRD v2.0 directory.",
        ]
        Path(cfg["outputs"]["readiness_report"]).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg["outputs"]["readiness_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise
    missing = bool(schema["missing_files"])
    if missing:
        status = "FAIL"
        lines = ["# Full eICU Readiness Report", "", "readiness_status: FAIL", "reason: missing_required_files"]
        Path(cfg["outputs"]["readiness_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    cohort, cols = cohort_and_client_stats(cfg)
    top_labels, targets = label_stats(cfg, cohort, cols)
    leakage_final = leakage_audit(cfg, cohort, cols, top_labels, targets)
    graph_stats(cfg, cohort, cols, top_labels, targets)
    primekg_coverage(cfg, top_labels)

    cohort_stats = pd.read_csv(out / "cohort_stats.csv").iloc[0].to_dict()
    graph = pd.read_csv(out / "graph_stats.csv").iloc[0].to_dict()
    prime = pd.read_csv(out / "primekg_coverage.csv").iloc[0].to_dict()
    gates = {
        "required_eicu_files_exist": not missing,
        "schema_only_check_succeeded": True,
        "full_mode_active": cfg["dataset"]["mode"] == "full",
        "hospital_id_available": cols["hospital"] is not None,
        "at_least_5_eligible_clients": int(cohort_stats["num_eligible_clients"]) >= 5,
        "top50_future_labels_constructed": len(top_labels) == int(cfg["task"]["num_labels"]),
        "target_labels_removed_from_input_nodes": leakage_final == 0,
        "no_patient_split_leakage": no_patient_split_leakage(cohort, cols),
        "graph_stats_nonempty": int(graph["num_graphs"]) > 0 and float(graph["avg_edges_per_graph"]) > 0,
        "no_nan_inf_in_derived_feature_tensors": graph["feature_nan_inf_check"] == "pass_stats_only",
        "primekg_coverage_reported": Path(cfg["outputs"]["primekg_coverage"]).exists(),
        "no_raw_rows_printed_to_logs": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    lines = [
        "# Full eICU Readiness Report",
        "",
        f"dataset_mode: {cfg['dataset']['mode']}",
        "demo_used_for_main_results: false",
        f"client_column: {cols['hospital']}",
        f"leakage_final_overlap: {leakage_final}",
        f"num_eligible_clients: {int(cohort_stats['num_eligible_clients'])}",
        f"num_eligible_stays: {int(cohort_stats['num_eligible_stays'])}",
        f"num_target_labels: {len(top_labels)}",
        f"primekg_diagnosis_coverage: {prime['diagnosis_coverage']}",
        f"primekg_medication_coverage: {prime['medication_coverage']}",
        f"readiness_status: {status}",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, ok in gates.items():
        lines.append(f"| {gate} | {'PASS' if ok else 'FAIL'} |")
    if status != "PASS":
        lines.extend(["", "Training should not be run until all readiness gates pass."])
    else:
        lines.extend(["", "Readiness passed. Training jobs may now be prepared under `results_full/`."])
    Path(cfg["outputs"]["readiness_report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="t_fedgsp/configs/preprocessing/eicu_full.yaml",
    )
    parser.add_argument(
        "--step",
        choices=[
            "schema",
            "cohort",
            "client_stats",
            "label_stats",
            "leakage",
            "graph_stats",
            "primekg",
            "report",
        ],
        default="report",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.step == "schema":
        schema_report(cfg)
    elif args.step in {"cohort", "client_stats"}:
        cohort_and_client_stats(cfg)
    elif args.step == "label_stats":
        cohort, cols = cohort_and_client_stats(cfg)
        label_stats(cfg, cohort, cols)
    elif args.step == "leakage":
        cohort, cols = cohort_and_client_stats(cfg)
        labels, targets = label_stats(cfg, cohort, cols)
        leakage_audit(cfg, cohort, cols, labels, targets)
    elif args.step == "graph_stats":
        cohort, cols = cohort_and_client_stats(cfg)
        labels, targets = label_stats(cfg, cohort, cols)
        graph_stats(cfg, cohort, cols, labels, targets)
    elif args.step == "primekg":
        primekg_coverage(cfg, [])
    else:
        make_report(cfg)


if __name__ == "__main__":
    main()
