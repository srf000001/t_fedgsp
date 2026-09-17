from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "t_fedgsp" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfedgsp_metrics import evaluate_multilabel


def clustered_micro_ap(
    labels_flat: np.ndarray,
    scores_flat: np.ndarray,
    pair_cluster: np.ndarray,
    cluster_counts: np.ndarray,
    batch_size: int = 8,
) -> np.ndarray:
    order = np.argsort(-scores_flat, kind="mergesort")
    y = labels_flat[order].astype(np.float64, copy=False)
    clusters = pair_cluster[order]
    output = np.empty(cluster_counts.shape[0], dtype=np.float64)
    for start in range(0, cluster_counts.shape[0], batch_size):
        counts = cluster_counts[start : start + batch_size]
        weights = counts[:, clusters].astype(np.float64, copy=False)
        positive_weight = weights * y[None, :]
        true_positive = np.cumsum(positive_weight, axis=1)
        predicted_positive = np.cumsum(weights, axis=1)
        precision = true_positive / np.maximum(predicted_positive, 1e-12)
        denominator = positive_weight.sum(axis=1)
        output[start : start + counts.shape[0]] = (
            (precision * positive_weight).sum(axis=1) / np.maximum(denominator, 1e-12)
        )
    return output


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only paired analysis for central confirmation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--output-root", default="t_fedgsp/results/central_confirmation_analysis")
    args = parser.parse_args()
    cache_root = PROJECT_ROOT / "t_fedgsp" / "data" / "temporal_v1"
    with np.load(cache_root / "dataset_arrays.npz") as arrays:
        labels_all = arrays["y"].copy()
        split = arrays["split_code"].copy()
        client_all = arrays["client_code"].copy()
        patient_all = arrays["patient_cluster_code"].copy()
    val = np.flatnonzero(split == 1)
    labels = labels_all[val]
    client = client_all[val]
    patient = patient_all[val]

    baseline_predictions = []
    proposed_predictions = []
    per_seed = []
    for seed in args.seeds:
        baseline_path = (
            PROJECT_ROOT
            / "t_fedgsp"
            / "results"
            / f"central_confirm_seed{seed}_graph"
            / "validation_predictions"
            / f"graph_gru_seed{seed}.npy"
        )
        proposed_path = (
            PROJECT_ROOT
            / "t_fedgsp"
            / "results"
            / f"central_confirm_seed{seed}_tfedgsp"
            / "validation_predictions"
            / f"tfedgsp_logitres_seed{seed}.npy"
        )
        if not baseline_path.exists() or not proposed_path.exists():
            raise FileNotFoundError(f"missing confirmation predictions for seed {seed}")
        baseline = np.load(baseline_path)
        proposed = np.load(proposed_path)
        if baseline.shape != labels.shape or proposed.shape != labels.shape:
            raise ValueError(f"prediction alignment failed for seed {seed}")
        baseline_metrics = evaluate_multilabel(labels, baseline)
        proposed_metrics = evaluate_multilabel(labels, proposed)
        per_seed.append(
            {
                "seed": seed,
                "baseline_micro_auprc": baseline_metrics["micro_auprc"],
                "proposed_micro_auprc": proposed_metrics["micro_auprc"],
                "delta_micro_auprc": proposed_metrics["micro_auprc"]
                - baseline_metrics["micro_auprc"],
                "baseline_macro_auprc": baseline_metrics["macro_auprc"],
                "proposed_macro_auprc": proposed_metrics["macro_auprc"],
            }
        )
        baseline_predictions.append(baseline.astype(np.float64))
        proposed_predictions.append(proposed.astype(np.float64))

    baseline_mean = np.mean(baseline_predictions, axis=0)
    proposed_mean = np.mean(proposed_predictions, axis=0)
    point_baseline = float(average_precision_score(labels.ravel(), baseline_mean.ravel()))
    point_proposed = float(average_precision_score(labels.ravel(), proposed_mean.ravel()))
    point_delta = point_proposed - point_baseline

    unique_patients, patient_inverse = np.unique(patient, return_inverse=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    probabilities = np.full(unique_patients.size, 1.0 / unique_patients.size)
    counts = rng.multinomial(unique_patients.size, probabilities, size=args.bootstrap).astype(
        np.int16
    )
    pair_cluster = np.repeat(patient_inverse, labels.shape[1])
    labels_flat = labels.ravel().astype(np.uint8)
    baseline_bootstrap = clustered_micro_ap(
        labels_flat, baseline_mean.ravel(), pair_cluster, counts
    )
    proposed_bootstrap = clustered_micro_ap(
        labels_flat, proposed_mean.ravel(), pair_cluster, counts
    )
    bootstrap_delta = proposed_bootstrap - baseline_bootstrap
    ci_low, ci_high = np.percentile(bootstrap_delta, [2.5, 97.5])

    hospital_rows = []
    for code in np.unique(client):
        selected = client == code
        y = labels[selected]
        if y.sum() == 0:
            continue
        base_ap = float(average_precision_score(y.ravel(), baseline_mean[selected].ravel()))
        proposed_ap = float(average_precision_score(y.ravel(), proposed_mean[selected].ravel()))
        hospital_rows.append((int(code), base_ap, proposed_ap, proposed_ap - base_ap))
    differences = np.asarray([row[3] for row in hospital_rows], dtype=np.float64)
    wilcoxon_result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    bottom_count = max(1, int(np.ceil(0.2 * len(hospital_rows))))
    baseline_hospitals = np.asarray([row[1] for row in hospital_rows])
    proposed_hospitals = np.asarray([row[2] for row in hospital_rows])

    seed_deltas = np.asarray([row["delta_micro_auprc"] for row in per_seed])
    gate = {
        "all_seed_deltas_positive": bool(np.all(seed_deltas > 0)),
        "mean_seed_delta_at_least_0_001": bool(seed_deltas.mean() >= 0.001),
        "cluster_bootstrap_ci_above_zero": bool(ci_low > 0),
    }
    gate["clean_validation_gate_pass"] = bool(all(gate.values()))
    report = {
        "status": "PASS",
        "scope": "locked validation partition only",
        "test_evaluated": False,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "seed_delta_mean": float(seed_deltas.mean()),
        "seed_delta_sample_sd": float(seed_deltas.std(ddof=1)),
        "pooled_probability_average": {
            "baseline_micro_auprc": point_baseline,
            "proposed_micro_auprc": point_proposed,
            "delta_micro_auprc": point_delta,
        },
        "patient_cluster_bootstrap": {
            "clusters": int(unique_patients.size),
            "resamples": args.bootstrap,
            "seed": args.bootstrap_seed,
            "delta_ci_95": [float(ci_low), float(ci_high)],
        },
        "hospital_analysis": {
            "hospitals": len(hospital_rows),
            "median_paired_delta": float(np.median(differences)),
            "wilcoxon_statistic": float(wilcoxon_result.statistic),
            "wilcoxon_p": float(wilcoxon_result.pvalue),
            "rank_biserial": float(rank_biserial(differences)),
            "baseline_worst_20pct_mean": float(np.sort(baseline_hospitals)[:bottom_count].mean()),
            "proposed_worst_20pct_mean": float(np.sort(proposed_hospitals)[:bottom_count].mean()),
        },
        "gate": gate,
    }
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "confirmation_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Central confirmation analysis",
        "",
        "Validation only; the test partition was not evaluated.",
        "",
        "| Seed | Graph-GRU | T-FedGSP | Delta |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['seed']} | {row['baseline_micro_auprc']:.6f} | "
        f"{row['proposed_micro_auprc']:.6f} | {row['delta_micro_auprc']:+.6f} |"
        for row in per_seed
    )
    lines.extend(
        [
            "",
            f"Mean paired delta: {seed_deltas.mean():+.6f} ± {seed_deltas.std(ddof=1):.6f} SD.",
            f"Patient-cluster bootstrap 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}].",
            f"Clean validation gate: {'PASS' if gate['clean_validation_gate_pass'] else 'FAIL'}.",
        ]
    )
    (output_root / "confirmation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
