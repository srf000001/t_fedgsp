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
    """Exact weighted AP for patient-cluster bootstrap samples."""
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


def prediction_path(
    seed: int,
    model: str,
    root_patterns: dict[str, str],
    filenames: dict[str, str],
) -> Path:
    if model not in root_patterns:
        raise ValueError(f"unsupported model: {model}")
    root = root_patterns[model].format(seed=seed)
    filename = filenames[model]
    return PROJECT_ROOT / "t_fedgsp" / "results" / root / "validation_predictions" / filename


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    proposed: np.ndarray,
    pair_cluster: np.ndarray,
    counts: np.ndarray,
) -> dict[str, object]:
    labels_flat = labels.ravel().astype(np.uint8)
    baseline_samples = clustered_micro_ap(
        labels_flat, baseline.ravel(), pair_cluster, counts
    )
    proposed_samples = clustered_micro_ap(
        labels_flat, proposed.ravel(), pair_cluster, counts
    )
    delta = proposed_samples - baseline_samples
    low, high = np.percentile(delta, [2.5, 97.5])
    return {
        "baseline_micro_auprc": float(
            average_precision_score(labels_flat, baseline.ravel())
        ),
        "proposed_micro_auprc": float(
            average_precision_score(labels_flat, proposed.ravel())
        ),
        "delta_micro_auprc": float(
            average_precision_score(labels_flat, proposed.ravel())
            - average_precision_score(labels_flat, baseline.ravel())
        ),
        "delta_ci_95": [float(low), float(high)],
    }


def hospital_analysis(
    labels: np.ndarray,
    client: np.ndarray,
    baseline: np.ndarray,
    proposed: np.ndarray,
) -> dict[str, float | int]:
    rows = []
    for code in np.unique(client):
        selected = client == code
        y = labels[selected]
        if y.sum() == 0:
            continue
        base_ap = float(average_precision_score(y.ravel(), baseline[selected].ravel()))
        prop_ap = float(average_precision_score(y.ravel(), proposed[selected].ravel()))
        rows.append((int(code), base_ap, prop_ap, prop_ap - base_ap))
    differences = np.asarray([row[3] for row in rows], dtype=np.float64)
    test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    bottom_count = max(1, int(np.ceil(0.2 * len(rows))))
    baseline_hospitals = np.asarray([row[1] for row in rows])
    proposed_hospitals = np.asarray([row[2] for row in rows])
    return {
        "hospitals": len(rows),
        "median_paired_delta": float(np.median(differences)),
        "positive_hospital_fraction": float(np.mean(differences > 0)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p": float(test.pvalue),
        "rank_biserial": float(rank_biserial(differences)),
        "baseline_worst_20pct_mean": float(
            np.sort(baseline_hospitals)[:bottom_count].mean()
        ),
        "proposed_worst_20pct_mean": float(
            np.sort(proposed_hospitals)[:bottom_count].mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked-validation analysis for federated confirmation seeds"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument(
        "--graph-root-pattern", default="federated_confirm_seed{seed}_backbone"
    )
    parser.add_argument("--graph-filename", default="fedavg_graph_continue.npy")
    parser.add_argument(
        "--rank1-root-pattern", default="federated_confirm_seed{seed}_rank1"
    )
    parser.add_argument("--rank1-filename", default="fedavg_separable_logitres.npy")
    parser.add_argument(
        "--rank4-root-pattern", default="federated_confirm_seed{seed}_rank4"
    )
    parser.add_argument("--rank4-filename", default="fedavg_tfedgsp_logitres.npy")
    parser.add_argument(
        "--output-root", default="t_fedgsp/results/federated_confirmation_analysis"
    )
    args = parser.parse_args()
    root_patterns = {
        "graph": args.graph_root_pattern,
        "rank1": args.rank1_root_pattern,
        "rank4": args.rank4_root_pattern,
    }
    filenames = {
        "graph": args.graph_filename,
        "rank1": args.rank1_filename,
        "rank4": args.rank4_filename,
    }

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

    predictions: dict[str, list[np.ndarray]] = {"graph": [], "rank1": [], "rank4": []}
    per_seed = []
    for seed in args.seeds:
        seed_predictions = {}
        seed_metrics = {}
        for model in predictions:
            path = prediction_path(seed, model, root_patterns, filenames)
            if not path.exists():
                raise FileNotFoundError(f"missing confirmation prediction: {path}")
            probability = np.load(path)
            if probability.shape != labels.shape:
                raise ValueError(
                    f"prediction alignment failed for seed {seed}, {model}: "
                    f"{probability.shape} != {labels.shape}"
                )
            probability = probability.astype(np.float64)
            seed_predictions[model] = probability
            predictions[model].append(probability)
            seed_metrics[model] = evaluate_multilabel(labels, probability)
        per_seed.append(
            {
                "seed": seed,
                "graph_micro_auprc": seed_metrics["graph"]["micro_auprc"],
                "rank1_micro_auprc": seed_metrics["rank1"]["micro_auprc"],
                "rank4_micro_auprc": seed_metrics["rank4"]["micro_auprc"],
                "rank4_minus_graph": seed_metrics["rank4"]["micro_auprc"]
                - seed_metrics["graph"]["micro_auprc"],
                "rank4_minus_rank1": seed_metrics["rank4"]["micro_auprc"]
                - seed_metrics["rank1"]["micro_auprc"],
                "graph_macro_auprc": seed_metrics["graph"]["macro_auprc"],
                "rank1_macro_auprc": seed_metrics["rank1"]["macro_auprc"],
                "rank4_macro_auprc": seed_metrics["rank4"]["macro_auprc"],
            }
        )

    mean_predictions = {
        model: np.mean(model_predictions, axis=0)
        for model, model_predictions in predictions.items()
    }
    unique_patients, patient_inverse = np.unique(patient, return_inverse=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    probabilities = np.full(unique_patients.size, 1.0 / unique_patients.size)
    counts = rng.multinomial(
        unique_patients.size, probabilities, size=args.bootstrap
    ).astype(np.int16)
    pair_cluster = np.repeat(patient_inverse, labels.shape[1])

    graph_vs_rank4 = paired_bootstrap(
        labels,
        mean_predictions["graph"],
        mean_predictions["rank4"],
        pair_cluster,
        counts,
    )
    rank1_vs_rank4 = paired_bootstrap(
        labels,
        mean_predictions["rank1"],
        mean_predictions["rank4"],
        pair_cluster,
        counts,
    )
    seed_deltas = np.asarray([row["rank4_minus_graph"] for row in per_seed])
    rank_deltas = np.asarray([row["rank4_minus_rank1"] for row in per_seed])
    ci_low = float(graph_vs_rank4["delta_ci_95"][0])
    gate = {
        "all_seed_deltas_positive": bool(np.all(seed_deltas > 0)),
        "mean_seed_delta_at_least_0_001": bool(seed_deltas.mean() >= 0.001),
        "patient_cluster_bootstrap_ci_above_zero": bool(ci_low > 0),
    }
    gate["clean_validation_gate_pass"] = bool(all(gate.values()))
    report = {
        "status": "PASS",
        "scope": "locked validation partition only",
        "test_evaluated": False,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "rank4_minus_graph_seed_mean": float(seed_deltas.mean()),
        "rank4_minus_graph_seed_sample_sd": float(seed_deltas.std(ddof=1)),
        "rank4_minus_rank1_seed_mean": float(rank_deltas.mean()),
        "rank4_minus_rank1_seed_sample_sd": float(rank_deltas.std(ddof=1)),
        "patient_cluster_bootstrap": {
            "clusters": int(unique_patients.size),
            "resamples": args.bootstrap,
            "seed": args.bootstrap_seed,
            "graph_vs_rank4": graph_vs_rank4,
            "rank1_vs_rank4": rank1_vs_rank4,
        },
        "hospital_analysis_graph_vs_rank4": hospital_analysis(
            labels,
            client,
            mean_predictions["graph"],
            mean_predictions["rank4"],
        ),
        "gate": gate,
    }

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "confirmation_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Federated confirmation analysis",
        "",
        "Locked validation only; the test partition was not evaluated.",
        "",
        "| Seed | Graph-GRU | Rank 1 | Rank 4 | Rank 4 - graph | Rank 4 - rank 1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['seed']} | {row['graph_micro_auprc']:.6f} | "
        f"{row['rank1_micro_auprc']:.6f} | {row['rank4_micro_auprc']:.6f} | "
        f"{row['rank4_minus_graph']:+.6f} | {row['rank4_minus_rank1']:+.6f} |"
        for row in per_seed
    )
    lines.extend(
        [
            "",
            f"Mean rank-4 minus graph delta: {seed_deltas.mean():+.6f} +/- "
            f"{seed_deltas.std(ddof=1):.6f} SD.",
            "Patient-cluster bootstrap 95% CI (rank 4 minus graph): "
            f"[{graph_vs_rank4['delta_ci_95'][0]:+.6f}, "
            f"{graph_vs_rank4['delta_ci_95'][1]:+.6f}].",
            f"Clean validation gate: {'PASS' if gate['clean_validation_gate_pass'] else 'FAIL'}.",
        ]
    )
    (output_root / "confirmation_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
