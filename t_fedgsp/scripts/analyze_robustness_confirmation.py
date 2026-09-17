from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate raw-event validation perturbations across model seeds"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--output-root",
        default="t_fedgsp/results/validation_robustness_confirmation_analysis",
    )
    args = parser.parse_args()
    frames = []
    for model_seed in args.seeds:
        path = (
            PROJECT_ROOT
            / "t_fedgsp"
            / "results"
            / f"validation_robustness_strong_seed{model_seed}_fulljoint"
            / "perturbation_metrics.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["model_seed"] = model_seed
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    per_model_seed = (
        data.groupby(["condition", "level", "model", "model_seed"], as_index=False)
        .agg(
            micro_auprc=("micro_auprc", "mean"),
            macro_auprc=("macro_auprc", "mean"),
            micro_auroc=("micro_auroc", "mean"),
            brier=("brier", "mean"),
            ece_15=("ece_15", "mean"),
        )
    )
    aggregate = (
        per_model_seed.groupby(["condition", "level", "model"], as_index=False)
        .agg(
            micro_auprc_mean=("micro_auprc", "mean"),
            micro_auprc_seed_sd=("micro_auprc", "std"),
            macro_auprc_mean=("macro_auprc", "mean"),
            micro_auroc_mean=("micro_auroc", "mean"),
            brier_mean=("brier", "mean"),
            ece_15_mean=("ece_15", "mean"),
        )
    )
    wide = aggregate.pivot(
        index=["condition", "level"], columns="model", values="micro_auprc_mean"
    ).reset_index()
    for baseline in ["graph_gru", "rank1", "rank4"]:
        wide[f"fulljoint_minus_{baseline}"] = wide["fulljoint"] - wide[baseline]

    seed_wide = per_model_seed.pivot(
        index=["condition", "level", "model_seed"],
        columns="model",
        values="micro_auprc",
    ).reset_index()
    seed_wide["fulljoint_minus_graph_gru"] = (
        seed_wide["fulljoint"] - seed_wide["graph_gru"]
    )
    condition_gate = (
        seed_wide.groupby(["condition", "level"], as_index=False)
        .agg(
            mean_delta=("fulljoint_minus_graph_gru", "mean"),
            seed_delta_sd=("fulljoint_minus_graph_gru", "std"),
            minimum_seed_delta=("fulljoint_minus_graph_gru", "min"),
            positive_seed_fraction=(
                "fulljoint_minus_graph_gru",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
        )
    )
    perturb = wide[wide["condition"] != "clean"]
    report = {
        "status": "PASS",
        "scope": "locked validation partition only",
        "test_evaluated": False,
        "model_seeds": args.seeds,
        "perturbation_conditions": int(perturb.shape[0]),
        "mean_fulljoint_minus_graph_over_perturbations": float(
            perturb["fulljoint_minus_graph_gru"].mean()
        ),
        "mean_fulljoint_minus_rank1_over_perturbations": float(
            perturb["fulljoint_minus_rank1"].mean()
        ),
        "all_conditions_positive_vs_graph": bool(
            (condition_gate["minimum_seed_delta"] > 0).all()
        ),
        "minimum_condition_seed_delta_vs_graph": float(
            condition_gate["minimum_seed_delta"].min()
        ),
    }

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    per_model_seed.to_csv(output_root / "per_model_seed_metrics.csv", index=False)
    aggregate.to_csv(output_root / "aggregate_metrics.csv", index=False)
    wide.to_csv(output_root / "aggregate_micro_auprc_wide.csv", index=False)
    condition_gate.to_csv(output_root / "condition_seed_deltas.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Raw-event robustness confirmation",
        "",
        "Locked validation only; test predictions were not evaluated.",
        "",
        "| Condition | Level | Graph-GRU | Rank 1 | Rank 4 | Full joint | Full joint - graph |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in wide.iterrows():
        lines.append(
            f"| {row['condition']} | {row['level']:.3g} | {row['graph_gru']:.6f} | "
            f"{row['rank1']:.6f} | {row['rank4']:.6f} | {row['fulljoint']:.6f} | "
            f"{row['fulljoint_minus_graph_gru']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Mean full-joint minus graph delta across perturbed conditions: "
            f"{report['mean_fulljoint_minus_graph_over_perturbations']:+.6f}.",
            "All condition-by-model-seed deltas positive versus graph: "
            f"{report['all_conditions_positive_vs_graph']}.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
