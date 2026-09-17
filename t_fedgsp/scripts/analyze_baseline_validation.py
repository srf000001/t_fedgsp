from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def final_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("test_evaluated") is not False:
        raise RuntimeError(f"unexpected or test-contaminated result: {path}")
    results = payload.get("results", [])
    if len(results) != 2 or not results[-1]["stage"].endswith("_continue"):
        raise RuntimeError(f"missing completed continuation stage: {path}")
    return results[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate locked validation results for static and temporal baselines"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--output-root",
        default="t_fedgsp/results/federated_baseline_validation_analysis",
    )
    args = parser.parse_args()

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model, directory in [
        ("static_mlp", "federated_static_seed{seed}"),
        ("temporal_gru", "federated_temporal_seed{seed}"),
    ]:
        for seed in args.seeds:
            path = (
                PROJECT_ROOT
                / "t_fedgsp"
                / "results"
                / directory.format(seed=seed)
                / "summary.json"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            result = final_result(path)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "val_micro_auprc": result["val_micro_auprc"],
                    "val_macro_auprc": result["val_macro_auprc"],
                    "val_micro_auroc": result["val_micro_auroc"],
                    "val_macro_auroc": result["val_macro_auroc"],
                    "val_precision_at_5": result["val_precision_at_5"],
                    "val_recall_at_5": result["val_recall_at_5"],
                    "val_brier": result["val_brier"],
                    "val_ece_15": result["val_ece_15"],
                    "trainable_parameters": result["trainable_parameters"],
                    "communication_bytes": result[
                        "total_protocol_communication_bytes"
                    ],
                    "best_round": result["best_round"],
                }
            )

    metric_names = [
        "val_micro_auprc",
        "val_macro_auprc",
        "val_micro_auroc",
        "val_macro_auroc",
        "val_precision_at_5",
        "val_recall_at_5",
        "val_brier",
        "val_ece_15",
    ]
    aggregate = {}
    for model in ["static_mlp", "temporal_gru"]:
        selected = [row for row in rows if row["model"] == model]
        aggregate[model] = {
            name: {
                "mean": float(np.mean([row[name] for row in selected])),
                "sample_sd": float(np.std([row[name] for row in selected], ddof=1)),
            }
            for name in metric_names
        }
        aggregate[model]["trainable_parameters"] = selected[0][
            "trainable_parameters"
        ]
        aggregate[model]["communication_bytes"] = selected[0][
            "communication_bytes"
        ]

    summary = {
        "scope": "locked eICU validation only",
        "test_evaluated": False,
        "seeds": args.seeds,
        "per_seed": rows,
        "aggregate_mean_and_sample_sd": aggregate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_root / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
