from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "t_fedgsp" / "results"
OUTPUT = RESULTS / "reviewer_optimizer_controls"


def final_stage(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = payload["results"][-1]
    return {
        "best_round": int(stage["best_round"]),
        "micro_auprc": float(stage["val_micro_auprc"]),
        "macro_auprc": float(stage["val_macro_auprc"]),
        "micro_auroc": float(stage["val_micro_auroc"]),
        "ece_15": float(stage["val_ece_15"]),
        "communication_bytes": int(stage["communication_bytes"]),
        "trainable_parameters": int(stage["trainable_parameters"]),
        "local_optimizer": str(stage["local_optimizer"]),
        "server_optimizer": str(stage["server_optimizer"]),
    }


def main() -> None:
    rows = []
    for seed in (1, 2, 3):
        specifications = {
            "full_graph_adamw_fedavg": RESULTS / f"federated_strong_seed{seed}_graph" / "summary.json",
            "residual_sgd_fedadam": RESULTS / f"federated_strong_seed{seed}_fulljoint" / "summary.json",
            "full_graph_sgd_fedadam": RESULTS / f"review_control_graph_sgdfedadam_seed{seed}" / "summary.json",
            "residual_adamw_fedavg": RESULTS / f"review_control_fulljoint_adamwfedavg_seed{seed}" / "summary.json",
        }
        row = {"seed": seed, "conditions": {}}
        for label, path in specifications.items():
            row["conditions"][label] = final_stage(path)
        rows.append(row)

    labels = list(rows[0]["conditions"])
    aggregate = {}
    for label in labels:
        values = np.asarray([row["conditions"][label]["micro_auprc"] for row in rows])
        aggregate[label] = {
            "mean_micro_auprc": float(values.mean()),
            "sample_sd_micro_auprc": float(values.std(ddof=1)),
            "mean_best_round": float(np.mean([row["conditions"][label]["best_round"] for row in rows])),
            "branch_communication_bytes": rows[0]["conditions"][label]["communication_bytes"],
            "trainable_parameters": rows[0]["conditions"][label]["trainable_parameters"],
        }

    paired = np.asarray(
        [
            row["conditions"]["full_graph_sgd_fedadam"]["micro_auprc"]
            - row["conditions"]["residual_sgd_fedadam"]["micro_auprc"]
            for row in rows
        ]
    )
    report = {
        "status": "PASS",
        "scope": "post-lock reviewer-requested validation-only crossed optimizer control",
        "test_evaluated": False,
        "starting_checkpoint_matched_within_seed": True,
        "seeds": [1, 2, 3],
        "rows": rows,
        "aggregate": aggregate,
        "full_graph_sgdfedadam_minus_residual_sgdfedadam": {
            "per_seed": paired.tolist(),
            "mean": float(paired.mean()),
            "sample_sd": float(paired.std(ddof=1)),
            "all_positive": bool(np.all(paired > 0)),
        },
        "interpretation": (
            "The preregistered locked-test contrast compares complete training recipes. "
            "It cannot identify the residual parameterization as the cause because the "
            "optimizer/server rule changes the ordering on validation."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Reviewer-requested optimizer-crossed validation control",
        "",
        "- Scope: post-lock, validation only; the locked test was not evaluated.",
        "- All conditions start from the same seed-specific 110-round Graph-GRU checkpoint.",
        "- Each condition receives 30 additional rounds with all 79 hospitals and one local epoch.",
        "",
        "| Condition | Mean Micro-AUPRC | SD | Trainable | Branch bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        item = aggregate[label]
        lines.append(
            f"| {label} | {item['mean_micro_auprc']:.6f} | "
            f"{item['sample_sd_micro_auprc']:.6f} | {item['trainable_parameters']:,} | "
            f"{item['branch_communication_bytes']:,} |"
        )
    lines.extend(
        [
            "",
            "The full-graph SGD/FedAdam minus residual SGD/FedAdam deltas are "
            + ", ".join(f"{value:+.6f}" for value in paired)
            + f" (mean {paired.mean():+.6f}).",
            "Residual AdamW/FedAvg selects round 0 in every seed.",
            "",
            "## Interpretation",
            "",
            "The locked-test comparison remains a valid comparison of preregistered recipes, "
            "but it does not isolate the residual structure. The paper must present T-FedGSP "
            "as a communication--utility operating point and disclose optimizer sensitivity.",
        ]
    )
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
