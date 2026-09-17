from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "t_fedgsp" / "results"
SEEDS = (1, 2, 3)
HYBRID_STAGE = "fedavg_tfedgsp_hybrid_adapter"
GENERIC_STAGE = "fedavg_generic_hybrid_adapter"
FULL_STAGE = "fedavg_graph_gru_continue"


def result_by_stage(path: Path, stage: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(item for item in payload["results"] if item["stage"] == stage)


def full_model_at_budget(seed: int, budget: int) -> dict:
    history = pd.read_csv(
        RESULTS / f"review_control_graph_sgdfedadam_seed{seed}" / "round_history.csv"
    )
    rows = history[history["stage"] == FULL_STAGE].copy()
    per_round = int(rows.loc[rows["round"] == 1, "communication_bytes"].iloc[0])
    max_round = min(int(rows["round"].max()), budget // per_round)
    eligible = rows[rows["round"] <= max_round]
    selected = eligible.loc[eligible["val_micro_auprc"].idxmax()]
    return {
        "rounds_affordable": int(max_round),
        "selected_round": int(selected["round"]),
        "communication_bytes": int(max_round * per_round),
        "micro_auprc": float(selected["val_micro_auprc"]),
        "macro_auprc": float(selected["val_macro_auprc"]),
        "micro_auroc": float(selected["val_micro_auroc"]),
    }


def main() -> None:
    rows = []
    for seed in SEEDS:
        summary_path = (
            RESULTS / f"review_control_hybrid_adapters_seed{seed}" / "summary.json"
        )
        hybrid = result_by_stage(summary_path, HYBRID_STAGE)
        generic = result_by_stage(summary_path, GENERIC_STAGE)
        budget = int(hybrid["communication_bytes"])
        full = full_model_at_budget(seed, budget)
        rows.append(
            {
                "seed": seed,
                "hybrid_micro_auprc": float(hybrid["val_micro_auprc"]),
                "generic_micro_auprc": float(generic["val_micro_auprc"]),
                "full_budget_micro_auprc": full["micro_auprc"],
                "hybrid_minus_generic": float(hybrid["val_micro_auprc"])
                - float(generic["val_micro_auprc"]),
                "hybrid_minus_full_budget": float(hybrid["val_micro_auprc"])
                - full["micro_auprc"],
                "hybrid_best_round": int(hybrid["best_round"]),
                "generic_best_round": int(generic["best_round"]),
                "full_rounds_affordable": full["rounds_affordable"],
            }
        )

    frame = pd.DataFrame(rows)
    first_summary = RESULTS / "review_control_hybrid_adapters_seed1" / "summary.json"
    hybrid_first = result_by_stage(first_summary, HYBRID_STAGE)
    generic_first = result_by_stage(first_summary, GENERIC_STAGE)
    aggregate = {
        "hybrid": {
            "mean_micro_auprc": float(frame["hybrid_micro_auprc"].mean()),
            "sample_sd_micro_auprc": float(frame["hybrid_micro_auprc"].std(ddof=1)),
            "trainable_parameters": int(hybrid_first["trainable_parameters"]),
            "communication_bytes": int(hybrid_first["communication_bytes"]),
        },
        "generic": {
            "mean_micro_auprc": float(frame["generic_micro_auprc"].mean()),
            "sample_sd_micro_auprc": float(frame["generic_micro_auprc"].std(ddof=1)),
            "trainable_parameters": int(generic_first["trainable_parameters"]),
            "communication_bytes": int(generic_first["communication_bytes"]),
        },
        "full_model_at_hybrid_budget": {
            "mean_micro_auprc": float(frame["full_budget_micro_auprc"].mean()),
            "sample_sd_micro_auprc": float(frame["full_budget_micro_auprc"].std(ddof=1)),
            "trainable_parameters": 18365,
            "communication_bytes": 8 * 11606680,
            "rounds": 8,
        },
    }
    output = {
        "status": "PASS",
        "scope": "post-lock validation-only equal-byte and generic-adapter controls",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "rows": rows,
        "aggregate": aggregate,
        "hybrid_minus_generic": {
            "per_seed": frame["hybrid_minus_generic"].tolist(),
            "mean": float(frame["hybrid_minus_generic"].mean()),
            "positive_seeds": int((frame["hybrid_minus_generic"] > 0).sum()),
        },
        "hybrid_minus_full_budget": {
            "per_seed": frame["hybrid_minus_full_budget"].tolist(),
            "mean": float(frame["hybrid_minus_full_budget"].mean()),
            "positive_seeds": int((frame["hybrid_minus_full_budget"] > 0).sum()),
        },
        "interpretation": (
            "The hybrid time--graph adapter is not directionally superior to either "
            "the near-parameter-matched generic adapter or the full Graph-GRU at an "
            "equal final-stage byte budget. The seed-0 screening advantage did not replicate."
        ),
    }

    output_root = RESULTS / "reviewer_budget_controls"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    frame.to_csv(output_root / "per_seed.csv", index=False)

    lines = [
        "# Reviewer-requested equal-byte controls",
        "",
        "All results are validation-only and use seeds 1--3. The locked test was not reopened.",
        "",
        "| Condition | Mean Micro-AUPRC | SD | Trainable parameters | Final-stage bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "hybrid": "Time--graph hybrid adapter (30 rounds)",
        "generic": "Generic hybrid adapter (30 rounds)",
        "full_model_at_hybrid_budget": "Full Graph-GRU (8 affordable rounds)",
    }
    for key in ("hybrid", "generic", "full_model_at_hybrid_budget"):
        item = aggregate[key]
        lines.append(
            f"| {labels[key]} | {item['mean_micro_auprc']:.6f} | "
            f"{item['sample_sd_micro_auprc']:.6f} | {item['trainable_parameters']:,} | "
            f"{item['communication_bytes']:,} |"
        )
    lines.extend(
        [
            "",
            "Hybrid minus generic per seed: "
            + ", ".join(f"{value:+.6f}" for value in output["hybrid_minus_generic"]["per_seed"])
            + f" (mean {output['hybrid_minus_generic']['mean']:+.6f}).",
            "",
            "Hybrid minus equal-byte full model per seed: "
            + ", ".join(
                f"{value:+.6f}" for value in output["hybrid_minus_full_budget"]["per_seed"]
            )
            + f" (mean {output['hybrid_minus_full_budget']['mean']:+.6f}).",
            "",
            "**Decision.** The seed-0 screening result did not replicate. The time--graph "
            "hybrid is retained only as a negative diagnostic control; it is not promoted "
            "as a superior method.",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
