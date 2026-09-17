from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    PROJECT_ROOT
    / "t_fedgsp"
    / "results"
    / "locked_test_evaluation"
    / "robustness_metrics.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "t_fedgsp"
    / "results"
    / "locked_test_evaluation"
    / "figures"
)


def curve(frame: pd.DataFrame, condition: str, levels: list[float], model: str):
    clean = frame[(frame.condition == "clean") & (frame.model == model)]
    means = [float(clean.micro_auprc.iloc[0])]
    sds = [0.0]
    for level in levels[1:]:
        selected = frame[
            (frame.condition == condition)
            & (frame.model == model)
            & np.isclose(frame.level, level)
        ].micro_auprc.to_numpy(dtype=float)
        means.append(float(selected.mean()))
        sds.append(float(selected.std(ddof=1)))
    return np.asarray(means), np.asarray(sds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot locked-test robustness curves")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    frame = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), sharey=True)
    specifications = [
        ("event_deletion", [0.0, 0.1, 0.3, 0.5], "Deleted events (%)", 100.0),
        (
            "raw_timestamp_jitter",
            [0.0, 30.0, 60.0, 120.0],
            "Jitter SD (min)",
            1.0,
        ),
    ]
    styles = {
        "graph_gru": ("Graph-GRU", "#0072B2", "o", "-"),
        "fulljoint": ("T-FedGSP", "#D55E00", "s", "--"),
    }
    handles = []
    for panel, (ax, spec) in enumerate(zip(axes, specifications)):
        condition, levels, xlabel, scale = spec
        x = np.asarray(levels) * scale
        for model, (label, color, marker, linestyle) in styles.items():
            mean, sd = curve(frame, condition, levels, model)
            handle = ax.errorbar(
                x,
                mean,
                yerr=sd,
                color=color,
                marker=marker,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.7,
                linewidth=1.0,
                linestyle=linestyle,
                capsize=1.6,
                capthick=0.6,
                label=label,
            )
            if panel == 0:
                handles.append(handle)
        ax.set_xlabel(xlabel, labelpad=1.5)
        ax.set_xticks(x)
        ax.set_ylim(0.035, 0.047)
        ax.set_yticks([0.036, 0.040, 0.044])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.5, pad=1.5)
        ax.text(
            -0.15,
            1.02,
            f"({chr(ord('a') + panel)})",
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            va="bottom",
        )
    axes[0].set_ylabel("Micro-AUPRC", labelpad=2.0)
    fig.legend(
        handles,
        ["Graph-GRU", "T-FedGSP"],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.08),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.24, top=0.82, wspace=0.22)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(
            output_dir / f"locked_test_robustness.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.01,
        )
    plt.close(fig)
    print(output_dir / "locked_test_robustness.pdf")


if __name__ == "__main__":
    main()
