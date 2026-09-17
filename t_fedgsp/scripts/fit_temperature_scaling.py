from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "t_fedgsp" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfedgsp_metrics import evaluate_multilabel


def fit_scalar_temperature(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    targets = labels.astype(np.float64)

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - targets * scaled))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = (np.log(clipped) - np.log1p(-clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit validation-only scalar temperature scaling")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="label=path-to-validation-probabilities.npy",
    )
    parser.add_argument(
        "--output", default="t_fedgsp/results/temperature_scaling/temperature_scaling.json"
    )
    args = parser.parse_args()
    with np.load(PROJECT_ROOT / "t_fedgsp/data/temporal_v1/dataset_arrays.npz") as arrays:
        labels = arrays["y"][arrays["split_code"] == 1].copy()
    report = {"scope": "validation only", "test_evaluated": False, "models": {}}
    for specification in args.prediction:
        if "=" not in specification:
            raise ValueError("prediction must use label=path syntax")
        label, raw_path = specification.split("=", 1)
        path = (PROJECT_ROOT / raw_path).resolve()
        probabilities = np.load(path)
        if probabilities.shape != labels.shape:
            raise ValueError(f"prediction alignment failed for {label}: {probabilities.shape}")
        temperature = fit_scalar_temperature(labels, probabilities)
        before = evaluate_multilabel(labels, probabilities)
        calibrated = apply_temperature(probabilities, temperature)
        after = evaluate_multilabel(labels, calibrated)
        report["models"][label] = {
            "source": str(path.relative_to(PROJECT_ROOT)),
            "temperature": temperature,
            "before": before,
            "after": after,
        }
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
