from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    target = y_true.reshape(-1)
    scores = probabilities.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = scores.size
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if not mask.any():
            continue
        result += (mask.sum() / total) * abs(float(target[mask].mean()) - float(scores[mask].mean()))
    return float(result)


def top_k_metrics(y_true: np.ndarray, probabilities: np.ndarray, k: int = 5) -> tuple[float, float]:
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    hits = np.take_along_axis(y_true, top, axis=1).sum(axis=1)
    precision = float(np.mean(hits / k))
    positives = y_true.sum(axis=1)
    valid = positives > 0
    recall = float(np.mean(hits[valid] / positives[valid])) if valid.any() else 0.0
    return precision, recall


def evaluate_multilabel(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    precision5, recall5 = top_k_metrics(y_true, probabilities, 5)
    valid_labels = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < y_true.shape[0])
    return {
        "micro_auprc": float(average_precision_score(y_true.ravel(), probabilities.ravel())),
        "macro_auprc": float(
            np.mean(
                [average_precision_score(y_true[:, index], probabilities[:, index]) for index in np.flatnonzero(valid_labels)]
            )
        ),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), probabilities.ravel())),
        "macro_auroc": float(roc_auc_score(y_true[:, valid_labels], probabilities[:, valid_labels], average="macro")),
        "precision_at_5": precision5,
        "recall_at_5": recall5,
        "brier": float(brier_score_loss(y_true.ravel(), probabilities.ravel())),
        "ece_15": expected_calibration_error(y_true, probabilities, 15),
    }

