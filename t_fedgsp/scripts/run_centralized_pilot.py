from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "t_fedgsp" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfedgsp_metrics import evaluate_multilabel
from tfedgsp_models import ModelConfig, create_model, trainable_parameter_count


RESIDUAL_PARAMETER_PREFIXES = (
    "temporal_raw",
    "graph_raw",
    "rank_logits",
    "residual_logit",
    "residual_channel_scale",
    "residual_head",
)


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool):
    order = indices.copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, order.size, batch_size):
        yield order[start : start + batch_size]


def fetch_batch(
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.asarray(basis[indices], dtype=np.float32)),
        torch.from_numpy(np.asarray(observations[indices], dtype=np.float32)),
        torch.from_numpy(np.asarray(labels[indices], dtype=np.float32)),
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    outputs: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for batch_indices in iter_batches(indices, batch_size, rng, False):
        x, m, _ = fetch_batch(basis, observations, labels, batch_indices)
        outputs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0)
    metrics = evaluate_multilabel(labels[indices], probabilities)
    return probabilities, metrics


def train_one(
    name: str,
    config: dict,
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    split_code: np.ndarray,
    rms: np.ndarray,
    output_root: Path,
) -> tuple[dict, list[dict]]:
    pilot = config["pilot"]
    seed = int(pilot["seed"])
    # Matched methods receive the same backbone initialization; architecture-
    # specific parameters are deterministic or consume randomness afterwards.
    set_seed(seed, int(pilot["torch_threads"]))
    model_config = ModelConfig(
        name=name,
        gamma=float(pilot["gamma"]),
        dropout=float(pilot["dropout"]),
        hidden=int(pilot["gru_hidden"]),
        temporal_order=int(pilot["temporal_order"]),
        graph_order=int(pilot["graph_order"]),
        rank=int(pilot["rank"]),
    )
    model = create_model(model_config, torch.from_numpy(rms), labels.shape[1])
    warm_start_path = pilot.get("warm_start_checkpoint")
    warm_started = bool(warm_start_path)
    if warm_started:
        warm_payload = torch.load(
            (Path.cwd() / str(warm_start_path)).resolve(), map_location="cpu", weights_only=False
        )
        source_state = warm_payload["model_state"]
        target_state = model.state_dict()
        transferred = 0
        for source_name, source_value in source_state.items():
            target_name = "base_graph_logits" if source_name == "graph_logits" else source_name
            if target_name in target_state and target_state[target_name].shape == source_value.shape:
                target_state[target_name] = source_value.clone()
                transferred += 1
        model.load_state_dict(target_state)
        if hasattr(model, "residual_logit"):
            with torch.no_grad():
                model.residual_logit.fill_(float(pilot.get("warm_start_residual_logit", -4.0)))
        if transferred == 0:
            raise RuntimeError(f"warm start transferred no parameters from {warm_start_path}")
    train_indices = np.flatnonzero(split_code == 0)
    val_indices = np.flatnonzero(split_code == 1)
    positives = labels[train_indices].sum(axis=0).astype(np.float64)
    negatives = train_indices.size - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1), 1.0, float(pilot["pos_weight_clip"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight.astype(np.float32)))
    if warm_started:
        residual_parameters = []
        backbone_parameters = []
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith(RESIDUAL_PARAMETER_PREFIXES):
                residual_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        base_lr = float(pilot["learning_rate"])
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": base_lr * float(pilot.get("backbone_lr_scale", 0.25)),
                },
                {
                    "params": residual_parameters,
                    "lr": base_lr * float(pilot.get("residual_lr_scale", 3.0)),
                },
            ],
            weight_decay=float(pilot["weight_decay"]),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(pilot["learning_rate"]), weight_decay=float(pilot["weight_decay"])
        )
    history: list[dict] = []
    best_metric = -np.inf
    best_state = None
    stale = 0
    start_time = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    if warm_started:
        _, initial_metrics = predict(
            model, basis, observations, labels, val_indices, int(pilot["batch_size"])
        )
        initial_row = {
            "method": name,
            "epoch": 0,
            "train_loss": None,
            "warm_start": True,
            **{f"val_{key}": value for key, value in initial_metrics.items()},
        }
        history.append(initial_row)
        print(json.dumps(initial_row), flush=True)
        best_metric = initial_metrics["micro_auprc"]
        best_state = copy.deepcopy(model.state_dict())
    freeze_backbone_epochs = int(pilot.get("freeze_backbone_epochs", 0)) if warm_started else 0
    frozen_backbone_eval = bool(pilot.get("frozen_backbone_eval", False)) if warm_started else False
    for epoch in range(1, int(pilot["max_epochs"]) + 1):
        backbone_trainable = True
        if warm_started:
            backbone_trainable = epoch > freeze_backbone_epochs
            for parameter_name, parameter in model.named_parameters():
                if not parameter_name.startswith(RESIDUAL_PARAMETER_PREFIXES):
                    parameter.requires_grad_(backbone_trainable)
        model.train()
        if warm_started and not backbone_trainable and frozen_backbone_eval:
            # A frozen transferred predictor should behave as the deterministic
            # inference-time backbone while its residual adapter is optimized.
            if hasattr(model, "gru"):
                model.gru.eval()
            if hasattr(model, "head"):
                model.head.eval()
        rng = np.random.default_rng(seed * 1000 + epoch)
        total_loss = 0.0
        examples = 0
        for batch_indices in iter_batches(train_indices, int(pilot["batch_size"]), rng, True):
            x, m, y = fetch_batch(basis, observations, labels, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_indices.size
            examples += batch_indices.size
        _, val_metrics = predict(
            model, basis, observations, labels, val_indices, int(pilot["batch_size"])
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
        row = {
            "method": name,
            "epoch": epoch,
            "train_loss": total_loss / max(1, examples),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["micro_auprc"] > best_metric + 1e-7:
            best_metric = val_metrics["micro_auprc"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(pilot["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"no valid checkpoint for {name}")
    model.load_state_dict(best_state)
    best_probabilities, best_metrics = predict(
        model, basis, observations, labels, val_indices, int(pilot["batch_size"])
    )
    elapsed = time.perf_counter() - start_time
    checkpoint = output_root / "checkpoints" / f"{name}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_config": model_config.__dict__,
            "rms_shape": list(rms.shape),
            "seed": seed,
            "validation_only": True,
        },
        checkpoint,
    )
    prediction_root = output_root / "validation_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    np.save(prediction_root / f"{name}_seed{seed}.npy", best_probabilities)
    result = {
        "method": name,
        "seed": seed,
        "status": "validation_pilot",
        "test_evaluated": False,
        "warm_started": warm_started,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "frozen_backbone_eval": frozen_backbone_eval,
        "trainable_parameters": trainable_parameter_count(model),
        "epochs_run": len(history),
        "best_epoch": int(max(history, key=lambda item: item["val_micro_auprc"])["epoch"]),
        "wall_seconds": elapsed,
        "peak_rss_mb": peak_rss / 1024**2,
        **{f"val_{key}": value for key, value in best_metrics.items()},
    }
    return result, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-train/validation-only centralized pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--warm-start-checkpoint", default=None)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["pilot"]["seed"] = int(args.seed)
    if args.output_root is not None:
        config["outputs"]["root"] = str(args.output_root)
    if args.warm_start_checkpoint is not None:
        config["pilot"]["warm_start_checkpoint"] = str(args.warm_start_checkpoint)
    set_seed(int(config["pilot"]["seed"]), int(config["pilot"]["torch_threads"]))
    data_cfg = config["data"]
    cache_root = (project_root / data_cfg["cache_root"]).resolve()
    basis_root = (project_root / data_cfg["basis_root"]).resolve()
    basis = np.load(basis_root / data_cfg["basis_file"], mmap_mode="r")
    observations = np.load(basis_root / data_cfg["observation_file"], mmap_mode="r")
    rms = np.load(basis_root / data_cfg["rms_file"])
    with np.load(cache_root / "dataset_arrays.npz") as arrays:
        labels = arrays["y"].copy()
        split_code = arrays["split_code"].copy()
    if basis.shape[0] != labels.shape[0] or observations.shape[0] != labels.shape[0]:
        raise ValueError("pilot data alignment failed")
    if np.any(split_code == 2) and args.methods == ["__touch_test__"]:
        raise RuntimeError("test evaluation is forbidden in this pilot")
    methods = args.methods or list(config["pilot"]["methods"])
    output_root = (project_root / config["outputs"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    all_history: list[dict] = []
    for name in methods:
        result, history = train_one(
            name, config, basis, observations, labels, split_code, rms, output_root
        )
        all_results.append(result)
        all_history.extend(history)
        pd.DataFrame(all_results).to_csv(output_root / "validation_results.csv", index=False)
        pd.DataFrame(all_history).to_csv(output_root / "training_history.csv", index=False)
    summary = {
        "status": "PASS",
        "scope": "full eICU train/validation pilot",
        "test_evaluated": False,
        "seed": int(config["pilot"]["seed"]),
        "methods": methods,
        "best_by_val_micro_auprc": max(all_results, key=lambda item: item["val_micro_auprc"])["method"],
        "results": all_results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "best": summary["best_by_val_micro_auprc"]}, indent=2))


if __name__ == "__main__":
    main()
