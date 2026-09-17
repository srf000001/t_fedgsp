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
from tfedgsp_models import ModelConfig, create_model


RESIDUAL_PARAMETER_PREFIXES = (
    "temporal_raw",
    "graph_raw",
    "rank_logits",
    "joint_raw",
    "residual_logit",
    "residual_channel_scale",
    "residual_head",
    "residual_joint",
)


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = indices.copy()
    rng.shuffle(order)
    for start in range(0, order.size, batch_size):
        yield order[start : start + batch_size]


def fetch(
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
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        x, m, _ = fetch(basis, observations, labels, selected)
        outputs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0)
    return probabilities, evaluate_multilabel(labels[indices], probabilities)


def class_weights(labels: np.ndarray, indices: np.ndarray, cap: float) -> torch.Tensor:
    positive = labels[indices].sum(axis=0).astype(np.float64)
    negative = indices.size - positive
    values = np.clip(negative / np.maximum(positive, 1.0), 1.0, cap)
    return torch.from_numpy(values.astype(np.float32))


def trainable_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def parameter_bytes(model: nn.Module, names: list[str]) -> int:
    named = dict(model.named_parameters())
    return int(sum(named[name].numel() * named[name].element_size() for name in names))


def configure_trainable(model: nn.Module, residual_only: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(RESIDUAL_PARAMETER_PREFIXES) if residual_only else True
        )


def initialize_weighted_label_bias(
    model: nn.Module,
    labels: np.ndarray,
    train_indices: np.ndarray,
    pos_weight_cap: float,
) -> None:
    positive = labels[train_indices].sum(axis=0).astype(np.float64)
    negative = train_indices.size - positive
    prevalence = (positive + 0.5) / (train_indices.size + 1.0)
    weights = np.clip(negative / np.maximum(positive, 1.0), 1.0, pos_weight_cap)
    weighted_odds = weights * prevalence / np.maximum(1.0 - prevalence, 1e-12)
    bias = torch.from_numpy(np.log(np.maximum(weighted_odds, 1e-12)).astype(np.float32))
    if hasattr(model, "head") and isinstance(model.head[-1], nn.Linear):
        prediction_head = model.head[-1]
    elif hasattr(model, "network") and isinstance(model.network[-1], nn.Linear):
        prediction_head = model.network[-1]
    else:
        raise TypeError("weighted label-bias initialization requires a linear prediction head")
    with torch.no_grad():
        prediction_head.bias.copy_(bias)


def transfer_graph_backbone(source_state: dict[str, torch.Tensor], target: nn.Module) -> None:
    target_state = target.state_dict()
    transferred = 0
    for source_name, value in source_state.items():
        target_name = "base_graph_logits" if source_name == "graph_logits" else source_name
        if target_name in target_state and target_state[target_name].shape == value.shape:
            target_state[target_name] = value.clone()
            transferred += 1
    if transferred == 0:
        raise RuntimeError("federated backbone transfer matched no tensors")
    target.load_state_dict(target_state)


def fedavg_round(
    model: nn.Module,
    global_state: dict[str, torch.Tensor],
    client_indices: list[tuple[int, np.ndarray]],
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight_cap: float,
    global_pos_weight: torch.Tensor | None,
    seed: int,
    round_index: int,
    optimizer_name: str,
    proximal_mu: float,
    client_weight_power: float,
) -> tuple[dict[str, torch.Tensor], float, int]:
    names = trainable_names(model)
    if not names:
        raise RuntimeError("FedAvg stage has no trainable parameters")
    raw_client_weights = np.asarray(
        [max(1, indices.size) ** client_weight_power for _, indices in client_indices],
        dtype=np.float64,
    )
    normalized_client_weights = raw_client_weights / raw_client_weights.sum()
    aggregated = {name: torch.zeros_like(global_state[name]) for name in names}
    weighted_loss = 0.0
    for client_position, (client_code, indices) in enumerate(client_indices):
        model.load_state_dict(global_state)
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=0.0,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"unsupported local optimizer {optimizer_name}")
        pos_weight = (
            global_pos_weight
            if global_pos_weight is not None
            else class_weights(labels, indices, pos_weight_cap)
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        client_loss = 0.0
        seen = 0
        for local_epoch in range(local_epochs):
            local_seed = seed * 1_000_003 + round_index * 10_007 + client_code * 101 + local_epoch
            torch.manual_seed(local_seed)
            rng = np.random.default_rng(local_seed)
            for selected in batches(indices, batch_size, rng):
                x, m, y = fetch(basis, observations, labels, selected)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x, m), y)
                if proximal_mu > 0:
                    named_parameters = dict(model.named_parameters())
                    proximal = sum(
                        torch.sum((named_parameters[name] - global_state[name]) ** 2)
                        for name in names
                    )
                    loss = loss + 0.5 * proximal_mu * proximal
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                client_loss += float(loss.item()) * selected.size
                seen += selected.size
        client_state = model.state_dict()
        weight = float(normalized_client_weights[client_position])
        for name in names:
            aggregated[name].add_(client_state[name], alpha=weight)
        weighted_loss += (client_loss / max(1, seen)) * weight
    next_state = {name: value.clone() for name, value in global_state.items()}
    next_state.update(aggregated)
    bytes_per_round = 2 * len(client_indices) * parameter_bytes(model, names)
    return next_state, weighted_loss, bytes_per_round


def run_stage(
    stage_name: str,
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
    rounds: int,
    client_indices: list[tuple[int, np.ndarray]],
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    val_indices: np.ndarray,
    config: dict,
    seed: int,
    round_offset: int,
    learning_rate: float,
) -> tuple[dict, list[dict], dict[str, torch.Tensor], np.ndarray]:
    fed = config["federated"]
    state = {name: value.clone() for name, value in initial_state.items()}
    model.load_state_dict(state)
    probabilities, initial_metrics = predict(
        model, basis, observations, labels, val_indices, int(fed["eval_batch_size"])
    )
    best_state = copy.deepcopy(state)
    best_probabilities = probabilities
    best_metric = initial_metrics["micro_auprc"]
    best_round = 0
    history = [
        {
            "stage": stage_name,
            "round": 0,
            "train_loss": None,
            "communication_bytes": 0,
            **{f"val_{key}": value for key, value in initial_metrics.items()},
        }
    ]
    cumulative_bytes = 0
    start = time.perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    global_weight = None
    server_optimizer = str(fed.get("server_optimizer", "fedavg")).lower()
    trainable = trainable_names(model)
    server_first = {name: torch.zeros_like(state[name]) for name in trainable}
    server_second = {name: torch.zeros_like(state[name]) for name in trainable}
    if str(fed.get("pos_weight_scope", "local")) == "global":
        all_train = np.concatenate([indices for _, indices in client_indices])
        global_weight = class_weights(labels, all_train, float(fed["pos_weight_clip"]))
    for stage_round in range(1, rounds + 1):
        previous_state = state
        candidate_state, loss, round_bytes = fedavg_round(
            model=model,
            global_state=state,
            client_indices=client_indices,
            basis=basis,
            observations=observations,
            labels=labels,
            batch_size=int(fed["batch_size"]),
            local_epochs=int(fed["local_epochs"]),
            learning_rate=learning_rate,
            weight_decay=float(fed["weight_decay"]),
            pos_weight_cap=float(fed["pos_weight_clip"]),
            global_pos_weight=global_weight,
            seed=seed,
            round_index=round_offset + stage_round,
            optimizer_name=str(fed.get("optimizer", "adamw")).lower(),
            proximal_mu=float(fed.get("proximal_mu", 0.0)),
            client_weight_power=float(fed.get("client_weight_power", 1.0)),
        )
        if server_optimizer == "fedavg":
            state = candidate_state
        elif server_optimizer == "fedadam":
            beta1 = float(fed.get("server_beta1", 0.9))
            beta2 = float(fed.get("server_beta2", 0.99))
            server_lr = float(fed.get("server_learning_rate", 0.01))
            tau = float(fed.get("server_tau", 1e-3))
            state = {name: value.clone() for name, value in previous_state.items()}
            for name in trainable:
                delta = candidate_state[name] - previous_state[name]
                server_first[name].mul_(beta1).add_(delta, alpha=1.0 - beta1)
                server_second[name].mul_(beta2).addcmul_(
                    delta, delta, value=1.0 - beta2
                )
                first_hat = server_first[name] / (1.0 - beta1**stage_round)
                second_hat = server_second[name] / (1.0 - beta2**stage_round)
                state[name] = previous_state[name] + server_lr * first_hat / (
                    torch.sqrt(second_hat) + tau
                )
        else:
            raise ValueError(f"unsupported server optimizer {server_optimizer}")
        model.load_state_dict(state)
        probabilities, metrics = predict(
            model, basis, observations, labels, val_indices, int(fed["eval_batch_size"])
        )
        cumulative_bytes += round_bytes
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        row = {
            "stage": stage_name,
            "round": stage_round,
            "train_loss": loss,
            "communication_bytes": round_bytes,
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["micro_auprc"] > best_metric + 1e-7:
            best_metric = metrics["micro_auprc"]
            best_state = copy.deepcopy(state)
            best_probabilities = probabilities
            best_round = stage_round
    model.load_state_dict(best_state)
    _, best_metrics = predict(
        model, basis, observations, labels, val_indices, int(fed["eval_batch_size"])
    )
    result = {
        "stage": stage_name,
        "seed": seed,
        "clients": len(client_indices),
        "rounds": rounds,
        "learning_rate": learning_rate,
        "local_optimizer": str(fed.get("optimizer", "adamw")).lower(),
        "server_optimizer": server_optimizer,
        "proximal_mu": float(fed.get("proximal_mu", 0.0)),
        "client_weight_power": float(fed.get("client_weight_power", 1.0)),
        "best_round": best_round,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "communication_bytes": cumulative_bytes,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_mb": peak_rss / 1024**2,
        **{f"val_{key}": value for key, value in best_metrics.items()},
    }
    return result, history, best_state, best_probabilities


def model_config(name: str, cfg: dict) -> ModelConfig:
    method = cfg["method"]
    return ModelConfig(
        name=name,
        gamma=float(method["gamma"]),
        dropout=float(method["dropout"]),
        hidden=int(method["gru_hidden"]),
        temporal_order=int(method["temporal_order"]),
        graph_order=int(method["graph_order"]),
        rank=int(method["rank"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only two-stage full-eICU FedAvg pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pretrained-backbone-checkpoint", default=None)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["federated"]["seed"] = int(args.seed)
    if args.output_root is not None:
        config["outputs"]["root"] = str(args.output_root)
    if args.pretrained_backbone_checkpoint is not None:
        config["federated"]["pretrained_backbone_checkpoint"] = str(
            args.pretrained_backbone_checkpoint
        )
    fed = config["federated"]
    seed = int(fed["seed"])
    set_seed(seed, int(fed["torch_threads"]))
    data = config["data"]
    basis_root = (project_root / data["basis_root"]).resolve()
    cache_root = (project_root / data["cache_root"]).resolve()
    basis = np.load(basis_root / data["basis_file"], mmap_mode="r")
    observations = np.load(basis_root / data["observation_file"], mmap_mode="r")
    rms = np.load(basis_root / data["rms_file"])
    with np.load(cache_root / "dataset_arrays.npz") as arrays:
        labels = arrays["y"].copy()
        split_code = arrays["split_code"].copy()
        client_code = arrays["client_code"].copy()
    train_indices = np.flatnonzero(split_code == 0)
    val_indices = np.flatnonzero(split_code == 1)
    client_indices = [
        (int(code), train_indices[client_code[train_indices] == code])
        for code in np.unique(client_code[train_indices])
    ]
    client_indices = [(code, indices) for code, indices in client_indices if indices.size]
    expected_clients = int(fed["expected_clients"])
    if len(client_indices) != expected_clients:
        raise RuntimeError(f"expected {expected_clients} train clients, found {len(client_indices)}")
    output_root = (project_root / config["outputs"]["root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    backbone_name = str(config["method"].get("backbone_name", "graph_gru"))
    backbone = create_model(
        model_config(backbone_name, config), torch.from_numpy(rms), labels.shape[1]
    )
    if bool(fed.get("initialize_weighted_label_bias", False)):
        initialize_weighted_label_bias(
            backbone,
            labels,
            train_indices,
            float(fed["pos_weight_clip"]),
        )
    pretrained_backbone = fed.get("pretrained_backbone_checkpoint")
    if pretrained_backbone:
        payload = torch.load(
            (project_root / str(pretrained_backbone)).resolve(),
            map_location="cpu",
            weights_only=False,
        )
        backbone.load_state_dict(payload["model_state"])
    configure_trainable(backbone, residual_only=False)
    backbone_result, backbone_history, backbone_state, backbone_probs = run_stage(
        f"fedavg_{backbone_name}_pretrain",
        backbone,
        copy.deepcopy(backbone.state_dict()),
        int(fed["backbone_rounds"]),
        client_indices,
        basis,
        observations,
        labels,
        val_indices,
        config,
        seed,
        0,
        float(fed["backbone_learning_rate"]),
    )

    branch_specs = []
    if bool(fed.get("include_graph_continue", True)):
        branch_specs.append(
            (f"fedavg_{backbone_name}_continue", backbone_name, False)
        )
    for name in config["method"].get("adapter_methods", ["tfedgsp_logitres"]):
        branch_specs.append((f"fedavg_{name}", name, True))
    results = [backbone_result]
    history = list(backbone_history)
    checkpoint_root = output_root / "checkpoints"
    prediction_root = output_root / "validation_predictions"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    backbone_stem = "graph_pretrain" if backbone_name == "graph_gru" else f"{backbone_name}_pretrain"
    torch.save(
        {"model_state": backbone_state, "seed": seed},
        checkpoint_root / f"{backbone_stem}.pt",
    )
    np.save(prediction_root / f"{backbone_stem}.npy", backbone_probs)

    for stage_name, name, residual_only in branch_specs:
        set_seed(seed, int(fed["torch_threads"]))
        branch = create_model(model_config(name, config), torch.from_numpy(rms), labels.shape[1])
        if residual_only:
            transfer_graph_backbone(backbone_state, branch)
            with torch.no_grad():
                branch.residual_logit.fill_(float(config["method"].get("residual_logit", 0.0)))
        else:
            branch.load_state_dict(backbone_state)
        configure_trainable(branch, residual_only=residual_only)
        result, rows, state, probabilities = run_stage(
            stage_name,
            branch,
            copy.deepcopy(branch.state_dict()),
            int(fed["branch_rounds"]),
            client_indices,
            basis,
            observations,
            labels,
            val_indices,
            config,
            seed,
            int(fed["backbone_rounds"]),
            float(
                fed["adapter_learning_rate"] if residual_only else fed["branch_learning_rate"]
            ),
        )
        result["total_protocol_communication_bytes"] = (
            backbone_result["communication_bytes"] + result["communication_bytes"]
        )
        results.append(result)
        history.extend(rows)
        torch.save({"model_state": state, "seed": seed}, checkpoint_root / f"{stage_name}.pt")
        np.save(prediction_root / f"{stage_name}.npy", probabilities)
        pd.DataFrame(results).to_csv(output_root / "validation_results.csv", index=False)
        pd.DataFrame(history).to_csv(output_root / "round_history.csv", index=False)

    summary = {
        "status": "PASS",
        "scope": "full eICU federated train/validation pilot",
        "test_evaluated": False,
        "seed": seed,
        "clients": len(client_indices),
        "weighted_label_bias_initialized": bool(
            fed.get("initialize_weighted_label_bias", False)
        ),
        "pretrained_backbone_checkpoint": pretrained_backbone,
        "results": results,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "stages": [item["stage"] for item in results]}, indent=2))


if __name__ == "__main__":
    main()
