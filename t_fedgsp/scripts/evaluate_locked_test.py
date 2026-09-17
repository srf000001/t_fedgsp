from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "t_fedgsp" / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in [SRC, SCRIPT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_federated_confirmation import clustered_micro_ap, rank_biserial
from evaluate_validation_perturbations import (
    perturb_events_to_two_hours,
    project_basis,
)
from fit_temperature_scaling import apply_temperature, fit_scalar_temperature
from tfedgsp_metrics import evaluate_multilabel
from tfedgsp_models import ModelConfig, create_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_for(seed: int, model: str) -> tuple[Path, Path, str]:
    if model == "static_mlp":
        root = f"federated_static_seed{seed}"
        filename = "fedavg_static_mlp_continue"
    elif model == "temporal_gru":
        root = f"federated_temporal_seed{seed}"
        filename = "fedavg_temporal_gru_continue"
    elif model == "graph_gru":
        root = f"federated_strong_seed{seed}_graph"
        filename = "fedavg_graph_continue"
    elif model == "fulljoint":
        root = f"federated_strong_seed{seed}_fulljoint"
        filename = "fedavg_tfedgsp_fulljoint_logitres"
    else:
        raise ValueError(model)
    result_root = PROJECT_ROOT / "t_fedgsp" / "results" / root
    return (
        result_root / "checkpoints" / f"{filename}.pt",
        result_root / "validation_predictions" / f"{filename}.npy",
        "tfedgsp_fulljoint_logitres" if model == "fulljoint" else model,
    )


def instantiate_model(model_name: str, rms: np.ndarray, num_labels: int) -> torch.nn.Module:
    return create_model(
        ModelConfig(
            name=model_name,
            gamma=1.0,
            dropout=0.2,
            hidden=42,
            temporal_order=1,
            graph_order=2,
            rank=1,
        ),
        torch.from_numpy(rms),
        num_labels,
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    basis: np.ndarray,
    observations: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        x = torch.from_numpy(np.asarray(basis[selected], dtype=np.float32))
        m = torch.from_numpy(np.asarray(observations[selected], dtype=np.float32))
        outputs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def hospital_analysis(
    labels: np.ndarray,
    clients: np.ndarray,
    baseline: np.ndarray,
    proposed: np.ndarray,
) -> dict[str, float | int]:
    differences = []
    for client in np.unique(clients):
        selected = clients == client
        if labels[selected].sum() == 0:
            continue
        baseline_ap = average_precision_score(
            labels[selected].ravel(), baseline[selected].ravel()
        )
        proposed_ap = average_precision_score(
            labels[selected].ravel(), proposed[selected].ravel()
        )
        differences.append(float(proposed_ap - baseline_ap))
    differences_array = np.asarray(differences, dtype=np.float64)
    test = wilcoxon(differences_array, alternative="two-sided", zero_method="wilcox")
    return {
        "hospitals": int(differences_array.size),
        "median_paired_delta": float(np.median(differences_array)),
        "positive_hospital_fraction": float(np.mean(differences_array > 0)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p": float(test.pvalue),
        "rank_biserial": float(rank_biserial(differences_array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time evaluation of the locked eICU test split")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-root", default="t_fedgsp/results/locked_test_evaluation")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate locked artifacts and model states without test inference",
    )
    args = parser.parse_args()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    if (output_root / "summary.json").exists():
        raise RuntimeError("locked test summary already exists; refusing repeated test evaluation")

    torch.set_num_threads(8)
    cache_root = PROJECT_ROOT / "t_fedgsp" / "data" / "temporal_v1"
    basis_root = PROJECT_ROOT / "t_fedgsp" / "data" / "model_basis_d64"
    with np.load(cache_root / "dataset_arrays.npz") as arrays:
        labels_all = arrays["y"].copy()
        split = arrays["split_code"].copy()
        patient_all = arrays["patient_cluster_code"].copy()
        client_all = arrays["client_code"].copy()
    val_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    val_labels = labels_all[val_indices]
    test_labels = labels_all[test_indices]
    test_patients = patient_all[test_indices]
    test_clients = client_all[test_indices]
    basis = np.load(basis_root / "basis_120min_d64_k2.npy", mmap_mode="r")
    observations = np.load(basis_root / "observations_120min.npy", mmap_mode="r")
    rms = np.load(basis_root / "train_rms.npy")

    models = ["static_mlp", "temporal_gru", "graph_gru", "fulljoint"]
    lock_entries = []
    specifications = {}
    for seed in args.seeds:
        for model_label in models:
            checkpoint, validation_prediction, model_name = paths_for(seed, model_label)
            if not checkpoint.exists() or not validation_prediction.exists():
                raise FileNotFoundError(
                    f"locked artifact missing for {model_label}, seed {seed}: "
                    f"{checkpoint} or {validation_prediction}"
                )
            val_probability = np.load(validation_prediction)
            if val_probability.shape != val_labels.shape:
                raise ValueError(f"validation prediction alignment failed: {validation_prediction}")
            temperature = fit_scalar_temperature(val_labels, val_probability)
            key = f"{model_label}_seed{seed}"
            specifications[key] = {
                "seed": seed,
                "model_label": model_label,
                "model_name": model_name,
                "checkpoint": checkpoint,
                "temperature": temperature,
            }
            lock_entries.append(
                {
                    "key": key,
                    "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "validation_prediction": str(validation_prediction.relative_to(PROJECT_ROOT)),
                    "validation_prediction_sha256": sha256(validation_prediction),
                    "validation_fitted_temperature": temperature,
                }
            )

    # Validate every serialized state before the checkpoint lock is written and
    # before any test inference can begin.
    for specification in specifications.values():
        preflight_model = instantiate_model(
            str(specification["model_name"]), rms, test_labels.shape[1]
        )
        preflight_payload = torch.load(
            specification["checkpoint"], map_location="cpu", weights_only=False
        )
        preflight_model.load_state_dict(preflight_payload["model_state"])
        del preflight_model, preflight_payload

    event_cache = PROJECT_ROOT / "t_fedgsp" / "data" / "test_event_cache"
    event_manifest_path = event_cache / "manifest.json"
    if not event_manifest_path.exists() or not (event_cache / "events.npz").exists():
        raise FileNotFoundError(
            "build the exact test event cache before the one-time evaluation with "
            "build_validation_event_cache.py --split test"
        )
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    if not (
        event_manifest.get("split") == "test"
        and event_manifest.get("reconstructs_locked_X_exactly")
        and event_manifest.get("reconstructs_locked_M_exactly")
        and int(event_manifest.get("stays", -1)) == test_labels.shape[0]
    ):
        raise RuntimeError("test event cache failed the exact reconstruction gate")
    with np.load(event_cache / "events.npz") as loaded_events:
        events = {name: loaded_events[name].copy() for name in loaded_events.files}
    powers = np.load(basis_root / "projection_powers.npy")
    clean_concepts, clean_modalities = perturb_events_to_two_hours(
        events,
        test_labels.shape[0],
        np.random.default_rng(0),
    )
    reconstructed_basis, reconstructed_observations = project_basis(
        clean_concepts, clean_modalities, powers
    )
    reconstruction_max_abs = 0.0
    observations_match = True
    for start in range(0, test_indices.size, int(args.batch_size)):
        stop = min(test_indices.size, start + int(args.batch_size))
        reconstruction_max_abs = max(
            reconstruction_max_abs,
            float(
                np.max(
                    np.abs(
                        reconstructed_basis[start:stop]
                        - np.asarray(basis[test_indices[start:stop]])
                    )
                )
            ),
        )
        observations_match = observations_match and np.array_equal(
            reconstructed_observations[start:stop],
            np.asarray(observations[test_indices[start:stop]]),
        )
    del clean_concepts, clean_modalities, reconstructed_basis, reconstructed_observations
    gc.collect()
    if reconstruction_max_abs > 1e-5 or not observations_match:
        raise RuntimeError(
            "test event cache failed projected-basis reconstruction: "
            f"max_abs={reconstruction_max_abs}, observations_match={observations_match}"
        )

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scope": "locked-test artifact preflight only",
                    "test_inference_executed": False,
                    "checkpoint_entries": len(lock_entries),
                    "test_event_cache_exact": True,
                    "projected_basis_max_abs_difference": reconstruction_max_abs,
                    "projected_observations_exact": observations_match,
                },
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-time locked test evaluation",
        "seeds": args.seeds,
        "test_stays": int(test_indices.size),
        "test_evaluated_when_lock_written": False,
        "test_event_cache_exact": True,
        "projected_basis_max_abs_difference": reconstruction_max_abs,
        "projected_observations_exact": observations_match,
        "entries": lock_entries,
    }
    (output_root / "checkpoint_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )

    raw_probabilities: dict[str, list[np.ndarray]] = {model: [] for model in models}
    calibrated_probabilities: dict[str, list[np.ndarray]] = {model: [] for model in models}
    robustness_models: dict[str, list[torch.nn.Module]] = {
        "graph_gru": [],
        "fulljoint": [],
    }
    per_seed = []
    for key, specification in specifications.items():
        model_label = str(specification["model_label"])
        model_name = str(specification["model_name"])
        model = instantiate_model(model_name, rms, test_labels.shape[1])
        payload = torch.load(
            specification["checkpoint"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(payload["model_state"])
        if model_label in robustness_models:
            robustness_models[model_label].append(model)
        probability = predict(
            model, basis, observations, test_indices, int(args.batch_size)
        )
        calibrated = apply_temperature(
            probability, float(specification["temperature"])
        )
        raw_metrics = evaluate_multilabel(test_labels, probability)
        calibrated_metrics = evaluate_multilabel(test_labels, calibrated)
        per_seed.append(
            {
                "key": key,
                "seed": int(specification["seed"]),
                "model": model_label,
                "temperature": float(specification["temperature"]),
                "raw": raw_metrics,
                "calibrated": calibrated_metrics,
            }
        )
        raw_probabilities[model_label].append(probability.astype(np.float64))
        calibrated_probabilities[model_label].append(calibrated.astype(np.float64))

    pooled_raw = {
        model: np.mean(probabilities, axis=0)
        for model, probabilities in raw_probabilities.items()
    }
    pooled_calibrated = {
        model: np.mean(probabilities, axis=0)
        for model, probabilities in calibrated_probabilities.items()
    }
    pooled = {
        model: {
            "raw": evaluate_multilabel(test_labels, pooled_raw[model]),
            "calibrated": evaluate_multilabel(
                test_labels, pooled_calibrated[model]
            ),
        }
        for model in models
    }

    # Execute the predeclared raw-event perturbation protocol in the same single
    # locked-test run. Only aggregate metrics leave memory; patient predictions
    # are never written to disk.
    local_indices = np.arange(test_labels.shape[0], dtype=np.int64)
    robustness_rows: list[dict] = []

    def add_robustness_condition(
        condition: str,
        level: float,
        perturbation_seed: int,
        condition_basis: np.ndarray,
        condition_observations: np.ndarray,
    ) -> None:
        for model_label in ["graph_gru", "fulljoint"]:
            probability_sum = np.zeros(test_labels.shape, dtype=np.float64)
            for trained_model in robustness_models[model_label]:
                probability_sum += predict(
                    trained_model,
                    condition_basis,
                    condition_observations,
                    local_indices,
                    int(args.batch_size),
                )
            probability_average = probability_sum / len(
                robustness_models[model_label]
            )
            robustness_rows.append(
                {
                    "condition": condition,
                    "level": float(level),
                    "perturbation_seed": int(perturbation_seed),
                    "model": model_label,
                    **evaluate_multilabel(test_labels, probability_average),
                }
            )

    for model_label in ["graph_gru", "fulljoint"]:
        robustness_rows.append(
            {
                "condition": "clean",
                "level": 0.0,
                "perturbation_seed": -1,
                "model": model_label,
                **pooled[model_label]["raw"],
            }
        )

    for deletion in [0.1, 0.3, 0.5]:
        for perturbation_seed in [0, 1, 2]:
            concept_matrix, modality_matrix = perturb_events_to_two_hours(
                events,
                test_labels.shape[0],
                np.random.default_rng(10_000 + perturbation_seed),
                deletion=deletion,
            )
            condition_basis, condition_observations = project_basis(
                concept_matrix, modality_matrix, powers
            )
            add_robustness_condition(
                "event_deletion",
                deletion,
                perturbation_seed,
                condition_basis,
                condition_observations,
            )
            del concept_matrix, modality_matrix, condition_basis, condition_observations
            gc.collect()

    for sigma in [30.0, 60.0, 120.0]:
        for perturbation_seed in [0, 1, 2]:
            concept_matrix, modality_matrix = perturb_events_to_two_hours(
                events,
                test_labels.shape[0],
                np.random.default_rng(20_000 + perturbation_seed),
                jitter_sigma_minutes=sigma,
            )
            condition_basis, condition_observations = project_basis(
                concept_matrix, modality_matrix, powers
            )
            add_robustness_condition(
                "raw_timestamp_jitter",
                sigma,
                perturbation_seed,
                condition_basis,
                condition_observations,
            )
            del concept_matrix, modality_matrix, condition_basis, condition_observations
            gc.collect()

    for modality_index, modality_name in enumerate(
        ["diagnosis", "medication", "lab", "treatment"]
    ):
        concept_matrix, modality_matrix = perturb_events_to_two_hours(
            events,
            test_labels.shape[0],
            np.random.default_rng(30_000 + modality_index),
            missing_modality=modality_index,
        )
        condition_basis, condition_observations = project_basis(
            concept_matrix, modality_matrix, powers
        )
        add_robustness_condition(
            f"missing_{modality_name}",
            1.0,
            -1,
            condition_basis,
            condition_observations,
        )
        del concept_matrix, modality_matrix, condition_basis, condition_observations
        gc.collect()

    def condition_mean(model: str, condition: str, level: float) -> float:
        values = [
            row["micro_auprc"]
            for row in robustness_rows
            if row["model"] == model
            and row["condition"] == condition
            and row["level"] == level
        ]
        if not values:
            raise RuntimeError(f"missing robustness rows: {model}, {condition}, {level}")
        return float(np.mean(values))

    robustness_curves = {}
    for condition, levels, normalizer in [
        ("event_deletion", [0.0, 0.1, 0.3, 0.5], 0.5),
        ("raw_timestamp_jitter", [0.0, 30.0, 60.0, 120.0], 120.0),
    ]:
        robustness_curves[condition] = {}
        for model_label in ["graph_gru", "fulljoint"]:
            values = [
                pooled[model_label]["raw"]["micro_auprc"]
                if level == 0.0
                else condition_mean(model_label, condition, level)
                for level in levels
            ]
            robustness_curves[condition][model_label] = {
                "levels": levels,
                "micro_auprc": values,
                "normalized_auc": float(np.trapezoid(values, levels) / normalizer),
            }
        graph_auc = robustness_curves[condition]["graph_gru"]["normalized_auc"]
        fulljoint_auc = robustness_curves[condition]["fulljoint"]["normalized_auc"]
        robustness_curves[condition]["fulljoint_minus_graph_auc"] = float(
            fulljoint_auc - graph_auc
        )
        robustness_curves[condition]["relative_auc_gain"] = float(
            (fulljoint_auc - graph_auc) / graph_auc
        )

    paired_robustness_deltas = []
    for row in robustness_rows:
        if row["model"] != "fulljoint" or row["condition"] == "clean":
            continue
        graph_row = next(
            candidate
            for candidate in robustness_rows
            if candidate["model"] == "graph_gru"
            and candidate["condition"] == row["condition"]
            and candidate["level"] == row["level"]
            and candidate["perturbation_seed"] == row["perturbation_seed"]
        )
        paired_robustness_deltas.append(
            float(row["micro_auprc"] - graph_row["micro_auprc"])
        )

    unique_patients, patient_inverse = np.unique(test_patients, return_inverse=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    counts = rng.multinomial(
        unique_patients.size,
        np.full(unique_patients.size, 1.0 / unique_patients.size),
        size=args.bootstrap,
    ).astype(np.int16)
    pair_cluster = np.repeat(patient_inverse, test_labels.shape[1])
    labels_flat = test_labels.ravel().astype(np.uint8)
    graph_bootstrap = clustered_micro_ap(
        labels_flat, pooled_raw["graph_gru"].ravel(), pair_cluster, counts
    )
    fulljoint_bootstrap = clustered_micro_ap(
        labels_flat, pooled_raw["fulljoint"].ravel(), pair_cluster, counts
    )
    delta_bootstrap = fulljoint_bootstrap - graph_bootstrap
    ci_low, ci_high = np.percentile(delta_bootstrap, [2.5, 97.5])
    seed_deltas = []
    for seed in args.seeds:
        graph_row = next(
            row for row in per_seed if row["seed"] == seed and row["model"] == "graph_gru"
        )
        fulljoint_row = next(
            row for row in per_seed if row["seed"] == seed and row["model"] == "fulljoint"
        )
        seed_deltas.append(
            fulljoint_row["raw"]["micro_auprc"]
            - graph_row["raw"]["micro_auprc"]
        )
    seed_deltas_array = np.asarray(seed_deltas, dtype=np.float64)

    report = {
        "status": "PASS",
        "scope": "one-time locked eICU test evaluation",
        "test_evaluated": True,
        "test_stays": int(test_indices.size),
        "test_patient_clusters": int(unique_patients.size),
        "seeds": args.seeds,
        "per_seed": per_seed,
        "pooled_probability_average": pooled,
        "fulljoint_minus_graph_seed_deltas": seed_deltas,
        "fulljoint_minus_graph_seed_mean": float(seed_deltas_array.mean()),
        "fulljoint_minus_graph_seed_sample_sd": float(seed_deltas_array.std(ddof=1)),
        "patient_cluster_bootstrap": {
            "resamples": args.bootstrap,
            "seed": args.bootstrap_seed,
            "delta_ci_95": [float(ci_low), float(ci_high)],
        },
        "hospital_analysis_fulljoint_vs_graph": hospital_analysis(
            test_labels,
            test_clients,
            pooled_raw["graph_gru"],
            pooled_raw["fulljoint"],
        ),
        "raw_event_robustness": {
            "source": "raw eICU minute offsets; shared event deletion/jitter",
            "model_seed_aggregation": "probability average over seeds 1, 2, and 3",
            "test_event_cache_manifest": event_manifest,
            "rows": robustness_rows,
            "curves": robustness_curves,
            "fulljoint_minus_graph_mean_over_perturbed_rows": float(
                np.mean(paired_robustness_deltas)
            ),
            "fulljoint_minus_graph_min_over_perturbed_rows": float(
                np.min(paired_robustness_deltas)
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Locked eICU test evaluation",
        "",
        "This is the single evaluation of the previously sealed test partition.",
        "",
        "| Model | Micro-AUPRC | Macro-AUPRC | Micro-AUROC | Brier (cal.) | ECE (cal.) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        raw = pooled[model]["raw"]
        calibrated = pooled[model]["calibrated"]
        lines.append(
            f"| {model} | {raw['micro_auprc']:.6f} | {raw['macro_auprc']:.6f} | "
            f"{raw['micro_auroc']:.6f} | {calibrated['brier']:.6f} | "
            f"{calibrated['ece_15']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Full-joint minus graph mean seed delta: {seed_deltas_array.mean():+.6f}.",
            f"Patient-cluster bootstrap 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}].",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output_root / "robustness_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(robustness_rows[0]))
        writer.writeheader()
        writer.writerows(robustness_rows)
    print(json.dumps({"status": "PASS", "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
