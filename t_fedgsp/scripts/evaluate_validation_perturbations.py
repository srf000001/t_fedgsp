from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "t_fedgsp" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfedgsp_metrics import evaluate_multilabel
from tfedgsp_models import ModelConfig, create_model


TARGET_BINS = 12
NUM_CONCEPTS = 1740
NUM_MODALITIES = 4


def perturb_events_to_two_hours(
    events: dict[str, np.ndarray],
    stays: int,
    rng: np.random.Generator,
    deletion: float = 0.0,
    jitter_sigma_minutes: float = 0.0,
    missing_modality: int | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Perturb raw event rows once, preserving concept/observation alignment."""
    observation_stay = events["observation_stay"]
    observation_offset = events["observation_offset"]
    observation_modality = events["observation_modality"]
    concept_event_id = events["concept_event_id"]
    concept_id = events["concept_id"]

    selected = np.ones(observation_stay.size, dtype=bool)
    if missing_modality is not None:
        selected &= observation_modality != missing_modality
    if deletion > 0:
        selected &= rng.random(observation_stay.size) >= deletion
    if jitter_sigma_minutes > 0:
        perturbed_offset = observation_offset.astype(np.float64)
        perturbed_offset += rng.normal(
            0.0, jitter_sigma_minutes, size=observation_stay.size
        )
        perturbed_offset = np.clip(
            perturbed_offset, 0.0, np.nextafter(1440.0, 0.0)
        )
    else:
        perturbed_offset = observation_offset
    observation_bin = np.clip(
        np.floor(perturbed_offset / 120.0).astype(np.int64), 0, TARGET_BINS - 1
    )

    observation_col = (
        observation_bin[selected] * NUM_MODALITIES + observation_modality[selected]
    )
    observations = sparse.coo_matrix(
        (
            np.ones(observation_col.size, dtype=np.float32),
            (observation_stay[selected], observation_col),
        ),
        shape=(stays, TARGET_BINS * NUM_MODALITIES),
        dtype=np.float32,
    ).tocsr()
    observations.sum_duplicates()

    concept_selected = selected[concept_event_id]
    selected_event_ids = concept_event_id[concept_selected]
    concept_col = (
        observation_bin[selected_event_ids] * NUM_CONCEPTS
        + concept_id[concept_selected]
    )
    concepts = sparse.coo_matrix(
        (
            np.ones(concept_col.size, dtype=np.float32),
            (observation_stay[selected_event_ids], concept_col),
        ),
        shape=(stays, TARGET_BINS * NUM_CONCEPTS),
        dtype=np.float32,
    ).tocsr()
    concepts.sum_duplicates()
    concepts.data = np.log1p(concepts.data).astype(np.float32)
    return concepts, observations


def project_basis(
    concept_two_hour: sparse.csr_matrix,
    observation_two_hour: sparse.csr_matrix,
    powers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coo = concept_two_hour.tocoo()
    bin_rows = coo.row.astype(np.int64) * TARGET_BINS + coo.col // NUM_CONCEPTS
    concept_cols = coo.col % NUM_CONCEPTS
    signal = sparse.coo_matrix(
        (coo.data, (bin_rows, concept_cols)),
        shape=(concept_two_hour.shape[0] * TARGET_BINS, NUM_CONCEPTS),
        dtype=np.float32,
    ).tocsr()
    basis = np.empty(
        (concept_two_hour.shape[0], TARGET_BINS, powers.shape[0], powers.shape[2]),
        dtype=np.float32,
    )
    for order in range(powers.shape[0]):
        projected = np.asarray(signal @ powers[order], dtype=np.float32)
        basis[:, :, order, :] = projected.reshape(
            concept_two_hour.shape[0], TARGET_BINS, powers.shape[2]
        )
    observations = np.asarray(observation_two_hour.toarray(), dtype=np.float32).reshape(
        concept_two_hour.shape[0], TARGET_BINS, NUM_MODALITIES
    )
    return basis, observations


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    basis: np.ndarray,
    observations: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    outputs = []
    for start in range(0, labels.shape[0], batch_size):
        stop = min(labels.shape[0], start + batch_size)
        x = torch.from_numpy(np.asarray(basis[start:stop], dtype=np.float32))
        m = torch.from_numpy(np.asarray(observations[start:stop], dtype=np.float32))
        outputs.append(torch.sigmoid(model(x, m)).cpu().numpy())
    return evaluate_multilabel(labels, np.concatenate(outputs, axis=0))


def load_models(
    rms: np.ndarray,
    seed: int,
    include_head_input_controls: bool = False,
) -> dict[str, torch.nn.Module]:
    if seed == 0:
        graph_root = "federated_graph_extra30_seed0"
        rank1_root = "federated_rank1_strongbase_seed0"
        rank4_root = "federated_adapter_strongbase_seed0"
        fulljoint_root = "federated_fulljoint_strongbase_seed0"
    else:
        graph_root = f"federated_strong_seed{seed}_graph"
        rank1_root = f"federated_strong_seed{seed}_rank1"
        rank4_root = f"federated_strong_seed{seed}_rank4"
        fulljoint_root = f"federated_strong_seed{seed}_fulljoint"
    specifications = {
        "graph_gru": (
            "graph_gru",
            PROJECT_ROOT
            / f"t_fedgsp/results/{graph_root}/checkpoints/fedavg_graph_continue.pt",
            4,
        ),
        "rank1": (
            "separable_logitres",
            PROJECT_ROOT
            / f"t_fedgsp/results/{rank1_root}/checkpoints/fedavg_separable_logitres.pt",
            1,
        ),
        "rank4": (
            "tfedgsp_logitres",
            PROJECT_ROOT
            / f"t_fedgsp/results/{rank4_root}/checkpoints/fedavg_tfedgsp_logitres.pt",
            4,
        ),
        "fulljoint": (
            "tfedgsp_fulljoint_logitres",
            PROJECT_ROOT
            / f"t_fedgsp/results/{fulljoint_root}/checkpoints/fedavg_tfedgsp_fulljoint_logitres.pt",
            1,
        ),
    }
    if include_head_input_controls:
        if seed != 0:
            raise ValueError("head/input controls currently exist only for seed 0")
        control_root = PROJECT_ROOT / "t_fedgsp/results/review_control_head_input_adapters_seed0"
        specifications.update(
            {
                "timegraph_head_input": (
                    "tfedgsp_head_inputres",
                    control_root
                    / "checkpoints/fedavg_tfedgsp_head_inputres.pt",
                    1,
                ),
                "generic_head_input": (
                    "generic_head_inputres",
                    control_root / "checkpoints/fedavg_generic_head_inputres.pt",
                    1,
                ),
            }
        )
    output = {}
    for label, (name, checkpoint, rank) in specifications.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = create_model(
            ModelConfig(
                name=name,
                gamma=1.0,
                dropout=0.2,
                hidden=42,
                temporal_order=1,
                graph_order=2,
                rank=rank,
            ),
            torch.from_numpy(rms),
            50,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        output[label] = model
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only robustness diagnostics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-root", default="t_fedgsp/results/validation_robustness_seed0")
    parser.add_argument("--include-head-input-controls", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    cache = PROJECT_ROOT / "t_fedgsp/data/temporal_v1"
    basis_root = PROJECT_ROOT / "t_fedgsp/data/model_basis_d64"
    with np.load(cache / "dataset_arrays.npz") as arrays:
        val_indices = np.flatnonzero(arrays["split_code"] == 1)
        labels = arrays["y"][val_indices].copy()
    powers = np.load(basis_root / "projection_powers.npy")
    rms = np.load(basis_root / "train_rms.npy")
    modality_names = ["diagnosis", "medication", "lab", "treatment"]
    event_cache = PROJECT_ROOT / "t_fedgsp/data/validation_event_cache"
    manifest_path = event_cache / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "build the exact validation event cache first with "
            "build_validation_event_cache.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("reconstructs_locked_X_exactly")
        and manifest.get("reconstructs_locked_M_exactly")
    ):
        raise RuntimeError("validation event cache failed the locked-cache reconstruction gate")
    with np.load(event_cache / "events.npz") as loaded_events:
        events = {name: loaded_events[name].copy() for name in loaded_events.files}
    if int(manifest["stays"]) != labels.shape[0]:
        raise ValueError("validation event cache stay alignment failed")
    models = load_models(rms, args.seed, args.include_head_input_controls)
    rows = []

    clean_basis = np.load(basis_root / "basis_120min_d64_k2.npy", mmap_mode="r")[val_indices]
    clean_observations = np.load(basis_root / "observations_120min.npy", mmap_mode="r")[val_indices]
    for model_name, model in models.items():
        metrics = evaluate_model(model, clean_basis, clean_observations, labels, args.batch_size)
        rows.append({"condition": "clean", "level": 0.0, "seed": -1, "model": model_name, **metrics})
    del clean_basis, clean_observations

    for deletion in [0.1, 0.3, 0.5]:
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(10_000 + seed)
            x, m = perturb_events_to_two_hours(
                events, labels.shape[0], rng, deletion=deletion
            )
            basis, observations = project_basis(x, m, powers)
            for model_name, model in models.items():
                metrics = evaluate_model(model, basis, observations, labels, args.batch_size)
                rows.append(
                    {
                        "condition": "event_deletion",
                        "level": deletion,
                        "seed": seed,
                        "model": model_name,
                        **metrics,
                    }
                )
            del x, m, basis, observations
            gc.collect()

    for sigma in [30.0, 60.0, 120.0]:
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(20_000 + seed)
            x, m = perturb_events_to_two_hours(
                events,
                labels.shape[0],
                rng,
                jitter_sigma_minutes=sigma,
            )
            basis, observations = project_basis(x, m, powers)
            for model_name, model in models.items():
                metrics = evaluate_model(model, basis, observations, labels, args.batch_size)
                rows.append(
                    {
                        "condition": "raw_timestamp_jitter",
                        "level": sigma,
                        "seed": seed,
                        "model": model_name,
                        **metrics,
                    }
                )
            del x, m, basis, observations
            gc.collect()

    for modality_index, modality_name in enumerate(modality_names):
        x, m = perturb_events_to_two_hours(
            events,
            labels.shape[0],
            np.random.default_rng(30_000 + modality_index),
            missing_modality=modality_index,
        )
        basis, observations = project_basis(x, m, powers)
        for model_name, model in models.items():
            metrics = evaluate_model(model, basis, observations, labels, args.batch_size)
            rows.append(
                {
                    "condition": f"missing_{modality_name}",
                    "level": 1.0,
                    "seed": -1,
                    "model": model_name,
                    **metrics,
                }
            )
        del x, m, basis, observations
        gc.collect()

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "perturbation_metrics.csv", index=False)
    summary = {
        "status": "PASS",
        "scope": "validation only",
        "test_evaluated": False,
        "model_seed": args.seed,
        "perturbation_source": "raw event offsets with shared event-level deletion/jitter",
        "timestamp_jitter_resolution": "raw eICU minute offsets",
        "locked_cache_reconstruction_required": True,
        "models": list(models),
        "rows": len(rows),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
