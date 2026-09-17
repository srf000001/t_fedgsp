from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_concept_time(
    x: sparse.csr_matrix,
    source_bins: int,
    num_concepts: int,
    factor: int,
) -> sparse.csr_matrix:
    if x.shape[1] != source_bins * num_concepts:
        raise ValueError(f"unexpected concept matrix shape {x.shape}")
    if source_bins % factor:
        raise ValueError("source bins must be divisible by aggregation factor")
    target_bins = source_bins // factor
    coo = x.tocoo()
    source_bin = coo.col // num_concepts
    concept = coo.col % num_concepts
    new_col = (source_bin // factor) * num_concepts + concept
    out = sparse.coo_matrix(
        (coo.data.astype(np.float32, copy=False), (coo.row, new_col)),
        shape=(x.shape[0], target_bins * num_concepts),
        dtype=np.float32,
    ).tocsr()
    out.sum_duplicates()
    out.data = np.log1p(out.data).astype(np.float32)
    return out


def aggregate_modality_time(
    m: sparse.csr_matrix,
    source_bins: int,
    num_modalities: int,
    factor: int,
) -> sparse.csr_matrix:
    if m.shape[1] != source_bins * num_modalities:
        raise ValueError(f"unexpected modality matrix shape {m.shape}")
    target_bins = source_bins // factor
    coo = m.tocoo()
    source_bin = coo.col // num_modalities
    modality = coo.col % num_modalities
    new_col = (source_bin // factor) * num_modalities + modality
    out = sparse.coo_matrix(
        (coo.data.astype(np.float32, copy=False), (coo.row, new_col)),
        shape=(m.shape[0], target_bins * num_modalities),
        dtype=np.float32,
    ).tocsr()
    out.sum_duplicates()
    return out


def achlioptas_projection(num_concepts: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        size=(num_concepts, dimension),
        p=[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    )
    return draws * np.float32(np.sqrt(3.0 / dimension))


def run_prepare(
    x: sparse.csr_matrix,
    m: sparse.csr_matrix,
    graph: sparse.csr_matrix,
    split_codes: np.ndarray,
    output_root: Path,
    source_bin_minutes: int,
    model_bin_minutes: int,
    num_concepts: int,
    num_modalities: int,
    projection_dim: int,
    graph_order: int,
    projection_seed: int,
    train_split_code: int,
) -> dict[str, Any]:
    if model_bin_minutes % source_bin_minutes:
        raise ValueError("model bin width must be a multiple of source bin width")
    factor = model_bin_minutes // source_bin_minutes
    source_bins = x.shape[1] // num_concepts
    target_bins = source_bins // factor
    if x.shape[0] != m.shape[0] or x.shape[0] != split_codes.shape[0]:
        raise ValueError("row alignment failed")
    if graph.shape != (num_concepts, num_concepts):
        raise ValueError("graph shape does not match concept vocabulary")
    x_agg = aggregate_concept_time(x, source_bins, num_concepts, factor)
    m_agg = aggregate_modality_time(m, source_bins, num_modalities, factor)
    coo = x_agg.tocoo()
    bin_rows = coo.row.astype(np.int64) * target_bins + (coo.col // num_concepts)
    concept_cols = coo.col % num_concepts
    signal = sparse.coo_matrix(
        (coo.data, (bin_rows, concept_cols)),
        shape=(x.shape[0] * target_bins, num_concepts),
        dtype=np.float32,
    ).tocsr()
    projection = achlioptas_projection(num_concepts, projection_dim, projection_seed)
    powers = np.empty((graph_order + 1, num_concepts, projection_dim), dtype=np.float32)
    powers[0] = projection
    for order in range(1, graph_order + 1):
        powers[order] = np.asarray(graph @ powers[order - 1], dtype=np.float32)
    output_root.mkdir(parents=True, exist_ok=True)
    basis_path = output_root / f"basis_{model_bin_minutes}min_d{projection_dim}_k{graph_order}.npy"
    basis = np.lib.format.open_memmap(
        basis_path,
        mode="w+",
        dtype=np.float32,
        shape=(x.shape[0], target_bins, graph_order + 1, projection_dim),
    )
    train_bin_mask = np.repeat(split_codes == train_split_code, target_bins)
    rms = np.empty((graph_order + 1, projection_dim), dtype=np.float32)
    for order in range(graph_order + 1):
        projected = np.asarray(signal @ powers[order], dtype=np.float32)
        basis[:, :, order, :] = projected.reshape(x.shape[0], target_bins, projection_dim)
        train_values = projected[train_bin_mask]
        rms[order] = np.sqrt(np.mean(np.square(train_values, dtype=np.float64), axis=0) + 1e-8).astype(
            np.float32
        )
    basis.flush()
    obs_dense = np.asarray(m_agg.toarray(), dtype=np.float32).reshape(
        x.shape[0], target_bins, num_modalities
    )
    np.save(output_root / f"observations_{model_bin_minutes}min.npy", obs_dense)
    np.save(output_root / "projection_powers.npy", powers)
    np.save(output_root / "train_rms.npy", rms)
    active_bins = (obs_dense.sum(axis=-1) > 0).sum(axis=1)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bin_minutes": source_bin_minutes,
        "model_bin_minutes": model_bin_minutes,
        "num_stays": int(x.shape[0]),
        "num_bins": int(target_bins),
        "num_concepts": int(num_concepts),
        "num_modalities": int(num_modalities),
        "projection": {
            "type": "Achlioptas",
            "seed": int(projection_seed),
            "dimension": int(projection_dim),
            "graph_order": int(graph_order),
            "nonzero_fraction": float(np.mean(projection != 0)),
        },
        "basis": {
            "path": basis_path.name,
            "shape": [int(value) for value in basis.shape],
            "dtype": str(basis.dtype),
            "finite": bool(np.isfinite(basis).all()),
            "file_size_bytes": int(basis_path.stat().st_size),
        },
        "observations": {
            "path": f"observations_{model_bin_minutes}min.npy",
            "shape": [int(value) for value in obs_dense.shape],
            "finite": bool(np.isfinite(obs_dense).all()),
            "active_bins_per_stay_median": float(np.median(active_bins)),
            "zero_active_stays": int((active_bins == 0).sum()),
        },
        "train_only_rms": {
            "shape": [int(value) for value in rms.shape],
            "finite": bool(np.isfinite(rms).all()),
            "strictly_positive": bool((rms > 0).all()),
            "train_stays": int((split_codes == train_split_code).sum()),
        },
        "checks": {
            "aligned_rows": bool(x.shape[0] == m.shape[0] == split_codes.shape[0]),
            "finite_basis": bool(np.isfinite(basis).all()),
            "finite_observations": bool(np.isfinite(obs_dense).all()),
            "no_empty_observation_stays": bool((active_bins > 0).all()),
            "train_only_scale": True,
            "fixed_projection": True,
        },
    }
    manifest["status"] = "PASS" if all(manifest["checks"].values()) else "FAIL"
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if manifest["status"] != "PASS":
        raise RuntimeError("model basis readiness failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fixed projected graph basis for matched models")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = config["dataset"]
    cache_root = (project_root / dataset["cache_root"]).resolve()
    output_root = (project_root / config["outputs"]["root"]).resolve()
    x = sparse.load_npz(cache_root / "X_time_concept_counts.npz").tocsr()
    m = sparse.load_npz(cache_root / "M_time_modality_counts.npz").tocsr()
    graph = sparse.load_npz(cache_root / "concept_graph_normalized.npz").tocsr()
    with np.load(cache_root / "dataset_arrays.npz") as arrays:
        split_codes = arrays["split_code"].copy()
    basis_cfg = config["basis"]
    manifest = run_prepare(
        x=x,
        m=m,
        graph=graph,
        split_codes=split_codes,
        output_root=output_root,
        source_bin_minutes=int(dataset["source_bin_minutes"]),
        model_bin_minutes=int(dataset["model_bin_minutes"]),
        num_concepts=int(dataset["num_concepts"]),
        num_modalities=int(dataset["num_modalities"]),
        projection_dim=int(basis_cfg["projection_dim"]),
        graph_order=int(basis_cfg["graph_order"]),
        projection_seed=int(basis_cfg["projection_seed"]),
        train_split_code=int(basis_cfg["train_split_code"]),
    )
    manifest["source_hashes"] = {
        name: sha256(cache_root / name)
        for name in [
            "X_time_concept_counts.npz",
            "M_time_modality_counts.npz",
            "concept_graph_normalized.npz",
            "dataset_arrays.npz",
        ]
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "basis_shape": manifest["basis"]["shape"],
                "basis_mb": round(manifest["basis"]["file_size_bytes"] / 1024**2, 2),
                "active_bins_median": manifest["observations"]["active_bins_per_stay_median"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

