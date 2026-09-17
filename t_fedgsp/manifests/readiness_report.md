# Temporal feature readiness report

- Generated: 2026-07-18T09:48:42.797930+00:00
- Dataset: eICU-CRD v2.0
- Shape: [107306, 41760]
- Sparse nonzeros: 3,544,842
- Modality-mask shape: [107306, 96]
- Stays without any raw observation: 0
- Concepts: 1740
- One-hour bins: 24
- Train-only concept graph edges: 14,716

| Gate | Status |
|---|---|
| full_eicu_not_demo | PASS |
| 107306_locked_stays | PASS |
| 50_future_labels | PASS |
| 79_hospital_clients | PASS |
| 84696_anonymous_patient_clusters | PASS |
| feature_target_alignment | PASS |
| observation_target_alignment | PASS |
| finite_sparse_values | PASS |
| nonempty_feature_matrix | PASS |
| finite_observation_values | PASS |
| no_empty_observation_stays | PASS |
| train_only_concept_graph | PASS |
| nonempty_concept_graph | PASS |
| input_target_diagnosis_policy_applied | PASS |
| no_patient_or_stay_ids_saved | PASS |

## Leakage and privacy boundary

- Input diagnosis events duplicating a future target were checked and removed; observed removals: 0.
- Split and label alignment are inherited from the locked full-eICU task and rechecked here.
- No stay ID, patient ID, or raw row is saved in the new cache.
- `dataset_arrays.npz` contains only aligned labels, split codes, anonymous client codes, and anonymous patient-cluster codes for local training/bootstrap inference.

## Intended use

This cache is a method-neutral one-hour event tensor. Candidate models may aggregate adjacent bins, but must report the resulting discrete resolution and may not call it continuous time.
