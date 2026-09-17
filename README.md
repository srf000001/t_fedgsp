# T-FedGSP anonymous reproducibility repository

This repository contains the experimental source code, locked configurations,
unit tests, aggregate manifests, and non-identifying result summaries for the
anonymous submission. It intentionally contains no author names, affiliations,
contact details, raw eICU rows, patient/stay/hospital identifiers, trained
checkpoints, or patient-level predictions.

## Repository layout

- `t_fedgsp/src/`: model definitions and multilabel metrics.
- `t_fedgsp/scripts/`: preprocessing, training, evaluation, robustness, and
  aggregate-analysis entry points.
- `t_fedgsp/tests/`: deterministic offline unit tests.
- `t_fedgsp/configs/`: the 18 explicit seed configurations and their locked
  templates.
- `t_fedgsp/configs/preprocessing/`: full-eICU readiness, temporal-feature,
  and graph-basis configurations.
- `t_fedgsp/manifests/`: non-identifying vocabulary, graph, and build metadata.
- `t_fedgsp/results/`: aggregate validation histories and locked audit
  summaries; no checkpoints or patient-level arrays are included.
- `docs/`: method and experiment protocol.

## Environment and offline verification

Python 3.13 was used for the archived runs. From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements-cpu.txt
python generate_seed_configs.py
python -m unittest discover -s t_fedgsp/tests -v
```

The tests use synthetic fixtures only and do not require eICU access.

## Restricted data boundary

The experiments use eICU-CRD v2.0, which must be obtained separately under its
credentialed data-use terms. Set `EICU_RAW_DIR` to the directory containing the
official compressed CSV tables. Keep all restricted or locally derived data
under `external_data/`; this directory is excluded by `.gitignore`.

The temporal builder expects three local inputs that are deliberately not
distributed:

- `external_data/derived/splits/split_assignments.csv`
- `external_data/derived/labels/targets_multihot.csv`
- `external_data/derived/vocab/`

They contain or derive from restricted record identifiers. Their expected
schema, deterministic patient-level split rule, cohort gates, label rule, and
train-only vocabulary policy are documented in `docs/EXPERIMENT_PROTOCOL.md`
and exercised by the readiness and unit-test code. Published manifests include
only aggregate counts and hashes.

## Reproduction order

1. Run the full-data readiness audit:

   ```bash
   python t_fedgsp/scripts/full_eicu_readiness.py \
     --config t_fedgsp/configs/preprocessing/eicu_full.yaml
   ```

2. Place the locally generated split, target, and vocabulary files at the
   paths above, then build temporal features and the fixed graph basis:

   ```bash
   python t_fedgsp/scripts/build_temporal_features.py \
     --config t_fedgsp/configs/preprocessing/temporal_features.yaml
   python t_fedgsp/scripts/prepare_model_basis.py \
     --config t_fedgsp/configs/preprocessing/model_basis_d64.yaml
   ```

3. Regenerate the seed matrix with `python generate_seed_configs.py`. Train the
   three `common_graphgru_seed*.yaml` configurations first. Then run the five
   corresponding final-branch/control configurations for each seed with:

   ```bash
   python t_fedgsp/scripts/run_federated_pilot.py --config CONFIG_PATH
   ```

4. Run aggregate analyses from the retained non-identifying summaries:

   ```bash
   python t_fedgsp/scripts/analyze_reviewer_optimizer_controls.py
   python t_fedgsp/scripts/analyze_budget_matched_controls.py
   ```

The archived one-time test report is under
`t_fedgsp/results/locked_audit/`. Re-executing locked inference requires new
locally trained checkpoints and credentialed eICU access; the repository does
not redistribute either.

## Anonymous-review status

See `ANONYMITY_AUDIT.md` for the packaging checks. Do not add a personal Git
configuration, author-bearing citation file, institutional badge, private
remote URL, or deanonymizing issue/commit history before the review period ends.
