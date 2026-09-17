# Anonymous repository audit

Audit scope: every file in this repository, including text sources, filenames,
configuration paths, aggregate CSV/JSON/Markdown outputs, and the packaged
graph manifest.

## Included

- Experimental Python source and tests.
- Locked preprocessing and seed-specific configurations.
- Aggregate-only vocabulary/graph manifests.
- Non-identifying per-seed histories and aggregate result summaries.
- Method and experiment-protocol documentation.

## Intentionally excluded

- Author names, affiliations, email addresses, ORCID records, funding details,
  acknowledgements, and author-bearing citation metadata.
- Original repository metadata or commit history.
- Raw eICU tables and any patient, stay, or hospital identifiers.
- Patient-level split files, labels, predictions, and event caches.
- Model checkpoints and optimizer states.
- Virtual environments, bytecode caches, local logs, manuscript sources, and
  internal working notes.

## Checks performed

- Text scan for personal names/contact patterns, institutional strings, funding
  identifiers, absolute user paths, internal workspace names, and private
  repository URLs: no retained matches.
- CSV header scan for patient/stay/hospital/client identifier fields: no
  retained matches in published result tables.
- File-type scan: no checkpoint, patient-prediction, raw-table, or archive files
  are present. The single `.npz` file is the aggregate normalized concept graph
  listed in `t_fedgsp/manifests/`.
- Offline unit tests: 7/7 passed under Python 3.13.
- `MANIFEST.sha256`: generated after final packaging for integrity verification.

Before creating the public anonymous remote, upload only the contents of this
folder and verify that the hosting account and repository owner are anonymous.
