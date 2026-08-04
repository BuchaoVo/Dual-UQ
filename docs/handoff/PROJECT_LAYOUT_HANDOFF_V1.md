# Project Layout Handoff V1

## Scope and safety

This migration is engineering-only. It did not recompute or change A0/Dataset-A candidates, thresholds, final-panel state, mappings, fragment/model identity, random seeds, statistics, or human-review decisions. No model or data was downloaded and no training was started.

Migration branch: `chore/project-layout-v1`
Starting commit: `0cb971b6b3a4eb4fb94245c627e5a255a7638be5`

## Completed

- Established canonical `configs/`, `data/`, `experiments/`, `runs/`, `artifacts/`, `docs/`, `third_party/`, `envs/`, and test-category skeletons.
- Moved tracked defaults/examples and legacy A0 configuration to `configs/defaults/`, `configs/examples/`, and `configs/legacy/a0_screening/`.
- Retained the machine-local path file as ignored `configs/local/paths.yaml`; the old tracked file was removed.
- Moved the smoke manifest to `tests/fixtures/manifests/smoke_pairs.tsv` and updated consumers.
- Organized Dataset-A design, protocol, archive, and handoff documents.
- Moved round1, V4, and V5 census entrypoints to `scripts/dataset_a/census/` and updated tests/documentation.
- Registered the root numbered scripts as legacy A0 entrypoints in `scripts/legacy/a0_screening/README.md` without duplicating them.
- Moved the clean ProteinMPNN repository to `third_party/ProteinMPNN`, pinned commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, recorded 11 weight hashes, and retained a root compatibility symlink.
- Removed the generated `src/dual_uq_inverse_folding.egg-info/` tree and expanded `.gitignore` for local config, runtime state, large data, checkpoints, and third-party outputs.
- Defined immutable run records in `docs/protocols/RUN_DIRECTORY_SPEC.md`.
- Added minimal offline helpers for deterministic inventory, terminal-run archival, and release digest verification under `scripts/maintenance/`.
- Indexed active H2 material and legacy Dataset-A stage outputs instead of moving path-bound evidence.
- Rewrote the root README with installation, configuration, pipeline, run, dependency, legacy, and Git policies.

## Verification

Migration baseline:

- `python -m compileall -q src`: passed.
- `pytest -q`: 707 passed.
- `pytest -q tests/dataset_a_scale`: 387 passed.
- `git diff --check`: pre-existing failure from trailing whitespace in dirty `data/manifests/screening_pool_preflight.tsv`.

Post-migration verification before handoff:

- `python -m compileall -q src`: passed.
- `pytest -q tests/dataset_a_scale`: 387 passed.
- `pytest -q`: 710 passed, including three new maintenance-tool tests.
- V4/V5 moved-entrypoint focused tests: 15 passed.
- Maintenance-tool Ruff scope: passed.

## Deferred

- Frozen `data/raw/`, `data/processed/`, and `data/manifests/` paths were not moved because freeze manifests, lifecycle/resume state, and evidence locators bind them.
- `reports/a0_dev_v0_9_freeze_manifest.json`, `data/manifests/a0_final_panel.tsv`, P0 manifests, lifecycle files, and status files remain untouched.
- Root `scripts/00_*.py` through `scripts/23_*.py` remain in place; wrappers or a full move are deferred until Dataset-A v1 release.
- Legacy reports and 108 log files remain in place because complete run metadata is unavailable.
- Both round1 stage-output trees remain under `reports/dataset_a_census/`; V4 and H2 evidence hard-code v2 paths.
- Active H2 evidence packet and human review sheet remain at historical paths; `artifacts/dataset/reports/h2_review/INDEX.md` records them.
- `src/dual_uq` was not split into core/data/design/evaluators/analysis packages.
- `docs/superpowers/plans/` remains in place because its documents span design, execution history, and handoff roles.
- Existing user worktree modifications and untracked experiment outputs/configurations were preserved and not normalized into commits.
- `configs/defaults/base.yaml` keeps the legacy `report_dir: reports` value to avoid changing current runtime behavior.

## Compatibility

`ProteinMPNN` at repository root is a tracked symlink to `third_party/ProteinMPNN`. It supports historical commands and existing uncommitted experiment records. New configuration must use `third_party.proteinmpnn_root: third_party/ProteinMPNN`.

No compatibility links were added for moved configuration, documentation, fixture, or census paths; their tracked references and tests were updated to canonical paths.

## Experiment path map

| Stage | Protocol definition | Runtime records | Curated deliverables |
| --- | --- | --- | --- |
| P0 Resolve & Freeze | `experiments/dataset_a_scale/` | `runs/dataset_a/p0/` | `artifacts/audits/dataset_a/` |
| P1 Build & Validate Backbones | `experiments/dataset_a_scale/` | `runs/dataset_a/p1/` | `artifacts/audits/dataset_a/` |
| P2 ProteinMPNN clean baseline | `experiments/p2_design_baseline/` | `runs/design_baseline/` | `artifacts/metrics/design_baseline/` |
| P3 Structure-UQ | `experiments/p3_structure_uq/` | `runs/structure_uq/` | `artifacts/metrics/structure_uq/` |
| P4 Evaluator-UQ | `experiments/p4_evaluator_uq/` | `runs/evaluator_uq/` | `artifacts/metrics/evaluator_uq/` |
| P5 Joint-UQ | `experiments/p5_joint_uq/` | `runs/joint_uq/` | `artifacts/metrics/joint_uq/` |
| P6 Pareto reliability | `experiments/p6_pareto_reliability/` | `runs/pareto_reliability/` | `artifacts/figures/pareto/` |
| P7 Mechanism analysis | `experiments/p7_mechanism_analysis/` | `runs/mechanism_analysis/` | `artifacts/reports/mechanism_analysis/` |
| P8 Residue counterfactual | `experiments/p8_counterfactual/` | `runs/counterfactual/` | `artifacts/reports/counterfactual/` |

## Next collaborator

Start with this handoff, then read `docs/protocols/RUN_DIRECTORY_SPEC.md`, `docs/handoff/DATASET_A_PIPELINE_HANDOFF_A8.md`, and `docs/protocols/PDR-01_D1-D2_条款草案_v0.2.md`. The next experimental implementation stage is P2 ProteinMPNN clean baseline after the required human gates; do not treat layout completion as authorization to train.
