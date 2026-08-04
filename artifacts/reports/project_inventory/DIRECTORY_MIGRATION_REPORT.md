# Directory Migration Report V1

## 1. Baseline

- Starting Git SHA: `0cb971b6b3a4eb4fb94245c627e5a255a7638be5`
- Working branch: `chore/project-layout-v1`
- Initial repository size: 296 MB
- Initial ProteinMPNN size: 180 MB
- Initial filesystem availability: 2.6 TB on `/mnt/data`
- Baseline compile: passed
- Baseline full tests: 707 passed
- Baseline Dataset-A tests: 387 passed
- Baseline `git diff --check`: failed only on existing trailing whitespace in dirty `data/manifests/screening_pool_preflight.tsv`

The migration began with 102 porcelain-status entries, including user-modified scientific artifacts, deleted historical files, an untracked ProteinMPNN checkout, active experiments, reports, and H2 review material. Those unrelated changes were preserved.

## 2. Post-migration verification

- Compile: passed
- Dataset-A tests: 387 passed
- Full tests: 710 passed
- New maintenance tests: 3 passed after verified RED (3 missing-script failures)
- V4/V5 moved-entrypoint focused tests: 15 passed
- Maintenance and touched regression-test Ruff scope: passed
- Scientific pipelines rerun: none
- Training started: no

## 3. File movement map

| Before | After | Method/status |
| --- | --- | --- |
| `configs/base.yaml` | `configs/defaults/base.yaml` | tracked rename |
| `configs/paths.example.yaml` | `configs/examples/paths.example.yaml` | tracked rename |
| `configs/paths.yaml` | `configs/local/paths.yaml` | local file retained; removed from tracking |
| seven A0 YAML files under `configs/` | `configs/legacy/a0_screening/` | tracked renames |
| `configs/smoke_pairs.tsv` | `tests/fixtures/manifests/smoke_pairs.tsv` | tracked rename |
| Dataset-A design document | `docs/design/` | existing untracked file moved and versioned |
| Dataset-A task specification | `docs/handoff/` | existing untracked file moved and versioned |
| `docs/DATASET_A_PIPELINE_HANDOFF_A8.md` | `docs/handoff/DATASET_A_PIPELINE_HANDOFF_A8.md` | tracked rename |
| `docs/MODEL_STAGE_HANDOFF.md` | `docs/handoff/MODEL_STAGE_HANDOFF.md` | tracked rename |
| PDR v0.2 | `docs/protocols/` | existing untracked file moved and versioned |
| PDR v0.1 | `docs/archive/` | existing untracked file moved and versioned |
| round1/V4/V5 census scripts | `scripts/dataset_a/census/` | tracked renames for round1/V4; existing V5 moved and versioned |
| `ProteinMPNN/` | `third_party/ProteinMPNN/` | same-disk move; registered at pinned gitlink |
| `src/dual_uq_inverse_folding.egg-info/` | removed | generated build artifact |

## 4. Commits before final documentation commit

- `280d578` — `chore: establish canonical project layout`
- `ac4934c` — `refactor: organize configuration and protocol documents`
- `030d8ca` — `chore: isolate pinned ProteinMPNN dependency`
- `8a4eb68` — `refactor: organize Dataset-A census entrypoints`

## 5. Deferred items

- Frozen raw, processed, and manifest paths are unchanged.
- Legacy A0 numbered scripts remain at root.
- Legacy audits, CSV metrics, reports, and logs were not bulk-moved because freeze, lifecycle, code, test, or unknown-use references could not be ruled out.
- Two 95-file/796-KB round1 stage-output trees remain under `reports/dataset_a_census/`.
- H2 evidence and decision-sheet files remain under `reports/dataset_a_census/` while human review is active.
- Existing experimental configs, logs, metrics, outputs, and checkpoints remain user worktree state.
- Package decomposition below `src/dual_uq` is deferred.

## 6. Compatibility links

`ProteinMPNN -> third_party/ProteinMPNN` is the only compatibility symlink. It preserves old A0 and uncommitted experiment paths. The canonical dependency root is declared in `configs/defaults/base.yaml`.

## 7. Hard-coded path updates

Updated references include A0 CLI defaults, A0 summary/selection tests, smoke fixture consumers, Dataset-A source docstrings, V4/V5 dynamic test imports, protocol cross-links, and census task commands. The before/after reference inventories are stored beside this report.

Remaining `reports/`, `data/processed/`, and `data/manifests/` references are intentionally retained where they define frozen scientific inputs, lifecycle compatibility, or legacy outputs. Existing untracked run logs and local experiment environment files may still contain the root ProteinMPNN path and are supported by the compatibility symlink.

## 8. Disk and Git tracking

- Final working-tree size before the final commit: 297 MB.
- Final observed filesystem availability: 2.8 TB; the mount is shared, so the difference from baseline is not attributed to this migration.
- No raw data, NPZ, model weight, checkpoint, or log directory was copied.
- ProteinMPNN is tracked as a submodule pointer, not as 180 MB of embedded content.
- Eleven weight checksums are recorded in `third_party/ProteinMPNN.version`.
- New run/log/checkpoint and third-party-output ignore rules prevent accidental bulk additions.
- `configs/local/paths.yaml` remains present and ignored.

## 9. Risks and recommendations

- Removing the root ProteinMPNN symlink would break historical commands; migrate active experiment configs before removing it.
- Moving round1 stage outputs would invalidate H2 source locators and V4 defaults; perform that only with a digest-aware evidence migration.
- The dirty scientific worktree prevents interpreting global `git diff --check` as a migration-only signal; use cached/commit-scoped checks and retain the baseline failure record.
- Do not normalize or recommit active report/manifests as part of layout cleanup.
- Begin future work from `docs/handoff/PROJECT_LAYOUT_HANDOFF_V1.md`; implement P2 only after human authorization and with a fresh run directory.
