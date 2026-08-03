# Third-party dependencies

`third_party/` contains pinned external dependencies; project code must not modify upstream sources in place.

## ProteinMPNN

- Source: <https://github.com/dauparas/ProteinMPNN>
- Pinned commit: `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`
- Version record: [`ProteinMPNN.version`](ProteinMPNN.version)
- License: [`ProteinMPNN/LICENSE`](ProteinMPNN/LICENSE)

The canonical project-relative root is `third_party/ProteinMPNN`, configured by `third_party.proteinmpnn_root` in `configs/defaults/base.yaml`.

The repository-root `ProteinMPNN` path is a temporary compatibility symlink for legacy A0 commands and uncommitted experiment records.

New ProteinMPNN outputs belong under `runs/design_baseline/`, `runs/structure_uq/`, or `runs/joint_uq/`, never under the dependency checkout.

Existing `third_party/ProteinMPNN/outputs/` content is retained as legacy/example output and ignored for future Git additions.
