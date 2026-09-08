# Dual-UQ Protein Inverse Folding

Dual-UQ studies structure-conditioned protein sequence design under backbone and evaluator uncertainty. The repository separates reusable code, scientific protocols, runtime records, and curated deliverables without changing frozen Dataset-A or A0 scientific decisions.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/` | reusable Python implementation |
| `configs/` | declarative defaults, examples, local overrides, experiment, and legacy configuration |
| `data/` | raw/interim/processed assets, manifests, and small fixtures |
| `experiments/` | scientific protocols and entrypoint definitions, not run output |
| `runs/` | untracked, immutable execution records |
| `artifacts/` | curated audits, metrics, figures, reports, checkpoints, and releases |
| `docs/` | design, protocol, handoff, decision, and archive documents |
| `third_party/` | pinned external dependencies |
| `tests/` | unit, integration, regression, Dataset-A, and fixture coverage |

## Installation

```bash
conda create -n dual-uq python=3.11 -y
conda activate dual-uq
pip install -e ".[dev]"
python scripts/maintenance/check_environment.py
```

ProteinMPNN is a pinned submodule. After cloning, initialize it with `git submodule update --init third_party/ProteinMPNN`.

## Configuration

Tracked defaults live in `configs/defaults/base.yaml`. Copy `configs/examples/paths.example.yaml` to the ignored local path and edit only the copy:

```bash
mkdir -p configs/local
cp configs/examples/paths.example.yaml configs/local/paths.yaml
```

Machine-specific absolute paths must not be committed.

## Dataset pipeline

- Reusable dataset logic: `src/dual_uq/dataset/`
- Dataset stage and contract tests: `tests/dataset/`
- Dataset entrypoints: `scripts/dataset/`
- Design and protocol context: `docs/design/` and `docs/protocols/`
- Current collaboration handoff: `docs/handoff/`

Current code consumes published benchmark releases; retired Dataset-A and
Stage0/Scale1 exploration entrypoints remain available through Git history.

## Experiments and runs

`experiments/` defines protocols for P2 design baseline through P8 counterfactual analysis. Every execution writes to a fresh ignored directory under `runs/` using `docs/protocols/RUN_DIRECTORY_SPEC.md`. Promote only reviewed, digest-bound outputs into `artifacts/`.

## Tests

```bash
python -m compileall -q src
pytest -q tests/dataset
pytest -q
ruff check src scripts tests
```

## Third-party ProteinMPNN

The canonical checkout is `third_party/ProteinMPNN`; `third_party/ProteinMPNN.version` records its commit, weights, checksums, and license. Historical root-level compatibility links are no longer part of the repository. New model output must go to `runs/`, not the dependency checkout.

## Data and Git policy

Frozen paths under `data/raw/`, `data/processed/`, and `data/manifests/` remain accessible at their historical locations. Raw data, large processed structures, run logs, checkpoints, and model outputs are ignored. Manifests, fixtures, curated artifacts, protocol documents, and dependency version records are eligible for review and version control.
