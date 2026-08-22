# StructCal v1 Schema Implementation

## Release layers

The canonical release root is `artifacts/releases/structcal_v1/`:

```text
core/         model-independent benchmark identity and generic eligibility
annotations/  frozen structural and ligand descriptors
tracks/       derived auditable Track-I and Track-II membership views
metadata/     benchmark, clustering, split, statistical, generation, evaluator,
              provenance, and validation contracts
```

Track-III has no Parquet cohort. Its metadata defines joint Track-I × Track-II
evaluation.

## Public Core

The public Core contains exactly:

- `proteins.parquet`: one row per biological protein; canonical sequence and length;
- `structures.parquet`: one row per structural-condition instance, with explicit Arm,
  source identity, condition label, and parent identity where applicable;
- `condition_pairs.parquet`: frozen orientation, formal role, comparability, mapping
  coverage, ligand context, and requested perturbation fields;
- `residue_mappings.parquet`: canonical-position correspondence and coordinate
  visibility only;
- `benchmark_instances.parquet`: generic task/geometry/ligand eligibility only;
- `splits.parquet`: the sole owner of global cluster and release split identity.

`condition_1` and `condition_2` are serialization fields whose meanings are fixed by
the Arm contract: PDB→AFDB, APO→HOLO, the declared functional-state orientation, or
REFERENCE→PERTURBED. Domain construction APIs retain meaningful role names.

## Separation rules

Public Core rejects:

- `*_sha256`, `*_checksum`, or equivalent hash fields;
- `proteinmpnn_*`, `esm_if1_*`, or other model-specific eligibility;
- model scores, model responses, `R_local`, `D_excess`, or `J_full`;
- structural geometry fields inside residue mappings.

Requested perturbation dose remains in pair/protocol identity; realized geometry is
stored separately in `perturbation_descriptors.parquet`. Model capability metadata is
combined with generic benchmark eligibility only at execution time.

## Relational validation

The authoritative JSON schemas live in `schemas/`. Release validation enforces primary
keys, foreign keys, protein/structure/pair identity agreement, canonical amino-acid and
position validity, mapping-flag consistency, one cluster and split per protein,
cluster-atomic split assignment, deterministic Track membership, annotation-to-Core
foreign keys, absence of a Track-III duplicate cohort, and absence of model-dependent
inputs from clustering/splitting.

The formal materializer is `scripts/dataset/materialize_structcal_v1.py`, backed by
`dual_uq.dataset.structcal_release`. It reads only frozen model-independent cohort,
mapping, and structural-annotation artifacts. Historical files are never overwritten.
