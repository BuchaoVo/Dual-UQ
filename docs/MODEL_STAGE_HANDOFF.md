# Dual-UQ Model-Stage Handoff

## Handoff state

The data pipeline is handed off as **A0-dev v0.9**. The diagnostic and
classification interfaces are available for development work, but the final
balanced 12-protein A0 panel does not exist.

The final selection audit has `audit_pass=false`, the panel has zero rows, and
the blocking conditions are:

1. `fewer_than_24_complete_quality_pass_diagnostics`;
2. `insufficient_low_confidence_local_candidates`;
3. `insufficient_high_pae_long_range_candidates`.

The current audited counts are 23 complete quality-pass diagnostics against a
minimum of 24. Eligible primary-category counts are 15 easy controls, 1
low-confidence-local candidate, 2 high-PAE-long-range candidates, and 4
high-confidence-state-disagreement candidates.

## What may proceed

Humans may use A0-dev v0.9 for:

- method implementation and integration testing;
- exploratory mechanism analysis;
- diagnostic recovery planning;
- development-only experiments that are clearly labeled as such.

## What must not be claimed

A0-dev v0.9 must not be treated as:

- the final A0 panel;
- a balanced four-category evaluation set;
- a frozen held-out benchmark;
- evidence of final model generalization.

The empty panel is a deliberate Gate result. It must not be filled manually,
by duplicating candidates, by reassigning multi-label candidates to secondary
categories, or by lowering quality thresholds.

## Human actions required before a later model stage

Before any final benchmark training or evaluation, a human owner must:

1. add or recover enough candidates to reach at least 24 complete quality-pass
   diagnostics;
2. obtain at least three eligible candidates in each primary category;
3. rerun the deterministic final selection audit;
4. confirm that `audit_pass=true` and the panel contains exactly 12 rows;
5. review and authorize model code, configuration, splits, seeds, compute, and
   output locations.

## Training ownership

All later model training is executed manually by a human operator. Codex does
not launch GearNet, ProteinMPNN, Joint-UQ, or related training jobs, and does
not create checkpoints or training logs during this handoff.

At this freeze:

- training started: **no**;
- checkpoint created: **no**;
- training configuration added: **no**;
- final held-out benchmark available: **no**.

Artifact identities for the handoff are fixed in
`reports/a0_dev_v0_9_freeze_manifest.json`.
