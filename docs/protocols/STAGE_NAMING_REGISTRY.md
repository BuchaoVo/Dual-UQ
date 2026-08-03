# Stage Naming Registry

## Authority and scope

This registry distinguishes Dataset-A protocol stage identifiers from project
workspace labels. It is an engineering naming reference only and does not
change protocol decisions, stage gates, scientific thresholds, or execution
authorization.

Protocol P0–P9 IDs remain defined by the Dataset-A protocol.

Structure-UQ / Evaluator-UQ / Joint-UQ / Pareto reliability /
Mechanism analysis / Counterfactual are scientific workstream labels.

Workspace directory names do not redefine frozen or draft protocol stage
semantics.

The authoritative stage definitions are in
`docs/design/Dual-UQ_Dataset-A_实验设计方案_v0.1.md`.

## Dataset-A protocol stage IDs

| Protocol ID | Authoritative protocol meaning |
| --- | --- |
| P0 | Resolve & Freeze Inputs |
| P1 | Build & Validate Paired Backbones |
| P2 | Generation |
| P3 | Cross-score |
| P4 | Matched neutral controls |
| P5 | Residue probability localization |
| P6 | Evidence join |
| P7 | Mechanism-blind site freeze |
| P8 | Counterfactual scoring |
| P9 | Protein summary |

The evaluator extensions P3′ (multi-evaluator scoring) and P3″ (Pareto and
interaction analysis) retain the meanings assigned by the same protocol.

## Workspace workstream labels

The numeric prefixes below provide a stable directory order. They are not a
second protocol-stage registry and do not establish protocol equivalence.

| Workspace directory | Scientific workstream label | Naming relationship |
| --- | --- | --- |
| `experiments/p2_design_baseline/` | Design baseline | Supports future design-generation work; it does not redefine protocol P2. |
| `experiments/p3_structure_uq/` | Structure-UQ | May consume evidence from several protocol stages; it does not redefine protocol P3. |
| `experiments/p4_evaluator_uq/` | Evaluator-UQ | Related analyses may include protocol P3′/P3″; it does not redefine protocol P4. |
| `experiments/p5_joint_uq/` | Joint-UQ | Cross-workstream modeling label; it does not redefine protocol P5. |
| `experiments/p6_pareto_reliability/` | Pareto reliability | Related analyses may include protocol P3″ and P9; it does not redefine protocol P6. |
| `experiments/p7_mechanism_analysis/` | Mechanism analysis | May use protocol P6/P9 evidence; it does not redefine protocol P7. |
| `experiments/p8_counterfactual/` | Counterfactual | Related to counterfactual analysis, but the directory name alone does not authorize or redefine protocol P8. |

## Interpretation rules

1. Protocol artifacts, gates, and decisions must use P0–P9 according to the
   authoritative Dataset-A protocol.
2. Workspace labels describe where engineering and scientific work is
   organized; they do not confer stage completion or execution authorization.
3. If a workspace label and a protocol-stage description appear inconsistent,
   the authoritative Dataset-A protocol controls.
4. This registry does not modify PDR-01 or any H1/H2/H3/H4 decision.
5. The presence of a workspace directory does not mean that its workstream or
   protocol P2 has started.

At this checkpoint, H2 remains HUMAN REVIEW REQUIRED and PDR-01 remains DRAFT
and NOT FROZEN.
