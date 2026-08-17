# APO/HOLO Cross-Model Local Response

## VERDICT

`PASS`

Both authorized model-specific local-response branches completed on the
standard-20-AA runtime cohort. No upstream admission, mapping, geometry, or
primary-pair artifact was changed. The A5YV76 nonstandard canonical residue
was excluded from `MODEL_EVALUABLE` for the full pair, with reason
`nonstandard_canonical_residue`; the original primary release remains intact.

## COHORT AND MATERIALIZED OUTPUTS

- Primary admitted pairs: 1,410.
- ProteinMPNN model-evaluable pairs: 1,393.
- ESM-IF1 model-evaluable pairs: 1,359.
- ESM-IF1 additional exclusions: 34 `true_uniprot_gap` pairs.
- Position-level rows: ProteinMPNN 975,550; ESM-IF1 1,041,024.
- Protein-level rows: ProteinMPNN 1,393; ESM-IF1 1,359.
- Cross-model shared proteins: 1,359.
- Cross-model shared position keys: 468,661.
- Both position tables have finite JS values and no duplicate
  `(pair_id, canonical_position, condition)` keys.

Materialized model outputs and manifest hashes are recorded in the respective
`proteinmpnn/manifest.json` and `esm_if1/manifest.json` files. ProteinMPNN
uses 16 matched deterministic realizations; its convergence summaries cover
4/8/16 realizations. Mean JS across proteins was 0.087447, 0.087431, and
0.087421 bits at 4, 8, and 16 realizations, respectively. ESM-IF1 uses the
frozen WT teacher-forced single realization.

## PROTEINMPNN LOCAL RESPONSE

Protein-level local JS burden (bits): median 0.078406, q10 0.055679, q90
0.129624. All 1,393 proteins have a positive mean local JS burden under the
finite-value definition. This is a model-native local distribution response,
not a biological or functional endpoint.

## ESM-IF1 LOCAL RESPONSE

Protein-level local JS burden (bits): median 0.115839, q10 0.058264, q90
0.193906. All 1,359 evaluated proteins have a positive mean local JS burden.
The ESM-IF1 result uses the official `esm_if1_gvp4_t16_142M_UR50` checkpoint
and the WT teacher-forced context.

## CROSS-MODEL CONCORDANCE

On the 1,359 shared proteins, ProteinMPNN and ESM-IF1 local burden have
protein-level Spearman correlation `rho = 0.783822`. The median within-protein
residue-level Spearman correlation across the 1,359 shared proteins is 0.344321
(q10 0.195990, q90 0.511664). Thus both architectures respond on the shared
cohort, with substantial but non-identical protein ranking concordance.

## GEOMETRY ASSOCIATION

Associations are descriptive within the available released geometry intersection;
they do not establish causality. For ProteinMPNN / ESM-IF1 respectively,
Spearman correlations between local JS and geometry descriptors were:

- aligned C-alpha displacement: 0.252440 / 0.299114;
- local pairwise-distance change: 0.408221 / 0.352131;
- contact-turnover fraction: 0.299547 / 0.182480.

The local pairwise-distance association is larger than the aligned-displacement
association for both models in this released intersection. No new geometry
metric was introduced.

## LIGAND LOCALIZATION

Mean local JS by released ligand label (ProteinMPNN / ESM-IF1, bits):

- `proximal`: 0.101543 / 0.114526;
- `explicit_zero_proximal`: 0.092534 / 0.130509;
- `distal_or_nonproximal`: 0.087293 / 0.117755.

These are position-level descriptive contrasts with unequal coverage and are
not interpreted as ligand causality or allostery.

## INTERPRETATION BOUNDARY

The completed results support a cross-model local-response observation: paired
apo/holo structural representations alter model-native inverse-folding amino-acid
preferences under both ProteinMPNN and ESM-IF1. The result is scoped to the
model-evaluable cohorts above and should not be read as biological fitness,
stability, functional failure, or causal ligand mechanism.

No free-sequence generation, DynamicMPNN, ESMFold, EvoEF2, or controlled
perturbation stage was started.

## NEXT

`APO_HOLO_GENERATIVE_PROPAGATION`
