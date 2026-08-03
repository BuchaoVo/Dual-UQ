# H2 Human Variant Evidence Packet

Status: **H2 preparation only — no authorization decision**

This packet uses only files already present in the repository. No network query or download was performed. Every statement is separated into observed fact, explicit local annotation, derived consistency check, or missing evidence.

## Summary

| Metric | Result |
| --- | --- |
| candidate_count | 4 |
| evidence_available_for_human_review | 3: Index 103, 22, 25 |
| evidence_incomplete | 1: Index 28 |
| no_local_evidence | 0 |
| V3 substitutions | all four match |
| AFDB AA == mapping AA | all four pass |
| V4 buckets | recovered 4; blocked missing-PDB 1; blocked other 1; still mismatching 0 |

Index 111 and Index 101 remain in the six-protein audit history but are not authorization-review candidates. Index 111 fails the H1 count/paired-identity gates; Index 101 has an AFDB-side mismatch.

## Candidate overview

| Index | Pair | Output / UniProt | Mapping→PDB | AFDB | PDB auth key | PDB label key | Readiness |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 28 | `1muw_A__P15587` | 175 / 176 | T→I | T | A:175 | A:175 | `evidence_incomplete` |
| 103 | `5avd_A__P00772` | 66 / 92 | D→N | D | A:81 | A:66 | `evidence_available_for_human_review` |
| 22 | `7af2_AAA__P29768` | 59 / 62 | D→G | D | AAA:62 | A:59 | `evidence_available_for_human_review` |
| 25 | `7bbx_A__Q5F6E9` | 8 / 8 | K→A | K | A:8 | A:9 | `evidence_available_for_human_review` |

All insertion codes are empty. PDB and AFDB residue records are `ATOM`; all four have selected N/CA/C/O atoms.

## Index 28 — 1muw_A__P15587

Observed fact: round1 V3 records output 175 / UniProt 176, mapping T, PDB I; local AFDB model position 176 is THR/T.

Explicit local annotations:

- `data/raw/mappings/1muw.xml.gz`, SIFTS residue key PDB chain A auth 175 / UniProt P15587 position 176: PDB ILE vs UniProt T; annotation “See remark 999”.
- `data/raw/pdb/1muw.cif`, `_struct_ref_seq_dif.pdbx_ordinal=1`: ILE vs THR, details “SEE REMARK 999”.
- Same mmCIF, `_pdbx_database_remark.id=999`: states the sequence was not deposited and cites a corrected-sequence dissertation; it does not explicitly assign a cause to T176I.
- Same mmCIF, `_entity.pdbx_mutation`: missing (`?`).

Evidence gaps: the mmCIF sequence-difference row uses UNP position 175 while SIFTS/round1 use UniProt 176; the cause of T176I is not explicit; no cached full UniProt variant record or P1 paired-backbone output is present.

Review readiness: **evidence_incomplete**.

## Index 103 — 5avd_A__P00772

Observed fact: round1 V3 records output 66 / UniProt 92, mapping D, PDB N; local AFDB model position 92 is ASP/D.

Explicit local annotations:

- `data/raw/mappings/5avd.xml.gz`, SIFTS residue key PDB chain A auth 81 / UniProt P00772 position 92: PDB ASN vs UniProt D; annotation “See sequence details”.
- `data/raw/pdb/5avd.cif`, `_struct_ref_seq_dif.pdbx_ordinal=1`: ASN vs ASP, details “see sequence details”.
- Same mmCIF, `_pdbx_entry_details.sequence_details`: “THE SEQUENCE CONFLICT D92N IS BASED ON REFERENCE PUBMED IS 4578945 ACCORDING TO DATABASE P00772 (CELA1_PIG)”.
- Same mmCIF, `_entity.pdbx_mutation`: missing (`?`).

Evidence gaps: the cited publication is not cached locally and was not retrieved; the local text does not itself decide an authorization category; no cached full UniProt variant record or P1 paired-backbone output is present.

Review readiness: **evidence_available_for_human_review**.

## Index 22 — 7af2_AAA__P29768

Observed fact: round1 V3 records output 59 / UniProt 62, mapping D, PDB G; local AFDB model position 62 is ASP/D.

Explicit local annotations:

- `data/raw/mappings/7af2.xml.gz`, SIFTS residue key PDB chain AAA auth 62 / UniProt P29768 position 62: PDB GLY vs UniProt D; annotation “Engineered mutation”.
- `data/raw/pdb/7af2.cif`, `_struct_ref_seq_dif.pdbx_ordinal=1`: GLY vs ASP, details “engineered mutation”.
- Same mmCIF, `_entity.pdbx_mutation`: `D62G`.
- Same mmCIF, `_struct.title` and `_citation.title`: “Salmonella typhimurium neuraminidase mutant (D62G)”.

Evidence gaps: no locally cached supporting experimental narrative beyond mmCIF/SIFTS metadata; no cached full UniProt variant record or P1 paired-backbone output is present.

Review readiness: **evidence_available_for_human_review**.

## Index 25 — 7bbx_A__Q5F6E9

Observed fact: round1 V3 records output 8 / UniProt 8, mapping K, PDB A; local AFDB model position 8 is LYS/K.

Explicit local annotations:

- `data/raw/mappings/7bbx.xml.gz`, SIFTS residue key PDB chain A auth 8 / UniProt Q5F6E9 position 8: PDB ALA vs UniProt K; annotation “Engineered mutation”.
- `data/raw/pdb/7bbx.cif`, `_struct_ref_seq_dif.pdbx_ordinal=2`: ALA vs LYS, details “engineered mutation”.
- Same mmCIF, `_struct.title`: “Neisseria gonorrhoeae transaldolase, variant K8A”.
- Same mmCIF, `_entity.pdbx_mutation`: missing (`?`).

Evidence gaps: the mmCIF sequence-difference row names UniProt accession A0A1D3FXY0, whereas SIFTS, pair QC and AFDB metadata use Q5F6E9; this requires human reconciliation. No cached full UniProt variant record or P1 paired-backbone output is present.

Review readiness: **evidence_available_for_human_review**.

## Shared provenance

For each candidate:

- mapping source: `data/processed/pairs/<pair_id>/residue_mapping.parquet`, target row identified by UniProt position and auth residue key; provenance is `sifts_auth`;
- pair QC: `data/processed/pairs/<pair_id>/pair_qc.json`, including PDB/SIFTS/mapping/AFDB paths and model identity;
- P0: `reports/dataset_a_census/round1_stage_outputs_v2/index<index>/P0_resolve_and_freeze_inputs/outputs/input_lock.json`, status complete and validation pass;
- strict P1 observation: `reports/dataset_a_census/round1_report.json`, record selected by screening index;
- recovery probe: `reports/dataset_a_census/v4_recovery_probe.json`, result selected by `protein_id=index<index>`.

The JSON companion contains the exact source field or residue-key locator for every evidence item and the complete auth/label/raw-residue provenance.

## Human-review boundary

No classification or authorization field is populated in this packet. Readiness means only that local evidence is available (or incomplete) for a human reviewer.
