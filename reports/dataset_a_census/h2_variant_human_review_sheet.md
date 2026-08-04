# H2 Human Variant Decision Sheet

Status: **human decision fields empty**
Evidence policy: local files only; no network, download, recommendation, classification, or authorization.

## Readiness summary

| Status | Count | Candidates |
| --- | ---: | --- |
| `ready_for_human_decision` | 2 | Index 103, 22 |
| `needs_additional_local_evidence` | 1 | Index 28 |
| `blocked_by_local_conflict` | 1 | Index 25 |

V3 mismatch values are unchanged; AFDB AA equals mapping AA for all four; V4 remains recovered 4 / blocked missing-PDB 1 / blocked other 1 / still mismatching 0.

## Human-review table

| Index | Pair | Mismatch | Position provenance | Accession provenance | Readiness |
| ---: | --- | --- | --- | --- | --- |
| 28 | `1muw_A__P15587` | T176I | output 175; PDB auth/label 175; SIFTS UniProt 176; mmCIF annotation reports UNP 175 | P15587 throughout | `needs_additional_local_evidence` |
| 103 | `5avd_A__P00772` | D92N | output/label 66; PDB auth 81; UniProt/SIFTS 92 | P00772 throughout | `ready_for_human_decision` |
| 22 | `7af2_AAA__P29768` | D62G | output/label 59; PDB auth 62; UniProt/SIFTS 62 | P29768 throughout | `ready_for_human_decision` |
| 25 | `7bbx_A__Q5F6E9` | K8A | output/UniProt/PDB auth 8; PDB label 9 | pair/SIFTS/P0/AFDB Q5F6E9; mmCIF A0A1D3FXY0; no local relation | `blocked_by_local_conflict` |

## Index 28 numbering closure

Closure label: **`numbering_relation_explicit_but_annotation_cause_unresolved`**.

| Coordinate/source | Exact local value | Locator |
| --- | --- | --- |
| Output | 175 | `round1_report.json: records[screening_index=28].mismatches[0]` |
| PDB auth | A:175, no insertion | `data/raw/pdb/1muw.cif: _atom_site auth A/175` |
| PDB label | A:175 | `data/raw/pdb/1muw.cif: _pdbx_poly_seq_scheme asym A, seq_id 175` |
| UniProt/SIFTS | P15587:176, T | `data/raw/mappings/1muw.xml.gz: PDB A:175 / UniProt P15587:176` |
| mmCIF difference annotation | auth 175 ILE vs UNP P15587 position 175 THR; “SEE REMARK 999” | `data/raw/pdb/1muw.cif: _struct_ref_seq_dif.pdbx_ordinal=1` |
| Local reference sequence | 386 residues beginning `SYQ...`; canonical cached sequence begins `MSYQ...` | `data/raw/pdb/1muw.cif: _struct_ref.pdbx_seq_one_letter_code`; AFDB metadata `uniprotSequence` |
| REMARK 999 | SEQRES not deposited; cites corrected-sequence dissertation; no target-specific cause | `data/raw/pdb/1muw.cif: _pdbx_database_remark.id=999` |

SIFTS explicitly maps the local/PDB numbering to UniProt with a +1 offset around the target. The mmCIF difference row nominally reports UNP position 175, so that discrepancy remains visible; no local text explains the biological or construct cause of T176I.

## Index 25 accession closure

| Source | Exact accession/value | Locator |
| --- | --- | --- |
| Pair identity | Q5F6E9 | `data/processed/pairs/7bbx_A__Q5F6E9/pair_qc.json: uniprot_id` |
| P0/mapping | Q5F6E9 | `residue_mapping.parquet: target row uniprot_id` |
| SIFTS | Q5F6E9 | `data/raw/mappings/7bbx.xml.gz: PDB A:8 / UniProt Q5F6E9:8` |
| AFDB metadata | Q5F6E9 / TAL_NEIG1 | `data/raw/afdb/Q5F6E9/AF-Q5F6E9-F1/metadata.json` |
| mmCIF | A0A1D3FXY0 / A0A1D3FXY0_NEIGO | `data/raw/pdb/7bbx.cif: _struct_ref.id=1, _struct_ref_seq.align_id=1` |
| mmCIF difference | A0A1D3FXY0 position 8, LYS→ALA, “engineered mutation” | `data/raw/pdb/7bbx.cif: _struct_ref_seq_dif.pdbx_ordinal=2` |

No explicit accession cross-reference, isoform relation, or secondary-accession relation between A0A1D3FXY0 and Q5F6E9 exists in the inspected local files. The discrepancy is not silently resolved; Index 25 is `blocked_by_local_conflict`.

Local relation status: **`no_explicit_local_relation`**. Accession evidence status: **`true_accession_conflict`**.

## Exact annotations and unresolved questions

### Index 103

- `data/raw/pdb/5avd.cif:_pdbx_entry_details.sequence_details`: “THE SEQUENCE CONFLICT D92N IS BASED ON REFERENCE PUBMED IS 4578945 ACCORDING TO DATABASE P00772 (CELA1_PIG)”.
- SIFTS target record: PDB ASN at A:81, UniProt P00772 D92, “See sequence details”.
- Missing: the cited publication content is not cached locally.
- Unresolved: whether human review accepts the deposited annotation under an allowed evidence category.

### Index 22

- `data/raw/pdb/7af2.cif:_entity.pdbx_mutation`: `D62G`.
- `_struct_ref_seq_dif.pdbx_ordinal=1`: GLY vs P29768 ASP62, “engineered mutation”.
- SIFTS target record: PDB GLY at AAA:62, UniProt P29768 D62, “Engineered mutation”.
- Missing: no supporting narrative beyond deposited local mmCIF/SIFTS metadata.
- Unresolved: whether the deposited local annotation meets the human evidence standard.

The JSON companion preserves `explicit_local_annotations`, a consolidated `source_locations` list, every supporting item, contradictory item, missing item, and unresolved question for all four candidates.

## Human decision fields

The following fields are present and `null` for every candidate:

- `human_decision`
- `evidence_class`
- `evidence_note`
- `reviewer`
- `review_date`
- `authorize_variant`
