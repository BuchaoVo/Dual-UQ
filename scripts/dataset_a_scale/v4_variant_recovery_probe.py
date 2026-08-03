"""V4 read-only variant-recovery probe (TASK-B, docs/handoff/Dataset-A_后续任务规格包_v0.1.md).

Answers, WITHOUT modifying p1.py: if the known pdb_amino_acid_mismatch sites
for a protein (from the TASK-A round-1 site table) are excluded from the
AA-identity check, does the protein actually reach a state equivalent to
p1_pairing_ok, or does it hit a different real P1 failure first?

Reuses the exact same frozen parsing/selection primitives p1.py uses
(_load_atom_records, _load_mapping, _index_records, structures.
select_backbone_atoms/group_residue_records) and replicates every real P1
per-row check (backbone completeness, label identity, AFDB pairing, AFDB
AA identity) except the PDB AA-identity check at the known mismatch
positions. Unlike P1 itself, this walks every mapped position instead of
stopping at the first failure, so it can report the complete downstream
picture for each protein.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dual_uq.dataset_a_scale.stages.p0 import P0_STAGE_NAME
from dual_uq.dataset_a_scale.stages.p1 import (
    _index_records,
    _load_atom_records,
    _load_mapping,
    _optional_mapping_label,
)
from dual_uq.dataset_a_scale.structures import (
    ResidueKey,
    group_residue_records,
    select_backbone_atoms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUND1_REPORT_PATH = PROJECT_ROOT / "reports/dataset_a_census/round1_report.json"
CENSUS_STAGE_ROOT = PROJECT_ROOT / "reports/dataset_a_census/round1_stage_outputs_v2"
OUTPUT_PATH = PROJECT_ROOT / "reports/dataset_a_census/v4_recovery_probe.json"

_BUCKETS = (
    "recovered",
    "blocked_downstream_missing_pdb",
    "blocked_downstream_other",
    "still_mismatching",
)


@dataclass(frozen=True)
class ProbeIssue:
    output_position: int
    code: str
    detail: str


@dataclass(frozen=True)
class ProbeResult:
    pair_id: str
    protein_id: str
    bucket: str
    known_mismatch_positions: tuple[int, ...]
    unexpected_mismatches: tuple[int, ...] = ()
    missing_pdb_positions: tuple[int, ...] = ()
    other_issues: tuple[ProbeIssue, ...] = field(default_factory=tuple)


def load_mismatch_proteins(round1_report_path: Path) -> list[dict[str, Any]]:
    """Read the TASK-A site table for every pdb_amino_acid_mismatch-hit protein."""
    data = json.loads(round1_report_path.read_text(encoding="utf-8"))
    return [r for r in data["records"] if r["failure_code"] == "pdb_amino_acid_mismatch"]


def _load_stage_paths(p0_stage_dir: Path) -> dict[str, Any]:
    lock = json.loads((p0_stage_dir / "outputs" / "input_lock.json").read_text(encoding="utf-8"))
    paths = {entry["logical_name"]: Path(entry["original_path"]) for entry in lock["input_files"]}
    fragment = lock["fragment_interval"]
    return {
        "pdb_path": paths["pdb_structure_path"],
        "afdb_path": paths["afdb_model_path"],
        "fragment_start": fragment["uniprot_start"],
        "model_length": fragment["model_length"],
    }


def _backbone_issue_code(exc: ValueError) -> str:
    return "ambiguous_backbone_atom" if "ambiguous duplicate atom" in str(exc) else "invalid_backbone_selection"


def probe_protein(
    *,
    pair_id: str,
    protein_id: str,
    known_mismatch_output_positions: set[int],
    pdb_path: Path,
    afdb_path: Path,
    p0_stage_dir: Path,
    fragment_start: int,
    model_length: int,
) -> ProbeResult:
    mapping = _load_mapping(p0_stage_dir / "outputs" / "residue_mapping.tsv")
    pdb_index = _index_records(_load_atom_records(pdb_path, pair_id))
    afdb_local_positions: dict[int, tuple] = {}
    for group in group_residue_records(_load_atom_records(afdb_path, pair_id)):
        key = group.residue.key
        if key.insertion_code or key.auth_seq_id in afdb_local_positions:
            continue
        afdb_local_positions[key.auth_seq_id] = group.atoms

    missing_pdb_positions: list[int] = []
    unexpected_mismatches: list[int] = []
    other_issues: list[ProbeIssue] = []

    for row in mapping.itertuples(index=False):
        output_position = int(row.output_position)
        uniprot_position = int(row.uniprot_residue_number)
        mapping_aa = str(row.canonical_aa)
        skip_aa_check = output_position in known_mismatch_output_positions

        # --- PDB side ---
        key = ResidueKey(pair_id, str(row.auth_asym_id), int(row.auth_seq_id), str(row.insertion_code))
        pdb_atoms = pdb_index.get(key)
        if pdb_atoms is None:
            missing_pdb_positions.append(output_position)
        else:
            try:
                pdb_selection = select_backbone_atoms(pdb_atoms)
            except ValueError as exc:
                other_issues.append(
                    ProbeIssue(output_position, _backbone_issue_code(exc), f"pdb: {exc}")
                )
                pdb_selection = None
            if pdb_selection is not None:
                if pdb_selection.missing_atoms:
                    other_issues.append(
                        ProbeIssue(
                            output_position,
                            "missing_backbone_atom",
                            f"pdb missing {list(pdb_selection.missing_atoms)}",
                        )
                    )
                else:
                    expected_label_chain = _optional_mapping_label(row.label_asym_id)
                    expected_label_seq = (
                        None
                        if _optional_mapping_label(row.label_seq_id) is None
                        else int(row.label_seq_id)
                    )
                    provenance = pdb_selection.residue
                    if expected_label_chain is not None and (
                        provenance.label_chain_id != expected_label_chain
                        or provenance.label_seq_id != expected_label_seq
                    ):
                        other_issues.append(
                            ProbeIssue(output_position, "pdb_label_identity_mismatch", "label identity conflict")
                        )
                    elif not skip_aa_check and provenance.canonical_aa != mapping_aa:
                        unexpected_mismatches.append(output_position)

        # --- AFDB side (never exempted, PDR-01 §1.5) ---
        model_position = uniprot_position - fragment_start + 1
        if not (1 <= model_position <= model_length):
            other_issues.append(
                ProbeIssue(output_position, "uniprot_position_mismatch", "outside frozen AFDB fragment")
            )
            continue
        afdb_atoms = afdb_local_positions.get(model_position)
        if afdb_atoms is None:
            other_issues.append(
                ProbeIssue(output_position, "residue_count_mismatch", "afdb missing mapped residue")
            )
            continue
        try:
            afdb_selection = select_backbone_atoms(afdb_atoms)
        except ValueError as exc:
            other_issues.append(ProbeIssue(output_position, _backbone_issue_code(exc), f"afdb: {exc}"))
            continue
        if afdb_selection.missing_atoms:
            other_issues.append(
                ProbeIssue(
                    output_position,
                    "missing_backbone_atom",
                    f"afdb missing {list(afdb_selection.missing_atoms)}",
                )
            )
        elif afdb_selection.residue.canonical_aa != mapping_aa:
            other_issues.append(ProbeIssue(output_position, "afdb_amino_acid_mismatch", "afdb AA differs"))

    if unexpected_mismatches:
        bucket = "still_mismatching"
    elif missing_pdb_positions:
        bucket = "blocked_downstream_missing_pdb"
    elif other_issues:
        bucket = "blocked_downstream_other"
    else:
        bucket = "recovered"

    return ProbeResult(
        pair_id=pair_id,
        protein_id=protein_id,
        bucket=bucket,
        known_mismatch_positions=tuple(sorted(known_mismatch_output_positions)),
        unexpected_mismatches=tuple(unexpected_mismatches),
        missing_pdb_positions=tuple(missing_pdb_positions),
        other_issues=tuple(other_issues),
    )


def run_probe(round1_report_path: Path, census_stage_root: Path) -> list[ProbeResult]:
    results = []
    for record in load_mismatch_proteins(round1_report_path):
        protein_id = record["protein_id"]
        p0_stage_dir = census_stage_root / protein_id / P0_STAGE_NAME
        stage_paths = _load_stage_paths(p0_stage_dir)
        known_positions = {site["output_position"] for site in record["mismatches"]}
        results.append(
            probe_protein(
                pair_id=record["pair_id"],
                protein_id=protein_id,
                known_mismatch_output_positions=known_positions,
                pdb_path=stage_paths["pdb_path"],
                afdb_path=stage_paths["afdb_path"],
                p0_stage_dir=p0_stage_dir,
                fragment_start=stage_paths["fragment_start"],
                model_length=stage_paths["model_length"],
            )
        )
    return results


def summarize(results: list[ProbeResult]) -> dict[str, Any]:
    counts = Counter(result.bucket for result in results)
    return {
        "bucket_counts": {bucket: counts.get(bucket, 0) for bucket in _BUCKETS},
        "still_mismatching_pair_ids": [r.pair_id for r in results if r.bucket == "still_mismatching"],
    }


def main() -> None:
    results = run_probe(ROUND1_REPORT_PATH, CENSUS_STAGE_ROOT)
    summary = summarize(results)
    if summary["bucket_counts"]["still_mismatching"] > 0:
        raise SystemExit(
            "V4 probe found still_mismatching proteins -- this bucket must be empty. "
            f"Affected: {summary['still_mismatching_pair_ids']}. Stopping without writing "
            "a V4 conclusion: this means the probe's residue-selection handling diverges "
            "from census.py's analyze_sequence_mismatches and must be reconciled first."
        )
    payload = {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
