from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dual_uq.core.hashing import sha256_bytes
from dual_uq.dataset import fixed_probes as fixed_probes_module
from dual_uq.dataset.fixed_probes import (
    ADMITTED_SUBSET_SHA256,
    FIXED_PROBE_COLUMNS,
    MASK_COORDINATE_SYSTEM,
    FixedProbeError,
    build_fixed_probe_candidates,
    build_protein_manifest,
    render_fixed_probe_parquet,
    render_protein_manifest,
)
from dual_uq.dataset.storage.proteinmpnn import (
    STANDARD_AMINO_ACIDS,
    sequence_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/design_baseline/stage0.yaml"
STAGE0_DIR = REPOSITORY_ROOT / "experiments/p2_design_baseline/stage0"
SUBSET_PATH = STAGE0_DIR / "stage0_intervention_admitted_v1.jsonl"
PANEL_PATH = STAGE0_DIR / "stage0_intervention_panel_v1.jsonl"
ADMISSION_PATH = STAGE0_DIR / "stage0_intervention_admission_v1.jsonl"
MANIFEST_PATH = STAGE0_DIR / "protein_manifest.json"
PROBES_PATH = STAGE0_DIR / "fixed_probe_candidates.parquet"

EXPECTED_IDENTITIES = (
    (21, "5gv8_A__P83686"),
    (26, "5mn1_A__P00760"),
    (49, "1fn8_A__P35049"),
    (88, "1pjx_A__Q7SIG4"),
    (96, "3pyp_A__P16113"),
    (125, "6s2s_A__P02689"),
    (196, "4ce8_A__Q9HYN5"),
    (200, "5avh_A__P24300"),
)
EXPECTED_LENGTHS = {
    "5gv8_A__P83686": (272, 272),
    "5mn1_A__P00760": (246, 223),
    "1fn8_A__P35049": (248, 224),
    "1pjx_A__Q7SIG4": (314, 314),
    "3pyp_A__P16113": (125, 125),
    "6s2s_A__P02689": (132, 132),
    "4ce8_A__Q9HYN5": (115, 114),
    "5avh_A__P24300": (388, 386),
}
EXPECTED_PROBE_COUNTS = {
    "5gv8_A__P83686": 5168,
    "5mn1_A__P00760": 4237,
    "1fn8_A__P35049": 4256,
    "1pjx_A__Q7SIG4": 5966,
    "3pyp_A__P16113": 2375,
    "6s2s_A__P02689": 2508,
    "4ce8_A__Q9HYN5": 2166,
    "5avh_A__P24300": 7334,
}
UPSTREAM_HASHES = {
    PANEL_PATH: "62d6e8576ed21663947c9bef893bcb9e74f7e6fba7b347910a4c0479b4e07e0d",
    ADMISSION_PATH: "1fa86cde49cdac68433572d929a99c964574c8edc7f9f62832e0c2b3d4a529c0",
    SUBSET_PATH: ADMITTED_SUBSET_SHA256,
}


@pytest.fixture(scope="module")
def protein_manifest() -> dict[str, object]:
    return build_protein_manifest(CONFIG_PATH, REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def fixed_probes(protein_manifest: dict[str, object]) -> pd.DataFrame:
    return build_fixed_probe_candidates(protein_manifest)


def test_config_resolves_only_exact_frozen_admitted_subset(
    protein_manifest: dict[str, object],
) -> None:
    cohort = protein_manifest["cohort"]
    proteins = protein_manifest["proteins"]

    assert cohort["admitted_subset_path"] == (
        "experiments/p2_design_baseline/stage0/"
        "stage0_intervention_admitted_v1.jsonl"
    )
    assert cohort["admitted_subset_sha256"] == ADMITTED_SUBSET_SHA256
    assert cohort["record_count"] == 8
    assert tuple(
        (record["candidate_index"], record["protein_id"]) for record in proteins
    ) == EXPECTED_IDENTITIES
    assert all(record["intervention_admission_status"] == "ADMITTED" for record in proteins)
    provenance = protein_manifest["provenance"]
    assert provenance["pdr01_protocol_path"] == (
        "docs/protocols/PDR-01_D1-D2_条款草案_v0.2.md"
    )
    assert len(provenance["pdr01_protocol_sha256"]) == 64


def test_all_eight_frozen_scientific_assets_and_sequences_resolve(
    protein_manifest: dict[str, object],
) -> None:
    for protein in protein_manifest["proteins"]:
        assert protein["canonical_sequence_length"] == len(
            protein["canonical_wt_sequence"]
        )
        assert protein["canonical_sequence_sha256"] == sequence_sha256(
            protein["canonical_wt_sequence"]
        )
        assert protein["canonical_sequence_provenance"]["provenance"] == (
            "full_span_exact_prediction_record"
        )
        assert protein["pdb_chain"] == "A"
        assert protein["pdb_entity_id"]
        assert protein["afdb_model_id"].startswith("AF-")
        for prefix in ("pdb_backbone", "afdb_backbone"):
            logical_path = protein[f"{prefix}_path"]
            assert not logical_path.startswith(("/home/", "/mnt/"))
            assert len(protein[f"{prefix}_sha256"]) == 64
            assert (REPOSITORY_ROOT / logical_path).is_file()


def test_common_masks_reproduce_frozen_uniprot_domain(
    protein_manifest: dict[str, object],
) -> None:
    observed_total = 0
    for protein in protein_manifest["proteins"]:
        canonical_length, expected_mask_length = EXPECTED_LENGTHS[protein["protein_id"]]
        positions = protein["mask_positions"]
        assert protein["canonical_sequence_length"] == canonical_length
        assert protein["mask_coordinate_system"] == MASK_COORDINATE_SYSTEM
        assert protein["mask_length"] == expected_mask_length == len(positions)
        assert positions == sorted(set(positions))
        assert all(1 <= position <= canonical_length for position in positions)
        assert protein["mapping_provenance"]["segment_count"] == 1
        assert protein["mapping_provenance"]["gap_count"] == 0
        observed_total += len(positions)

    assert observed_total == 1790
    assert protein_manifest["summary"]["total_mask_positions"] == 1790


def test_fixed_probes_are_exact_single_mutants_in_stable_order(
    protein_manifest: dict[str, object], fixed_probes: pd.DataFrame
) -> None:
    assert list(fixed_probes.columns) == list(FIXED_PROBE_COLUMNS)
    assert len(fixed_probes) == 34010 == 19 * 1790
    assert not fixed_probes.duplicated(["protein_id", "position", "mut_aa"]).any()

    manifest_by_id = {
        record["protein_id"]: record for record in protein_manifest["proteins"]
    }
    counts = Counter(fixed_probes["protein_id"])
    for protein_id, protein in manifest_by_id.items():
        assert counts[protein_id] == EXPECTED_PROBE_COUNTS[protein_id]
        assert counts[protein_id] == 19 * protein["mask_length"]
    assert sum(counts.values()) == sum(EXPECTED_PROBE_COUNTS.values()) == 34010

    for row in fixed_probes.itertuples(index=False):
        protein = manifest_by_id[row.protein_id]
        wt = protein["canonical_wt_sequence"]
        assert row.position in protein["mask_positions"]
        assert row.wt_aa == wt[row.position - 1]
        assert row.mut_aa != row.wt_aa
        assert set(row.full_sequence) <= set(STANDARD_AMINO_ACIDS)
        assert len(row.full_sequence) == len(wt)
        differences = [
            index
            for index, (left, right) in enumerate(
                zip(wt, row.full_sequence, strict=True), start=1
            )
            if left != right
        ]
        assert differences == [row.position]
        assert row.sequence_hash == sequence_sha256(row.full_sequence)

    for protein in protein_manifest["proteins"]:
        for position in (
            protein["mask_positions"][0],
            protein["mask_positions"][len(protein["mask_positions"]) // 2],
            protein["mask_positions"][-1],
        ):
            rows = fixed_probes.loc[
                (fixed_probes["protein_id"] == protein["protein_id"])
                & (fixed_probes["position"] == position)
            ]
            wt_aa = protein["canonical_wt_sequence"][position - 1]
            assert rows["mut_aa"].tolist() == [
                amino_acid for amino_acid in STANDARD_AMINO_ACIDS if amino_acid != wt_aa
            ]


def test_scientific_records_and_serialization_are_deterministic(
    protein_manifest: dict[str, object], fixed_probes: pd.DataFrame
) -> None:
    second_manifest = build_protein_manifest(CONFIG_PATH, REPOSITORY_ROOT)
    second_probes = build_fixed_probe_candidates(second_manifest)

    assert render_protein_manifest(protein_manifest) == render_protein_manifest(
        second_manifest
    )
    pd.testing.assert_frame_equal(fixed_probes, second_probes)
    assert render_fixed_probe_parquet(fixed_probes) == render_fixed_probe_parquet(
        second_probes
    )


def test_materialized_outputs_match_release_api(
    protein_manifest: dict[str, object], fixed_probes: pd.DataFrame
) -> None:
    assert MANIFEST_PATH.read_bytes() == render_protein_manifest(protein_manifest)
    materialized = pd.read_parquet(PROBES_PATH)
    pd.testing.assert_frame_equal(materialized, fixed_probes)
    assert PROBES_PATH.read_bytes() == render_fixed_probe_parquet(fixed_probes)


def test_wrong_subset_binding_is_blocked_before_materialization(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["inputs"]["stage0_intervention_admitted_subset"]["path"] = (
        "experiments/p2_design_baseline/stage0/not_the_frozen_subset.jsonl"
    )
    config_path = tmp_path / "stage0.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(FixedProbeError, match="admitted subset binding differs"):
        build_protein_manifest(config_path, REPOSITORY_ROOT)

    assert not (tmp_path / "protein_manifest.json").exists()
    assert not (tmp_path / "fixed_probe_candidates.parquet").exists()


def test_subset_bound_panel_hash_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sha256_file = fixed_probes_module.sha256_file

    def drift_panel_only(path: Path) -> str:
        if path.resolve() == PANEL_PATH.resolve():
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(fixed_probes_module, "sha256_file", drift_panel_only)
    with pytest.raises(FixedProbeError, match="Panel SHA differs"):
        build_protein_manifest(CONFIG_PATH, REPOSITORY_ROOT)


def test_upstream_panel_admission_and_subset_remain_byte_immutable() -> None:
    assert {path: sha256_bytes(path.read_bytes()) for path in UPSTREAM_HASHES} == (
        UPSTREAM_HASHES
    )


def test_outputs_contain_no_absolute_machine_paths(
    protein_manifest: dict[str, object], fixed_probes: pd.DataFrame
) -> None:
    manifest_bytes = render_protein_manifest(protein_manifest)
    assert b"/home/" not in manifest_bytes
    assert b"/mnt/" not in manifest_bytes
    assert all(
        not value.startswith(("/home/", "/mnt/"))
        for column in ("protein_id", "uniprot_accession", "sequence_hash")
        for value in fixed_probes[column]
    )
