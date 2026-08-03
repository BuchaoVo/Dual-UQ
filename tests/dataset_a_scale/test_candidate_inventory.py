from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dataset_a"
    / "census"
    / "candidate_inventory.py"
)


def _load_inventory_module():
    spec = importlib.util.spec_from_file_location("candidate_inventory", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_row in range(220):
        rows.append(
            {
                "polymer_entity_id": f"{source_row + 1:04d}_1",
                "pdb_id": f"{source_row + 1:04d}",
                "entity_id": "1",
                "chain_id": "A",
                "uniprot_id": f"P{source_row + 1:05d}",
                "sequence_cluster": f"cluster-{source_row + 1}",
                "length": 100 + source_row,
                "resolution": 1.0 + source_row / 1000,
                "organism": "synthetic",
                "afdb_available": True,
                "afdb_global_plddt": 60.0 + source_row % 40,
                "discovery_status": "eligible"
                if source_row not in {2, 8, 31, 77, 100, 155, 219}
                else "failed",
                "error": None,
            }
        )
    return pd.DataFrame(rows)


def test_canonical_universe_uses_source_order_and_records_provenance(tmp_path: Path) -> None:
    inventory = _load_inventory_module()
    source = tmp_path / "data/processed/discovery/discovered_candidates.parquet"
    source.parent.mkdir(parents=True)
    frame = _canonical_source()
    frame.to_parquet(source, index=False)

    candidates, metadata = inventory.load_canonical_universe(
        source,
        project_root=tmp_path,
    )

    assert metadata == {
        "canonical_source_path": "data/processed/discovery/discovered_candidates.parquet",
        "source_sha256": inventory.sha256_file(source),
        "total_row_count": 220,
        "eligible_row_count": 213,
        "indexing_policy_version": "inventory-source-order-v1",
    }
    assert candidates["candidate_index"].tolist() == list(range(1, 214))
    assert candidates["canonical_source_row"].head(4).tolist() == [0, 1, 3, 4]
    assert candidates.iloc[0]["pair_id"] == "0001_A__P00001"
    assert candidates.iloc[-1]["canonical_source_row"] == 218
    assert candidates["polymer_entity_id"].is_unique
    assert candidates["pair_id"].is_unique
    assert not candidates.duplicated(["PDB", "chain", "UniProt"]).any()


def test_canonical_universe_rejects_count_or_identity_drift(tmp_path: Path) -> None:
    inventory = _load_inventory_module()
    source = tmp_path / "discovered_candidates.parquet"
    frame = _canonical_source()
    frame.loc[1, "uniprot_id"] = frame.loc[0, "uniprot_id"]
    frame.loc[1, "pdb_id"] = frame.loc[0, "pdb_id"]
    frame.to_parquet(source, index=False)

    try:
        inventory.load_canonical_universe(source, project_root=tmp_path)
    except inventory.InventoryInvariantError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("duplicate canonical identity must stop inventory construction")


def test_missing_local_evidence_stays_null_or_unknown(tmp_path: Path) -> None:
    inventory = _load_inventory_module()
    row = {
        "candidate_index": 1,
        "canonical_source_row": 0,
        "polymer_entity_id": "1ABC_1",
        "pair_id": "1abc_A__P00001",
        "PDB": "1abc",
        "chain": "A",
        "UniProt": "P00001",
        "sequence_cluster": "30:1",
        "length": 120,
        "afdb_global_plddt": 93.0,
    }

    record = inventory.build_candidate_record(row, project_root=tmp_path)

    assert record["historical_screening_index"] is None
    assert record["protein_family_if_available"] is None
    assert record["canonical_uniprot_length"] is None
    assert record["mapped_length"] is None
    assert record["mapping_coverage"] is None
    assert record["low_confidence_fraction_proxy"] is None
    assert record["long_range_PAE_proxy"] is None
    assert record["existing_PDB_AFDB_disagreement_proxy"] is None
    assert record["known_sequence_mismatch_flag"] is None
    assert record["known_provenance_issue"] is None
    assert record["estimated_P0_readiness"] == "insufficient_local_evidence"
    assert record["estimated_P1_risk"] == "unknown"
    assert record["local_data_complete"] is False
    assert record["sampling_stratum_prior"] == "uncertain_or_unclassified"
    assert record["prior_evidence_status"] == "unobserved"
    assert record["prior_evidence_sources"] == []
    assert record["prior_observability_complete"] is False
    assert set(record["missing_local_inputs"]) == {
        "pdb_structure",
        "sifts_mapping",
        "pair_qc",
        "residue_mapping",
        "afdb_metadata",
        "afdb_model",
        "afdb_pae",
        "afdb_plddt",
    }


def test_partial_local_inputs_are_distinguished_and_preferred_in_batch(tmp_path: Path) -> None:
    inventory = _load_inventory_module()
    pdb_path = tmp_path / "data/raw/pdb/1abc.cif"
    sifts_path = tmp_path / "data/raw/mappings/1abc.xml.gz"
    pdb_path.parent.mkdir(parents=True)
    sifts_path.parent.mkdir(parents=True)
    pdb_path.write_text("data_test\n#\n", encoding="utf-8")
    sifts_path.write_bytes(b"local-test-evidence")
    row = {
        "candidate_index": 1,
        "canonical_source_row": 0,
        "polymer_entity_id": "1ABC_1",
        "pair_id": "1abc_A__P00001",
        "PDB": "1abc",
        "chain": "A",
        "UniProt": "P00001",
        "sequence_cluster": "30:1",
        "length": 120,
        "afdb_global_plddt": 93.0,
    }

    record = inventory.build_candidate_record(row, project_root=tmp_path)
    assert record["estimated_P0_readiness"] == "partial_local_inputs"

    sparse = {
        **record,
        "candidate_index": 1,
        "pair_id": "sparse",
        "UniProt": "P00001",
        "sequence_cluster": "cluster-1",
        "missing_local_inputs": [f"missing-{index}" for index in range(8)],
    }
    richer = {
        **record,
        "candidate_index": 2,
        "pair_id": "richer",
        "UniProt": "P00002",
        "sequence_cluster": "cluster-2",
        "missing_local_inputs": ["pair_qc", "residue_mapping"],
    }
    selected = inventory.select_batch1([sparse, richer], target=1)

    assert selected[0]["pair_id"] == "richer"


def test_batch1_is_deterministic_acquisition_panel_with_evidence_priority() -> None:
    inventory = _load_inventory_module()
    candidates = []
    for index in range(1, 41):
        if index <= 4:
            prior = (
                "low_confidence_local_prior",
                "high_pae_long_range_prior",
                "state_disagreement_prior",
                "low_confidence_local_prior",
            )[index - 1]
            evidence_status = "positive_complete"
        elif index <= 30:
            prior = "uncertain_or_unclassified"
            evidence_status = "unobserved"
        else:
            prior = "easy_control_prior"
            evidence_status = "positive_complete"
        candidates.append(
            {
                "candidate_index": index,
                "pair_id": f"pair-{index}",
                "UniProt": f"P{index:05d}",
                "sequence_cluster": None if index in {39, 40} else f"cluster-{index}",
                "sampling_stratum_prior": prior,
                "prior_evidence_status": evidence_status,
                "estimated_P0_readiness": (
                    "likely_ready" if index % 3 == 0 else "insufficient_local_evidence"
                ),
                "local_data_complete": index % 3 == 0,
                "missing_local_inputs": [] if index % 3 == 0 else ["pair_qc"],
                "canonical_uniprot_length": 90 + index * 20,
                "global_pLDDT_proxy": 70.0 + index / 10,
                "round1_member": False,
            }
        )

    selected = inventory.select_batch1(candidates, target=12)
    reversed_selected = inventory.select_batch1(list(reversed(candidates)), target=12)

    assert len(selected) == 12
    assert [row["candidate_index"] for row in selected] == [
        row["candidate_index"] for row in reversed_selected
    ]
    assert len({row["UniProt"] for row in selected}) == 12
    known_clusters = [row["sequence_cluster"] for row in selected if row["sequence_cluster"]]
    assert len(known_clusters) == len(set(known_clusters))
    assert all(
        row["sampling_stratum_prior"] != "easy_control_prior" for row in selected
    )
    assert {
        row["candidate_index"] for row in selected[:4]
    } == {1, 2, 3, 4}
    assert any(row["estimated_P0_readiness"] == "likely_ready" for row in selected)
    assert any(row["sampling_stratum_prior"] == "uncertain_or_unclassified" for row in selected)
    assert all("acquisition_panel" in row["selection_reason"] for row in selected)


def test_batch1_length_diversity_falls_back_to_discovery_length() -> None:
    inventory = _load_inventory_module()
    candidate = {
        "candidate_index": 1,
        "pair_id": "pair-1",
        "UniProt": "P00001",
        "sequence_cluster": "cluster-1",
        "sampling_stratum_prior": "uncertain_or_unclassified",
        "prior_evidence_status": "unobserved",
        "estimated_P0_readiness": "partial_local_inputs",
        "local_data_complete": False,
        "missing_local_inputs": ["pair_qc"],
        "canonical_uniprot_length": None,
        "discovery_pdb_entity_length": 420,
        "global_pLDDT_proxy": 90.0,
        "round1_member": False,
    }

    selected = inventory.select_batch1([candidate], target=1)

    assert "length_bucket=300_599" in selected[0]["selection_reason"]


def _complete_local_availability() -> dict[str, bool]:
    return {
        "pdb_structure": True,
        "residue_mapping": True,
        "afdb_model": True,
        "afdb_pae": True,
        "afdb_plddt": True,
    }


def _complete_a0_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "screening_index": 101,
        "primary_category": "easy_control",
        "classification_status": "complete_classification",
        "full_diagnostic_complete": True,
        "main_quality_pass": True,
        "is_easy_control": True,
        "is_low_conf_local": False,
        "is_high_pae_long_range": False,
        "is_high_conf_state_disagreement": False,
        "easy_control_evidence_available": True,
        "low_conf_evidence_available": True,
        "high_pae_evidence_available": True,
        "state_disagreement_evidence_available": True,
    }
    evidence.update(overrides)
    return evidence


def test_global_plddt_alone_cannot_establish_easy_or_low_confidence_prior() -> None:
    inventory = _load_inventory_module()

    for global_plddt in (69.0, 95.0):
        assessment = inventory.assess_sampling_prior(
            a0_summary={},
            local_availability={},
            global_plddt=global_plddt,
        )
        assert assessment["sampling_stratum_prior"] == "uncertain_or_unclassified"
        assert assessment["prior_evidence_status"] == "unobserved"
        assert assessment["prior_observability_complete"] is False
        assert assessment["supported_sampling_priors"] == []


def test_easy_prior_requires_complete_classification_and_provenance() -> None:
    inventory = _load_inventory_module()

    complete = inventory.assess_sampling_prior(
        a0_summary=_complete_a0_evidence(),
        local_availability=_complete_local_availability(),
        global_plddt=95.0,
    )
    assert complete["sampling_stratum_prior"] == "easy_control_prior"
    assert complete["prior_evidence_status"] == "positive_complete"
    assert complete["prior_observability_complete"] is True
    assert complete["supported_sampling_priors"] == ["easy_control_prior"]
    assert "reports/a0_candidate_summary.csv:screening_index=101" in complete[
        "prior_evidence_sources"
    ]

    for missing in ("pdb_structure", "residue_mapping", "afdb_model", "afdb_pae"):
        availability = _complete_local_availability()
        availability[missing] = False
        incomplete = inventory.assess_sampling_prior(
            a0_summary=_complete_a0_evidence(),
            local_availability=availability,
            global_plddt=95.0,
        )
        assert incomplete["sampling_stratum_prior"] == "uncertain_or_unclassified"
        assert incomplete["prior_evidence_status"] == "positive_partial"
        assert incomplete["prior_observability_complete"] is False


def test_non_unknown_priors_require_matching_positive_evidence() -> None:
    inventory = _load_inventory_module()
    cases = [
        (
            "low_confidence_local",
            "is_low_conf_local",
            "low_conf_evidence_available",
            "low_confidence_local_prior",
        ),
        (
            "high_pae_long_range",
            "is_high_pae_long_range",
            "high_pae_evidence_available",
            "high_pae_long_range_prior",
        ),
        (
            "high_confidence_state_disagreement",
            "is_high_conf_state_disagreement",
            "state_disagreement_evidence_available",
            "state_disagreement_prior",
        ),
    ]
    for primary, label, availability_field, expected in cases:
        evidence = _complete_a0_evidence(
            primary_category=primary,
            is_easy_control=False,
            **{label: True, availability_field: True},
        )
        complete = inventory.assess_sampling_prior(
            a0_summary=evidence,
            local_availability=_complete_local_availability(),
            global_plddt=95.0,
        )
        assert complete["sampling_stratum_prior"] == expected
        assert complete["prior_evidence_status"] == "positive_complete"

        evidence[availability_field] = False
        unsupported = inventory.assess_sampling_prior(
            a0_summary=evidence,
            local_availability=_complete_local_availability(),
            global_plddt=95.0,
        )
        assert unsupported["sampling_stratum_prior"] == "uncertain_or_unclassified"
        assert unsupported["prior_evidence_status"] == "positive_partial"


def test_multi_prior_overlap_uses_documented_primary_and_preserves_all_support() -> None:
    inventory = _load_inventory_module()
    evidence = _complete_a0_evidence(
        primary_category="high_confidence_state_disagreement",
        is_easy_control=False,
        is_low_conf_local=True,
        is_high_pae_long_range=True,
        is_high_conf_state_disagreement=True,
    )

    assessment = inventory.assess_sampling_prior(
        a0_summary=evidence,
        local_availability=_complete_local_availability(),
        global_plddt=95.0,
    )

    assert assessment["sampling_stratum_prior"] == "state_disagreement_prior"
    assert assessment["supported_sampling_priors"] == [
        "state_disagreement_prior",
        "high_pae_long_range_prior",
        "low_confidence_local_prior",
    ]
    assert assessment["sampling_stratum_prior_source"] == (
        "documented_a0_primary_category_precedence_sampling_only"
    )


def test_real_former_proxy_easy_candidate_becomes_unclassified() -> None:
    inventory = _load_inventory_module()
    payload = inventory.build_inventory(
        inventory.PROJECT_ROOT,
        inventory.PROJECT_ROOT / inventory.CANONICAL_SOURCE,
    )
    candidate = next(row for row in payload["candidates"] if row["candidate_index"] == 1)

    assert candidate["pair_id"] == "2vb1_A__P00698"
    assert candidate["global_pLDDT_proxy"] >= 90.0
    assert candidate["sampling_stratum_prior"] == "uncertain_or_unclassified"
    assert candidate["prior_evidence_status"] == "unobserved"


def test_repository_inventory_preserves_source_binding_and_positive_prior_contract() -> None:
    inventory = _load_inventory_module()
    payload = inventory.build_inventory(
        inventory.PROJECT_ROOT,
        inventory.PROJECT_ROOT / inventory.CANONICAL_SOURCE,
    )

    assert payload["source_metadata"] == {
        "canonical_source_path": "data/processed/discovery/discovered_candidates.parquet",
        "source_sha256": "dddb21ef1e41babb71827ba729eabe5c568ab6a7d6d6c5953f6da2e10bd98436",
        "total_row_count": 220,
        "eligible_row_count": 213,
        "indexing_policy_version": "inventory-source-order-v1",
    }
    assert [row["candidate_index"] for row in payload["candidates"]] == list(range(1, 214))
    positive = [
        row
        for row in payload["candidates"]
        if row["sampling_stratum_prior"] != "uncertain_or_unclassified"
    ]
    assert positive
    assert all(row["prior_evidence_status"] == "positive_complete" for row in positive)
    assert all(row["prior_observability_complete"] is True for row in positive)
    assert all(row["prior_evidence_sources"] for row in positive)


def test_sampling_prior_is_explicitly_non_inferential() -> None:
    inventory = _load_inventory_module()

    assert inventory.SAMPLING_PRIOR_DISCLAIMER == (
        "sampling_stratum_prior is a non-inferential sampling aid; it is not a "
        "formal P6 mechanism label and does not change candidate admission."
    )
    assert inventory.EVIDENCE_POLICY["identity_source_available"] == (
        "True only when local pair-QC or Round-1 provides a numeric sequence-identity value; "
        "canonical discovery identity is tracked separately."
    )


def test_round1_comparison_names_local_data_selection_enrichment() -> None:
    inventory = _load_inventory_module()
    base = {
        "canonical_uniprot_length": 200,
        "discovery_pdb_entity_length": 190,
        "global_pLDDT_proxy": 90.0,
        "sampling_stratum_prior": "easy_control_prior",
        "prior_evidence_status": "positive_complete",
        "prior_observability_complete": True,
        "supported_sampling_priors": ["easy_control_prior"],
        "pair_id": "synthetic_A__P00001",
        "candidate_index": 1,
        "mapping_available": False,
        "pair_qc_status": None,
        "fragment_metadata_available": False,
        "single_fragment_full_coverage_possible": None,
        "estimated_P0_readiness": "partial_local_inputs",
        "mapped_length": None,
        "missing_local_inputs": ["pair_qc"],
        "known_sequence_mismatch_flag": None,
        "known_provenance_issue": None,
    }
    candidates = [
        {**base, "round1_member": True, "local_data_complete": True},
        {**base, "round1_member": False, "local_data_complete": False},
    ]

    comparison = inventory.summarize_inventory(candidates)[
        "round1_vs_remaining_descriptive_comparison"
    ]

    assert comparison["observed_selection_enrichment"] == (
        "Round-1 is strongly enriched for locally complete inputs; its global-pLDDT "
        "distribution is modestly higher. Mechanism-rich priors in Round-1 reflect existing "
        "completed diagnostics and must not be interpreted as remaining-pool prevalence."
    )


def test_tsv_writer_uses_repository_lf_line_endings(tmp_path: Path) -> None:
    inventory = _load_inventory_module()
    output = tmp_path / "inventory.tsv"

    inventory._write_tsv(output, [{"candidate_index": 1}], ["candidate_index"])

    assert output.read_bytes() == b"candidate_index\n1\n"
