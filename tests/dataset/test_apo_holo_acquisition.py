from __future__ import annotations

from pathlib import Path

import pandas as pd

from dual_uq.dataset import apo_holo


def test_metadata_census_is_paginated_and_records_operational_truncation(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_post(url: str, payload: dict) -> dict:
        calls.append(payload["request_options"]["paginate"]["start"])
        return {
            "total_count": 3,
            "result_set": [{"identifier": "1abc_1"}, {"identifier": "2abc_1"}],
        }

    def fake_metadata(identifier: str, *, require_single_uniprot: bool) -> dict:
        accession = "P00001" if identifier == "1abc_1" else "P00002"
        return {
            "polymer_entity_id": identifier,
            "pdb_id": identifier[:4],
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_id": accession,
            "length": 100,
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution": 2.0,
            "sequence_cluster": "30:1",
        }

    def fake_entry(url: str) -> dict:
        return {
            "rcsb_entry_container_identifiers": {
                "non_polymer_entity_ids": [],
                "assembly_ids": ["1"],
            },
            "rcsb_entry_info": {"nonpolymer_entity_count": 0},
        }

    monkeypatch.setattr(apo_holo, "post_json", fake_post)
    monkeypatch.setattr(apo_holo, "fetch_candidate_metadata", fake_metadata)
    monkeypatch.setattr(apo_holo, "request_json", fake_entry)

    config = apo_holo.ApoHoloConfig(tmp_path, census_page_size=2, max_polymer_entities=2)
    census = apo_holo.run_metadata_census(config)

    assert calls == [0]
    assert census["polymer_entity_id"].tolist() == ["1abc_1", "2abc_1"]
    assert census.attrs["total_count"] == 3
    assert census.attrs["census_truncated"] is True


def test_candidate_groups_keep_all_structures_for_same_uniprot() -> None:
    census = pd.DataFrame(
        [
            {"polymer_entity_id": "1abc_1", "uniprot_id": "P00001", "nonpolymer_entity_count": 1},
            {"polymer_entity_id": "2abc_1", "uniprot_id": "P00001", "nonpolymer_entity_count": 0},
            {"polymer_entity_id": "3abc_1", "uniprot_id": "P00002", "nonpolymer_entity_count": 0},
        ]
    )
    groups = apo_holo.identify_candidate_groups(census)
    assert groups["polymer_entity_id"].tolist() == ["1abc_1", "2abc_1"]
    assert groups["candidate_group_status"].eq("potential_apo_holo_group").all()
