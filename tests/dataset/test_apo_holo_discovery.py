from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dual_uq.dataset import apo_holo


def test_holo_anchor_query_requires_experimental_protein_and_nonpolymer() -> None:
    query = apo_holo.build_holo_anchor_query(25)
    assert query["return_type"] == "polymer_entity"
    assert query["request_options"]["paginate"]["rows"] == 25
    text = json.dumps(query)
    assert "selected_polymer_entity_types" in text
    assert "non_polymer" in text or "nonpolymer" in text


def test_uniprot_query_is_exact_and_experimental() -> None:
    query = apo_holo.build_uniprot_entity_query("P00001", 10)
    text = json.dumps(query)
    assert query["return_type"] == "polymer_entity"
    assert "P00001" in text
    assert "experimental" in text
    assert "nonpolymer_entity_count" not in text


def test_anchor_id_discovery_deduplicates_and_preserves_order() -> None:
    response = {
        "result_set": [
            {"identifier": "1abc_1"},
            {"identifier": "1abc_1"},
            {"identifier": "2def_1"},
            {"identifier": ""},
        ]
    }
    assert apo_holo.discover_holo_anchor_ids(response) == ["1abc_1", "2def_1"]


def test_graphql_normalization_preserves_missing_fields_as_unknown() -> None:
    result = apo_holo.normalize_graphql_entities(
        {"data": {"entries": [{"rcsb_id": "1abc"}]}}
    )
    assert result.loc[0, "pdb_id"] == "1abc"
    assert pd.isna(result.loc[0, "uniprot_id"])
    assert result.loc[0, "metadata_status"] == "partial"


def test_graphql_batch_reuses_completed_chunk(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_post(url: str, payload: dict) -> dict:
        calls.append(payload)
        return {"data": {"entries": [{"rcsb_id": "1abc"}]}}

    monkeypatch.setattr(apo_holo, "post_json", fake_post)
    cache = tmp_path / "metadata.json"
    first = apo_holo.fetch_graphql_metadata_batch(["1abc_1"], cache)
    second = apo_holo.fetch_graphql_metadata_batch(["1abc_1"], cache)
    assert len(calls) == 1
    assert first.equals(second)


def test_graphql_batch_uses_shard_cache_without_rewriting_parent(monkeypatch, tmp_path: Path) -> None:
    def fake_post(url: str, payload: dict) -> dict:
        return {"data": {"entries": [{"rcsb_id": "1abc"}]}}

    monkeypatch.setattr(apo_holo, "post_json", fake_post)
    cache = tmp_path / "metadata.json"
    apo_holo.fetch_graphql_metadata_batch(["1abc_1"], cache)
    assert not cache.exists()
    shards = list((tmp_path / "metadata.chunks").glob("*.json"))
    assert len(shards) == 1


def test_failed_graphql_chunk_is_recorded_not_negative_evidence(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(apo_holo, "post_json", fail)
    result = apo_holo.fetch_graphql_metadata_batch(["1abc_1"], tmp_path / "metadata.json")
    assert result.loc[0, "metadata_status"] == "network_unresolved"


def test_graphql_errors_are_retriable_and_not_cached_as_complete(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_post(url: str, payload: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"errors": [{"message": "schema rejected"}]}

    monkeypatch.setattr(apo_holo, "post_json", fake_post)
    cache = tmp_path / "metadata.json"
    first = apo_holo.fetch_graphql_metadata_batch(["1abc_1"], cache)
    second = apo_holo.fetch_graphql_metadata_batch(["1abc_1"], cache)
    assert calls == 2
    assert first.loc[0, "metadata_status"] == "network_unresolved"
    assert second.loc[0, "metadata_status"] == "network_unresolved"


def test_expanded_discovery_builds_anchor_and_counterpart_funnel(
    monkeypatch, tmp_path: Path
) -> None:
    search_calls: list[dict] = []

    def fake_post(url: str, payload: dict) -> dict:
        if url == apo_holo.RCSB_SEARCH_API:
            search_calls.append(payload)
            has_accession = any(
                node.get("parameters", {}).get("attribute")
                == "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession"
                for node in payload["query"]["nodes"]
            )
            return {
                "total_count": 2,
                "result_set": (
                    [{"identifier": "1abc_1"}, {"identifier": "1abc_1"}]
                    if has_accession
                    else [{"identifier": "1abc_1"}]
                ),
            }
        return {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "1abc",
                        "rcsb_entry_info": {"experimental_method": ["X-RAY"]},
                        "rcsb_entry_container_identifiers": {
                            "non_polymer_entity_ids": ["2"],
                            "assembly_ids": ["1"],
                        },
                        "polymer_entities": [
                            {
                                "rcsb_id": "1abc_1",
                                "entity_id": "1",
                                "entity_poly": {
                                    "pdbx_seq_one_letter_code_can": "A" * 100,
                                    "rcsb_sample_sequence_length": 100,
                                },
                                "rcsb_polymer_entity_container_identifiers": {
                                    "entity_id": "1",
                                    "auth_asym_ids": ["A"],
                                    "reference_sequence_identifiers": [
                                        {
                                            "database_name": "UniProt",
                                            "database_accession": "P00001",
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ]
            }
        }

    monkeypatch.setattr(apo_holo, "post_json", fake_post)
    census, anchors = apo_holo.run_expanded_discovery(
        apo_holo.ApoHoloConfig(tmp_path, max_polymer_entities=10)
    )
    assert {"discovery_stage", "uniprot_id", "metadata_status"}.issubset(census.columns)
    assert set(anchors["uniprot_id"]) == {"P00001"}
    assert len(search_calls) == 2
    assert census.attrs["census_truncated"] is True


def test_counterpart_deduplication_does_not_mark_complete_pagination_truncated(monkeypatch, tmp_path: Path) -> None:
    def fake_search(query: dict, *, cache_path=None, cache_key=None):
        return ["1abc_1", "2abc_1"], 3

    def fake_graphql(ids: list[str], cache_path: Path) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "polymer_entity_id": ["1abc_1", "2abc_1"],
                "pdb_id": ["1abc", "2abc"],
                "uniprot_id": ["P00001", "P00002"],
                "metadata_status": ["complete", "complete"],
            }
        )

    monkeypatch.setattr(apo_holo, "_search_identifiers", fake_search)
    monkeypatch.setattr(apo_holo, "fetch_graphql_metadata_batch", fake_graphql)
    result = apo_holo.discover_uniprot_counterparts(
        ["P00001", "P00002"], tmp_path, rows=1000
    )
    assert len(result) == 2
    assert result.attrs["census_truncated"] is False


def test_expanded_multi_page_anchor_census_is_not_marked_truncated(monkeypatch, tmp_path: Path) -> None:
    pages = [
        pd.DataFrame({"polymer_entity_id": ["1abc_1"], "uniprot_id": ["P00001"]}),
        pd.DataFrame({"polymer_entity_id": ["2abc_1"], "uniprot_id": ["P00001"]}),
    ]
    for frame in pages:
        frame["metadata_status"] = "complete"
        frame.attrs.update(total_count=2, retrieved_count=1, census_truncated=True)
    pages[-1].attrs["census_truncated"] = False
    calls = iter(pages)

    def fake_anchor(*args, **kwargs):
        return next(calls)

    def fake_counterparts(*args, **kwargs):
        result = pd.DataFrame(columns=["polymer_entity_id", "uniprot_id", "metadata_status"])
        result.attrs.update(total_count=0, retrieved_count=0, census_truncated=False)
        return result

    monkeypatch.setattr(apo_holo, "discover_holo_anchor_accessions", fake_anchor)
    monkeypatch.setattr(apo_holo, "discover_uniprot_counterparts", fake_counterparts)
    census, _ = apo_holo.run_expanded_discovery(
        apo_holo.ApoHoloConfig(tmp_path, census_page_size=1, max_polymer_entities=None)
    )
    assert len(census) == 2
    assert census.attrs["census_truncated"] is False
