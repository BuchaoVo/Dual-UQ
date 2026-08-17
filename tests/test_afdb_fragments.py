from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dual_uq import pairing
from dual_uq.afdb import (
    UnsupportedAFDBFragment,
    fetch_afdb_prediction,
    get_afdb_prediction_metadata,
    select_prediction_for_interval,
)
from dual_uq.preflight import (
    evaluate_afdb_fragment_support,
    finalize_preflight_status,
)


def _record(model_id: str, start: int, end: int, *, version: int = 6) -> dict:
    return {
        "modelEntityId": model_id,
        "entryId": model_id,
        "sequenceStart": start,
        "sequenceEnd": end,
        "uniprotStart": start,
        "uniprotEnd": end,
        "latestVersion": version,
        "isComplex": False,
        "cifUrl": f"https://example.test/{model_id}.cif",
        "plddtDocUrl": f"https://example.test/{model_id}-confidence.json",
        "paeDocUrl": f"https://example.test/{model_id}-pae.json",
    }


def test_selects_f1_when_it_fully_covers_interval() -> None:
    records = [_record("AF-P69441-F1", 1, 214)]

    selected = select_prediction_for_interval(records, (1, 214))

    assert selected["modelEntityId"] == "AF-P69441-F1"


def test_selects_covering_f2_or_f3_when_f1_does_not_cover() -> None:
    records = [
        _record("AF-X-F1", 1, 100),
        _record("AF-X-F3", 180, 400),
        _record("AF-X-F2", 190, 310),
    ]

    selected = select_prediction_for_interval(records, (200, 300))

    assert selected["modelEntityId"] == "AF-X-F2"


def test_multiple_covering_fragments_choose_minimum_redundancy() -> None:
    records = [
        _record("AF-X-WIDE", 50, 350),
        _record("AF-X-TIGHT", 95, 205),
        _record("AF-X-MEDIUM", 80, 220),
    ]

    selected = select_prediction_for_interval(records, (100, 200))

    assert selected["modelEntityId"] == "AF-X-TIGHT"


def test_tie_break_is_deterministic_independent_of_input_order() -> None:
    records = [
        _record("AF-X-F3", 90, 210, version=2),
        _record("AF-X-F2", 90, 210, version=1),
    ]

    forward = select_prediction_for_interval(records, (100, 200))
    reverse = select_prediction_for_interval(list(reversed(records)), (100, 200))

    assert forward["modelEntityId"] == "AF-X-F2"
    assert reverse["modelEntityId"] == "AF-X-F2"


def test_partial_coverage_raises_unsupported_afdb_fragment() -> None:
    records = [
        _record("AF-X-LEFT", 1, 120),
        _record("AF-X-RIGHT", 180, 300),
    ]

    with pytest.raises(UnsupportedAFDBFragment) as caught:
        select_prediction_for_interval(records, (100, 200))

    assert caught.value.mapped_interval == (100, 200)
    assert caught.value.fragment_intervals == ((1, 120), (180, 300))


def test_index9_real_metadata_is_unsupported() -> None:
    root = Path(__file__).resolve().parents[1]
    diagnosis = json.loads(
        (root / "reports/index9_numbering_diagnosis.json").read_text(encoding="utf-8")
    )
    records = [
        _record(
            str(fragment["model_entity_id"]),
            int(fragment["uniprot_start"]),
            int(fragment["uniprot_end"]),
            version=int(fragment["version"]),
        )
        for fragment in diagnosis["afdb_fragment_intervals"]
    ]

    with pytest.raises(UnsupportedAFDBFragment):
        select_prediction_for_interval(records, (1024, 1192))


def test_1ake_real_metadata_still_selects_f1() -> None:
    record = _record("AF-P69441-F1", 1, 214, version=6)

    selected = select_prediction_for_interval([record], (1, 214))

    assert selected["modelEntityId"] == "AF-P69441-F1"
    assert selected["latestVersion"] == 6


def test_unmapped_metadata_selection_prefers_f1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record("AF-X-F2", 1, 100),
        _record("AF-X-F1", 1, 100),
    ]
    monkeypatch.setattr(
        "dual_uq.afdb.get_afdb_prediction_records",
        lambda accession: records,
    )

    selected = get_afdb_prediction_metadata("P00001")

    assert selected["modelEntityId"] == "AF-X-F1"


def test_fetch_downloads_only_selected_fragment_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            **_record("AF-X-F1", 1, 100),
            "sequence": "A" * 100,
        },
        {
            **_record("AF-X-F2", 101, 220),
            "sequence": "C" * 120,
        },
    ]
    downloaded_urls: list[str] = []

    monkeypatch.setattr(
        "dual_uq.afdb.get_afdb_prediction_records",
        lambda accession: records,
    )

    def fake_download(url: str, output: Path) -> Path:
        downloaded_urls.append(url)
        return output

    monkeypatch.setattr("dual_uq.afdb.download_file", fake_download)

    result = fetch_afdb_prediction(
        "P00001",
        tmp_path,
        mapped_interval=(120, 180),
    )

    assert result["prediction"]["modelEntityId"] == "AF-X-F2"
    assert result["canonical_uniprot_length"] == 220
    assert result["fragment_length"] == 120
    assert len(downloaded_urls) == 3
    assert all("AF-X-F2" in url for url in downloaded_urls)
    assert not any("AF-X-F1" in url for url in downloaded_urls)


def test_build_pair_maps_sifts_before_selecting_afdb_fragment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    mapping = pd.DataFrame(
        {
            "pdb_chain_id": ["A", "A", "A"],
            "pdb_residue_number": ["1", "2", "3"],
            "pdb_residue_name": ["ALA", "CYS", "ASP"],
            "uniprot_id": ["P00001", "P00001", "P00001"],
            "uniprot_residue_number": [10, 11, 12],
            "uniprot_residue_name": ["A", "C", "D"],
        }
    )

    monkeypatch.setattr(
        pairing,
        "fetch_pdb_mmcif",
        lambda pdb_id, output_dir: tmp_path / "1abc.cif",
    )

    def fake_fetch_sifts(pdb_id: str, output_dir: Path) -> Path:
        call_order.append("sifts")
        return tmp_path / "1abc.xml.gz"

    def fake_parse_sifts(
        path: Path,
        *,
        chain_id: str,
        uniprot_id: str,
    ) -> pd.DataFrame:
        call_order.append("mapping")
        return mapping

    def fake_fetch_afdb(
        accession: str,
        output_dir: Path,
        *,
        mapped_interval: tuple[int, int],
    ) -> dict:
        call_order.append("afdb")
        assert mapped_interval == (10, 12)
        return {
            "model_path": str(tmp_path / "model.cif"),
            "plddt_path": str(tmp_path / "plddt.json"),
            "pae_path": str(tmp_path / "pae.json"),
            "prediction": {
                **_record("AF-P00001-F2", 10, 12),
                "sequence": "ACD",
            },
            "canonical_uniprot_length": 12,
            "fragment_length": 3,
            "fragment_interval": (10, 12),
        }

    monkeypatch.setattr(pairing, "fetch_sifts_xml", fake_fetch_sifts)
    monkeypatch.setattr(pairing, "parse_sifts_residue_mapping", fake_parse_sifts)
    monkeypatch.setattr(pairing, "fetch_afdb_prediction", fake_fetch_afdb)

    report = pairing.build_pair(
        project_root=tmp_path,
        pdb_id="1abc",
        chain_id="A",
        uniprot_id="P00001",
    )

    assert call_order == ["sifts", "mapping", "afdb"]
    assert report["mapped_uniprot_start"] == 10
    assert report["mapped_uniprot_end"] == 12
    assert report["uniprot_length"] == 12
    assert report["afdb_fragment_length"] == 3


def test_preflight_separates_canonical_and_fragment_lengths() -> None:
    records = [
        _record("AF-X-F1", 1, 100),
        _record("AF-X-F2", 101, 220),
    ]

    support = evaluate_afdb_fragment_support(records, (120, 180))

    assert support["afdb_fragment_status"] == "supported"
    assert support["canonical_uniprot_length"] == 220
    assert support["afdb_fragment_length"] == 120
    assert support["afdb_fragment_start"] == 101
    assert support["afdb_fragment_end"] == 220


def test_preflight_returns_structured_unsupported_status() -> None:
    records = [
        _record("AF-X-LEFT", 1, 120),
        _record("AF-X-RIGHT", 180, 300),
    ]

    support = evaluate_afdb_fragment_support(records, (100, 200))

    assert support["afdb_fragment_status"] == "unsupported_afdb_fragment"
    assert support["preflight_status"] == "unsupported_afdb_fragment"
    assert support["preflight_reason"] == "no_afdb_fragment_covers_mapped_interval"
    assert "failed_runtime" not in support.values()


def test_mapping_quality_and_afdb_coverage_remain_orthogonal() -> None:
    support = evaluate_afdb_fragment_support(
        [
            _record("AF-X-LEFT", 1, 1050),
            _record("AF-X-RIGHT", 1233, 7095),
        ],
        (1024, 1192),
    )
    metrics = {
        "full_length_mapping_coverage": 169 / 7095,
        "entity_mapping_coverage": 169 / 173,
        "sequence_identity": 1.0,
        "observed_ca_fraction_of_mapped": 1.0,
        "internal_unmapped_fraction": 0.0,
    }
    thresholds = {
        "min_full_length_mapping_coverage": 0.90,
        "min_entity_mapping_coverage": 0.90,
        "min_sequence_identity": 0.95,
        "min_observed_ca_fraction": 0.90,
        "warn_full_length_mapping_coverage": 0.70,
        "max_internal_unmapped_fraction": 0.05,
    }

    result = finalize_preflight_status(support, metrics, thresholds)

    assert result["preflight_status"] == "unsupported_afdb_fragment"
    assert result["afdb_coverage_status"] == "unsupported_afdb_fragment"
    assert result["mapping_quality_status"] == "fail_preflight"
    assert "low_full_length_mapping_coverage" in result["mapping_quality_reason"]
    assert result["geometry_launched"] is False
