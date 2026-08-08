from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from dual_uq.core.hashing import sha256_bytes
from dual_uq.dataset.releases.intervention_admission import (
    ADMISSION_VERSION,
    ORIGINAL_PANEL_SHA256,
    InterventionAdmissionError,
    build_admission_records,
    build_admitted_subset,
    build_release_metadata,
    render_admission_jsonl,
    render_variant_review_packets,
    validate_stage0_admission_bindings,
    write_immutable_release,
)
from dual_uq.dataset.releases.intervention_panel import EXPECTED_STAGE0_PANEL

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STAGE0_DIR = REPOSITORY_ROOT / "experiments/p2_design_baseline/stage0"
PANEL_PATH = STAGE0_DIR / "stage0_intervention_panel_v1.jsonl"
ADMISSION_PATH = STAGE0_DIR / "stage0_intervention_admission_v1.jsonl"
ADMITTED_PATH = STAGE0_DIR / "stage0_intervention_admitted_v1.jsonl"
METADATA_PATH = STAGE0_DIR / "stage0_intervention_admission_v1.meta.json"
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/design_baseline/stage0.yaml"


def _panel_records() -> list[dict[str, object]]:
    return [json.loads(line) for line in PANEL_PATH.read_text().splitlines()]


def test_frozen_panel_is_immutable_and_admission_materializes_exact_11() -> None:
    before = PANEL_PATH.read_bytes()

    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=sha256_bytes(before),
    )

    assert sha256_bytes(before) == ORIGINAL_PANEL_SHA256
    assert PANEL_PATH.read_bytes() == before
    assert tuple((row["candidate_index"], row["protein_id"]) for row in records) == (
        EXPECTED_STAGE0_PANEL
    )
    assert all(row["declared_panel_member"] is True for row in records)
    assert all(row["admission_version"] == ADMISSION_VERSION for row in records)


def test_admission_records_preserve_frozen_classifications_and_values() -> None:
    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=ORIGINAL_PANEL_SHA256,
    )
    by_id = {row["protein_id"]: row for row in records}

    assert Counter(row["intervention_admission_status"] for row in records) == {
        "ADMITTED": 8,
        "PENDING_HUMAN_VARIANT_REVIEW": 2,
        "IDENTITY_CONTRACT_FAIL": 1,
    }
    assert by_id["2ykz_A__P00138"]["pdr01_mismatch_count"] == 1
    assert by_id["2ykz_A__P00138"]["pdr01_sequence_identity_paired"] == 0.9920634921
    assert by_id["2ykz_A__P00138"]["observed_sequence_variants"] == ["A39V"]
    assert by_id["1ix9_A__P00448"]["pdr01_mismatch_count"] == 1
    assert by_id["1ix9_A__P00448"]["pdr01_sequence_identity_paired"] == 0.9951219512
    assert by_id["1ix9_A__P00448"]["observed_sequence_variants"] == ["Y175F"]
    assert by_id["6jgj_A__P42212"]["pdr01_mismatch_count"] == 5
    assert by_id["6jgj_A__P42212"]["pdr01_sequence_identity_paired"] == 0.9779735683
    assert by_id["6jgj_A__P42212"]["observed_sequence_variants"] == [
        "Q80R",
        "F99S",
        "M153T",
        "V163A",
        "E222Q",
    ]
    for protein_id in ("2ykz_A__P00138", "1ix9_A__P00448", "6jgj_A__P42212"):
        assert by_id[protein_id]["authorized_sequence_variants"] is None

    admitted = [row for row in records if row["intervention_admission_status"] == "ADMITTED"]
    assert all(row["pdr01_mismatch_count"] is None for row in admitted)
    assert all(row["pdr01_sequence_identity_paired"] is None for row in admitted)
    assert all(row["metric_resolution_status"] == "not_serialized_in_stage0_1a" for row in admitted)
    assert all(row["observed_sequence_variants"] is None for row in admitted)


def test_admitted_subset_is_mechanical_deterministic_filter() -> None:
    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=ORIGINAL_PANEL_SHA256,
    )
    ledger = render_admission_jsonl(records)
    ledger_sha256 = sha256_bytes(ledger)

    first = build_admitted_subset(
        records,
        admission_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        admission_sha256=ledger_sha256,
    )
    second = build_admitted_subset(
        list(reversed(records)),
        admission_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        admission_sha256=ledger_sha256,
    )

    assert tuple((row["candidate_index"], row["protein_id"]) for row in first) == tuple(
        (row["candidate_index"], row["protein_id"])
        for row in records
        if row["intervention_admission_status"] == "ADMITTED"
    )
    assert first == second
    assert all(row["intervention_admission_status"] == "ADMITTED" for row in first)
    assert not {22, 77, 179} & {row["candidate_index"] for row in first}

    swapped = [dict(row) for row in records]
    next(row for row in swapped if row["candidate_index"] == 21)[
        "intervention_admission_status"
    ] = "PENDING_HUMAN_VARIANT_REVIEW"
    next(row for row in swapped if row["candidate_index"] == 77)[
        "intervention_admission_status"
    ] = "ADMITTED"
    swapped_subset = build_admitted_subset(
        swapped,
        admission_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        admission_sha256=ledger_sha256,
    )
    assert {row["candidate_index"] for row in swapped_subset} == {
        row["candidate_index"]
        for row in swapped
        if row["intervention_admission_status"] == "ADMITTED"
    }


def test_rendering_and_review_packets_are_deterministic_and_decision_free() -> None:
    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=ORIGINAL_PANEL_SHA256,
    )
    first = render_admission_jsonl(records)
    second = render_admission_jsonl(list(reversed(records)))
    packets = render_variant_review_packets(records)

    assert first == second
    assert first.endswith(b"\n")
    assert b"/home/" not in first
    assert set(packets) == {
        "variant_review_2ykz_A39V.md",
        "variant_review_1ix9_Y175F.md",
    }
    for payload in packets.values():
        lowered = payload.decode("utf-8").lower()
        assert "/home/" not in lowered
        assert "approve" not in lowered
        assert "reject" not in lowered
        assert "recommend" not in lowered
        assert "no exact human variant authorization currently exists" in lowered

    packet_2ykz = packets["variant_review_2ykz_A39V.md"].decode("utf-8")
    assert "`2ykz_A__P00138`" in packet_2ykz
    assert "`A39V`" in packet_2ykz
    assert "UniProt `1–127`" in packet_2ykz
    assert "`0.9920634921`" in packet_2ykz
    assert "U1: canonical Q" in packet_2ykz
    assert "U127: K lacks" in packet_2ykz
    assert "PDR-01_D1-D2_条款草案_v0.2.md" in packet_2ykz
    assert "frozen `P2P3-STAGE0-1A` audit" in packet_2ykz

    packet_1ix9 = packets["variant_review_1ix9_Y175F.md"].decode("utf-8")
    assert "`1ix9_A__P00448`" in packet_1ix9
    assert "`Y175F`" in packet_1ix9
    assert "UniProt `2–206`" in packet_1ix9
    assert "`0.9951219512`" in packet_1ix9
    assert "PDR-01_D1-D2_条款草案_v0.2.md" in packet_1ix9
    assert "frozen `P2P3-STAGE0-1A` audit" in packet_1ix9


def test_immutable_release_reuses_identical_and_rejects_conflict(tmp_path: Path) -> None:
    path = tmp_path / "release.jsonl"

    assert write_immutable_release(path, b"same\n") == "created"
    assert write_immutable_release(path, b"same\n") == "reused_identical"
    with pytest.raises(InterventionAdmissionError, match="Immutable output differs"):
        write_immutable_release(path, b"different\n")


def test_stage0_config_binds_panel_admission_and_mechanical_subset() -> None:
    resolved = validate_stage0_admission_bindings(CONFIG_PATH)

    assert resolved == {
        "stage0_intervention_panel": (
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_panel_v1.jsonl"
        ),
        "stage0_intervention_admission": (
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        "stage0_intervention_admitted_subset": (
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admitted_v1.jsonl"
        ),
    }
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "canonical_census" not in config["inputs"]


def test_stage0_config_rejects_wrong_admission_version(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["inputs"]["stage0_intervention_admission"]["admission_version"] = "wrong"
    path = tmp_path / "stage0.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(InterventionAdmissionError, match="version differs"):
        validate_stage0_admission_bindings(path)


def test_release_metadata_distinguishes_declaration_admission_and_execution() -> None:
    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=ORIGINAL_PANEL_SHA256,
    )
    ledger = render_admission_jsonl(records)
    subset = build_admitted_subset(
        records,
        admission_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        admission_sha256=sha256_bytes(ledger),
    )
    subset_bytes = render_admission_jsonl(subset)

    metadata = build_release_metadata(
        panel_sha256=ORIGINAL_PANEL_SHA256,
        admission_sha256=sha256_bytes(ledger),
        admitted_subset_sha256=sha256_bytes(subset_bytes),
        pdr01_protocol_sha256="a" * 64,
        admission_records=records,
        admitted_records=subset,
    )

    assert metadata["record_counts"] == {
        "historical_declaration": 11,
        "admission_ledger": 11,
        "executable_subset": 8,
    }
    assert metadata["semantic_distinction"] == {
        "stage0_intervention_panel_v1": (
            "historical frozen 11-protein experimental declaration"
        ),
        "stage0_intervention_admission_v1": (
            "current formal admission state of all 11 declared proteins"
        ),
        "stage0_intervention_admitted_v1": (
            "current executable subset under this admission version"
        ),
    }
    assert "timestamp" not in metadata


def test_materialized_outputs_match_release_api() -> None:
    records = build_admission_records(
        _panel_records(),
        panel_path="experiments/p2_design_baseline/stage0/stage0_intervention_panel_v1.jsonl",
        panel_sha256=ORIGINAL_PANEL_SHA256,
    )
    expected_admission = render_admission_jsonl(records)
    subset = build_admitted_subset(
        records,
        admission_path=(
            "experiments/p2_design_baseline/stage0/"
            "stage0_intervention_admission_v1.jsonl"
        ),
        admission_sha256=sha256_bytes(expected_admission),
    )

    assert ADMISSION_PATH.read_bytes() == expected_admission
    assert ADMITTED_PATH.read_bytes() == render_admission_jsonl(subset)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["source_panel_sha256"] == ORIGINAL_PANEL_SHA256
    assert metadata["admission_ledger_sha256"] == sha256_bytes(expected_admission)
    assert metadata["admitted_subset_sha256"] == sha256_bytes(
        render_admission_jsonl(subset)
    )
