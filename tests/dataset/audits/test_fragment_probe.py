from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/dual_uq/dataset/audits/fragments.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("v5_fragment_empirical_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(
    protein_id: str,
    full_length: int | None,
    status: str,
    *,
    screening_index: int,
    interval: tuple[int, int] | None = None,
    fragment_intervals: tuple[tuple[int, int], ...] = (),
) -> dict[str, object]:
    start, end = interval if interval is not None else (None, None)
    return {
        "protein_id": protein_id,
        "screening_index": screening_index,
        "uniprot_full_length": full_length,
        "selected_model_entity_id": f"AF-{protein_id}-F1",
        "fragment_number": 1,
        "fragment_start": start,
        "fragment_end": end,
        "fragment_length": None if start is None else end - start + 1,
        "fragmentation_status": status,
        "is_fragmented_protein": (
            False if status == "single_fragment" else True if status == "multi_fragment" else None
        ),
        "fragment_count": len(fragment_intervals) or None,
        "fragment_intervals": [list(item) for item in fragment_intervals],
    }


def test_pure_single_fragment_cohort_does_not_infer_threshold() -> None:
    probe = _load_probe()
    observations = [
        _observation("p2", 200, "single_fragment", screening_index=2, interval=(1, 200)),
        _observation("p1", 100, "single_fragment", screening_index=1, interval=(1, 100)),
    ]

    summary = probe.summarize_observations(observations)

    assert summary["single_fragment_count"] == 2
    assert summary["multi_fragment_count"] == 0
    assert summary["unknown_fragmentation_count"] == 0
    assert summary["max_full_length_single_fragment"] == 200
    assert summary["min_full_length_multi_fragment"] is None
    assert summary["empirical_transition_interval"] is None
    assert summary["threshold_identifiable"] is False
    assert summary["fragmentation_threshold"] is None


def test_clear_single_to_multi_transition_reports_interval_not_point_threshold() -> None:
    probe = _load_probe()
    observations = [
        _observation("s1", 900, "single_fragment", screening_index=1, interval=(1, 900)),
        _observation("s2", 1000, "single_fragment", screening_index=2, interval=(1, 1000)),
        _observation(
            "m1",
            1200,
            "multi_fragment",
            screening_index=3,
            interval=(1, 800),
            fragment_intervals=((1, 800), (401, 1200)),
        ),
        _observation(
            "m2",
            1400,
            "multi_fragment",
            screening_index=4,
            interval=(1, 800),
            fragment_intervals=((1, 800), (601, 1400)),
        ),
    ]

    summary = probe.summarize_observations(observations)

    assert summary["max_full_length_single_fragment"] == 1000
    assert summary["min_full_length_multi_fragment"] == 1200
    assert summary["empirical_transition_interval"] == {
        "lower_exclusive": 1000,
        "upper_inclusive": 1200,
        "notation": "(1000, 1200]",
    }
    assert summary["threshold_identifiable"] is True
    assert summary["fragmentation_threshold"] is None


def test_overlapping_single_and_multi_lengths_are_not_identifiable() -> None:
    probe = _load_probe()
    observations = [
        _observation("s1", 1300, "single_fragment", screening_index=1, interval=(1, 1300)),
        _observation("s2", 1400, "single_fragment", screening_index=2, interval=(1, 1400)),
        _observation(
            "m1",
            1200,
            "multi_fragment",
            screening_index=3,
            interval=(1, 800),
            fragment_intervals=((1, 800), (401, 1200)),
        ),
        _observation(
            "m2",
            1500,
            "multi_fragment",
            screening_index=4,
            interval=(1, 800),
            fragment_intervals=((1, 800), (701, 1500)),
        ),
    ]

    summary = probe.summarize_observations(observations)

    assert summary["empirical_transition_interval"] is None
    assert summary["threshold_identifiable"] is False
    assert "overlap" in summary["conclusion"]


def test_missing_metadata_is_structured_as_unknown_with_nulls() -> None:
    probe = _load_probe()
    record = {"protein_id": "index9", "screening_index": 9, "uniprot_full_length": None}

    observation = probe.build_fragment_observation(record, pair_qc={}, metadata=None)

    assert observation["fragmentation_status"] == "unknown"
    assert observation["is_fragmented_protein"] is None
    assert observation["selected_model_entity_id"] is None
    assert observation["fragment_start"] is None
    assert observation["fragment_end"] is None
    assert observation["fragment_length"] is None
    assert observation["fragment_count"] is None
    assert observation["evidence_note"] == "missing selected AFDB metadata"


def test_explicit_full_length_metadata_classifies_single_fragment() -> None:
    probe = _load_probe()
    record = {"protein_id": "index2", "screening_index": 2, "uniprot_full_length": 246}
    metadata = {
        "modelEntityId": "AF-P00760-F1",
        "uniprotStart": 1,
        "uniprotEnd": 246,
    }

    observation = probe.build_fragment_observation(record, pair_qc={}, metadata=metadata)

    assert observation["fragment_number"] == 1
    assert observation["fragment_start"] == 1
    assert observation["fragment_end"] == 246
    assert observation["fragment_length"] == 246
    assert observation["fragmentation_status"] == "single_fragment"
    assert observation["is_fragmented_protein"] is False
    assert observation["fragment_count"] is None


def test_explicit_multiple_intervals_classify_multi_and_override_fragment_length_proxy() -> None:
    probe = _load_probe()
    record = {"protein_id": "index9", "screening_index": 9, "uniprot_full_length": 126}
    metadata = {
        "modelEntityId": "AF-0000000365840311",
        "uniprotStart": 1368,
        "uniprotEnd": 1493,
    }
    preflight = {
        "canonical_uniprot_length": 7095,
        "afdb_fragment_intervals": "[[13, 127], [880, 1050], [1368, 1493]]",
        "mapped_uniprot_start": 1024,
        "mapped_uniprot_end": 1192,
        "afdb_coverage_status": "unsupported_afdb_fragment",
    }

    observation = probe.build_fragment_observation(
        record, pair_qc={}, metadata=metadata, preflight=preflight
    )

    assert observation["uniprot_full_length"] == 7095
    assert observation["uniprot_full_length_source"] == "preflight.canonical_uniprot_length"
    assert observation["fragment_number"] is None
    assert observation["fragment_count"] == 3
    assert observation["fragmentation_status"] == "multi_fragment"
    assert observation["is_fragmented_protein"] is True


def test_one_multi_fragment_sample_is_insufficient_near_boundary() -> None:
    probe = _load_probe()
    observations = [
        _observation("s1", 500, "single_fragment", screening_index=1, interval=(1, 500)),
        _observation("s2", 1000, "single_fragment", screening_index=2, interval=(1, 1000)),
        _observation(
            "m1",
            1200,
            "multi_fragment",
            screening_index=3,
            interval=(1, 800),
            fragment_intervals=((1, 800), (401, 1200)),
        ),
    ]

    summary = probe.summarize_observations(observations)

    assert summary["empirical_transition_interval"]["notation"] == "(1000, 1200]"
    assert summary["threshold_identifiable"] is False
    assert summary["fragmentation_threshold"] is None
    assert "insufficient" in summary["conclusion"]


def test_uniform_fragment_geometry_is_reported_but_irregular_geometry_is_null() -> None:
    probe = _load_probe()
    uniform = _observation(
        "m1",
        1800,
        "multi_fragment",
        screening_index=1,
        interval=(1, 800),
        fragment_intervals=((1, 800), (501, 1300), (1001, 1800)),
    )
    irregular = _observation(
        "m2",
        1700,
        "multi_fragment",
        screening_index=2,
        interval=(1, 700),
        fragment_intervals=((1, 700), (450, 1200), (1000, 1700)),
    )

    uniform_summary = probe.summarize_observations([uniform])
    irregular_summary = probe.summarize_observations([irregular])

    assert uniform_summary["observed_fragment_length"] == 800
    assert uniform_summary["observed_step"] == 500
    assert irregular_summary["observed_fragment_length"] is None
    assert irregular_summary["observed_step"] is None


def test_output_order_is_deterministic_and_index9_regression_is_preserved() -> None:
    probe = _load_probe()
    records = [
        _observation("index10", 227, "single_fragment", screening_index=10, interval=(1, 227)),
        {
            **_observation(
                "index9",
                7095,
                "multi_fragment",
                screening_index=9,
                interval=(1368, 1493),
                fragment_intervals=((880, 1050), (1368, 1493)),
            ),
            "selected_model_entity_id": "AF-0000000365840311",
            "fragment_number": None,
            "mapped_uniprot_start": 1024,
            "mapped_uniprot_end": 1192,
            "afdb_coverage_status": "unsupported_afdb_fragment",
        },
    ]

    payload = probe.build_report(records, head="0cb971b")

    assert [item["screening_index"] for item in payload["proteins"]] == [9, 10]
    assert payload["source"] == {"head": "0cb971b", "cohort_size": 2}
    assert payload["index9_regression"] == {
        "screening_index": 9,
        "mapped_interval": [1024, 1192],
        "selected_fragment_interval": [1368, 1493],
        "afdb_coverage_status": "unsupported_afdb_fragment",
        "regression_pass": True,
    }
