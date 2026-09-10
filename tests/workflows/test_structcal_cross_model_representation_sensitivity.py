from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.workflows.structcal_cross_model_representation_sensitivity import (
    MODEL_SPECS,
    build_structcal_scorer,
    filter_dynamicmpnn_contiguous_cases,
    load_structcal_model,
    response_shard_path,
    safe_output_path,
    score_cases_with_callback,
)


def _cases() -> pd.DataFrame:
    coordinates = np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
    return pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "protein_id": "P1",
                "identity_cluster_id": "C1",
                "split": "LOCKED_TEST",
                "track_or_diagnostic": "TRACK_I",
                "state_family": None,
                "condition": condition,
                "condition_label": label,
                "canonical_position": position,
                "wt_sequence_projection": "AC",
                "n_canonical_positions": 2,
                "coordinates": coordinates if position == 1 else None,
            }
            for condition, label in (("CONDITION_1", "PDB"), ("CONDITION_2", "AFDB"))
            for position in (1, 2)
        ]
    )


def test_score_cases_uses_same_sequence_axis_for_both_views() -> None:
    calls: list[tuple[str, str, tuple[int, ...], str]] = []

    def score(**kwargs: object) -> np.ndarray:
        calls.append(
            (
                str(kwargs["pair_id"]),
                str(kwargs["condition"]),
                tuple(kwargs["positions"]),  # type: ignore[arg-type]
                str(kwargs["sequence"]),
            )
        )
        result = np.full((2, 20), 0.01 / 19.0)
        result[:, 0] = 0.99
        return result

    residue, pair, quality = score_cases_with_callback(
        _cases(),
        score,
        model_id="fixture",
        checkpoint_id="checkpoint",
        semantic_class="L0",
        regime="operational_pdb_afdb",
    )

    assert calls == [
        ("pair-1", "CONDITION_1", (1, 2), "AC"),
        ("pair-1", "CONDITION_2", (1, 2), "AC"),
    ]
    assert len(residue) == 2
    assert len(pair) == 1
    assert len(quality) == 1


def test_structcal_scorer_keeps_native_pifold_input_contract() -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    class Adapter:
        def probability_distributions(self, structure: object) -> np.ndarray:
            calls.append((tuple(structure.coordinates.shape), structure.sequence_length))  # type: ignore[attr-defined]
            return np.full((2, 20), 0.05)

    scorer = build_structcal_scorer("pifold", Adapter())
    result = scorer(
        sequence="AC",
        coordinates=np.zeros((2, 4, 3), dtype=np.float32),
        positions=(1, 2),
    )

    assert result.shape == (2, 20)
    assert calls == [((2, 4, 3), 2)]


def test_model_loader_forwards_explicit_proteinmpnn_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    adapter = object()

    def load_adapter(**kwargs: object) -> object:
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(
        "dual_uq.workflows.structcal_cross_model_representation_sensitivity."
        "load_authorized_proteinmpnn_adapter",
        load_adapter,
    )
    checkpoint = tmp_path / "custom.pt"
    observed, checkpoint_id = load_structcal_model(
        "proteinmpnn",
        "v_48_020",
        tmp_path,
        tmp_path / "run",
        "cpu",
        checkpoint_path=checkpoint,
        backbone_noise=0.2,
    )

    assert observed is adapter
    assert checkpoint_id == "custom"
    assert captured["checkpoint_path"] == checkpoint
    assert captured["backbone_noise"] == 0.2


def test_model_loader_reads_machine_specific_pifold_paths_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    adapter = object()
    source = tmp_path / "PiFold"
    checkpoint = tmp_path / "checkpoint.pth"

    monkeypatch.setenv("PIFOLD_SOURCE_PATH", str(source))
    monkeypatch.setenv("PIFOLD_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(
        "dual_uq.workflows.structcal_cross_model_representation_sensitivity."
        "load_authorized_pifold_adapter",
        lambda **kwargs: captured.update(kwargs) or adapter,
    )

    observed, checkpoint_id = load_structcal_model(
        "pifold", "default", tmp_path, tmp_path / "run", "cpu"
    )

    assert observed is adapter
    assert checkpoint_id == "official_checkpoint_pth"
    assert captured["implementation_path"] == source
    assert captured["checkpoint_path"] == checkpoint


def test_score_cases_accepts_coordinates_after_parquet_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cases.parquet"
    _cases().to_parquet(path, index=False)
    observed_shapes: list[tuple[int, ...]] = []

    def score(**kwargs: object) -> np.ndarray:
        observed_shapes.append(np.asarray(kwargs["coordinates"]).shape)
        return np.full((2, 20), 0.05)

    score_cases_with_callback(
        pd.read_parquet(path),
        score,
        model_id="fixture",
        checkpoint_id="checkpoint",
        semantic_class="L0",
        regime="controlled",
    )

    assert observed_shapes == [(2, 4, 3), (2, 4, 3)]


def test_safe_output_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "run"

    assert safe_output_path(root, "shards/model") == root.resolve() / "shards/model"
    with pytest.raises(ValueError, match="escape"):
        safe_output_path(root, "../outside")


def test_model_specs_freeze_native_semantics_and_atom_requirements() -> None:
    assert MODEL_SPECS["pifold"]["atoms"] == ("N", "CA", "C", "O")
    assert MODEL_SPECS["dynamicmpnn"]["probe_semantics"] == (
        "teacher_forced_natural_order_native_prefix"
    )
    assert MODEL_SPECS["kwdesign"]["semantic_class"] == "L0_composite"


def test_response_shard_path_is_stable_and_scoped(tmp_path: Path) -> None:
    path = response_shard_path(
        tmp_path,
        model="proteinmpnn",
        checkpoint="v_48_020",
        regime="controlled",
    )

    assert path == tmp_path / "shards/proteinmpnn/v_48_020/controlled"

    identical = response_shard_path(
        tmp_path,
        model="proteinmpnn",
        checkpoint="v_48_020",
        regime="identical",
    )
    assert identical == tmp_path / "shards/proteinmpnn/v_48_020/identical"


def test_dynamicmpnn_filter_records_noncontiguous_pairs_without_splitting() -> None:
    cases = pd.concat(
        [
            _cases().assign(pair_id="contiguous"),
            _cases()
            .assign(pair_id="gapped")
            .replace({"canonical_position": {2: 3}}),
        ],
        ignore_index=True,
    )

    evaluable, exclusions = filter_dynamicmpnn_contiguous_cases(
        cases, regime="operational_pdb_afdb"
    )

    assert evaluable["pair_id"].unique().tolist() == ["contiguous"]
    assert exclusions[["pair_id", "status", "reason"]].to_dict("records") == [
        {
            "pair_id": "gapped",
            "status": "DATA_UNRESOLVED",
            "reason": "NONCONTIGUOUS_CANONICAL_AXIS",
        }
    ]
