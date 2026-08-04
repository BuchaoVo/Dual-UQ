from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def summary_script() -> ModuleType:
    path = ROOT / "scripts" / "12_build_a0_candidate_summary.py"
    spec = spec_from_file_location("a0_geometry_evidence_script", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(
    *,
    index: int = 113,
    pdb_id: str = "3ip0",
    chain_id: str = "A",
    uniprot_id: str = "P26281",
) -> dict[str, object]:
    return {
        "screening_index": index,
        "source": "replacement_pool",
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": f"{pdb_id}_{chain_id}__{uniprot_id}",
        "provisional_stratum": "lower_global_confidence_replacement",
    }


def _write_geometry(
    root: Path,
    identity: dict[str, object],
    values: list[float],
    *,
    field: str = "ca_disagreement",
    include_qc_statistics: bool = True,
) -> Path:
    pair_dir = root / "data/processed/pairs" / str(identity["pair_name"])
    pair_dir.mkdir(parents=True, exist_ok=True)
    positions = list(range(1, len(values) + 1))
    pd.DataFrame(
        {
            "uniprot_residue_number": positions,
            "auth_asym_id": [identity["chain_id"]] * len(values),
            "auth_seq_id": positions,
            "insertion_code": [""] * len(values),
            "label_asym_id": ["X"] * len(values),
            "label_seq_id": positions,
            field: values,
        }
    ).to_parquet(pair_dir / "residue_geometry.parquet", index=False)
    qc: dict[str, object] = {
        "pdb_id": identity["pdb_id"],
        "chain_id": identity["chain_id"],
        "uniprot_id": identity["uniprot_id"],
        "mapped_ca_count": len(values),
    }
    if include_qc_statistics:
        array = np.asarray(values, dtype=float)
        qc.update(
            {
                "median_aligned_ca_distance": float(np.median(array)),
                "p90_aligned_ca_distance": float(
                    np.quantile(array, 0.90, method="linear")
                ),
                "max_aligned_ca_distance": float(np.max(array)),
            }
        )
    (pair_dir / "pair_geometry_qc.json").write_text(
        json.dumps(qc),
        encoding="utf-8",
    )
    return pair_dir


def test_replacement_new_schema_loads_without_pilot_or_legacy_number(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_geometry(tmp_path, identity, [0.0, 1.0, 2.0, 3.0, 4.0])

    result = summary_script.load_pair_geometry_evidence(
        project_root=tmp_path,
        candidate_identity=identity,
        geometry_status="complete",
        pilot_mechanism={},
        strict=True,
    )

    assert result.available is True
    assert result.source == "pair_geometry_qc+residue_geometry"
    assert result.ca_disagreement_median == pytest.approx(2.0)
    assert result.ca_disagreement_p90 == pytest.approx(3.6)
    assert result.ca_disagreement_max == pytest.approx(4.0)
    assert result.mapped_ca_count == 5
    assert result.identity_match is True


def test_qc_missing_statistics_uses_deterministic_linear_quantile(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_geometry(
        tmp_path,
        identity,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        include_qc_statistics=False,
    )

    result = summary_script.load_pair_geometry_evidence(
        project_root=tmp_path,
        candidate_identity=identity,
        geometry_status="complete",
        pilot_mechanism={},
        strict=True,
    )

    assert result.source == "residue_geometry"
    assert result.ca_disagreement_median == pytest.approx(2.0)
    assert result.ca_disagreement_p90 == pytest.approx(3.6)
    assert result.ca_disagreement_max == pytest.approx(4.0)


@pytest.mark.parametrize(
    "values",
    [
        [0.1, np.nan, 0.3],
        [0.1, np.inf, 0.3],
        [0.1, -0.2, 0.3],
    ],
)
def test_invalid_ca_disagreement_is_rejected(
    summary_script: ModuleType,
    tmp_path: Path,
    values: list[float],
) -> None:
    identity = _identity()
    _write_geometry(
        tmp_path,
        identity,
        values,
        include_qc_statistics=False,
    )

    with pytest.raises(ValueError, match="CA disagreement"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=identity,
            geometry_status="complete",
            pilot_mechanism={},
            strict=True,
        )


def test_complete_geometry_missing_artifact_fails_strictly(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="geometry artifact"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=_identity(),
            geometry_status="complete",
            pilot_mechanism={},
            strict=True,
        )


def test_incomplete_geometry_does_not_read_stale_artifact(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])

    result = summary_script.load_pair_geometry_evidence(
        project_root=tmp_path,
        candidate_identity=identity,
        geometry_status="not_started",
        pilot_mechanism={
            "ca_disagreement_median": 9.0,
            "ca_disagreement_p90": 9.0,
            "ca_disagreement_max": 9.0,
        },
        strict=True,
    )

    assert result.available is False
    assert result.ca_disagreement_p90 is None
    assert result.error_reason == "geometry_not_complete"


def test_artifact_identity_mismatch_fails_strictly(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    pair_dir = _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])
    qc = json.loads((pair_dir / "pair_geometry_qc.json").read_text())
    qc["uniprot_id"] = "WRONG"
    (pair_dir / "pair_geometry_qc.json").write_text(json.dumps(qc))

    with pytest.raises(ValueError, match="geometry identity mismatch"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=identity,
            geometry_status="complete",
            pilot_mechanism={},
            strict=True,
        )


def test_duplicate_uniprot_positions_are_rejected(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    pair_dir = _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])
    table = pd.read_parquet(pair_dir / "residue_geometry.parquet")
    table.loc[1, "uniprot_residue_number"] = 1
    table.to_parquet(pair_dir / "residue_geometry.parquet", index=False)

    with pytest.raises(ValueError, match="duplicate.*UniProt"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=identity,
            geometry_status="complete",
            pilot_mechanism={},
            strict=True,
        )


def test_pilot_conflict_does_not_override_canonical_artifact(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="pilot geometry conflict"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=identity,
            geometry_status="complete",
            pilot_mechanism={"ca_disagreement_p90": 99.0},
            strict=True,
        )


@pytest.mark.parametrize("invalid_value", [None, "not-a-number", np.nan, np.inf])
def test_present_invalid_qc_statistic_is_not_treated_as_missing(
    summary_script: ModuleType,
    tmp_path: Path,
    invalid_value: object,
) -> None:
    identity = _identity()
    pair_dir = _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])
    qc_path = pair_dir / "pair_geometry_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["p90_aligned_ca_distance"] = invalid_value
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid geometry QC statistic"):
        summary_script.load_pair_geometry_evidence(
            project_root=tmp_path,
            candidate_identity=identity,
            geometry_status="complete",
            pilot_mechanism={},
            strict=True,
        )


def test_malformed_canonical_artifact_does_not_fall_back_to_pilot(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identity = _identity()
    pair_dir = _write_geometry(tmp_path, identity, [0.1, 0.2, 0.3])
    qc_path = pair_dir / "pair_geometry_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["uniprot_id"] = "WRONG"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")

    result = summary_script.load_pair_geometry_evidence(
        project_root=tmp_path,
        candidate_identity=identity,
        geometry_status="complete",
        pilot_mechanism={"ca_disagreement_p90": 0.5},
        strict=False,
    )

    assert result.available is False
    assert result.ca_disagreement_p90 is None
    assert result.source is None
    assert "geometry identity mismatch" in str(result.error_reason)


@pytest.mark.parametrize(
    ("index", "pair_name"),
    [
        (6, "1gci_A__P29600"),
        (8, "3w5h_A__P83686"),
        (24, "3o4p_A__Q7SIG4"),
        (36, "3zoj_A__F2QVG4"),
        (35, "1x8p_A__Q94734"),
        (None, "1ake_A__P69441"),
    ],
)
def test_existing_pilot_geometry_statistics_regress(
    summary_script: ModuleType,
    index: int | None,
    pair_name: str,
) -> None:
    pilot = pd.read_csv(ROOT / "reports/geometry_pilot_mechanisms.csv")
    row = pilot.loc[pilot["pair_name"] == pair_name].iloc[0].to_dict()
    pdb_id, remainder = pair_name.split("_", maxsplit=1)
    chain_id, uniprot_id = remainder.split("__", maxsplit=1)
    identity = {
        "screening_index": index,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "uniprot_id": uniprot_id,
        "pair_name": pair_name,
    }

    result = summary_script.load_pair_geometry_evidence(
        project_root=ROOT,
        candidate_identity=identity,
        geometry_status="complete",
        pilot_mechanism=row,
        strict=True,
    )

    assert result.ca_disagreement_median == pytest.approx(
        row["ca_disagreement_median"],
        abs=1e-6,
    )
    assert result.ca_disagreement_p90 == pytest.approx(
        row["ca_disagreement_p90"],
        abs=1e-6,
    )
    assert result.ca_disagreement_max == pytest.approx(
        row["ca_disagreement_max"],
        abs=1e-6,
    )


def _write_classification_artifacts(
    root: Path,
    candidate: dict[str, object],
    *,
    high_pae: bool,
    state_disagreement: bool,
) -> None:
    pair_dir = _write_geometry(
        root,
        candidate,
        [0.2, 0.3, 0.4, 0.5, 0.6],
    )
    positions = np.array([1, 50, 100], dtype=int)
    off_diagonal = 20.0 if high_pae else 3.0
    pae = np.full((3, 3), off_diagonal, dtype=float)
    np.fill_diagonal(pae, 0.0)
    np.savez(
        pair_dir / "pairwise_geometry.npz",
        uniprot_positions=positions,
        symmetric_pae=pae,
    )
    if state_disagreement:
        pd.DataFrame(
            [
                {
                    "start_position": 10,
                    "end_position": 15,
                    "residue_count": 6,
                    "median_plddt": 95.0,
                    "median_disagreement": 2.0,
                }
            ]
        ).to_csv(pair_dir / "disagreement_segments.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "start_position",
                "end_position",
                "residue_count",
                "median_plddt",
                "median_disagreement",
            ]
        ).to_csv(pair_dir / "disagreement_segments.csv", index=False)


def _complete_lifecycle(candidate: dict[str, object]) -> dict[str, object]:
    return {
        **candidate,
        "preflight_status": "pass_full_length",
        "full_length_mapping_coverage": 1.0,
        "entity_mapping_coverage": 1.0,
        "sequence_identity": 1.0,
        "observed_ca_fraction_of_mapped": 1.0,
        "pair_status": "complete",
        "geometry_status": "complete",
        "robust_status": "complete",
        "segment_context_status": "complete",
        "complete_diagnostics": True,
        "quality_pass": True,
    }


def _mapped_confidence() -> dict[str, object]:
    return {
        "scoring_status": "success",
        "confidence_model_match": True,
        "plddt_bfactor_match": True,
        "mapped_plddt_min": 60.0,
        "mapped_plddt_q10": 90.0,
        "mapped_plddt_median": 95.0,
        "longest_internal_below_70_length": 6,
        "longest_internal_below_80_length": 6,
        "terminal_only_below_80": False,
    }


def test_synthetic_replacement_complete_geometry_propagates_classification(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    candidate = _identity()
    _write_classification_artifacts(
        tmp_path,
        candidate,
        high_pae=False,
        state_disagreement=False,
    )
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )

    row = summary_script._assemble_candidate(
        candidate,
        root=tmp_path,
        config=config,
        preflight={},
        lifecycle=_complete_lifecycle(candidate),
        replacement=candidate,
        pilot={},
        mechanism={},
        mapped_confidence=_mapped_confidence(),
        strict=True,
    )

    assert row["ca_disagreement_p90"] == pytest.approx(0.56)
    assert row["classification_status"] == "complete_classification"
    assert row["primary_category"] == "low_confidence_local"
    assert row["selection_eligible"] is True


def test_synthetic_index101_multilabel_priority_is_unchanged(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    candidate = _identity(
        index=101,
        pdb_id="8pb5",
        uniprot_id="P0DPA9",
    )
    _write_classification_artifacts(
        tmp_path,
        candidate,
        high_pae=True,
        state_disagreement=True,
    )
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )

    row = summary_script._assemble_candidate(
        candidate,
        root=tmp_path,
        config=config,
        preflight={},
        lifecycle=_complete_lifecycle(candidate),
        replacement=candidate,
        pilot={},
        mechanism={},
        mapped_confidence=_mapped_confidence(),
        strict=True,
    )

    assert row["is_low_conf_local"] is True
    assert row["is_high_pae_long_range"] is True
    assert row["is_high_conf_state_disagreement"] is True
    assert row["primary_category"] == "high_confidence_state_disagreement"
    assert row["classification_status"] == "complete_classification"


def test_lifecycle_geometry_status_precedes_stale_pilot_status(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    candidate = _identity()
    _write_classification_artifacts(
        tmp_path,
        candidate,
        high_pae=False,
        state_disagreement=False,
    )
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )
    lifecycle = {
        **_complete_lifecycle(candidate),
        "geometry_status": "not_started",
    }

    row = summary_script._assemble_candidate(
        candidate,
        root=tmp_path,
        config=config,
        preflight={},
        lifecycle=lifecycle,
        replacement=candidate,
        pilot={},
        mechanism={
            "geometry_status": "complete",
            "ca_disagreement_p90": 0.56,
        },
        mapped_confidence=_mapped_confidence(),
        strict=False,
    )

    assert row["geometry_status"] == "not_started"
    assert row["ca_disagreement_p90"] is None
    assert row["classification_status"] == "insufficient_evidence"


def test_strict_summary_rejects_lifecycle_pilot_geometry_status_conflict(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    candidate = _identity()
    _write_classification_artifacts(
        tmp_path,
        candidate,
        high_pae=False,
        state_disagreement=False,
    )
    config = yaml.safe_load(
        (ROOT / "configs/legacy/a0_screening/a0_selection.yaml").read_text(encoding="utf-8")
    )
    lifecycle = _complete_lifecycle(candidate)

    with pytest.raises(ValueError, match="geometry status conflict"):
        summary_script._assemble_candidate(
            candidate,
            root=tmp_path,
            config=config,
            preflight={},
            lifecycle=lifecycle,
            replacement=candidate,
            pilot={},
            mechanism={"geometry_status": "not_started"},
            mapped_confidence=_mapped_confidence(),
            strict=True,
        )


def test_geometry_evidence_is_independent_of_candidate_input_order(
    summary_script: ModuleType,
    tmp_path: Path,
) -> None:
    identities = [
        _identity(index=113, pdb_id="3ip0", uniprot_id="P26281"),
        _identity(index=101, pdb_id="8pb5", uniprot_id="P0DPA9"),
    ]
    for offset, identity in enumerate(identities):
        _write_geometry(
            tmp_path,
            identity,
            [0.1 + offset, 0.2 + offset, 0.3 + offset],
        )

    def load(items: list[dict[str, object]]) -> dict[int, float]:
        return {
            int(identity["screening_index"]): (
                summary_script.load_pair_geometry_evidence(
                    project_root=tmp_path,
                    candidate_identity=identity,
                    geometry_status="complete",
                    pilot_mechanism={},
                    strict=True,
                ).ca_disagreement_p90
            )
            for identity in items
        }

    assert load(identities) == load(list(reversed(identities)))
