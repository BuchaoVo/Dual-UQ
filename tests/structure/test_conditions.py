from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from dual_uq.structure import StructuralIntervention, StructureCondition
from dual_uq.workflows.final_confirmatory_protocol import (
    frozen_plan_scientific_fingerprint,
    interpret_frozen_scoring_plan,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_SCORING_PLAN_PATH = (
    REPOSITORY_ROOT
    / "experiments/p2_design_baseline/scale1/scale1b_v2/"
    "scale1b_v2_proteinmpnn_scoring_plan.parquet"
)


def _condition(
    *,
    protein_id: str = "fixture_A__P00001",
    condition_id: str = "PDB",
    source: str = "PDB",
    structure_sha256: str = "a" * 64,
    structure_locator: str | None = "data/raw/pdb/fixture.cif",
) -> StructureCondition:
    return StructureCondition(
        protein_id=protein_id,
        condition_id=condition_id,
        source=source,
        structure_sha256=structure_sha256,
        structure_locator=structure_locator,
    )


def test_structure_condition_is_immutable_and_has_explicit_identity() -> None:
    condition = _condition()

    assert condition.protein_id == "fixture_A__P00001"
    assert condition.condition_id == "PDB"
    assert condition.source == "PDB"
    assert condition.structure_sha256 == "a" * 64
    assert condition.scientific_identity == (
        "fixture_A__P00001",
        "PDB",
        "PDB",
        "a" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        condition.condition_id = "AFDB"  # type: ignore[misc]


def test_structure_locator_does_not_change_scientific_identity() -> None:
    first = _condition(structure_locator="data/raw/pdb/fixture.cif")
    relocated = _condition(structure_locator="artifacts/relocated/fixture.cif")

    assert first == relocated
    assert hash(first) == hash(relocated)
    assert first.scientific_identity == relocated.scientific_identity


def test_structural_intervention_is_neutral_and_requires_one_protein() -> None:
    pdb = _condition()
    afdb = _condition(
        condition_id="AFDB",
        source="AlphaFoldDB",
        structure_sha256="b" * 64,
        structure_locator="data/raw/afdb/P00001/model.cif",
    )

    intervention = StructuralIntervention(
        intervention_id="fixture_A__P00001::PDB__AFDB",
        condition_a=pdb,
        condition_b=afdb,
    )

    assert intervention.protein_id == "fixture_A__P00001"
    assert intervention.conditions == (pdb, afdb)
    assert not hasattr(intervention, "control")
    assert not hasattr(intervention, "treatment")
    assert not hasattr(intervention, "common_mask_binding")

    with pytest.raises(ValueError, match="same protein"):
        StructuralIntervention(
            intervention_id="invalid",
            condition_a=pdb,
            condition_b=_condition(protein_id="other_A__P00002"),
        )


def test_same_source_conditions_can_form_an_intervention() -> None:
    apo = _condition(condition_id="apo_structure_1", source="PDB")
    holo = _condition(
        condition_id="holo_structure_1",
        source="PDB",
        structure_sha256="c" * 64,
        structure_locator="data/raw/pdb/holo.cif",
    )

    intervention = StructuralIntervention(
        intervention_id="fixture_A__P00001::apo__holo",
        condition_a=apo,
        condition_b=holo,
    )

    assert intervention.condition_a.source == intervention.condition_b.source == "PDB"
    assert intervention.condition_a.condition_id != intervention.condition_b.condition_id


def test_common_mask_is_an_external_measurement_binding() -> None:
    intervention = StructuralIntervention(
        intervention_id="fixture_A__P00001::PDB__AFDB",
        condition_a=_condition(),
        condition_b=_condition(
            condition_id="AFDB",
            source="AlphaFoldDB",
            structure_sha256="b" * 64,
        ),
    )

    first_measurement = (intervention, "1" * 64)
    second_measurement = (intervention, "2" * 64)
    assert first_measurement[0] == second_measurement[0]
    assert first_measurement[1] != second_measurement[1]


def test_frozen_plan_adaptation_is_lossless_for_all_rows() -> None:
    frozen = pd.read_parquet(FROZEN_SCORING_PLAN_PATH)

    interpreted = interpret_frozen_scoring_plan(frozen)

    assert len(interpreted) == len(frozen) == 7_620
    for row, view in zip(frozen.to_dict("records"), interpreted, strict=True):
        assert view.condition.protein_id == row["protein_id"]
        assert view.condition.condition_id == row["structure_condition"]
        assert view.condition.structure_sha256 == row["structure_sha256"]
        assert view.condition.structure_locator == row["structure_artifact_reference"]
        assert view.intervention.protein_id == row["protein_id"]
        assert view.intervention.condition_a.condition_id == "PDB"
        assert view.intervention.condition_b.condition_id == "AFDB"
        assert view.common_mask_binding == row["common_mask_binding"]
        assert frozen_plan_scientific_fingerprint(view.scientific_identity) == (
            frozen_plan_scientific_fingerprint(row)
        )

    sources = {
        view.condition.condition_id: view.condition.source for view in interpreted
    }
    assert sources == {"PDB": "PDB", "AFDB": "AlphaFoldDB"}
