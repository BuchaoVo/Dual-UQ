from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.core.paths import ProjectPaths
from dual_uq.dataset.planning import (
    BASE_SEED,
    CURRENT_POOL_SUFFICIENT,
    ENDPOINT_SPECS,
    HETEROGENEITY_SCENARIOS,
    N_GRID,
    PRODUCTION_REPLICATES,
    PowerPlanningConfig,
    PowerPlanningError,
    apply_heterogeneity_inflation,
    audit_power_planning,
    build_feasibility_table,
    build_stage0_planning_effects,
    draw_protein_block_indices,
    materialize_power_planning,
    simulate_power_grid,
    validate_power_planning_inputs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_inputs():
    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    return validate_power_planning_inputs(paths, PowerPlanningConfig())


def test_frozen_input_gate_requires_exact_hashes_counts_and_h1_stop(
    repository_inputs,
) -> None:
    assert len(repository_inputs.admission_census) == 72
    assert len(repository_inputs.common_masks) == 22_200
    assert len(repository_inputs.admitted_census) == 52
    assert len(repository_inputs.stage0_protein_summary) == 8
    assert repository_inputs.stage0_manifest["h1_verdict"] == (
        "H1_PARTIALLY_SUPPORTED"
    )
    assert repository_inputs.stage0_manifest["stage0_local_analysis_stop"] is True

    paths = ProjectPaths.discover(project_root=REPOSITORY_ROOT)
    bad = replace(
        PowerPlanningConfig(),
        expected_admission_census_sha256="0" * 64,
    )
    with pytest.raises(PowerPlanningError) as exc:
        validate_power_planning_inputs(paths, bad)
    assert exc.value.code == "upstream_sha256_mismatch"


def test_stage0_effect_adapter_uses_exact_frozen_protein_fields(
    repository_inputs,
) -> None:
    effects = build_stage0_planning_effects(repository_inputs)
    source = repository_inputs.stage0_protein_summary.set_index("protein_id")

    assert len(effects) == 8
    assert effects["protein_id"].tolist() == repository_inputs.protein_order
    assert set(effects.columns) == {
        "protein_id",
        "position_count",
        *(spec.endpoint_id for spec in ENDPOINT_SPECS),
    }
    for spec in ENDPOINT_SPECS:
        np.testing.assert_allclose(
            effects.set_index("protein_id")[spec.endpoint_id],
            source[spec.source_column],
            rtol=0,
            atol=0,
        )


def test_feasibility_geometry_is_preoutcome_and_deterministic(repository_inputs) -> None:
    feasibility = build_feasibility_table(repository_inputs)

    assert len(feasibility) == 52
    assert feasibility["common_mask_count"].min() == 95
    assert feasibility["common_mask_count"].median() == 207.5
    assert feasibility["common_mask_count"].max() == 498
    assert (
        feasibility["low_tertile_size"]
        + feasibility["mid_tertile_size"]
        + feasibility["high_tertile_size"]
        == feasibility["common_mask_count"]
    ).all()
    assert (
        feasibility["low_high_matching_capacity"]
        == feasibility[["low_tertile_size", "high_tertile_size"]].min(axis=1)
    ).all()
    assert (
        feasibility["balanced_3x3_reference_cell_size"]
        == feasibility["common_mask_count"] // 9
    ).all()
    for threshold in (60, 75, 90, 100, 120, 150):
        assert f"retained_at_common_mask_{threshold}" in feasibility
    assert not {
        "p",
        "m",
        "sdfi",
        "top1",
        "regret",
        "structural_excess",
    }.intersection({column.lower() for column in feasibility.columns})


def test_feasibility_rejects_mask_geometry_drift(repository_inputs) -> None:
    masks = repository_inputs.common_masks.copy()
    admitted_id = str(repository_inputs.admitted_census.iloc[0]["pair_id"])
    index = masks.index[
        masks["pair_id"].astype(str).eq(admitted_id) & masks["common_mask"].eq(True)
    ][0]
    masks.loc[index, "common_mask"] = False
    drifted = replace(repository_inputs, common_masks=masks)

    with pytest.raises(PowerPlanningError) as exc:
        build_feasibility_table(drifted)
    assert exc.value.code == "common_mask_geometry_mismatch"


def test_median_centered_heterogeneity_uses_fisher_z_for_correlations() -> None:
    effects = pd.DataFrame(
        {
            "protein_id": ["p1", "p2", "p3"],
            "natural": [1.0, 2.0, 5.0],
            "correlation": [-0.5, 0.0, 0.5],
        }
    )
    transformed = apply_heterogeneity_inflation(
        effects,
        endpoint_ids=("natural", "correlation"),
        correlation_endpoint_ids=("correlation",),
        multiplier=2.0,
    )

    np.testing.assert_allclose(transformed["natural"], [0.0, 2.0, 8.0])
    expected_z = 2.0 * np.arctanh(np.asarray([-0.5, 0.0, 0.5]))
    np.testing.assert_allclose(transformed["correlation"], np.tanh(expected_z))
    assert transformed["natural"].median() == 2.0
    assert transformed["correlation"].median() == 0.0


def test_block_indices_are_deterministic_and_shared_across_endpoints() -> None:
    first = draw_protein_block_indices(
        protein_count=8, cohort_size=20, replicate_count=25, seed=123
    )
    second = draw_protein_block_indices(
        protein_count=8, cohort_size=20, replicate_count=25, seed=123
    )
    assert first.shape == (25, 20)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0
    assert first.max() < 8

    linked = np.column_stack((np.arange(8), np.arange(8) * 10))
    sampled = linked[first]
    np.testing.assert_array_equal(sampled[:, :, 1], sampled[:, :, 0] * 10)


def test_small_simulation_grid_obeys_exact_stability_contract() -> None:
    effects = pd.DataFrame(
        {
            "protein_id": [f"p{i}" for i in range(8)],
            "endpoint": [1.0] * 8,
        }
    )
    result = simulate_power_grid(
        effects,
        endpoint_directions={"endpoint": 1},
        primary_endpoint_ids={"endpoint"},
        correlation_endpoint_ids=set(),
        n_grid=(16,),
        heterogeneity_scenarios={"H1.0": 1.0},
        replicate_count=100,
        base_seed=20260730,
    )
    row = result.iloc[0]
    assert row["recurrence_count_threshold"] == 10
    assert row["median_direction_stability"] == 1.0
    assert row["recurrence_stability"] == 1.0
    assert row["adequacy_status"] == "ADEQUATE_FOR_CONFIRMATORY_REPLICATION"
    assert row["median_effect_p05"] == 1.0
    assert row["median_effect_p95"] == 1.0


@pytest.fixture(scope="module")
def small_repository_result(repository_inputs):
    return audit_power_planning(
        ProjectPaths.discover(project_root=REPOSITORY_ROOT),
        config=PowerPlanningConfig(
            n_grid=(16, 20),
            heterogeneity_scenarios={"H1.0": 1.0, "H2.0": 2.0},
            replicate_count=200,
        ),
        inputs=repository_inputs,
    )


def test_structured_result_has_no_pseudoreplication_or_h1c_size_override(
    small_repository_result,
) -> None:
    result = small_repository_result
    assert len(result.feasibility) == 52
    assert len(result.stage0_effects) == 8
    assert len(result.simulation) == len(ENDPOINT_SPECS) * 2 * 2
    assert set(result.simulation["biological_unit"]) == {"PROTEIN"}
    assert not any("p_value" in column.lower() for column in result.simulation)
    assert result.manifest["POWER_UNIT"] == "PROTEIN"
    assert result.manifest["H1C_NOT_USED_TO_FORCE_POSITIVE_RESULT"] is True
    assert result.manifest["SCALE1B_COHORT_NOT_SELECTED"] is True
    assert result.manifest["N_grid"] == [16, 20]
    assert result.manifest["heterogeneity_transformations"][
        "protein_dependence"
    ] == "whole protein-level endpoint vector block resampling"
    assert result.manifest["manifest_self_hash_policy"] == (
        "reported_by_cli_after_write_to_avoid_recursive_self_hash"
    )
    assert result.manifest["quantile_method"] == "numpy_linear"
    assert any(
        "left-censored by the evaluated N-grid floor of 16" in limitation
        for limitation in result.summary["limitations"]
    )


def test_immutable_materialization_and_portable_manifest(
    tmp_path: Path, small_repository_result
) -> None:
    paths = ProjectPaths.discover(
        project_root=REPOSITORY_ROOT,
        reports_root=tmp_path / "reports",
    )
    config = PowerPlanningConfig(
        output_root_ref="reports/planning",
        n_grid=(16, 20),
        heterogeneity_scenarios={"H1.0": 1.0, "H2.0": 2.0},
        replicate_count=200,
    )
    first = materialize_power_planning(
        small_repository_result, paths, config=config
    )
    second = materialize_power_planning(
        small_repository_result, paths, config=config
    )
    assert set(first["write_status"].values()) == {"created"}
    assert set(second["write_status"].values()) == {"reused_identical"}
    assert all(not Path(record["path"]).is_absolute() for record in first["outputs"].values())

    target = tmp_path / "reports/planning/scale1_power_simulation.parquet"
    target.write_bytes(b"conflict")
    with pytest.raises(PowerPlanningError) as exc:
        materialize_power_planning(
            small_repository_result, paths, config=config
        )
    assert exc.value.code == "immutable_planning_artifact_conflict"


def test_frozen_production_constants() -> None:
    assert BASE_SEED == 20260730
    assert PRODUCTION_REPLICATES == 20_000
    assert N_GRID == (16, 20, 24, 30, 32, 40, 48, 52, 60, 72)
    assert HETEROGENEITY_SCENARIOS == {"H1.0": 1.0, "H1.5": 1.5, "H2.0": 2.0}
    assert CURRENT_POOL_SUFFICIENT == (
        "CURRENT_52_POOL_SUFFICIENT_FOR_SCALE1B_PLANNING"
    )
