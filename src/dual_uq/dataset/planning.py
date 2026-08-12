"""Offline protein-level feasibility and empirical stability planning."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import derive_seed, derive_seed_digest, sha256_file
from dual_uq.core.paths import ProjectPathError, ProjectPaths

SCHEMA_VERSION = "dual-uq.scale1-protein-power-planning.v1"
DEFAULT_OUTPUT_ROOT = "experiments/p2_design_baseline/scale1/planning"
BASE_SEED = 20260730
PRODUCTION_REPLICATES = 20_000
N_GRID = (16, 20, 24, 30, 32, 40, 48, 52, 60, 72)
HETEROGENEITY_SCENARIOS = {"H1.0": 1.0, "H1.5": 1.5, "H2.0": 2.0}
COMMON_MASK_THRESHOLDS = (60, 75, 90, 100, 120, 150)
RECURRENCE_FRACTION = 0.625
MEDIAN_STABILITY_MIN = 0.90
RECURRENCE_STABILITY_MIN = 0.80

CURRENT_POOL_SUFFICIENT = "CURRENT_52_POOL_SUFFICIENT_FOR_SCALE1B_PLANNING"
ADDITIONAL_ACQUISITION_REQUIRED = "ADDITIONAL_ACQUISITION_REQUIRED_BEFORE_SCALE1B"
POWER_PLANNING_INCONCLUSIVE = "POWER_PLANNING_INCONCLUSIVE"


class PowerPlanningError(RuntimeError):
    """One structured input, simulation, or immutable-output failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_id: str
    source_column: str
    family: str
    role: str
    scale: str
    predicted_direction: int


ENDPOINT_SPECS = (
    EndpointSpec(
        "h1a_conditional_p_top1",
        "p_gradient_flip_mean",
        "H1a",
        "primary",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1a_conditional_p_regret",
        "p_gradient_regret_mean",
        "H1a",
        "primary",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1a_m_protective_top1",
        "m_gradient_flip_mean",
        "H1a",
        "primary",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1a_m_protective_regret",
        "m_gradient_regret_mean",
        "H1a",
        "primary",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1b_p_e_flip_spearman",
        "p_e_flip_spearman",
        "H1b",
        "primary",
        "fisher_z_correlation",
        1,
    ),
    EndpointSpec(
        "h1b_p_e_regret_spearman",
        "p_e_regret_spearman",
        "H1b",
        "primary",
        "fisher_z_correlation",
        1,
    ),
    EndpointSpec(
        "h1b_c2_excess_top1",
        "c2_delta_e_flip_mean",
        "H1b",
        "supporting",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1b_c2_excess_regret",
        "c2_delta_e_regret_mean",
        "H1b",
        "supporting",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1c_m_e_flip_spearman",
        "m_e_flip_spearman",
        "H1c",
        "descriptive_only",
        "fisher_z_correlation",
        -1,
    ),
    EndpointSpec(
        "h1c_m_e_regret_spearman",
        "m_e_regret_spearman",
        "H1c",
        "descriptive_only",
        "fisher_z_correlation",
        -1,
    ),
    EndpointSpec(
        "h1c_c1_excess_top1",
        "c1_delta_e_flip_mean",
        "H1c",
        "descriptive_only",
        "natural",
        1,
    ),
    EndpointSpec(
        "h1c_c1_excess_regret",
        "c1_delta_e_regret_mean",
        "H1c",
        "descriptive_only",
        "natural",
        1,
    ),
)


@dataclass(frozen=True)
class PowerPlanningConfig:
    admission_census_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_census.parquet"
    )
    common_masks_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_common_masks.parquet"
    )
    admission_summary_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_summary.json"
    )
    admission_manifest_ref: str = (
        "experiments/p2_design_baseline/scale1/scale1a1/"
        "scale1_formal_admission_manifest.json"
    )
    h1_manifest_ref: str = (
        "experiments/p2_design_baseline/stage0/"
        "h1_confirmatory_audit_manifest.json"
    )
    h1_protein_summary_ref: str = (
        "experiments/p2_design_baseline/stage0/"
        "h1_confirmatory_protein_summary.parquet"
    )
    output_root_ref: str = DEFAULT_OUTPUT_ROOT
    expected_admission_census_sha256: str = (
        "ca01e17f55637d0f4584aa7bbb8940e50ea4aaf25b9c7ee7f741bb51a4095a7f"
    )
    expected_common_masks_sha256: str = (
        "face3d6f211585d452f474b090adb618d121aada849a4cbd492e92e9b60a03b9"
    )
    expected_admission_summary_sha256: str = (
        "50afd84f6ac4a3104fc817015c89b9e1127b16bb8b5c8f688fb385cee02ecc0f"
    )
    expected_admission_manifest_sha256: str = (
        "54227ee11e67fdb6da12cbf9001b603875b1fd3c28d603608d384f0c50988223"
    )
    expected_h1_manifest_sha256: str = (
        "d74252a4393f9e0ebf0b58d7025c7bca0569701921c592bacb425bd6bfc07fbf"
    )
    expected_h1_protein_summary_sha256: str = (
        "104399ab030a49692a1971a0d3654ea03fe94c1ad2cf904c2d3153f16647f49f"
    )
    n_grid: tuple[int, ...] = N_GRID
    heterogeneity_scenarios: Mapping[str, float] = field(
        default_factory=lambda: dict(HETEROGENEITY_SCENARIOS)
    )
    replicate_count: int = PRODUCTION_REPLICATES
    base_seed: int = BASE_SEED


@dataclass(frozen=True)
class PowerPlanningInputs:
    admission_census: pd.DataFrame
    common_masks: pd.DataFrame
    admitted_census: pd.DataFrame
    admission_summary: dict[str, Any]
    admission_manifest: dict[str, Any]
    stage0_manifest: dict[str, Any]
    stage0_protein_summary: pd.DataFrame
    protein_order: list[str]
    input_artifacts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PowerPlanningResult:
    feasibility: pd.DataFrame
    stage0_effects: pd.DataFrame
    simulation: pd.DataFrame
    summary: dict[str, Any]
    manifest: dict[str, Any]
    input_artifacts: tuple[dict[str, Any], ...]


def _resolve(paths: ProjectPaths, logical_ref: str) -> Path:
    try:
        return paths.resolve_logical(logical_ref)
    except ProjectPathError as exc:
        raise PowerPlanningError(
            "nonportable_input_path", f"Invalid logical path: {logical_ref}"
        ) from exc


def _logical(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.logical_ref(path)
    except ProjectPathError as exc:
        raise PowerPlanningError(
            "nonportable_output_path", f"Path is outside project roots: {path.name}"
        ) from exc


def _require_hash(
    paths: ProjectPaths, logical_ref: str, expected_sha256: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(paths, logical_ref)
    if not path.is_file():
        raise PowerPlanningError(
            "missing_upstream_input", f"Missing {label}: {logical_ref}"
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise PowerPlanningError(
            "upstream_sha256_mismatch",
            f"{label} SHA256 mismatch",
            expected_sha256=expected_sha256,
            observed_sha256=observed,
        )
    return path, {"path": logical_ref, "sha256": observed, "label": label}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PowerPlanningError(
            "unreadable_upstream_input", f"Unable to read {label}"
        ) from exc
    if not isinstance(payload, dict):
        raise PowerPlanningError(
            "invalid_upstream_schema", f"{label} must be a JSON object"
        )
    return payload


def _read_parquet(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise PowerPlanningError(
            "unreadable_upstream_input", f"Unable to read {label}"
        ) from exc


def validate_power_planning_inputs(
    paths: ProjectPaths, config: PowerPlanningConfig
) -> PowerPlanningInputs:
    """Validate the frozen Scale-1A1 and Stage-0 H1 trust chain."""
    bindings = (
        (
            config.admission_census_ref,
            config.expected_admission_census_sha256,
            "Scale-1A1 admission census",
        ),
        (
            config.common_masks_ref,
            config.expected_common_masks_sha256,
            "Scale-1A1 common masks",
        ),
        (
            config.admission_summary_ref,
            config.expected_admission_summary_sha256,
            "Scale-1A1 summary",
        ),
        (
            config.admission_manifest_ref,
            config.expected_admission_manifest_sha256,
            "Scale-1A1 manifest",
        ),
        (
            config.h1_manifest_ref,
            config.expected_h1_manifest_sha256,
            "frozen Stage-0 H1 confirmatory manifest",
        ),
        (
            config.h1_protein_summary_ref,
            config.expected_h1_protein_summary_sha256,
            "frozen Stage-0 H1 protein summary",
        ),
    )
    artifacts: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    for logical_ref, expected_sha, label in bindings:
        path, record = _require_hash(paths, logical_ref, expected_sha, label)
        resolved[logical_ref] = path
        artifacts.append(record)

    census = _read_parquet(
        resolved[config.admission_census_ref], "Scale-1A1 admission census"
    )
    masks = _read_parquet(resolved[config.common_masks_ref], "Scale-1A1 common masks")
    summary = _read_json(
        resolved[config.admission_summary_ref], "Scale-1A1 summary"
    )
    admission_manifest = _read_json(
        resolved[config.admission_manifest_ref], "Scale-1A1 manifest"
    )
    h1_manifest = _read_json(
        resolved[config.h1_manifest_ref], "Stage-0 H1 confirmatory manifest"
    )
    h1_proteins = _read_parquet(
        resolved[config.h1_protein_summary_ref], "Stage-0 H1 protein summary"
    )

    expected_status_counts = {
        "FORMALLY_ADMITTED": 52,
        "PENDING_HUMAN_VARIANT_REVIEW": 10,
        "IDENTITY_CONTRACT_FAIL": 4,
        "MAPPING_FAIL": 4,
        "PROVENANCE_FAIL": 2,
    }
    if (
        len(census) != 72
        or census["pair_id"].duplicated().any()
        or dict(Counter(census["admission_status"])) != expected_status_counts
    ):
        raise PowerPlanningError(
            "admission_census_mismatch", "Scale-1A1 terminal census is not frozen"
        )
    if len(masks) != 22_200 or masks.duplicated(
        ["pair_id", "canonical_position"]
    ).any():
        raise PowerPlanningError(
            "common_mask_census_mismatch", "Scale-1A1 common-mask table is not exact"
        )
    if (
        summary.get("sampling_frame_count") != 213
        or summary.get("evaluated_count") != 72
        or summary.get("unevaluated_count") != 141
        or summary.get("terminal_status_counts") != expected_status_counts
    ):
        raise PowerPlanningError(
            "admission_summary_mismatch", "Scale-1A1 summary counts differ"
        )
    if (
        admission_manifest.get("scale1a1_status")
        != "SCALE1A1_FORMAL_CENSUS_COMPLETE"
        or admission_manifest.get("SCALE1_SCORING_NOT_STARTED") is not True
        or admission_manifest.get("SCALE1B_ELIGIBILITY_NOT_DEFINED") is not True
    ):
        raise PowerPlanningError(
            "admission_manifest_mismatch", "Scale-1A1 manifest scope is not frozen"
        )
    manifest_outputs = admission_manifest.get("output_artifacts", {})
    expected_output_bindings = {
        "formal_admission_census": (
            config.admission_census_ref,
            config.expected_admission_census_sha256,
        ),
        "common_masks": (
            config.common_masks_ref,
            config.expected_common_masks_sha256,
        ),
        "formal_admission_summary": (
            config.admission_summary_ref,
            config.expected_admission_summary_sha256,
        ),
    }
    for name, (expected_path, expected_sha) in expected_output_bindings.items():
        record = manifest_outputs.get(name, {})
        if record.get("path") != expected_path or record.get("sha256") != expected_sha:
            raise PowerPlanningError(
                "admission_manifest_binding_mismatch",
                f"Scale-1A1 manifest does not bind {name}",
            )

    protein_order = [str(value) for value in h1_manifest.get("protein_order", [])]
    h1_output = h1_manifest.get("outputs", {}).get("protein_summary", {})
    if (
        h1_manifest.get("status") != "complete"
        or h1_manifest.get("h1_verdict") != "H1_PARTIALLY_SUPPORTED"
        or h1_manifest.get("stage0_local_analysis_stop") is not True
        or len(protein_order) != 8
        or len(set(protein_order)) != 8
        or h1_output.get("path") != config.h1_protein_summary_ref
        or h1_output.get("sha256") != config.expected_h1_protein_summary_sha256
        or h1_output.get("rows") != 8
    ):
        raise PowerPlanningError(
            "stage0_h1_contract_mismatch", "Frozen Stage-0 H1 contract differs"
        )
    if (
        len(h1_proteins) != 8
        or h1_proteins["protein_id"].duplicated().any()
        or h1_proteins["protein_id"].astype(str).tolist() != protein_order
    ):
        raise PowerPlanningError(
            "stage0_protein_unit_mismatch", "Stage-0 protein units are not exact"
        )
    required_effects = {spec.source_column for spec in ENDPOINT_SPECS}
    if missing := sorted(required_effects - set(h1_proteins.columns)):
        raise PowerPlanningError(
            "stage0_effect_schema_mismatch",
            f"Stage-0 protein summary lacks frozen effects: {missing}",
        )
    values = h1_proteins[list(required_effects)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PowerPlanningError(
            "stage0_effect_nonfinite", "Frozen Stage-0 planning effects must be finite"
        )

    admitted = census.loc[census["admission_status"].eq("FORMALLY_ADMITTED")].copy()
    admitted_ids = admitted["pair_id"].astype(str).tolist()
    mask_ids = set(masks["pair_id"].astype(str))
    if len(admitted) != 52 or not set(admitted_ids).issubset(mask_ids):
        raise PowerPlanningError(
            "admitted_mask_binding_mismatch", "Admitted proteins lack common masks"
        )
    return PowerPlanningInputs(
        admission_census=census,
        common_masks=masks,
        admitted_census=admitted.reset_index(drop=True),
        admission_summary=summary,
        admission_manifest=admission_manifest,
        stage0_manifest=h1_manifest,
        stage0_protein_summary=h1_proteins,
        protein_order=protein_order,
        input_artifacts=tuple(artifacts),
    )


def _rank_split_sizes(count: int, groups: int) -> tuple[int, ...]:
    if count < 0 or groups < 1:
        raise PowerPlanningError(
            "invalid_feasibility_geometry", "Rank-split inputs are invalid"
        )
    base, remainder = divmod(count, groups)
    return tuple(base + int(index < remainder) for index in range(groups))


def build_feasibility_table(inputs: PowerPlanningInputs) -> pd.DataFrame:
    """Build one pre-outcome mechanical-geometry row per admitted protein."""
    identity_columns = [
        "candidate_id",
        "sampling_frame_index",
        "canonical_source_row",
        "polymer_entity_id",
        "pair_id",
        "pdb_id",
        "pdb_chain",
        "canonical_accession",
        "canonical_sequence_length",
        "mapped_residue_count",
        "common_mask_count",
        "common_mask_fraction_of_mapped",
    ]
    missing = sorted(set(identity_columns) - set(inputs.admitted_census.columns))
    if missing:
        raise PowerPlanningError(
            "admission_geometry_schema_mismatch",
            f"Admission census lacks geometry fields: {missing}",
        )
    table = inputs.admitted_census[identity_columns].copy()
    table = table.rename(
        columns={"common_mask_fraction_of_mapped": "common_mask_fraction"}
    )
    if table["common_mask_count"].isna().any() or table[
        "common_mask_fraction"
    ].isna().any():
        raise PowerPlanningError(
            "missing_admitted_geometry", "Admitted proteins require complete geometry"
        )
    admitted_ids = set(table["pair_id"].astype(str))
    mask_columns = {"pair_id", "canonical_position", "mapping_present", "common_mask"}
    if missing_mask_columns := sorted(mask_columns - set(inputs.common_masks.columns)):
        raise PowerPlanningError(
            "common_mask_geometry_mismatch",
            f"Common-mask table lacks geometry fields: {missing_mask_columns}",
        )
    masks = inputs.common_masks.loc[
        inputs.common_masks["pair_id"].astype(str).isin(admitted_ids)
    ].copy()
    if (
        set(masks["pair_id"].astype(str)) != admitted_ids
        or masks[["mapping_present", "common_mask"]].isna().any().any()
    ):
        raise PowerPlanningError(
            "common_mask_geometry_mismatch",
            "Admitted common-mask evidence is incomplete",
        )
    observed = masks.groupby("pair_id", sort=False).agg(
        canonical_row_count=("canonical_position", "size"),
        observed_mapped_residue_count=("mapping_present", "sum"),
        observed_common_mask_count=("common_mask", "sum"),
    )
    expected = table.set_index("pair_id").join(observed, how="left")
    observed_fraction = (
        expected["observed_common_mask_count"]
        / expected["observed_mapped_residue_count"]
    )
    if (
        not expected["canonical_row_count"].eq(
            expected["canonical_sequence_length"]
        ).all()
        or not expected["observed_mapped_residue_count"].eq(
            expected["mapped_residue_count"]
        ).all()
        or not expected["observed_common_mask_count"].eq(
            expected["common_mask_count"]
        ).all()
        or not np.allclose(
            observed_fraction.to_numpy(dtype=float),
            expected["common_mask_fraction"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
    ):
        raise PowerPlanningError(
            "common_mask_geometry_mismatch",
            "Admission census geometry does not match canonical common-mask rows",
        )
    sizes = [
        _rank_split_sizes(int(count), 3) for count in table["common_mask_count"]
    ]
    table["low_tertile_size"] = [value[0] for value in sizes]
    table["mid_tertile_size"] = [value[1] for value in sizes]
    table["high_tertile_size"] = [value[2] for value in sizes]
    table["low_high_matching_capacity"] = table[
        ["low_tertile_size", "high_tertile_size"]
    ].min(axis=1)
    table["balanced_3x3_reference_cell_size"] = (
        table["common_mask_count"].astype(int) // 9
    )
    for threshold in COMMON_MASK_THRESHOLDS:
        table[f"retained_at_common_mask_{threshold}"] = table[
            "common_mask_count"
        ].ge(threshold)
    return table.reset_index(drop=True)


def build_stage0_planning_effects(inputs: PowerPlanningInputs) -> pd.DataFrame:
    """Adapt only the frozen eight protein-level H1 endpoint fields."""
    source = inputs.stage0_protein_summary
    output = source[["protein_id", "position_count"]].copy()
    for spec in ENDPOINT_SPECS:
        output[spec.endpoint_id] = source[spec.source_column].astype(float)
    return output


def apply_heterogeneity_inflation(
    effects: pd.DataFrame,
    *,
    endpoint_ids: Sequence[str],
    correlation_endpoint_ids: Sequence[str] | set[str],
    multiplier: float,
) -> pd.DataFrame:
    """Inflate centered protein effects around each endpoint's empirical median."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise PowerPlanningError(
            "invalid_heterogeneity_multiplier", "Heterogeneity multiplier must be positive"
        )
    result = effects.copy()
    correlations = set(correlation_endpoint_ids)
    for endpoint_id in endpoint_ids:
        values = effects[endpoint_id].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise PowerPlanningError(
                "nonfinite_planning_effect", f"Nonfinite effect: {endpoint_id}"
            )
        if endpoint_id in correlations:
            if np.any(np.abs(values) > 1):
                raise PowerPlanningError(
                    "invalid_correlation_effect", f"Correlation outside [-1,1]: {endpoint_id}"
                )
            transformed = np.arctanh(np.clip(values, -1 + 1e-12, 1 - 1e-12))
            center = float(np.median(transformed))
            inflated = center + multiplier * (transformed - center)
            result[endpoint_id] = np.tanh(inflated)
        else:
            center = float(np.median(values))
            result[endpoint_id] = center + multiplier * (values - center)
    return result


def draw_protein_block_indices(
    *, protein_count: int, cohort_size: int, replicate_count: int, seed: int
) -> np.ndarray:
    """Draw one shared protein-row index matrix for every endpoint."""
    if protein_count < 1 or cohort_size < 1 or replicate_count < 1:
        raise PowerPlanningError(
            "invalid_simulation_shape", "Simulation dimensions must be positive"
        )
    return np.random.default_rng(seed).integers(
        0,
        protein_count,
        size=(replicate_count, cohort_size),
        dtype=np.int64,
    )


def _stream_identity(base_seed: int, scenario: str, cohort_size: int) -> tuple[str, int]:
    identity = {
        "base_seed": base_seed,
        "pipeline_version": SCHEMA_VERSION,
        "protein_id": "STAGE0_8_PROTEIN_BLOCKS",
        "stage": "scale1_power_simulation",
        "candidate_id": f"{scenario}|N={cohort_size}",
        "replicate": 0,
    }
    return derive_seed_digest(**identity), derive_seed(**identity)


def simulate_power_grid(
    effects: pd.DataFrame,
    *,
    endpoint_directions: Mapping[str, int],
    primary_endpoint_ids: set[str],
    correlation_endpoint_ids: set[str],
    n_grid: Sequence[int],
    heterogeneity_scenarios: Mapping[str, float],
    replicate_count: int,
    base_seed: int,
) -> pd.DataFrame:
    """Vectorized whole-protein block bootstrap over the frozen planning grid."""
    endpoint_ids = tuple(endpoint_directions)
    if effects["protein_id"].duplicated().any() or len(effects) != 8:
        raise PowerPlanningError(
            "invalid_power_unit", "Power simulation requires exactly eight protein rows"
        )
    if any(direction not in {-1, 1} for direction in endpoint_directions.values()):
        raise PowerPlanningError(
            "invalid_endpoint_direction", "Endpoint directions must be -1 or +1"
        )
    spec_by_id = {spec.endpoint_id: spec for spec in ENDPOINT_SPECS}
    rows: list[dict[str, Any]] = []
    for scenario, multiplier in heterogeneity_scenarios.items():
        inflated = apply_heterogeneity_inflation(
            effects,
            endpoint_ids=endpoint_ids,
            correlation_endpoint_ids=correlation_endpoint_ids,
            multiplier=float(multiplier),
        )
        for cohort_size in n_grid:
            digest, seed = _stream_identity(base_seed, str(scenario), int(cohort_size))
            indices = draw_protein_block_indices(
                protein_count=len(inflated),
                cohort_size=int(cohort_size),
                replicate_count=replicate_count,
                seed=seed,
            )
            recurrence_count = math.ceil(RECURRENCE_FRACTION * int(cohort_size))
            for endpoint_id in endpoint_ids:
                values = inflated[endpoint_id].to_numpy(dtype=float)
                sampled = values[indices]
                direction = int(endpoint_directions[endpoint_id])
                oriented = sampled * direction
                median_effects = np.median(sampled, axis=1)
                directional_fractions = np.mean(oriented > 0, axis=1)
                median_stability = float(np.mean(median_effects * direction > 0))
                recurrence_stability = float(
                    np.mean(np.sum(oriented > 0, axis=1) >= recurrence_count)
                )
                quantiles = np.quantile(
                    median_effects,
                    [0.05, 0.25, 0.5, 0.75, 0.95],
                    method="linear",
                )
                fraction_quantiles = np.quantile(
                    directional_fractions,
                    [0.05, 0.5, 0.95],
                    method="linear",
                )
                primary = endpoint_id in primary_endpoint_ids
                adequate = (
                    primary
                    and median_stability >= MEDIAN_STABILITY_MIN
                    and recurrence_stability >= RECURRENCE_STABILITY_MIN
                )
                spec = spec_by_id.get(endpoint_id)
                rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "endpoint_family": spec.family if spec else "synthetic_test",
                        "endpoint_role": spec.role if spec else (
                            "primary" if primary else "descriptive_only"
                        ),
                        "endpoint_scale": spec.scale if spec else (
                            "fisher_z_correlation"
                            if endpoint_id in correlation_endpoint_ids
                            else "natural"
                        ),
                        "predicted_direction": direction,
                        "heterogeneity_scenario": str(scenario),
                        "heterogeneity_multiplier": float(multiplier),
                        "cohort_size_n": int(cohort_size),
                        "replicate_count": int(replicate_count),
                        "biological_unit": "PROTEIN",
                        "seed_digest": digest,
                        "seed_int": seed,
                        "recurrence_fraction_target": RECURRENCE_FRACTION,
                        "recurrence_count_threshold": recurrence_count,
                        "median_direction_stability": median_stability,
                        "recurrence_stability": recurrence_stability,
                        "median_effect_p05": float(quantiles[0]),
                        "median_effect_p25": float(quantiles[1]),
                        "median_effect_median": float(quantiles[2]),
                        "median_effect_p75": float(quantiles[3]),
                        "median_effect_p95": float(quantiles[4]),
                        "directional_fraction_p05": float(fraction_quantiles[0]),
                        "directional_fraction_median": float(fraction_quantiles[1]),
                        "directional_fraction_p95": float(fraction_quantiles[2]),
                        "adequacy_status": (
                            "ADEQUATE_FOR_CONFIRMATORY_REPLICATION"
                            if adequate
                            else (
                                "NOT_ADEQUATE_FOR_CONFIRMATORY_REPLICATION"
                                if primary
                                else "NOT_PRIMARY_ENDPOINT"
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    keys = ("min", "q10", "q25", "median", "q75", "q90", "max")
    quantiles = np.quantile(
        array,
        (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
        method="linear",
    )
    output: dict[str, float | int] = {
        key: float(value) for key, value in zip(keys, quantiles, strict=True)
    }
    if np.all(array == np.floor(array)):
        output["min"] = int(output["min"])
        output["max"] = int(output["max"])
    return output


def _threshold_analysis(feasibility: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in COMMON_MASK_THRESHOLDS:
        retained = feasibility.loc[
            feasibility[f"retained_at_common_mask_{threshold}"]
        ]
        rows.append(
            {
                "common_mask_threshold": threshold,
                "retained_admitted_proteins": len(retained),
                "minimum_tertile_size": int(
                    retained[
                        ["low_tertile_size", "mid_tertile_size", "high_tertile_size"]
                    ].min(axis=1).min()
                )
                if len(retained)
                else 0,
                "minimum_low_high_matching_capacity": int(
                    retained["low_high_matching_capacity"].min()
                )
                if len(retained)
                else 0,
                "minimum_balanced_3x3_reference_cell_size": int(
                    retained["balanced_3x3_reference_cell_size"].min()
                )
                if len(retained)
                else 0,
            }
        )
    return rows


def _minimum_adequate_by_endpoint(simulation: pd.DataFrame) -> list[dict[str, Any]]:
    primary = simulation.loc[simulation["endpoint_role"].eq("primary")]
    rows: list[dict[str, Any]] = []
    for (endpoint, scenario), group in primary.groupby(
        ["endpoint_id", "heterogeneity_scenario"], sort=False
    ):
        adequate = group.loc[
            group["adequacy_status"].eq("ADEQUATE_FOR_CONFIRMATORY_REPLICATION")
        ]
        rows.append(
            {
                "endpoint_id": str(endpoint),
                "heterogeneity_scenario": str(scenario),
                "minimum_adequate_n": (
                    int(adequate["cohort_size_n"].min()) if len(adequate) else None
                ),
            }
        )
    return rows


def _overall_recommended_n(simulation: pd.DataFrame, scenario: str) -> int | None:
    primary_ids = {
        spec.endpoint_id for spec in ENDPOINT_SPECS if spec.role == "primary"
    }
    subset = simulation.loc[
        simulation["heterogeneity_scenario"].eq(scenario)
        & simulation["endpoint_id"].isin(primary_ids)
    ]
    for cohort_size in sorted(subset["cohort_size_n"].unique()):
        rows = subset.loc[subset["cohort_size_n"].eq(cohort_size)]
        if set(rows["endpoint_id"]) == primary_ids and rows[
            "adequacy_status"
        ].eq("ADEQUATE_FOR_CONFIRMATORY_REPLICATION").all():
            return int(cohort_size)
    return None


def _current_pool_decision(recommended_n: int | None, admitted_count: int) -> tuple[str, int | None]:
    if recommended_n is None:
        return POWER_PLANNING_INCONCLUSIVE, None
    additional = max(0, recommended_n - admitted_count)
    if additional == 0:
        return CURRENT_POOL_SUFFICIENT, 0
    return ADDITIONAL_ACQUISITION_REQUIRED, additional


def audit_power_planning(
    paths: ProjectPaths,
    *,
    config: PowerPlanningConfig | None = None,
    inputs: PowerPlanningInputs | None = None,
) -> PowerPlanningResult:
    """Build the complete structured planning result without writing outputs."""
    config = config or PowerPlanningConfig()
    inputs = inputs or validate_power_planning_inputs(paths, config)
    feasibility = build_feasibility_table(inputs)
    effects = build_stage0_planning_effects(inputs)
    directions = {spec.endpoint_id: spec.predicted_direction for spec in ENDPOINT_SPECS}
    primary = {spec.endpoint_id for spec in ENDPOINT_SPECS if spec.role == "primary"}
    correlations = {
        spec.endpoint_id
        for spec in ENDPOINT_SPECS
        if spec.scale == "fisher_z_correlation"
    }
    simulation = simulate_power_grid(
        effects,
        endpoint_directions=directions,
        primary_endpoint_ids=primary,
        correlation_endpoint_ids=correlations,
        n_grid=config.n_grid,
        heterogeneity_scenarios=config.heterogeneity_scenarios,
        replicate_count=config.replicate_count,
        base_seed=config.base_seed,
    )
    recommended_n = _overall_recommended_n(simulation, "H2.0")
    decision, additional = _current_pool_decision(recommended_n, len(feasibility))
    h1c = simulation.loc[
        simulation["endpoint_family"].eq("H1c")
        & simulation["heterogeneity_scenario"].eq("H2.0")
    ]
    h1c_summary = [
        {
            "endpoint_id": str(row.endpoint_id),
            "cohort_size_n": int(row.cohort_size_n),
            "median_effect": float(row.median_effect_median),
            "median_effect_uncertainty_width_p05_p95": float(
                row.median_effect_p95 - row.median_effect_p05
            ),
            "directional_fraction_median": float(row.directional_fraction_median),
            "directional_fraction_p05": float(row.directional_fraction_p05),
            "directional_fraction_p95": float(row.directional_fraction_p95),
        }
        for row in h1c.itertuples(index=False)
    ]
    threshold_analysis = _threshold_analysis(feasibility)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "planning_status": "SCALE1_PROTEIN_POWER_PLANNING_COMPLETE",
        "admitted_count": len(feasibility),
        "feasibility_distribution": {
            "common_mask_count": _quantiles(
                feasibility["common_mask_count"].astype(int).tolist()
            ),
            "common_mask_fraction": _quantiles(
                feasibility["common_mask_fraction"].astype(float).tolist()
            ),
        },
        "threshold_analysis": threshold_analysis,
        "recommended_mechanical_rule": (
            "NO_ADDITIONAL_COMMON_MASK_FEASIBILITY_FILTER_REQUIRED"
        ),
        "minimum_adequate_n_by_primary_endpoint_and_scenario": (
            _minimum_adequate_by_endpoint(simulation)
        ),
        "primary_decision_scenario": "H2.0",
        "recommended_n_under_h2_0": recommended_n,
        "current_pool_decision": decision,
        "additional_admitted_needed": additional,
        "h1c_precision_summary": h1c_summary,
        "diversity_capacity": "DIVERSITY_CAPACITY_NOT_YET_ASSESSED",
        "limitations": [
            "Only eight Stage-0 pilot proteins inform the empirical effect distribution.",
            "Simulation is empirical planning, not formal significance power.",
            "Diversity may still constrain the final cohort.",
            "No Scale-1 outcomes have been observed.",
            (
                "The recommended N is left-censored by the evaluated N-grid floor "
                "of 16; 16 is the smallest evaluated adequate N, not a proven global "
                "minimum."
            ),
        ],
        "planning_interpretation": (
            "The calculated N is a planning target conditioned on the Stage-0 "
            "empirical effect distribution and heterogeneity stress model, not a "
            "formal population-level power guarantee."
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "planning_status": "SCALE1_PROTEIN_POWER_PLANNING_COMPLETE",
        "upstream_artifacts": list(inputs.input_artifacts),
        "frozen_h1": {
            "verdict": "H1_PARTIALLY_SUPPORTED",
            "H1a": "structural decision fragility",
            "H1b": "perturbation specificity",
            "H1c": "UNRESOLVED; descriptive precision only",
        },
        "planning_endpoints": [
            {
                "endpoint_id": spec.endpoint_id,
                "source_column": spec.source_column,
                "family": spec.family,
                "role": spec.role,
                "scale": spec.scale,
                "predicted_direction": spec.predicted_direction,
            }
            for spec in ENDPOINT_SPECS
        ],
        "N_grid": list(config.n_grid),
        "heterogeneity_scenarios": dict(config.heterogeneity_scenarios),
        "heterogeneity_center": "per_endpoint_stage0_protein_median",
        "heterogeneity_transformations": {
            "correlations": "Fisher-z inflate around median z, then inverse transform",
            "non_correlations": "natural-scale inflate around endpoint median",
            "protein_dependence": (
                "whole protein-level endpoint vector block resampling"
            ),
            "central_effect_shift": "none",
        },
        "base_seed": config.base_seed,
        "seed_derivation": "dual_uq.core.hashing SHA-derived per scenario and N",
        "replicate_count": config.replicate_count,
        "quantile_method": "numpy_linear",
        "adequacy_rule": {
            "primary_scenario": "H2.0",
            "median_direction_stability_min": MEDIAN_STABILITY_MIN,
            "recurrence_fraction": RECURRENCE_FRACTION,
            "recurrence_count_rule": "ceil(0.625 * N)",
            "recurrence_stability_min": RECURRENCE_STABILITY_MIN,
            "all_primary_H1a_H1b_endpoints_required": True,
        },
        "mechanical_feasibility": {
            "rank_split": "three deterministic groups with sizes differing by at most one",
            "low_high_matching_capacity": "min(LOW size, HIGH size)",
            "balanced_3x3_reference": "floor(common_mask_count / 9)",
            "future_occupancy_claimed_balanced": False,
            "candidate_thresholds": list(COMMON_MASK_THRESHOLDS),
            "selected_filter": None,
        },
        "outputs": {},
        "manifest_self_hash_policy": (
            "reported_by_cli_after_write_to_avoid_recursive_self_hash"
        ),
        "STAGE0_LOCAL_ANALYSIS_STOP": True,
        "SCALE1_SCORING_NOT_STARTED": True,
        "SCALE1B_COHORT_NOT_SELECTED": True,
        "POWER_UNIT": "PROTEIN",
        "H1C_NOT_USED_TO_FORCE_POSITIVE_RESULT": True,
        "scope_confirmations": {
            "network_access_used": False,
            "downloads_performed": False,
            "proteinmpnn_executed": False,
            "scale1_scores_computed": False,
            "fixed_probes_constructed": False,
            "scale1b_cohort_selected": False,
            "p_values_computed": False,
            "positions_or_mutations_used_as_power_units": False,
        },
    }
    return PowerPlanningResult(
        feasibility=feasibility,
        stage0_effects=effects,
        simulation=simulation,
        summary=summary,
        manifest=manifest,
        input_artifacts=inputs.input_artifacts,
    )


def _render_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PowerPlanningError(
            "invalid_planning_output", "Planning JSON cannot be serialized"
        ) from exc


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() == payload:
            return "reused_identical"
        raise PowerPlanningError(
            "immutable_planning_artifact_conflict",
            f"Immutable output differs: {path.name}",
        )
    atomic_write_new_bytes(path, payload)
    return "created"


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, index=False)
        if path.exists():
            if path.stat().st_size == temporary.stat().st_size and sha256_file(
                path
            ) == sha256_file(temporary):
                return "reused_identical"
            raise PowerPlanningError(
                "immutable_planning_artifact_conflict",
                f"Immutable output differs: {path.name}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PowerPlanningError(
                "immutable_planning_artifact_conflict",
                f"Output appeared concurrently: {path.name}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rehash_inputs(result: PowerPlanningResult, paths: ProjectPaths) -> None:
    for record in result.input_artifacts:
        path = _resolve(paths, str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise PowerPlanningError(
                "upstream_input_changed", f"Frozen input changed: {record['label']}"
            )


def materialize_power_planning(
    result: PowerPlanningResult,
    paths: ProjectPaths,
    *,
    config: PowerPlanningConfig | None = None,
) -> dict[str, Any]:
    """Write three Parquets and summary, then immutable manifest last."""
    config = config or PowerPlanningConfig()
    _rehash_inputs(result, paths)
    output_root = _resolve(paths, config.output_root_ref)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "feasibility": output_root / "scale1_feasibility_table.parquet",
        "stage0_planning_effects": output_root
        / "scale1_stage0_planning_effects.parquet",
        "power_simulation": output_root / "scale1_power_simulation.parquet",
        "summary": output_root / "scale1_power_planning_summary.json",
        "manifest": output_root / "scale1_power_planning_manifest.json",
    }
    write_status = {
        "feasibility": _write_immutable_parquet(targets["feasibility"], result.feasibility),
        "stage0_planning_effects": _write_immutable_parquet(
            targets["stage0_planning_effects"], result.stage0_effects
        ),
        "power_simulation": _write_immutable_parquet(
            targets["power_simulation"], result.simulation
        ),
        "summary": _write_immutable_bytes(
            targets["summary"], _render_json(result.summary)
        ),
    }
    outputs: dict[str, dict[str, Any]] = {
        "feasibility": {
            "path": _logical(paths, targets["feasibility"]),
            "rows": len(result.feasibility),
            "sha256": sha256_file(targets["feasibility"]),
        },
        "stage0_planning_effects": {
            "path": _logical(paths, targets["stage0_planning_effects"]),
            "rows": len(result.stage0_effects),
            "sha256": sha256_file(targets["stage0_planning_effects"]),
        },
        "power_simulation": {
            "path": _logical(paths, targets["power_simulation"]),
            "rows": len(result.simulation),
            "sha256": sha256_file(targets["power_simulation"]),
        },
        "summary": {
            "path": _logical(paths, targets["summary"]),
            "sha256": sha256_file(targets["summary"]),
        },
    }
    manifest = {**result.manifest, "outputs": outputs}
    write_status["manifest"] = _write_immutable_bytes(
        targets["manifest"], _render_json(manifest)
    )
    outputs["manifest"] = {
        "path": _logical(paths, targets["manifest"]),
        "sha256": sha256_file(targets["manifest"]),
    }
    return {
        "planning_status": result.summary["planning_status"],
        "current_pool_decision": result.summary["current_pool_decision"],
        "write_status": write_status,
        "outputs": outputs,
    }


def run_power_planning(
    paths: ProjectPaths, *, config: PowerPlanningConfig | None = None
) -> dict[str, Any]:
    """Validate, simulate once, and materialize the immutable planning release."""
    config = config or PowerPlanningConfig()
    inputs = validate_power_planning_inputs(paths, config)
    result = audit_power_planning(paths, config=config, inputs=inputs)
    return materialize_power_planning(result, paths, config=config)
