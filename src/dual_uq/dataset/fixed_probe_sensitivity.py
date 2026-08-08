"""Offline Stage-0 fixed-probe structural-sensitivity analysis.

This module consumes the immutable scoring release. It never imports or
executes ProteinMPNN and does not generate sequences.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_file

STAGE0_ROOT = Path("experiments/p2_design_baseline/stage0")
SCORING_MANIFEST_PATH = STAGE0_ROOT / "fixed_probe_scoring_manifest.json"
WT_SCORES_PATH = STAGE0_ROOT / "fixed_probe_wt_scores.parquet"
RAW_SCORES_PATH = STAGE0_ROOT / "fixed_probe_scores.parquet"
SCORING_NULL_PATH = STAGE0_ROOT / "fixed_probe_scoring_null.parquet"

SCORING_MANIFEST_SHA256 = (
    "ec9882604cdfad0d61fbd30c1314466fbc7cbbea1fe7173d84c9913309783b00"
)
WT_SCORES_SHA256 = (
    "373948042b6868611ba6bc9ea80253d47cf0b441e002dcc6c10eed57f23b8d31"
)
RAW_SCORES_SHA256 = (
    "ac0cf8e1f251770fd4ade08580b8df19000a47124531350e9af63dd71f773abf"
)
SCORING_NULL_SHA256 = (
    "fa248c180444498e89541582edab12ef6d42a822ddbb65bc92e358b383621403"
)

REPEAT_COUNT = 30
EXPECTED_PROTEIN_COUNT = 8
EXPECTED_CANDIDATE_COUNT = 34_010
EXPECTED_POSITION_COUNT = 1_790
EXPECTED_EFFECT_COUNT = EXPECTED_CANDIDATE_COUNT * REPEAT_COUNT
ANALYSIS_PROTOCOL_VERSION = "stage0_fixed_probe_structural_sensitivity_v1"
STRUCTURAL_ORIENTATION = "AFDB_minus_PDB"
ZERO_TOLERANCE = 1.0e-6

WT_REQUIRED_COLUMNS = {
    "protein_id",
    "backbone_condition",
    "backbone_sha256",
    "repeat_index",
    "seed",
    "decoding_realization_sha256",
    "score_sum_logp_mask",
    "score_mean_logp_mask",
    "scored_residue_count",
    "model_checkpoint_sha256",
    "scoring_protocol",
}
RAW_REQUIRED_COLUMNS = WT_REQUIRED_COLUMNS | {
    "sequence_hash",
    "position",
    "wt_aa",
    "mut_aa",
    "delta_score_vs_wt",
}
NULL_REQUIRED_COLUMNS = {
    "protein_id",
    "sequence_hash",
    "position",
    "wt_aa",
    "mut_aa",
    "backbone_condition",
    "backbone_sha256",
    "model_checkpoint_sha256",
    "scoring_protocol",
    "n_repeats",
    "scoring_null_type",
    "std_convention",
    "score_mean_logp_mask_mean",
    "score_mean_logp_mask_std_population",
    "score_mean_logp_mask_min",
    "score_mean_logp_mask_max",
    "delta_score_vs_wt_mean",
    "delta_score_vs_wt_std_population",
    "delta_score_vs_wt_min",
    "delta_score_vs_wt_max",
}


class FixedProbeSensitivityError(ValueError):
    """Structured input, pairing, analysis, or release failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class SensitivityInputs:
    project_root: Path
    scoring_manifest_path: Path
    scoring_manifest_sha256: str
    scoring_manifest: dict[str, Any]
    wt_scores_path: Path
    wt_scores_sha256: str
    wt_scores: pd.DataFrame
    raw_scores_path: Path
    raw_scores_sha256: str
    raw_scores: pd.DataFrame
    scoring_null_path: Path
    scoring_null_sha256: str
    scoring_null: pd.DataFrame
    protein_manifest_path: Path
    protein_manifest_sha256: str
    protein_manifest: dict[str, Any]
    fixed_probes_path: Path
    fixed_probes_sha256: str
    fixed_probes: pd.DataFrame
    protein_order: tuple[str, ...]


@dataclass(frozen=True)
class FixedProbeSensitivityResult:
    project_root: Path
    input_provenance: dict[str, dict[str, Any]]
    paired_effects: pd.DataFrame
    candidate_summary: pd.DataFrame
    position_summary: pd.DataFrame
    protein_summary: pd.DataFrame
    repeat_count: int
    protein_order: tuple[str, ...]


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, code: str, label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FixedProbeSensitivityError(
            code, f"{label} is missing required columns: {missing}"
        )


def _require_finite(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    if not np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all():
        raise FixedProbeSensitivityError(
            "nonfinite_scoring_input", f"{label} contains NaN or infinity"
        )


def validate_sensitivity_tables(
    wt_scores: pd.DataFrame,
    raw_scores: pd.DataFrame,
    scoring_null: pd.DataFrame,
    *,
    repeat_count: int,
    expected_protein_count: int,
    expected_candidate_count: int,
) -> None:
    """Validate score-table schemas, keys, grids, and finite numerical fields."""
    _require_columns(
        wt_scores,
        WT_REQUIRED_COLUMNS,
        code="wt_score_schema_mismatch",
        label="WT scores",
    )
    _require_columns(
        raw_scores,
        RAW_REQUIRED_COLUMNS,
        code="raw_score_schema_mismatch",
        label="raw scores",
    )
    _require_columns(
        scoring_null,
        NULL_REQUIRED_COLUMNS,
        code="scoring_null_schema_mismatch",
        label="same-state null",
    )
    if repeat_count <= 0:
        raise FixedProbeSensitivityError(
            "invalid_repeat_count", "Repeat count must be positive"
        )
    if wt_scores.duplicated(
        ["protein_id", "backbone_condition", "repeat_index"]
    ).any():
        raise FixedProbeSensitivityError(
            "duplicate_wt_score_key", "WT scientific keys repeat"
        )
    if raw_scores.duplicated(
        ["protein_id", "sequence_hash", "backbone_condition", "repeat_index"]
    ).any():
        raise FixedProbeSensitivityError(
            "duplicate_raw_score_key", "Raw-score scientific keys repeat"
        )
    if scoring_null.duplicated(
        ["protein_id", "sequence_hash", "backbone_condition"]
    ).any():
        raise FixedProbeSensitivityError(
            "duplicate_scoring_null_key", "Same-state-null scientific keys repeat"
        )

    if set(wt_scores["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise FixedProbeSensitivityError(
            "wt_condition_grid_mismatch", "WT structural conditions differ"
        )
    if set(raw_scores["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise FixedProbeSensitivityError(
            "raw_condition_grid_mismatch", "Raw structural conditions differ"
        )
    if set(scoring_null["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise FixedProbeSensitivityError(
            "scoring_null_condition_grid_mismatch",
            "Same-state-null structural conditions differ",
        )

    protein_count = wt_scores["protein_id"].astype(str).nunique()
    candidate_count = raw_scores[["protein_id", "sequence_hash"]].drop_duplicates().shape[0]
    if protein_count != expected_protein_count:
        raise FixedProbeSensitivityError(
            "protein_count_mismatch", "WT protein count differs"
        )
    if candidate_count != expected_candidate_count:
        raise FixedProbeSensitivityError(
            "candidate_count_mismatch", "Raw candidate count differs"
        )
    if len(wt_scores) != expected_protein_count * 2 * repeat_count:
        raise FixedProbeSensitivityError(
            "wt_repeat_grid_mismatch", "WT repeat grid is incomplete"
        )
    if len(raw_scores) != expected_candidate_count * 2 * repeat_count:
        raise FixedProbeSensitivityError(
            "raw_repeat_grid_mismatch", "Raw repeat grid is incomplete"
        )
    if len(scoring_null) != expected_candidate_count * 2:
        raise FixedProbeSensitivityError(
            "scoring_null_grid_mismatch", "Same-state-null grid is incomplete"
        )

    expected_repeats = set(range(repeat_count))
    if set(wt_scores["repeat_index"].astype(int)) != expected_repeats:
        raise FixedProbeSensitivityError(
            "wt_repeat_grid_mismatch", "WT repeat indices differ"
        )
    if set(raw_scores["repeat_index"].astype(int)) != expected_repeats:
        raise FixedProbeSensitivityError(
            "raw_repeat_grid_mismatch", "Raw repeat indices differ"
        )
    wt_group_sizes = wt_scores.groupby(
        ["protein_id", "backbone_condition"], sort=False
    ).size()
    if len(wt_group_sizes) != expected_protein_count * 2 or not (
        wt_group_sizes == repeat_count
    ).all():
        raise FixedProbeSensitivityError(
            "wt_repeat_grid_mismatch", "WT per-condition repeats differ"
        )
    raw_group_sizes = raw_scores.groupby(
        ["protein_id", "sequence_hash", "backbone_condition"], sort=False
    ).size()
    if len(raw_group_sizes) != expected_candidate_count * 2 or not (
        raw_group_sizes == repeat_count
    ).all():
        raise FixedProbeSensitivityError(
            "raw_repeat_grid_mismatch", "Raw per-candidate repeats differ"
        )
    if set(scoring_null["n_repeats"].astype(int)) != {repeat_count}:
        raise FixedProbeSensitivityError(
            "scoring_null_repeat_mismatch", "Same-state-null repeat count differs"
        )
    if set(scoring_null["std_convention"].astype(str)) != {"population_ddof0"}:
        raise FixedProbeSensitivityError(
            "scoring_null_std_mismatch", "Same-state-null standard deviation differs"
        )
    if set(scoring_null["scoring_null_type"].astype(str)) != {
        "empirical_repeat_distribution"
    }:
        raise FixedProbeSensitivityError(
            "scoring_null_type_mismatch", "Same-state-null type differs"
        )

    _require_finite(
        wt_scores,
        ["score_sum_logp_mask", "score_mean_logp_mask"],
        label="WT scores",
    )
    _require_finite(
        raw_scores,
        ["score_sum_logp_mask", "score_mean_logp_mask", "delta_score_vs_wt"],
        label="raw scores",
    )
    null_numerical = [
        column
        for column in scoring_null.columns
        if column.endswith(("_mean", "_std_population", "_min", "_max"))
    ]
    _require_finite(scoring_null, null_numerical, label="same-state null")


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeSensitivityError(code, f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FixedProbeSensitivityError(code, f"JSON is not an object: {path}")
    return payload


def _manifest_bound_path(project_root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise FixedProbeSensitivityError(
            "scoring_manifest_contract_mismatch", f"Missing manifest path for {label}"
        )
    path = (project_root / record["path"]).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise FixedProbeSensitivityError(
            "scoring_manifest_contract_mismatch",
            f"Manifest path escapes project root for {label}",
        ) from exc
    return path


def load_frozen_sensitivity_inputs(project_root: Path) -> SensitivityInputs:
    """Load the exact immutable Stage0-2A release after all integrity gates."""
    root = project_root.resolve()
    scoring_manifest_path = root / SCORING_MANIFEST_PATH
    if not scoring_manifest_path.is_file() or sha256_file(scoring_manifest_path) != (
        SCORING_MANIFEST_SHA256
    ):
        raise FixedProbeSensitivityError(
            "scoring_manifest_hash_mismatch", "Scoring manifest SHA256 differs"
        )
    manifest = _read_json(
        scoring_manifest_path, code="scoring_manifest_contract_mismatch"
    )
    if (
        manifest.get("schema_version") != "stage0_fixed_probe_scoring_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("cross_condition_analysis_performed") is not False
    ):
        raise FixedProbeSensitivityError(
            "scoring_manifest_contract_mismatch", "Scoring manifest contract differs"
        )

    outputs = manifest.get("outputs")
    upstream = manifest.get("upstream")
    if not isinstance(outputs, dict) or not isinstance(upstream, dict):
        raise FixedProbeSensitivityError(
            "scoring_manifest_contract_mismatch", "Scoring manifest bindings are absent"
        )
    output_bindings = {
        "wt_scores": (WT_SCORES_PATH, WT_SCORES_SHA256, 480),
        "fixed_probe_scores": (RAW_SCORES_PATH, RAW_SCORES_SHA256, 2_040_600),
        "same_state_null": (SCORING_NULL_PATH, SCORING_NULL_SHA256, 68_020),
    }
    resolved_outputs: dict[str, Path] = {}
    for key, (expected_path, expected_sha, expected_rows) in output_bindings.items():
        record = outputs.get(key)
        path = _manifest_bound_path(root, record, label=key)
        if path != (root / expected_path).resolve() or record.get("sha256") != expected_sha:
            raise FixedProbeSensitivityError(
                "scoring_manifest_contract_mismatch",
                f"Scoring manifest identity differs for {key}",
            )
        if record.get("rows") != expected_rows:
            raise FixedProbeSensitivityError(
                "scoring_manifest_contract_mismatch",
                f"Scoring manifest row count differs for {key}",
            )
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FixedProbeSensitivityError(
                f"{key}_hash_mismatch", f"Frozen SHA differs for {key}"
            )
        resolved_outputs[key] = path

    protein_record = upstream.get("protein_manifest")
    probe_record = upstream.get("fixed_probes")
    protein_manifest_path = _manifest_bound_path(
        root, protein_record, label="protein_manifest"
    )
    fixed_probes_path = _manifest_bound_path(root, probe_record, label="fixed_probes")
    for path, record, label in (
        (protein_manifest_path, protein_record, "protein_manifest"),
        (fixed_probes_path, probe_record, "fixed_probes"),
    ):
        expected_sha = record.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or not path.is_file()
            or sha256_file(path) != expected_sha
        ):
            raise FixedProbeSensitivityError(
                f"{label}_hash_mismatch", f"Manifest-bound SHA differs for {label}"
            )

    try:
        wt_scores = pd.read_parquet(resolved_outputs["wt_scores"])
        raw_scores = pd.read_parquet(resolved_outputs["fixed_probe_scores"])
        scoring_null = pd.read_parquet(resolved_outputs["same_state_null"])
        fixed_probes = pd.read_parquet(fixed_probes_path)
    except (OSError, ValueError) as exc:
        raise FixedProbeSensitivityError(
            "scoring_input_unreadable", "Unable to read a frozen Parquet input"
        ) from exc
    protein_manifest = _read_json(
        protein_manifest_path, code="protein_manifest_contract_mismatch"
    )
    proteins = protein_manifest.get("proteins")
    if not isinstance(proteins, list) or len(proteins) != EXPECTED_PROTEIN_COUNT:
        raise FixedProbeSensitivityError(
            "protein_manifest_contract_mismatch", "Protein manifest membership differs"
        )
    protein_order = tuple(str(record.get("protein_id")) for record in proteins)
    if len(set(protein_order)) != EXPECTED_PROTEIN_COUNT:
        raise FixedProbeSensitivityError(
            "protein_manifest_contract_mismatch", "Protein identities repeat"
        )
    if len(fixed_probes) != EXPECTED_CANDIDATE_COUNT or fixed_probes.duplicated(
        ["protein_id", "sequence_hash"]
    ).any():
        raise FixedProbeSensitivityError(
            "fixed_probe_contract_mismatch", "Fixed-probe membership differs"
        )
    if tuple(pd.unique(fixed_probes["protein_id"].astype(str))) != protein_order:
        raise FixedProbeSensitivityError(
            "fixed_probe_contract_mismatch", "Fixed-probe protein order differs"
        )

    validate_sensitivity_tables(
        wt_scores,
        raw_scores,
        scoring_null,
        repeat_count=REPEAT_COUNT,
        expected_protein_count=EXPECTED_PROTEIN_COUNT,
        expected_candidate_count=EXPECTED_CANDIDATE_COUNT,
    )
    expected_candidates = set(
        zip(
            fixed_probes["protein_id"].astype(str),
            fixed_probes["sequence_hash"].astype(str),
            strict=True,
        )
    )
    observed_candidates = set(
        raw_scores[["protein_id", "sequence_hash"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_candidates != expected_candidates:
        raise FixedProbeSensitivityError(
            "fixed_probe_join_mismatch", "Raw candidates differ from frozen probes"
        )

    return SensitivityInputs(
        project_root=root,
        scoring_manifest_path=scoring_manifest_path,
        scoring_manifest_sha256=SCORING_MANIFEST_SHA256,
        scoring_manifest=manifest,
        wt_scores_path=resolved_outputs["wt_scores"],
        wt_scores_sha256=WT_SCORES_SHA256,
        wt_scores=wt_scores,
        raw_scores_path=resolved_outputs["fixed_probe_scores"],
        raw_scores_sha256=RAW_SCORES_SHA256,
        raw_scores=raw_scores,
        scoring_null_path=resolved_outputs["same_state_null"],
        scoring_null_sha256=SCORING_NULL_SHA256,
        scoring_null=scoring_null,
        protein_manifest_path=protein_manifest_path,
        protein_manifest_sha256=str(protein_record["sha256"]),
        protein_manifest=protein_manifest,
        fixed_probes_path=fixed_probes_path,
        fixed_probes_sha256=str(probe_record["sha256"]),
        fixed_probes=fixed_probes,
        protein_order=protein_order,
    )


def _pair_conditions(
    frame: pd.DataFrame,
    *,
    identity: list[str],
    value_columns: list[str],
    label: str,
) -> pd.DataFrame:
    """Pair one PDB and one AFDB row while preserving explicit provenance."""
    condition_frames: dict[str, pd.DataFrame] = {}
    for condition, prefix in (("PDB", "pdb"), ("AFDB", "afdb")):
        selected = frame.loc[
            frame["backbone_condition"].astype(str) == condition,
            identity + value_columns,
        ].copy()
        rename = {column: f"{prefix}_{column}" for column in value_columns}
        condition_frames[condition] = selected.rename(columns=rename)
    paired = condition_frames["PDB"].merge(
        condition_frames["AFDB"],
        on=identity,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not (paired["_merge"] == "both").all():
        raise FixedProbeSensitivityError(
            "missing_structural_pair",
            f"{label} lacks exactly one PDB/AFDB pair",
            outcome="FAIL",
        )
    return paired.drop(columns="_merge")


def _require_matching_realization_fingerprints(
    frame: pd.DataFrame,
    *,
    base_identity: list[str],
    label: str,
) -> None:
    """Diagnose fingerprint mismatches before the exact-key condition join."""
    fingerprints: dict[str, pd.DataFrame] = {}
    for condition, prefix in (("PDB", "pdb"), ("AFDB", "afdb")):
        selected = frame.loc[
            frame["backbone_condition"].astype(str) == condition,
            base_identity + ["decoding_realization_sha256"],
        ].copy()
        fingerprints[condition] = selected.rename(
            columns={
                "decoding_realization_sha256": (
                    f"{prefix}_decoding_realization_sha256"
                )
            }
        )
    compared = fingerprints["PDB"].merge(
        fingerprints["AFDB"],
        on=base_identity,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    paired = compared["_merge"] == "both"
    if paired.any() and not (
        compared.loc[paired, "pdb_decoding_realization_sha256"].astype(str)
        == compared.loc[paired, "afdb_decoding_realization_sha256"].astype(str)
    ).all():
        raise FixedProbeSensitivityError(
            "realization_fingerprint_mismatch",
            f"{label} PDB/AFDB realization fingerprints differ",
            outcome="FAIL",
        )


def build_paired_structural_effects(
    inputs: SensitivityInputs, *, repeat_count: int = REPEAT_COUNT
) -> pd.DataFrame:
    """Build the sole canonical PDB/AFDB candidate-repeat comparison table."""
    raw_base_identity = ["protein_id", "sequence_hash", "repeat_index"]
    _require_matching_realization_fingerprints(
        inputs.raw_scores,
        base_identity=raw_base_identity,
        label="Candidate score grid",
    )
    raw_identity = raw_base_identity + ["decoding_realization_sha256"]
    raw_values = [
        "position",
        "wt_aa",
        "mut_aa",
        "seed",
        "backbone_sha256",
        "score_mean_logp_mask",
        "delta_score_vs_wt",
    ]
    paired = _pair_conditions(
        inputs.raw_scores,
        identity=raw_identity,
        value_columns=raw_values,
        label="Candidate score grid",
    )
    if len(paired) != len(inputs.fixed_probes) * repeat_count:
        raise FixedProbeSensitivityError(
            "missing_structural_pair",
            "Candidate paired-row count differs",
            outcome="FAIL",
        )
    provenance_fields = ["position", "wt_aa", "mut_aa", "seed"]
    if any(
        not (paired[f"pdb_{field}"] == paired[f"afdb_{field}"]).all()
        for field in provenance_fields
    ):
        raise FixedProbeSensitivityError(
            "paired_candidate_provenance_mismatch",
            "Candidate PDB/AFDB provenance differs",
            outcome="FAIL",
        )

    wt_base_identity = ["protein_id", "repeat_index"]
    _require_matching_realization_fingerprints(
        inputs.wt_scores,
        base_identity=wt_base_identity,
        label="WT score grid",
    )
    wt = _pair_conditions(
        inputs.wt_scores,
        identity=wt_base_identity + ["decoding_realization_sha256"],
        value_columns=[
            "seed",
            "backbone_sha256",
            "score_mean_logp_mask",
        ],
        label="WT score grid",
    )
    if len(wt) != len(inputs.protein_order) * repeat_count:
        raise FixedProbeSensitivityError(
            "missing_structural_pair", "WT paired-row count differs", outcome="FAIL"
        )
    if not (wt["pdb_seed"] == wt["afdb_seed"]).all():
        raise FixedProbeSensitivityError(
            "paired_wt_provenance_mismatch", "WT PDB/AFDB seeds differ", outcome="FAIL"
        )
    wt = wt.assign(
        decoding_realization_sha256=wt["decoding_realization_sha256"].astype(str),
        wt_structural_shift=(
            wt["afdb_score_mean_logp_mask"] - wt["pdb_score_mean_logp_mask"]
        ),
    )
    wt_columns = [
        "protein_id",
        "repeat_index",
        "decoding_realization_sha256",
        "wt_structural_shift",
        "pdb_score_mean_logp_mask",
        "afdb_score_mean_logp_mask",
    ]
    wt = wt[wt_columns].rename(
        columns={
            "pdb_score_mean_logp_mask": "pdb_wt_score_mean_logp_mask",
            "afdb_score_mean_logp_mask": "afdb_wt_score_mean_logp_mask",
        }
    )

    paired = paired.assign(
        decoding_realization_sha256=paired["decoding_realization_sha256"].astype(str),
        raw_structural_shift=(
            paired["afdb_score_mean_logp_mask"]
            - paired["pdb_score_mean_logp_mask"]
        ),
        structural_mutation_interaction=(
            paired["afdb_delta_score_vs_wt"]
            - paired["pdb_delta_score_vs_wt"]
        ),
    )
    paired = paired.merge(
        wt,
        on=[
            "protein_id",
            "repeat_index",
            "decoding_realization_sha256",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if paired["wt_structural_shift"].isna().any():
        raise FixedProbeSensitivityError(
            "realization_fingerprint_mismatch",
            "Candidate realization does not join to matching WT",
            outcome="FAIL",
        )
    algebraic = paired["raw_structural_shift"] - paired["wt_structural_shift"]
    if not np.allclose(
        paired["structural_mutation_interaction"].to_numpy(dtype=np.float64),
        algebraic.to_numpy(dtype=np.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise FixedProbeSensitivityError(
            "interaction_algebra_mismatch",
            "WT-relative and raw-shift interaction definitions differ",
            outcome="FAIL",
        )

    candidate_order = inputs.fixed_probes[
        ["protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"]
    ].rename(
        columns={
            "position": "position_frozen",
            "wt_aa": "wt_aa_frozen",
            "mut_aa": "mut_aa_frozen",
        }
    )
    candidate_order["_candidate_order"] = np.arange(
        len(candidate_order), dtype=np.int64
    )
    paired = paired.merge(
        candidate_order,
        on=["protein_id", "sequence_hash"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if paired["_candidate_order"].isna().any():
        raise FixedProbeSensitivityError(
            "fixed_probe_join_mismatch",
            "Paired candidate does not join to the frozen library",
            outcome="FAIL",
        )
    for field in ("position", "wt_aa", "mut_aa"):
        if not (paired[f"pdb_{field}"] == paired[f"{field}_frozen"]).all():
            raise FixedProbeSensitivityError(
                "paired_candidate_provenance_mismatch",
                f"Paired candidate {field} differs from frozen probes",
                outcome="FAIL",
            )
    paired = paired.sort_values(
        ["_candidate_order", "repeat_index"], kind="stable"
    ).reset_index(drop=True)
    result = pd.DataFrame(
        {
            "protein_id": paired["protein_id"].astype(str),
            "sequence_hash": paired["sequence_hash"].astype(str),
            "position": paired["pdb_position"].astype(int),
            "wt_aa": paired["pdb_wt_aa"].astype(str),
            "mut_aa": paired["pdb_mut_aa"].astype(str),
            "repeat_index": paired["repeat_index"].astype(int),
            "seed": paired["pdb_seed"].astype(int),
            "decoding_realization_sha256": paired[
                "decoding_realization_sha256"
            ].astype(str),
            "pdb_backbone_sha256": paired["pdb_backbone_sha256"].astype(str),
            "afdb_backbone_sha256": paired["afdb_backbone_sha256"].astype(str),
            "pdb_raw_score_mean_logp_mask": paired[
                "pdb_score_mean_logp_mask"
            ].astype(float),
            "afdb_raw_score_mean_logp_mask": paired[
                "afdb_score_mean_logp_mask"
            ].astype(float),
            "pdb_wt_score_mean_logp_mask": paired[
                "pdb_wt_score_mean_logp_mask"
            ].astype(float),
            "afdb_wt_score_mean_logp_mask": paired[
                "afdb_wt_score_mean_logp_mask"
            ].astype(float),
            "pdb_mutation_effect": paired["pdb_delta_score_vs_wt"].astype(float),
            "afdb_mutation_effect": paired["afdb_delta_score_vs_wt"].astype(float),
            "raw_structural_shift": paired["raw_structural_shift"].astype(float),
            "wt_structural_shift": paired["wt_structural_shift"].astype(float),
            "structural_mutation_interaction": paired[
                "structural_mutation_interaction"
            ].astype(float),
        }
    )
    numerical = [
        "pdb_raw_score_mean_logp_mask",
        "afdb_raw_score_mean_logp_mask",
        "pdb_wt_score_mean_logp_mask",
        "afdb_wt_score_mean_logp_mask",
        "pdb_mutation_effect",
        "afdb_mutation_effect",
        "raw_structural_shift",
        "wt_structural_shift",
        "structural_mutation_interaction",
    ]
    if not np.isfinite(result[numerical].to_numpy(dtype=np.float64)).all():
        raise FixedProbeSensitivityError(
            "nonfinite_structural_effect", "Paired effects contain NaN or infinity", outcome="FAIL"
        )
    return result


def summarize_candidate_sensitivity(
    paired: pd.DataFrame,
    scoring_null: pd.DataFrame,
    candidate_order: pd.DataFrame,
    *,
    repeat_count: int = REPEAT_COUNT,
    zero_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    """Summarize paired interactions and same-state technical variability."""
    keys = ["protein_id", "sequence_hash"]
    required_paired = {
        *keys,
        "position",
        "wt_aa",
        "mut_aa",
        "repeat_index",
        "structural_mutation_interaction",
    }
    missing = sorted(required_paired - set(paired.columns))
    if missing:
        raise FixedProbeSensitivityError(
            "paired_effect_schema_mismatch",
            f"Paired effects are missing columns: {missing}",
            outcome="FAIL",
        )
    if paired.duplicated(keys + ["repeat_index"]).any():
        raise FixedProbeSensitivityError(
            "duplicate_paired_effect_key",
            "Paired candidate-repeat keys repeat",
            outcome="FAIL",
        )
    group_sizes = paired.groupby(keys, sort=False).size()
    if len(group_sizes) != len(candidate_order) or not (
        group_sizes == repeat_count
    ).all():
        raise FixedProbeSensitivityError(
            "candidate_repeat_grid_mismatch",
            "Candidate interaction repeat grid differs",
            outcome="FAIL",
        )
    expected_repeats = set(range(repeat_count))
    repeat_sets = paired.groupby(keys, sort=False)["repeat_index"].agg(
        lambda value: {int(item) for item in value}
    )
    if not repeat_sets.map(lambda value: value == expected_repeats).all():
        raise FixedProbeSensitivityError(
            "candidate_repeat_grid_mismatch",
            "Candidate interaction repeat indices differ",
            outcome="FAIL",
        )
    provenance = paired.groupby(keys, sort=False)[
        ["position", "wt_aa", "mut_aa"]
    ].nunique()
    if not (provenance == 1).all().all():
        raise FixedProbeSensitivityError(
            "candidate_provenance_mismatch",
            "Candidate provenance varies across repeats",
            outcome="FAIL",
        )

    metric = "structural_mutation_interaction"
    grouped = paired.groupby(keys, sort=False)[metric]
    summary = grouped.agg(
        interaction_mean="mean",
        interaction_std_population=lambda values: float(
            np.asarray(values, dtype=np.float64).std(ddof=0)
        ),
        interaction_min="min",
        interaction_max="max",
        interaction_median="median",
        n_repeats="size",
    )
    quantile_levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    quantiles = grouped.quantile(quantile_levels, interpolation="linear").unstack()
    quantiles.columns = [f"interaction_q{int(level * 100):02d}" for level in quantile_levels]
    summary = summary.join(quantiles)
    signs = paired.assign(
        _positive=paired[metric] > zero_tolerance,
        _negative=paired[metric] < -zero_tolerance,
        _zero=paired[metric].abs() <= zero_tolerance,
    ).groupby(keys, sort=False)[["_positive", "_negative", "_zero"]].mean()
    summary = summary.join(
        signs.rename(
            columns={
                "_positive": "fraction_positive",
                "_negative": "fraction_negative",
                "_zero": "fraction_zero_within_tolerance",
            }
        )
    ).reset_index()

    frozen = candidate_order[
        ["protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"]
    ].copy()
    if frozen.duplicated(keys).any():
        raise FixedProbeSensitivityError(
            "duplicate_fixed_probe_key", "Frozen candidate identities repeat", outcome="FAIL"
        )
    frozen["_candidate_order"] = np.arange(len(frozen), dtype=np.int64)
    summary = frozen.merge(summary, on=keys, how="left", validate="one_to_one", sort=False)
    if summary["n_repeats"].isna().any():
        raise FixedProbeSensitivityError(
            "candidate_repeat_grid_mismatch",
            "A frozen candidate lacks an interaction summary",
            outcome="FAIL",
        )

    null_required = {
        *keys,
        "position",
        "wt_aa",
        "mut_aa",
        "backbone_condition",
        "n_repeats",
        "delta_score_vs_wt_std_population",
    }
    null_missing = sorted(null_required - set(scoring_null.columns))
    if null_missing:
        raise FixedProbeSensitivityError(
            "scoring_null_schema_mismatch",
            f"Same-state null is missing columns: {null_missing}",
            outcome="FAIL",
        )
    if scoring_null.duplicated(keys + ["backbone_condition"]).any():
        raise FixedProbeSensitivityError(
            "duplicate_scoring_null_key",
            "Same-state-null candidate conditions repeat",
            outcome="FAIL",
        )
    if set(scoring_null["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise FixedProbeSensitivityError(
            "scoring_null_condition_grid_mismatch",
            "Same-state-null structural conditions differ",
            outcome="FAIL",
        )
    if set(scoring_null["n_repeats"].astype(int)) != {repeat_count}:
        raise FixedProbeSensitivityError(
            "scoring_null_repeat_mismatch",
            "Same-state-null repeat count differs",
            outcome="FAIL",
        )
    technical = scoring_null.pivot(
        index=keys,
        columns="backbone_condition",
        values="delta_score_vs_wt_std_population",
    ).rename(columns={"PDB": "technical_sd_pdb", "AFDB": "technical_sd_afdb"})
    if list(technical.columns) != ["technical_sd_afdb", "technical_sd_pdb"]:
        technical = technical.reindex(columns=["technical_sd_pdb", "technical_sd_afdb"])
    else:
        technical = technical[["technical_sd_pdb", "technical_sd_afdb"]]
    technical = technical.reset_index()
    if len(technical) != len(frozen) or technical[
        ["technical_sd_pdb", "technical_sd_afdb"]
    ].isna().any().any():
        raise FixedProbeSensitivityError(
            "scoring_null_grid_mismatch",
            "Same-state-null candidate grid differs",
            outcome="FAIL",
        )
    summary = summary.merge(technical, on=keys, how="left", validate="one_to_one")
    if (summary[["technical_sd_pdb", "technical_sd_afdb"]] < 0).any().any():
        raise FixedProbeSensitivityError(
            "invalid_technical_scale",
            "Same-state technical standard deviation is negative",
            outcome="FAIL",
        )
    summary["technical_scale"] = np.sqrt(
        (
            summary["technical_sd_pdb"].pow(2)
            + summary["technical_sd_afdb"].pow(2)
        )
        / 2.0
    )
    defined = summary["technical_scale"] > 0.0
    summary["interaction_to_technical_scale"] = np.where(
        defined,
        summary["interaction_mean"].abs() / summary["technical_scale"],
        np.nan,
    )
    summary["technical_scale_status"] = np.where(
        defined, "defined", "zero_technical_scale"
    )
    summary["quantile_interval_semantics"] = (
        "decoding_realization_quantile_interval"
    )
    summary["zero_tolerance"] = float(zero_tolerance)
    summary = summary.sort_values("_candidate_order", kind="stable").drop(
        columns="_candidate_order"
    )
    summary["n_repeats"] = summary["n_repeats"].astype(int)
    fractions = summary[
        ["fraction_positive", "fraction_negative", "fraction_zero_within_tolerance"]
    ].sum(axis=1)
    if not np.allclose(fractions.to_numpy(), 1.0, atol=1.0e-12, rtol=0.0):
        raise FixedProbeSensitivityError(
            "sign_fraction_mismatch",
            "Candidate sign fractions do not partition repeats",
            outcome="FAIL",
        )
    return summary.reset_index(drop=True)


def summarize_position_sensitivity(
    candidate_summary: pd.DataFrame,
    *,
    mutations_per_position: int = 19,
) -> pd.DataFrame:
    """Aggregate canonical candidate summaries within UniProt positions."""
    required = {
        "protein_id",
        "sequence_hash",
        "position",
        "wt_aa",
        "mut_aa",
        "interaction_mean",
        "interaction_std_population",
        "technical_scale",
        "interaction_to_technical_scale",
        "zero_tolerance",
    }
    missing = sorted(required - set(candidate_summary.columns))
    if missing:
        raise FixedProbeSensitivityError(
            "candidate_summary_schema_mismatch",
            f"Candidate summary is missing columns: {missing}",
            outcome="FAIL",
        )
    if candidate_summary.duplicated(["protein_id", "position", "mut_aa"]).any():
        raise FixedProbeSensitivityError(
            "duplicate_position_mutation",
            "A protein-position mutation repeats",
            outcome="FAIL",
        )
    groups = candidate_summary.groupby(["protein_id", "position"], sort=False)
    sizes = groups.size()
    mutation_counts = groups["mut_aa"].nunique()
    if not (sizes == mutations_per_position).all() or not (
        mutation_counts == mutations_per_position
    ).all():
        raise FixedProbeSensitivityError(
            "position_mutation_grid_mismatch",
            "Every position must contain exactly 19 unique mutations",
            outcome="FAIL",
        )

    rows: list[dict[str, Any]] = []
    for (protein_id, position), group in groups:
        wt_values = tuple(pd.unique(group["wt_aa"].astype(str)))
        tolerance_values = tuple(pd.unique(group["zero_tolerance"].astype(float)))
        if len(wt_values) != 1 or len(tolerance_values) != 1:
            raise FixedProbeSensitivityError(
                "position_provenance_mismatch",
                "WT identity or zero tolerance varies within a position",
                outcome="FAIL",
            )
        values = group["interaction_mean"].to_numpy(dtype=np.float64)
        absolute = np.abs(values)
        maximum_index = int(np.argmax(absolute))
        maximum = group.iloc[maximum_index]
        tolerance = tolerance_values[0]
        defined_ratios = group["interaction_to_technical_scale"].dropna().to_numpy(
            dtype=np.float64
        )
        positive = int((values > tolerance).sum())
        negative = int((values < -tolerance).sum())
        rows.append(
            {
                "protein_id": str(protein_id),
                "position": int(position),
                "wt_aa": wt_values[0],
                "n_mutations": len(group),
                "position_interaction_mean": float(values.mean()),
                "position_mean_abs_interaction": float(absolute.mean()),
                "position_rms_interaction": float(np.sqrt(np.mean(values**2))),
                "position_median_abs_interaction": float(np.median(absolute)),
                "position_max_abs_interaction": float(absolute[maximum_index]),
                "max_abs_interaction_mut_aa": str(maximum["mut_aa"]),
                "max_abs_interaction_substitution": (
                    f"{wt_values[0]}>{maximum['mut_aa']}"
                ),
                "mean_technical_scale": float(
                    group["technical_scale"].to_numpy(dtype=np.float64).mean()
                ),
                "defined_ratio_count": int(defined_ratios.size),
                "mean_interaction_to_technical_scale_defined": (
                    float(defined_ratios.mean())
                    if defined_ratios.size
                    else np.nan
                ),
                "positive_mutation_count": positive,
                "negative_mutation_count": negative,
                "positive_mutation_fraction": float(positive / len(group)),
                "negative_mutation_fraction": float(negative / len(group)),
                "zero_tolerance": float(tolerance),
            }
        )
    return pd.DataFrame(rows)


def _linear_quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    quantiles = np.quantile(values, levels, method="linear")
    return {
        f"{prefix}_q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, quantiles, strict=True)
    }


def summarize_protein_sensitivity(
    candidate_summary: pd.DataFrame,
    position_summary: pd.DataFrame,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Build the eight-row pilot summary with protein as interpretation unit."""
    candidate_membership = tuple(pd.unique(candidate_summary["protein_id"].astype(str)))
    position_membership = tuple(pd.unique(position_summary["protein_id"].astype(str)))
    if (
        candidate_membership != protein_order
        or position_membership != protein_order
        or len(set(protein_order)) != len(protein_order)
    ):
        raise FixedProbeSensitivityError(
            "protein_membership_mismatch",
            "Protein summary membership/order differs",
            outcome="FAIL",
        )
    rows: list[dict[str, Any]] = []
    for protein_id in protein_order:
        candidates = candidate_summary.loc[
            candidate_summary["protein_id"].astype(str) == protein_id
        ]
        positions = position_summary.loc[
            position_summary["protein_id"].astype(str) == protein_id
        ]
        if (
            candidates.empty
            or positions.empty
            or int(positions["n_mutations"].sum()) != len(candidates)
        ):
            raise FixedProbeSensitivityError(
                "protein_count_reconciliation_mismatch",
                f"Candidate/position counts do not reconcile for {protein_id}",
                outcome="FAIL",
            )
        values = candidates["interaction_mean"].to_numpy(dtype=np.float64)
        absolute = np.abs(values)
        ratios = candidates["interaction_to_technical_scale"].dropna().to_numpy(
            dtype=np.float64
        )
        tolerances = tuple(pd.unique(candidates["zero_tolerance"].astype(float)))
        if len(tolerances) != 1:
            raise FixedProbeSensitivityError(
                "protein_tolerance_mismatch",
                f"Zero tolerance varies for {protein_id}",
                outcome="FAIL",
            )
        tolerance = tolerances[0]
        position_values = positions["position_mean_abs_interaction"].to_numpy(
            dtype=np.float64
        )
        row: dict[str, Any] = {
            "protein_id": protein_id,
            "candidate_count": len(candidates),
            "position_count": len(positions),
            "candidate_mean_abs_interaction": float(absolute.mean()),
            "candidate_rms_interaction": float(np.sqrt(np.mean(values**2))),
            "candidate_median_abs_interaction": float(np.median(absolute)),
            "candidate_max_abs_interaction": float(absolute.max()),
            "mean_interaction_std_population": float(
                candidates["interaction_std_population"].to_numpy(dtype=np.float64).mean()
            ),
            "mean_technical_scale": float(
                candidates["technical_scale"].to_numpy(dtype=np.float64).mean()
            ),
            "defined_ratio_count": int(ratios.size),
            "candidate_fraction_positive_mean_interaction": float(
                (values > tolerance).mean()
            ),
            "candidate_fraction_negative_mean_interaction": float(
                (values < -tolerance).mean()
            ),
            "position_mean_abs_interaction_mean": float(position_values.mean()),
            "position_mean_abs_interaction_median": float(
                np.median(position_values)
            ),
            "position_mean_abs_interaction_q75": float(
                np.quantile(position_values, 0.75, method="linear")
            ),
            "position_mean_abs_interaction_q95": float(
                np.quantile(position_values, 0.95, method="linear")
            ),
            "position_mean_abs_interaction_max": float(position_values.max()),
            "zero_tolerance": float(tolerance),
        }
        row.update(_linear_quantiles(absolute, "candidate_abs_interaction"))
        if ratios.size:
            row.update(
                {
                    "interaction_to_technical_scale_mean": float(ratios.mean()),
                    "interaction_to_technical_scale_median": float(
                        np.median(ratios)
                    ),
                    **_linear_quantiles(
                        ratios, "interaction_to_technical_scale"
                    ),
                }
            )
        else:
            row.update(
                {
                    "interaction_to_technical_scale_mean": np.nan,
                    "interaction_to_technical_scale_median": np.nan,
                    **{
                        f"interaction_to_technical_scale_q{quantile}": np.nan
                        for quantile in ("05", "25", "50", "75", "95")
                    },
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _logical_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise FixedProbeSensitivityError(
            "nonportable_analysis_path",
            f"Scientific artifact is outside the project root: {path}",
            outcome="FAIL",
        ) from exc


def build_fixed_probe_sensitivity_result(
    inputs: SensitivityInputs,
) -> FixedProbeSensitivityResult:
    """Run the single canonical raw-to-summary analysis dataflow."""
    paired = build_paired_structural_effects(inputs, repeat_count=REPEAT_COUNT)
    candidates = summarize_candidate_sensitivity(
        paired,
        inputs.scoring_null,
        inputs.fixed_probes,
        repeat_count=REPEAT_COUNT,
        zero_tolerance=ZERO_TOLERANCE,
    )
    positions = summarize_position_sensitivity(candidates)
    proteins = summarize_protein_sensitivity(
        candidates, positions, inputs.protein_order
    )
    root = inputs.project_root
    provenance = {
        "scoring_manifest": {
            "path": _logical_path(inputs.scoring_manifest_path, root),
            "sha256": inputs.scoring_manifest_sha256,
            "rows": None,
        },
        "wt_scores": {
            "path": _logical_path(inputs.wt_scores_path, root),
            "sha256": inputs.wt_scores_sha256,
            "rows": len(inputs.wt_scores),
        },
        "raw_scores": {
            "path": _logical_path(inputs.raw_scores_path, root),
            "sha256": inputs.raw_scores_sha256,
            "rows": len(inputs.raw_scores),
        },
        "same_state_null": {
            "path": _logical_path(inputs.scoring_null_path, root),
            "sha256": inputs.scoring_null_sha256,
            "rows": len(inputs.scoring_null),
        },
        "protein_manifest": {
            "path": _logical_path(inputs.protein_manifest_path, root),
            "sha256": inputs.protein_manifest_sha256,
            "rows": len(inputs.protein_manifest["proteins"]),
        },
        "fixed_probes": {
            "path": _logical_path(inputs.fixed_probes_path, root),
            "sha256": inputs.fixed_probes_sha256,
            "rows": len(inputs.fixed_probes),
        },
    }
    result = FixedProbeSensitivityResult(
        project_root=root,
        input_provenance=provenance,
        paired_effects=paired,
        candidate_summary=candidates,
        position_summary=positions,
        protein_summary=proteins,
        repeat_count=REPEAT_COUNT,
        protein_order=inputs.protein_order,
    )
    validate_sensitivity_result(result)
    return result


def _forbidden_fields(frames: tuple[pd.DataFrame, ...]) -> list[str]:
    forbidden_tokens = (
        "uncertainty_label",
        "binary_label",
        "p_value",
        "significance",
        "top_k",
        "top1",
        "rank",
        "regret",
    )
    return sorted(
        {
            column
            for frame in frames
            for column in frame.columns
            if any(token in column.lower() for token in forbidden_tokens)
        }
    )


def validate_sensitivity_result(
    result: FixedProbeSensitivityResult,
    *,
    expected_effect_rows: int = EXPECTED_EFFECT_COUNT,
    expected_candidate_rows: int = EXPECTED_CANDIDATE_COUNT,
    expected_position_rows: int = EXPECTED_POSITION_COUNT,
    expected_protein_rows: int = EXPECTED_PROTEIN_COUNT,
) -> None:
    """Apply the final release gate before any canonical output is written."""
    frames = (
        result.paired_effects,
        result.candidate_summary,
        result.position_summary,
        result.protein_summary,
    )
    forbidden = _forbidden_fields(frames)
    if forbidden:
        raise FixedProbeSensitivityError(
            "forbidden_analysis_field",
            f"Decision/significance fields are forbidden: {forbidden}",
            outcome="FAIL",
        )
    if tuple(len(frame) for frame in frames) != (
        expected_effect_rows,
        expected_candidate_rows,
        expected_position_rows,
        expected_protein_rows,
    ):
        raise FixedProbeSensitivityError(
            "sensitivity_output_count_mismatch",
            "Sensitivity output row counts differ",
            outcome="FAIL",
        )
    if result.paired_effects.duplicated(
        ["protein_id", "sequence_hash", "repeat_index"]
    ).any():
        raise FixedProbeSensitivityError(
            "duplicate_structural_effect_key",
            "Paired-effect scientific keys repeat",
            outcome="FAIL",
        )
    if result.candidate_summary.duplicated(["protein_id", "sequence_hash"]).any():
        raise FixedProbeSensitivityError(
            "duplicate_candidate_summary_key",
            "Candidate-summary scientific keys repeat",
            outcome="FAIL",
        )
    if result.position_summary.duplicated(["protein_id", "position"]).any():
        raise FixedProbeSensitivityError(
            "duplicate_position_summary_key",
            "Position-summary scientific keys repeat",
            outcome="FAIL",
        )
    if result.protein_summary["protein_id"].astype(str).duplicated().any():
        raise FixedProbeSensitivityError(
            "duplicate_protein_summary_key",
            "Protein-summary scientific keys repeat",
            outcome="FAIL",
        )
    if tuple(result.protein_summary["protein_id"].astype(str)) != result.protein_order:
        raise FixedProbeSensitivityError(
            "protein_membership_mismatch",
            "Protein summary does not preserve frozen order",
            outcome="FAIL",
        )
    if set(result.candidate_summary["n_repeats"].astype(int)) != {
        result.repeat_count
    }:
        raise FixedProbeSensitivityError(
            "candidate_repeat_grid_mismatch",
            "Candidate summary repeat counts differ",
            outcome="FAIL",
        )
    if set(result.position_summary["n_mutations"].astype(int)) != {19}:
        raise FixedProbeSensitivityError(
            "position_mutation_grid_mismatch",
            "Position summary mutation counts differ",
            outcome="FAIL",
        )
    if int(result.position_summary["n_mutations"].sum()) != len(
        result.candidate_summary
    ):
        raise FixedProbeSensitivityError(
            "protein_count_reconciliation_mismatch",
            "Candidate and position totals differ",
            outcome="FAIL",
        )
    if int(result.protein_summary["candidate_count"].sum()) != len(
        result.candidate_summary
    ) or int(result.protein_summary["position_count"].sum()) != len(
        result.position_summary
    ):
        raise FixedProbeSensitivityError(
            "protein_count_reconciliation_mismatch",
            "Protein summary totals differ",
            outcome="FAIL",
        )
    effect_columns = [
        "pdb_raw_score_mean_logp_mask",
        "afdb_raw_score_mean_logp_mask",
        "pdb_wt_score_mean_logp_mask",
        "afdb_wt_score_mean_logp_mask",
        "pdb_mutation_effect",
        "afdb_mutation_effect",
        "raw_structural_shift",
        "wt_structural_shift",
        "structural_mutation_interaction",
    ]
    if not np.isfinite(
        result.paired_effects[effect_columns].to_numpy(dtype=np.float64)
    ).all():
        raise FixedProbeSensitivityError(
            "nonfinite_structural_effect",
            "Paired structural effects contain NaN or infinity",
            outcome="FAIL",
        )
    candidate_finite = [
        column
        for column in result.candidate_summary.columns
        if column.startswith("interaction_")
        and column != "interaction_to_technical_scale"
    ] + [
        "technical_sd_pdb",
        "technical_sd_afdb",
        "technical_scale",
        "fraction_positive",
        "fraction_negative",
        "fraction_zero_within_tolerance",
    ]
    if not np.isfinite(
        result.candidate_summary[candidate_finite].to_numpy(dtype=np.float64)
    ).all():
        raise FixedProbeSensitivityError(
            "nonfinite_candidate_summary",
            "Candidate summary contains invalid required values",
            outcome="FAIL",
        )
    defined = result.candidate_summary["technical_scale_status"] == "defined"
    ratios = result.candidate_summary["interaction_to_technical_scale"]
    if (
        not np.isfinite(ratios.loc[defined].to_numpy(dtype=np.float64)).all()
        or ratios.loc[~defined].notna().any()
        or not set(result.candidate_summary["technical_scale_status"]).issubset(
            {"defined", "zero_technical_scale"}
        )
    ):
        raise FixedProbeSensitivityError(
            "invalid_technical_scale",
            "Technical-scale ratio/status semantics differ",
            outcome="FAIL",
        )


def write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    """Atomically create deterministic Parquet or reuse byte-identical output."""
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
            if (
                path.stat().st_size == temporary.stat().st_size
                and sha256_file(path) == sha256_file(temporary)
            ):
                return "reused_identical"
            raise FixedProbeSensitivityError(
                "immutable_analysis_artifact_conflict",
                f"Existing analysis artifact differs: {path}",
                outcome="FAIL",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FixedProbeSensitivityError(
                "immutable_analysis_artifact_conflict",
                f"Analysis artifact appeared concurrently: {path}",
                outcome="FAIL",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render_json(payload: dict[str, Any]) -> bytes:
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
        raise FixedProbeSensitivityError(
            "invalid_analysis_manifest",
            "Analysis manifest cannot be serialized",
            outcome="FAIL",
        ) from exc


def write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically create deterministic JSON or reuse byte-identical output."""
    rendered = _render_json(payload)
    if path.exists():
        if path.read_bytes() == rendered:
            return "reused_identical"
        raise FixedProbeSensitivityError(
            "immutable_analysis_artifact_conflict",
            f"Existing analysis manifest differs: {path}",
            outcome="FAIL",
        )
    atomic_write_new_bytes(path, rendered)
    return "created"


def _rehash_inputs(result: FixedProbeSensitivityResult) -> None:
    for label, record in result.input_provenance.items():
        path = result.project_root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise FixedProbeSensitivityError(
                "upstream_mutated_during_analysis",
                f"Frozen input changed during analysis: {label}",
            )


def _analysis_manifest(
    result: FixedProbeSensitivityResult,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    frames = {
        "paired_effects": result.paired_effects,
        "candidate_summary": result.candidate_summary,
        "position_summary": result.position_summary,
        "protein_summary": result.protein_summary,
    }
    outputs = {
        label: {
            "path": _logical_path(path, result.project_root),
            "sha256": sha256_file(path),
            "rows": len(frames[label]),
            "bytes": path.stat().st_size,
        }
        for label, path in output_paths.items()
    }
    return {
        "schema_version": "stage0_fixed_probe_sensitivity_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_fixed_probe_structural_condition_sensitivity_only",
        "inputs": result.input_provenance,
        "analysis_protocol": {
            "identity": ANALYSIS_PROTOCOL_VERSION,
            "structural_orientation": STRUCTURAL_ORIENTATION,
            "primary_estimand": "delta_score_vs_wt_AFDB_minus_delta_score_vs_wt_PDB",
            "equivalent_estimand": "candidate_raw_AFDB_minus_PDB_minus_WT_raw_AFDB_minus_PDB",
            "repeat_count": result.repeat_count,
            "repeat_seeds": list(range(result.repeat_count)),
            "standard_deviation": "population_ddof0",
            "quantile_levels": [0.05, 0.25, 0.50, 0.75, 0.95],
            "quantile_method": "linear",
            "quantile_interval_semantics": "decoding_realization_quantile_interval",
            "zero_tolerance_absolute": ZERO_TOLERANCE,
            "technical_scale": "sqrt((technical_sd_pdb^2 + technical_sd_afdb^2) / 2)",
        },
        "protein_order": list(result.protein_order),
        "outputs": outputs,
        "final_counts": {
            "paired_candidate_repeat_rows": len(result.paired_effects),
            "candidate_summary_rows": len(result.candidate_summary),
            "position_summary_rows": len(result.position_summary),
            "protein_summary_rows": len(result.protein_summary),
        },
        "scope_declarations": {
            "hypothesis_testing_performed": False,
            "binary_uncertainty_classification_performed": False,
            "top_k_or_rank_analysis_performed": False,
            "regret_analysis_performed": False,
            "proteinmpnn_executed": False,
            "sequence_generation_performed": False,
            "external_evaluator_executed": False,
            "arm_b_started": False,
            "stage0_2b_started": False,
        },
    }


def materialize_fixed_probe_sensitivity(
    result: FixedProbeSensitivityResult,
    output_root: Path,
) -> dict[str, Any]:
    """Validate and immutably render all analysis artifacts, manifest last."""
    validate_sensitivity_result(result)
    _rehash_inputs(result)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "paired_effects": output_root / "fixed_probe_structural_effects.parquet",
        "candidate_summary": output_root
        / "fixed_probe_candidate_sensitivity.parquet",
        "position_summary": output_root / "fixed_probe_position_sensitivity.parquet",
        "protein_summary": output_root / "fixed_probe_protein_sensitivity.parquet",
    }
    frames = {
        "paired_effects": result.paired_effects,
        "candidate_summary": result.candidate_summary,
        "position_summary": result.position_summary,
        "protein_summary": result.protein_summary,
    }
    write_status = {
        label: write_immutable_parquet(output_paths[label], frames[label])
        for label in output_paths
    }
    _rehash_inputs(result)
    manifest = _analysis_manifest(result, output_paths)
    manifest_path = output_root / "fixed_probe_sensitivity_manifest.json"
    write_status["manifest"] = write_immutable_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "manifest_path": manifest_path,
        "write_status": write_status,
        "paired_effect_rows": len(result.paired_effects),
        "candidate_summary_rows": len(result.candidate_summary),
        "position_summary_rows": len(result.position_summary),
        "protein_summary_rows": len(result.protein_summary),
    }


def run_fixed_probe_sensitivity(project_root: Path) -> FixedProbeSensitivityResult:
    """Load frozen inputs and construct the canonical structured result."""
    inputs = load_frozen_sensitivity_inputs(project_root)
    return build_fixed_probe_sensitivity_result(inputs)
