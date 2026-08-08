"""Offline local amino-acid decision sensitivity for frozen Stage0 scores."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.core.hashing import sha256_file
from dual_uq.dataset.fixed_probe_sensitivity import (
    STAGE0_ROOT,
    FixedProbeSensitivityError,
    SensitivityInputs,
    load_frozen_sensitivity_inputs,
    write_immutable_json,
    write_immutable_parquet,
)

SENSITIVITY_MANIFEST_PATH = STAGE0_ROOT / "fixed_probe_sensitivity_manifest.json"
SENSITIVITY_MANIFEST_SHA256 = (
    "148354c1ac1d4389ac8cb73240fb457dd193bee7c105e183b65f8d4db048a3ff"
)
STRUCTURAL_EFFECTS_SHA256 = (
    "171504e5dc75828b931ac91ee2c0deedf3583cf1f32de8f6902f18a897ff39e4"
)
CANDIDATE_SENSITIVITY_SHA256 = (
    "a7276db340e23334f9ee5c8d6287e0a0427c628b0bf15294170884d8162f50ca"
)
POSITION_SENSITIVITY_SHA256 = (
    "6faa79d1d1ff3a0999a878ad35aeb3e9ffdb22df058c308d727e95fc06ed0eb9"
)
PROTEIN_SENSITIVITY_SHA256 = (
    "91464052845e5c02250ca7dff5b27ef93c77279296a47736ffe8d6ba52f65e18"
)

EXPECTED_PROTEIN_COUNT = 8
EXPECTED_POSITION_COUNT = 1_790
EXPECTED_CANDIDATE_COUNT = 34_010
EXPECTED_STRUCTURAL_EFFECT_COUNT = 1_020_300
REPEAT_COUNT = 30
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
ANALYSIS_PROTOCOL_VERSION = "stage0_fixed_probe_local_decision_sensitivity_v1"
STRUCTURAL_ORIENTATION = "AFDB_versus_PDB"
EXPECTED_REPEAT_EFFECT_COUNT = EXPECTED_POSITION_COUNT * REPEAT_COUNT
EXPECTED_NULL_COUNT = EXPECTED_POSITION_COUNT * 2


class DecisionSensitivityError(ValueError):
    """Structured input, analysis, or immutable-release failure."""

    def __init__(self, code: str, message: str, *, outcome: str = "BLOCKED") -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


@dataclass(frozen=True)
class DecisionInputs:
    project_root: Path
    scoring: SensitivityInputs
    sensitivity_manifest_path: Path
    sensitivity_manifest_sha256: str
    sensitivity_manifest: dict[str, Any]
    structural_effects_path: Path
    structural_effects_sha256: str
    structural_effects: pd.DataFrame
    candidate_sensitivity_path: Path
    candidate_sensitivity_sha256: str
    candidate_sensitivity: pd.DataFrame
    position_sensitivity_path: Path
    position_sensitivity_sha256: str
    position_sensitivity: pd.DataFrame
    protein_sensitivity_path: Path
    protein_sensitivity_sha256: str
    protein_sensitivity: pd.DataFrame
    protein_order: tuple[str, ...]


@dataclass(frozen=True)
class DecisionSensitivityResult:
    project_root: Path
    input_provenance: dict[str, dict[str, Any]]
    repeat_effects: pd.DataFrame
    position_summary: pd.DataFrame
    technical_null: pd.DataFrame
    protein_summary: pd.DataFrame
    repeat_count: int
    protein_order: tuple[str, ...]


def _read_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionSensitivityError(code, f"Unable to read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DecisionSensitivityError(code, f"JSON is not an object: {path}")
    return payload


def _bound_path(project_root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise DecisionSensitivityError(
            "sensitivity_manifest_contract_mismatch",
            f"Missing manifest path for {label}",
        )
    path = (project_root / record["path"]).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise DecisionSensitivityError(
            "sensitivity_manifest_contract_mismatch",
            f"Manifest path escapes project root for {label}",
        ) from exc
    return path


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, code: str, label: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DecisionSensitivityError(code, f"{label} missing columns: {missing}")


def validate_fixed_probe_identity_table(
    fixed_probes: pd.DataFrame,
    *,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    expected_position_count: int = EXPECTED_POSITION_COUNT,
    mutations_per_position: int = 19,
) -> None:
    """Validate the frozen protein-position-mutant identity grid."""
    required = {"protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"}
    _require_columns(
        fixed_probes,
        required,
        code="fixed_probe_schema_mismatch",
        label="fixed probes",
    )
    if len(fixed_probes) != expected_candidate_count:
        raise DecisionSensitivityError(
            "fixed_probe_count_mismatch", "Fixed-probe candidate count differs"
        )
    if fixed_probes.duplicated(["protein_id", "position", "mut_aa"]).any():
        raise DecisionSensitivityError(
            "duplicate_position_amino_acid",
            "A protein-position mutant amino acid repeats",
        )
    groups = fixed_probes.groupby(["protein_id", "position"], sort=False)
    if len(groups) != expected_position_count:
        raise DecisionSensitivityError(
            "fixed_probe_position_count_mismatch", "Fixed-probe position count differs"
        )
    if not (groups.size() == mutations_per_position).all() or not (
        groups["mut_aa"].nunique() == mutations_per_position
    ).all():
        raise DecisionSensitivityError(
            "fixed_probe_mutation_grid_mismatch",
            "Each fixed position must contain the expected unique mutants",
        )


def _load_manifest_output(
    project_root: Path,
    outputs: dict[str, Any],
    *,
    label: str,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[Path, pd.DataFrame]:
    record = outputs.get(label)
    path = _bound_path(project_root, record, label=label)
    if not path.is_file():
        raise DecisionSensitivityError(
            f"{label}_missing", f"Frozen {label} is missing: {path}"
        )
    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        raise DecisionSensitivityError(
            f"{label}_unreadable", f"Frozen {label} is unreadable: {path}"
        ) from exc
    if record.get("sha256") != expected_sha256 or actual_sha256 != expected_sha256:
        raise DecisionSensitivityError(
            f"{label}_hash_mismatch", f"Frozen {label} SHA256 differs"
        )
    if int(record.get("rows", -1)) != expected_rows:
        raise DecisionSensitivityError(
            f"{label}_count_mismatch", f"Frozen {label} row count differs"
        )
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DecisionSensitivityError(
            f"{label}_unreadable", f"Frozen {label} cannot be parsed: {path}"
        ) from exc
    if len(frame) != expected_rows:
        raise DecisionSensitivityError(
            f"{label}_count_mismatch", f"Frozen {label} payload rows differ"
        )
    return path, frame


def load_frozen_decision_inputs(project_root: Path) -> DecisionInputs:
    """Load and validate every frozen input before local decision analysis."""
    root = project_root.resolve()
    try:
        scoring = load_frozen_sensitivity_inputs(root)
    except FixedProbeSensitivityError as exc:
        raise DecisionSensitivityError(exc.code, str(exc), outcome=exc.outcome) from exc
    validate_fixed_probe_identity_table(scoring.fixed_probes)

    manifest_path = (root / SENSITIVITY_MANIFEST_PATH).resolve()
    if not manifest_path.is_file():
        raise DecisionSensitivityError(
            "sensitivity_manifest_missing",
            f"Frozen sensitivity manifest is missing: {manifest_path}",
        )
    try:
        manifest_sha256 = sha256_file(manifest_path)
    except OSError as exc:
        raise DecisionSensitivityError(
            "sensitivity_manifest_unreadable",
            f"Frozen sensitivity manifest is unreadable: {manifest_path}",
        ) from exc
    if manifest_sha256 != SENSITIVITY_MANIFEST_SHA256:
        raise DecisionSensitivityError(
            "sensitivity_manifest_hash_mismatch",
            "Frozen sensitivity manifest SHA256 differs",
        )
    manifest = _read_json(
        manifest_path, code="sensitivity_manifest_contract_mismatch"
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version")
        != "stage0_fixed_probe_sensitivity_manifest_v1"
    ):
        raise DecisionSensitivityError(
            "sensitivity_manifest_contract_mismatch",
            "Sensitivity manifest is not the completed v1 release",
        )
    protein_order = tuple(str(value) for value in manifest.get("protein_order", []))
    if protein_order != scoring.protein_order:
        raise DecisionSensitivityError(
            "protein_membership_mismatch", "Sensitivity protein order differs"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise DecisionSensitivityError(
            "sensitivity_manifest_contract_mismatch", "Sensitivity outputs are absent"
        )
    structural_path, structural = _load_manifest_output(
        root,
        outputs,
        label="paired_effects",
        expected_sha256=STRUCTURAL_EFFECTS_SHA256,
        expected_rows=EXPECTED_STRUCTURAL_EFFECT_COUNT,
    )
    candidate_path, candidates = _load_manifest_output(
        root,
        outputs,
        label="candidate_summary",
        expected_sha256=CANDIDATE_SENSITIVITY_SHA256,
        expected_rows=EXPECTED_CANDIDATE_COUNT,
    )
    position_path, positions = _load_manifest_output(
        root,
        outputs,
        label="position_summary",
        expected_sha256=POSITION_SENSITIVITY_SHA256,
        expected_rows=EXPECTED_POSITION_COUNT,
    )
    protein_path, proteins = _load_manifest_output(
        root,
        outputs,
        label="protein_summary",
        expected_sha256=PROTEIN_SENSITIVITY_SHA256,
        expected_rows=EXPECTED_PROTEIN_COUNT,
    )
    _require_columns(
        positions,
        {
            "protein_id",
            "position",
            "position_mean_abs_interaction",
            "position_rms_interaction",
            "position_max_abs_interaction",
        },
        code="position_sensitivity_schema_mismatch",
        label="position sensitivity",
    )
    if positions.duplicated(["protein_id", "position"]).any():
        raise DecisionSensitivityError(
            "duplicate_position_sensitivity_key",
            "Position-sensitivity scientific keys repeat",
        )
    if tuple(proteins["protein_id"].astype(str)) != protein_order:
        raise DecisionSensitivityError(
            "protein_membership_mismatch", "Protein sensitivity order differs"
        )
    return DecisionInputs(
        project_root=root,
        scoring=scoring,
        sensitivity_manifest_path=manifest_path,
        sensitivity_manifest_sha256=SENSITIVITY_MANIFEST_SHA256,
        sensitivity_manifest=manifest,
        structural_effects_path=structural_path,
        structural_effects_sha256=STRUCTURAL_EFFECTS_SHA256,
        structural_effects=structural,
        candidate_sensitivity_path=candidate_path,
        candidate_sensitivity_sha256=CANDIDATE_SENSITIVITY_SHA256,
        candidate_sensitivity=candidates,
        position_sensitivity_path=position_path,
        position_sensitivity_sha256=POSITION_SENSITIVITY_SHA256,
        position_sensitivity=positions,
        protein_sensitivity_path=protein_path,
        protein_sensitivity_sha256=PROTEIN_SENSITIVITY_SHA256,
        protein_sensitivity=proteins,
        protein_order=protein_order,
    )


def build_local_amino_acid_scores(
    inputs: DecisionInputs | Any,
    *,
    repeat_count: int = REPEAT_COUNT,
) -> pd.DataFrame:
    """Construct the canonical WT-plus-19-mutant local score table once."""
    fixed = inputs.scoring.fixed_probes.copy()
    wt_scores = inputs.scoring.wt_scores.copy()
    raw_scores = inputs.scoring.raw_scores.copy()
    _require_columns(
        fixed,
        {"protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"},
        code="fixed_probe_schema_mismatch",
        label="fixed probes",
    )
    source_required = {
        "protein_id",
        "backbone_condition",
        "backbone_sha256",
        "repeat_index",
        "seed",
        "decoding_realization_sha256",
        "score_mean_logp_mask",
    }
    _require_columns(
        wt_scores,
        source_required,
        code="wt_score_schema_mismatch",
        label="WT scores",
    )
    _require_columns(
        raw_scores,
        source_required
        | {"sequence_hash", "position", "wt_aa", "mut_aa"},
        code="raw_score_schema_mismatch",
        label="raw scores",
    )
    if repeat_count <= 0:
        raise DecisionSensitivityError("invalid_repeat_count", "Repeat count must be positive")

    position_keys = ["protein_id", "position"]
    fixed_groups = fixed.groupby(position_keys, sort=False)
    if not (fixed_groups.size() == 19).all() or not (
        fixed_groups["mut_aa"].nunique() == 19
    ).all():
        raise DecisionSensitivityError(
            "local_amino_acid_grid_mismatch",
            "Every local position must provide 19 unique mutant amino acids",
            outcome="FAIL",
        )
    if not (fixed_groups["wt_aa"].nunique() == 1).all():
        raise DecisionSensitivityError(
            "local_wt_identity_mismatch",
            "WT amino acid varies within a local position",
            outcome="FAIL",
        )
    alphabet = set(STANDARD_AMINO_ACIDS)
    for _, group in fixed_groups:
        wt_aa = str(group["wt_aa"].iloc[0])
        mutants = set(group["mut_aa"].astype(str))
        if wt_aa not in alphabet or mutants != alphabet - {wt_aa}:
            raise DecisionSensitivityError(
                "local_amino_acid_grid_mismatch",
                "WT plus mutant amino acids do not form the standard alphabet",
                outcome="FAIL",
            )

    expected_repeats = set(range(repeat_count))
    for label, frame, keys in (
        ("WT", wt_scores, ["protein_id", "backbone_condition"]),
        (
            "mutant",
            raw_scores,
            ["protein_id", "sequence_hash", "backbone_condition"],
        ),
    ):
        if set(frame["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
            raise DecisionSensitivityError(
                "local_condition_grid_mismatch",
                f"{label} structural conditions differ",
                outcome="FAIL",
            )
        groups = frame.groupby(keys, sort=False)
        if not (groups.size() == repeat_count).all() or not groups[
            "repeat_index"
        ].agg(lambda values: {int(value) for value in values}).map(
            lambda values: values == expected_repeats
        ).all():
            raise DecisionSensitivityError(
                "local_amino_acid_grid_mismatch",
                f"{label} repeat grid is incomplete",
                outcome="FAIL",
            )

    fixed_identity = fixed[
        ["protein_id", "sequence_hash", "position", "wt_aa", "mut_aa"]
    ]
    raw = raw_scores.merge(
        fixed_identity,
        on=["protein_id", "sequence_hash"],
        how="inner",
        suffixes=("", "_frozen"),
        validate="many_to_one",
        sort=False,
    )
    if len(raw) != len(raw_scores):
        raise DecisionSensitivityError(
            "fixed_probe_join_mismatch",
            "Raw mutant scores do not match the frozen candidates",
            outcome="FAIL",
        )
    for field in ("position", "wt_aa", "mut_aa"):
        if not (raw[field] == raw[f"{field}_frozen"]).all():
            raise DecisionSensitivityError(
                "fixed_probe_join_mismatch",
                f"Raw mutant {field} differs from the frozen candidate",
                outcome="FAIL",
            )
    mutant_rows = pd.DataFrame(
        {
            "protein_id": raw["protein_id"].astype(str),
            "position": raw["position"].astype(int),
            "wt_aa": raw["wt_aa"].astype(str),
            "aa": raw["mut_aa"].astype(str),
            "is_wt": False,
            "sequence_hash": raw["sequence_hash"].astype(str),
            "backbone_condition": raw["backbone_condition"].astype(str),
            "backbone_sha256": raw["backbone_sha256"].astype(str),
            "repeat_index": raw["repeat_index"].astype(int),
            "seed": raw["seed"].astype(int),
            "decoding_realization_sha256": raw[
                "decoding_realization_sha256"
            ].astype(str),
            "score_mean_logp_mask": raw["score_mean_logp_mask"].astype(float),
        }
    )

    positions = fixed.groupby(position_keys, sort=False, as_index=False).agg(
        wt_aa=("wt_aa", "first")
    )
    wt = positions.merge(
        wt_scores,
        on="protein_id",
        how="inner",
        validate="many_to_many",
        sort=False,
    )
    expected_wt_rows = len(positions) * 2 * repeat_count
    if len(wt) != expected_wt_rows:
        raise DecisionSensitivityError(
            "local_amino_acid_grid_mismatch",
            "WT analytical expansion is incomplete",
            outcome="FAIL",
        )
    wt_rows = pd.DataFrame(
        {
            "protein_id": wt["protein_id"].astype(str),
            "position": wt["position"].astype(int),
            "wt_aa": wt["wt_aa"].astype(str),
            "aa": wt["wt_aa"].astype(str),
            "is_wt": True,
            "sequence_hash": pd.Series([pd.NA] * len(wt), dtype="string"),
            "backbone_condition": wt["backbone_condition"].astype(str),
            "backbone_sha256": wt["backbone_sha256"].astype(str),
            "repeat_index": wt["repeat_index"].astype(int),
            "seed": wt["seed"].astype(int),
            "decoding_realization_sha256": wt[
                "decoding_realization_sha256"
            ].astype(str),
            "score_mean_logp_mask": wt["score_mean_logp_mask"].astype(float),
        }
    )
    local = pd.concat([wt_rows, mutant_rows], ignore_index=True)
    local_keys = [
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
    ]
    groups = local.groupby(local_keys, sort=False)
    alphabet_string = "".join(sorted(STANDARD_AMINO_ACIDS))
    amino_sets = groups["aa"].agg(lambda values: "".join(sorted(values.astype(str))))
    if (
        not (groups.size() == 20).all()
        or not (groups["aa"].nunique() == 20).all()
        or not (groups["is_wt"].sum() == 1).all()
        or not (amino_sets == alphabet_string).all()
    ):
        raise DecisionSensitivityError(
            "local_amino_acid_grid_mismatch",
            "A local backbone/repeat set is not one WT plus 19 unique mutants",
            outcome="FAIL",
        )
    protein_rank = {protein: index for index, protein in enumerate(inputs.protein_order)}
    if not set(local["protein_id"]).issubset(protein_rank):
        raise DecisionSensitivityError(
            "protein_membership_mismatch", "Local scores include an unknown protein", outcome="FAIL"
        )
    local = local.assign(
        _protein_order=local["protein_id"].map(protein_rank).astype(int),
        _condition_order=local["backbone_condition"].map({"PDB": 0, "AFDB": 1}),
        _aa_order=local["aa"].map(
            {aa: index for index, aa in enumerate(STANDARD_AMINO_ACIDS)}
        ),
    ).sort_values(
        [
            "_protein_order",
            "position",
            "_condition_order",
            "repeat_index",
            "_aa_order",
        ],
        kind="stable",
    )
    return local.drop(
        columns=["_protein_order", "_condition_order", "_aa_order"]
    ).reset_index(drop=True)


def rank_local_amino_acid_scores(
    local_scores: pd.DataFrame,
    *,
    tie_atol: float = 1.0e-6,
    tie_rtol: float = 1.0e-6,
    protein_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Rank each 20-AA landscape once using score then canonical AA order."""
    required = {
        "protein_id",
        "position",
        "wt_aa",
        "aa",
        "is_wt",
        "backbone_condition",
        "repeat_index",
        "seed",
        "decoding_realization_sha256",
        "score_mean_logp_mask",
    }
    _require_columns(
        local_scores,
        required,
        code="local_score_schema_mismatch",
        label="local scores",
    )
    group_keys = [
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
    ]
    if local_scores.duplicated(group_keys + ["aa"]).any():
        raise DecisionSensitivityError(
            "duplicate_local_amino_acid",
            "A local amino acid repeats within a landscape",
            outcome="FAIL",
        )
    groups = local_scores.groupby(group_keys, sort=False)
    if not (groups.size() == 20).all() or not (groups["aa"].nunique() == 20).all():
        raise DecisionSensitivityError(
            "local_amino_acid_grid_mismatch",
            "Each ranked landscape must contain exactly 20 amino acids",
            outcome="FAIL",
        )
    scores = local_scores["score_mean_logp_mask"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise DecisionSensitivityError(
            "nonfinite_local_score", "Local scores contain NaN or infinity", outcome="FAIL"
        )
    amino_order = {aa: index for index, aa in enumerate(STANDARD_AMINO_ACIDS)}
    if set(local_scores["aa"].astype(str)) != set(STANDARD_AMINO_ACIDS):
        raise DecisionSensitivityError(
            "local_amino_acid_grid_mismatch",
            "Local scores contain nonstandard amino acids",
            outcome="FAIL",
        )
    order = protein_order or tuple(sorted(local_scores["protein_id"].astype(str).unique()))
    protein_rank = {protein: index for index, protein in enumerate(order)}
    if not set(local_scores["protein_id"].astype(str)).issubset(protein_rank):
        raise DecisionSensitivityError(
            "protein_membership_mismatch", "Ranked scores include unknown proteins", outcome="FAIL"
        )
    ranked = local_scores.copy().assign(
        _protein_order=local_scores["protein_id"].astype(str).map(protein_rank),
        _condition_order=local_scores["backbone_condition"].map({"PDB": 0, "AFDB": 1}),
        _aa_order=local_scores["aa"].astype(str).map(amino_order),
    )
    if ranked[["_protein_order", "_condition_order", "_aa_order"]].isna().any().any():
        raise DecisionSensitivityError(
            "local_score_provenance_mismatch",
            "Protein, condition, or amino-acid ordering is unresolved",
            outcome="FAIL",
        )
    ranked = ranked.sort_values(
        [
            "_protein_order",
            "position",
            "_condition_order",
            "repeat_index",
            "score_mean_logp_mask",
            "_aa_order",
        ],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.groupby(group_keys, sort=False).cumcount() + 1
    ranked["top_score"] = ranked.groupby(group_keys, sort=False)[
        "score_mean_logp_mask"
    ].transform("max")
    ranked["exact_top_tie_member"] = (
        ranked["score_mean_logp_mask"] == ranked["top_score"]
    )
    ranked["practical_top_tie_member"] = (
        (ranked["score_mean_logp_mask"] - ranked["top_score"]).abs()
        <= tie_atol + tie_rtol * ranked["top_score"].abs()
    )
    ranked["exact_top_tie_count"] = ranked.groupby(group_keys, sort=False)[
        "exact_top_tie_member"
    ].transform("sum").astype(int)
    ranked["practical_top_tie_count"] = ranked.groupby(group_keys, sort=False)[
        "practical_top_tie_member"
    ].transform("sum").astype(int)
    ranked["exact_top_tie"] = ranked["exact_top_tie_count"] > 1
    ranked["practical_top_tie"] = ranked["practical_top_tie_count"] > 1
    return ranked.drop(
        columns=["_protein_order", "_condition_order", "_aa_order"]
    )


def _nonnegative_regret(value: float, *, label: str) -> float:
    return float(
        _checked_nonnegative_regrets(
            np.asarray([value], dtype=np.float64), label=label
        )[0]
    )


def _checked_nonnegative_regrets(
    values: np.ndarray,
    *,
    label: str,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Fail on material negative regret and clamp floating roundoff only."""
    regrets = np.asarray(values, dtype=np.float64)
    if not np.isfinite(regrets).all():
        raise DecisionSensitivityError(
            "nonfinite_local_regret", f"{label} contains NaN or infinity", outcome="FAIL"
        )
    minimum = float(regrets.min()) if regrets.size else 0.0
    if minimum < -tolerance:
        raise DecisionSensitivityError(
            "negative_local_regret",
            f"{label} is below the floating tolerance: {minimum}",
            outcome="FAIL",
        )
    return np.maximum(regrets, 0.0)


def build_repeat_decision_effects(ranked_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare paired PDB/AFDB local rankings per position and realization."""
    required = {
        "protein_id",
        "position",
        "wt_aa",
        "aa",
        "backbone_condition",
        "repeat_index",
        "seed",
        "decoding_realization_sha256",
        "score_mean_logp_mask",
        "rank",
        "exact_top_tie",
        "exact_top_tie_count",
        "practical_top_tie",
        "practical_top_tie_count",
    }
    _require_columns(
        ranked_scores,
        required,
        code="ranked_score_schema_mismatch",
        label="ranked local scores",
    )
    group_keys = ["protein_id", "position", "repeat_index"]
    coarse_keys = group_keys + ["aa"]
    fingerprint = "decoding_realization_sha256"
    join_keys = group_keys + [fingerprint, "aa"]
    conditions: dict[str, pd.DataFrame] = {}
    for condition in ("PDB", "AFDB"):
        frame = ranked_scores.loc[
            ranked_scores["backbone_condition"].astype(str) == condition
        ].copy()
        groups = frame.groupby(group_keys, sort=False)
        if (
            frame.duplicated(coarse_keys).any()
            or not (groups.size() == 20).all()
            or not (groups["aa"].nunique() == 20).all()
            or not (groups[fingerprint].nunique() == 1).all()
        ):
            raise DecisionSensitivityError(
                "paired_decision_grid_mismatch",
                f"{condition} local rankings are incomplete",
                outcome="FAIL",
            )
        conditions[condition] = frame
    pdb = conditions["PDB"]
    afdb = conditions["AFDB"]
    coarse = pdb[coarse_keys + [fingerprint]].merge(
        afdb[coarse_keys + [fingerprint]],
        on=coarse_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_pdb", "_afdb"),
        indicator=True,
        sort=False,
    )
    if not (coarse["_merge"] == "both").all():
        raise DecisionSensitivityError(
            "paired_decision_grid_mismatch",
            "PDB/AFDB local amino-acid identities differ",
            outcome="FAIL",
        )
    if not (coarse[f"{fingerprint}_pdb"] == coarse[f"{fingerprint}_afdb"]).all():
        raise DecisionSensitivityError(
            "decision_realization_fingerprint_mismatch",
            "PDB/AFDB local decisions use different realizations",
            outcome="FAIL",
        )
    paired = pdb.merge(
        afdb,
        on=join_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_pdb", "_afdb"),
        sort=False,
    )
    if len(paired) != len(pdb) or len(paired) != len(afdb):
        raise DecisionSensitivityError(
            "paired_decision_grid_mismatch",
            "PDB/AFDB exact local decision pairing is incomplete",
            outcome="FAIL",
        )
    if not (paired["wt_aa_pdb"].astype(str) == paired["wt_aa_afdb"].astype(str)).all():
        raise DecisionSensitivityError(
            "local_wt_identity_mismatch",
            "WT amino acid differs across conditions",
            outcome="FAIL",
        )
    if not (paired["seed_pdb"].astype(int) == paired["seed_afdb"].astype(int)).all():
        raise DecisionSensitivityError(
            "decision_realization_fingerprint_mismatch",
            "PDB/AFDB local decisions use different seed provenance",
            outcome="FAIL",
        )

    grouped = paired.groupby(group_keys, sort=False)
    if not (grouped.size() == 20).all():
        raise DecisionSensitivityError(
            "paired_decision_grid_mismatch",
            "Each paired decision must contain exactly 20 amino acids",
            outcome="FAIL",
        )
    paired["_rank_displacement"] = (
        paired["rank_afdb"].astype(float) - paired["rank_pdb"].astype(float)
    ).abs() / 19.0
    paired["_top3_both"] = (paired["rank_pdb"] <= 3) & (
        paired["rank_afdb"] <= 3
    )
    paired["_top5_both"] = (paired["rank_pdb"] <= 5) & (
        paired["rank_afdb"] <= 5
    )
    paired["_rank_difference_squared"] = (
        paired["rank_afdb"].astype(float) - paired["rank_pdb"].astype(float)
    ) ** 2
    summary = grouped.agg(
        seed=("seed_pdb", "first"),
        decoding_realization_sha256=(fingerprint, "first"),
        wt_aa=("wt_aa_pdb", "first"),
        _top3_overlap=("_top3_both", "sum"),
        _top5_overlap=("_top5_both", "sum"),
        _rank_difference_squared_sum=("_rank_difference_squared", "sum"),
        mean_normalized_rank_displacement=("_rank_displacement", "mean"),
        max_normalized_rank_displacement=("_rank_displacement", "max"),
        _afdb_max=("score_mean_logp_mask_afdb", "max"),
        _pdb_max=("score_mean_logp_mask_pdb", "max"),
        pdb_exact_top_tie=("exact_top_tie_pdb", "first"),
        pdb_exact_top_tie_count=("exact_top_tie_count_pdb", "first"),
        pdb_practical_top_tie=("practical_top_tie_pdb", "first"),
        pdb_practical_top_tie_count=("practical_top_tie_count_pdb", "first"),
        afdb_exact_top_tie=("exact_top_tie_afdb", "first"),
        afdb_exact_top_tie_count=("exact_top_tie_count_afdb", "first"),
        afdb_practical_top_tie=("practical_top_tie_afdb", "first"),
        afdb_practical_top_tie_count=("practical_top_tie_count_afdb", "first"),
    )
    pdb_top = paired.loc[paired["rank_pdb"] == 1, group_keys + ["aa", "score_mean_logp_mask_afdb"]].set_index(group_keys)
    afdb_top = paired.loc[paired["rank_afdb"] == 1, group_keys + ["aa", "score_mean_logp_mask_pdb"]].set_index(group_keys)
    if len(pdb_top) != len(summary) or len(afdb_top) != len(summary):
        raise DecisionSensitivityError(
            "paired_decision_grid_mismatch",
            "A paired decision lacks a unique strict Top-1 amino acid",
            outcome="FAIL",
        )
    summary["top1_pdb"] = pdb_top["aa"]
    summary["top1_afdb"] = afdb_top["aa"]
    summary["top1_disagreement"] = summary["top1_pdb"] != summary["top1_afdb"]
    summary["top3_instability"] = 1.0 - summary["_top3_overlap"] / 3.0
    summary["top5_instability"] = 1.0 - summary["_top5_overlap"] / 5.0
    summary["spearman_rank"] = 1.0 - (
        6.0 * summary["_rank_difference_squared_sum"] / (20.0 * (20.0**2 - 1.0))
    )
    raw_pdb_to_afdb = (
        summary["_afdb_max"] - pdb_top["score_mean_logp_mask_afdb"]
    )
    raw_afdb_to_pdb = (
        summary["_pdb_max"] - afdb_top["score_mean_logp_mask_pdb"]
    )
    summary["regret_pdb_to_afdb"] = _checked_nonnegative_regrets(
        raw_pdb_to_afdb.to_numpy(dtype=np.float64),
        label="PDB-to-AFDB regret",
    )
    summary["regret_afdb_to_pdb"] = _checked_nonnegative_regrets(
        raw_afdb_to_pdb.to_numpy(dtype=np.float64),
        label="AFDB-to-PDB regret",
    )
    summary["symmetric_regret"] = (
        summary["regret_pdb_to_afdb"] + summary["regret_afdb_to_pdb"]
    ) / 2.0
    summary["wt_top1_pdb"] = summary["top1_pdb"] == summary["wt_aa"]
    summary["wt_top1_afdb"] = summary["top1_afdb"] == summary["wt_aa"]
    summary["wt_top1_changed"] = summary["wt_top1_pdb"] != summary["wt_top1_afdb"]
    output_columns = [
        "seed",
        "decoding_realization_sha256",
        "wt_aa",
        "top1_pdb",
        "top1_afdb",
        "top1_disagreement",
        "top3_instability",
        "top5_instability",
        "spearman_rank",
        "mean_normalized_rank_displacement",
        "max_normalized_rank_displacement",
        "regret_pdb_to_afdb",
        "regret_afdb_to_pdb",
        "symmetric_regret",
        "wt_top1_pdb",
        "wt_top1_afdb",
        "wt_top1_changed",
        "pdb_exact_top_tie",
        "pdb_exact_top_tie_count",
        "pdb_practical_top_tie",
        "pdb_practical_top_tie_count",
        "afdb_exact_top_tie",
        "afdb_exact_top_tie_count",
        "afdb_practical_top_tie",
        "afdb_practical_top_tie_count",
    ]
    result = summary[output_columns].reset_index()
    regrets = result[
        ["regret_pdb_to_afdb", "regret_afdb_to_pdb", "symmetric_regret"]
    ].to_numpy(dtype=np.float64)
    if (regrets < -1.0e-12).any():
        raise DecisionSensitivityError(
            "negative_local_regret",
            "Paired local decision regret is negative",
            outcome="FAIL",
        )
    return result


def summarize_same_state_decision_null(
    ranked_scores: pd.DataFrame,
    *,
    repeat_count: int = REPEAT_COUNT,
) -> pd.DataFrame:
    """Summarize all within-backbone repeat pairs without emitting pair rows."""
    required = {
        "protein_id",
        "position",
        "backbone_condition",
        "repeat_index",
        "aa",
        "score_mean_logp_mask",
        "rank",
        "exact_top_tie",
        "practical_top_tie",
    }
    _require_columns(
        ranked_scores,
        required,
        code="ranked_score_schema_mismatch",
        label="ranked local scores",
    )
    group_keys = ["protein_id", "position", "backbone_condition"]
    expected_repeats = set(range(repeat_count))
    amino_order = {aa: index for index, aa in enumerate(STANDARD_AMINO_ACIDS)}
    ordered = ranked_scores.assign(
        _aa_order=ranked_scores["aa"].astype(str).map(amino_order)
    )
    if ordered["_aa_order"].isna().any() or ordered.duplicated(
        group_keys + ["repeat_index", "aa"]
    ).any():
        raise DecisionSensitivityError(
            "same_state_repeat_grid_mismatch",
            "Same-state rankings contain invalid or duplicated amino acids",
            outcome="FAIL",
        )
    grouped = ordered.groupby(group_keys, sort=False)
    if not (grouped.size() == repeat_count * 20).all() or not grouped[
        "repeat_index"
    ].agg(lambda values: {int(value) for value in values}).map(
        lambda values: values == expected_repeats
    ).all():
        raise DecisionSensitivityError(
            "same_state_repeat_grid_mismatch",
            "Same-state local ranking repeat grid is incomplete",
            outcome="FAIL",
        )
    group_index = pd.MultiIndex.from_frame(ordered[group_keys])
    group_codes, unique_groups = pd.factorize(group_index, sort=False)
    ordered = ordered.assign(_group_code=group_codes).sort_values(
        ["_group_code", "repeat_index", "_aa_order"], kind="stable"
    )
    group_count = len(unique_groups)
    expected_rows = group_count * repeat_count * 20
    if len(ordered) != expected_rows:
        raise DecisionSensitivityError(
            "same_state_repeat_grid_mismatch",
            "Same-state ranking rows cannot form a dense array",
            outcome="FAIL",
        )
    ranks = ordered["rank"].to_numpy(dtype=np.float64).reshape(
        group_count, repeat_count, 20
    )
    scores = ordered["score_mean_logp_mask"].to_numpy(dtype=np.float64).reshape(
        group_count, repeat_count, 20
    )
    exact_ties = ordered["exact_top_tie"].to_numpy(dtype=bool).reshape(
        group_count, repeat_count, 20
    )[:, :, 0]
    practical_ties = ordered["practical_top_tie"].to_numpy(dtype=bool).reshape(
        group_count, repeat_count, 20
    )[:, :, 0]
    if not np.isfinite(ranks).all() or not np.isfinite(scores).all():
        raise DecisionSensitivityError(
            "same_state_repeat_grid_mismatch",
            "Same-state rankings contain nonfinite values",
            outcome="FAIL",
        )
    top1 = np.argmin(ranks, axis=2)
    top1_counts = np.stack(
        [(top1 == amino_index).sum(axis=1) for amino_index in range(20)], axis=1
    )
    modal_index = np.argmax(top1_counts, axis=1)
    maximum_count = top1_counts.max(axis=1)
    distinct_top1_count = (top1_counts > 0).sum(axis=1)
    left, right = np.triu_indices(repeat_count, k=1)
    expected_pair_count = repeat_count * (repeat_count - 1) // 2
    if len(left) != expected_pair_count:
        raise DecisionSensitivityError(
            "same_state_pair_count_mismatch",
            "Same-state repeat-pair count differs",
            outcome="FAIL",
        )
    pairwise_top1 = (top1[:, left] != top1[:, right]).mean(axis=1)
    top3 = ranks <= 3
    top5 = ranks <= 5
    pairwise_top3 = (
        1.0
        - np.logical_and(top3[:, left, :], top3[:, right, :]).sum(axis=2) / 3.0
    ).mean(axis=1)
    pairwise_top5 = (
        1.0
        - np.logical_and(top5[:, left, :], top5[:, right, :]).sum(axis=2) / 5.0
    ).mean(axis=1)
    pairwise_rank = (
        np.abs(ranks[:, left, :] - ranks[:, right, :]) / 19.0
    ).mean(axis=2).mean(axis=1)
    left_top1 = top1[:, left]
    right_top1 = top1[:, right]
    left_scores = scores[:, left, :]
    right_scores = scores[:, right, :]
    right_at_left = np.take_along_axis(
        right_scores, left_top1[:, :, None], axis=2
    ).squeeze(axis=2)
    left_at_right = np.take_along_axis(
        left_scores, right_top1[:, :, None], axis=2
    ).squeeze(axis=2)
    left_to_right = _checked_nonnegative_regrets(
        right_scores.max(axis=2) - right_at_left,
        label="same-state left-to-right regret",
    )
    right_to_left = _checked_nonnegative_regrets(
        left_scores.max(axis=2) - left_at_right,
        label="same-state right-to-left regret",
    )
    symmetric_regrets = (left_to_right + right_to_left) / 2.0

    identity = ordered.drop_duplicates("_group_code", keep="first").sort_values(
        "_group_code", kind="stable"
    )
    alphabet = np.asarray(list(STANDARD_AMINO_ACIDS), dtype=object)
    return pd.DataFrame(
        {
            "protein_id": identity["protein_id"].astype(str).to_numpy(),
            "position": identity["position"].astype(int).to_numpy(),
            "backbone_condition": identity["backbone_condition"]
            .astype(str)
            .to_numpy(),
            "n_repeats": repeat_count,
            "repeat_pair_count": expected_pair_count,
            "modal_top1_aa": alphabet[modal_index],
            "top1_modal_concentration": maximum_count / repeat_count,
            "distinct_top1_count": distinct_top1_count,
            "pairwise_top1_disagreement_rate": pairwise_top1,
            "pairwise_top3_instability_mean": pairwise_top3,
            "pairwise_top5_instability_mean": pairwise_top5,
            "pairwise_mean_normalized_rank_displacement_mean": pairwise_rank,
            "pairwise_symmetric_regret_mean": symmetric_regrets.mean(axis=1),
            "pairwise_symmetric_regret_median": np.median(
                symmetric_regrets, axis=1
            ),
            "pairwise_symmetric_regret_q75": np.quantile(
                symmetric_regrets, 0.75, axis=1, method="linear"
            ),
            "pairwise_symmetric_regret_q95": np.quantile(
                symmetric_regrets, 0.95, axis=1, method="linear"
            ),
            "pairwise_symmetric_regret_max": symmetric_regrets.max(axis=1),
            "exact_top_tie_fraction": exact_ties.mean(axis=1),
            "practical_top_tie_fraction": practical_ties.mean(axis=1),
        }
    )


def _canonical_mode(values: pd.Series) -> str:
    counts = Counter(values.astype(str))
    if not counts:
        raise DecisionSensitivityError(
            "empty_decision_group", "Cannot compute a modal amino acid", outcome="FAIL"
        )
    maximum_count = max(counts.values())
    return min(
        (aa for aa, count in counts.items() if count == maximum_count),
        key={aa: index for index, aa in enumerate(STANDARD_AMINO_ACIDS)}.__getitem__,
    )


def summarize_position_decisions(
    repeat_effects: pd.DataFrame,
    decision_null: pd.DataFrame,
    continuous_position_sensitivity: pd.DataFrame,
    *,
    repeat_count: int = REPEAT_COUNT,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize repeat decisions once and join continuous and null evidence."""
    effect_required = {
        "protein_id",
        "position",
        "repeat_index",
        "wt_aa",
        "top1_pdb",
        "top1_afdb",
        "top1_disagreement",
        "top3_instability",
        "top5_instability",
        "spearman_rank",
        "mean_normalized_rank_displacement",
        "max_normalized_rank_displacement",
        "regret_pdb_to_afdb",
        "regret_afdb_to_pdb",
        "symmetric_regret",
        "wt_top1_changed",
        "pdb_exact_top_tie",
        "pdb_practical_top_tie",
        "afdb_exact_top_tie",
        "afdb_practical_top_tie",
    }
    _require_columns(
        repeat_effects,
        effect_required,
        code="repeat_decision_schema_mismatch",
        label="repeat decision effects",
    )
    null_required = {
        "protein_id",
        "position",
        "backbone_condition",
        "n_repeats",
        "repeat_pair_count",
        "modal_top1_aa",
        "top1_modal_concentration",
        "distinct_top1_count",
        "pairwise_top1_disagreement_rate",
        "pairwise_top3_instability_mean",
        "pairwise_top5_instability_mean",
        "pairwise_mean_normalized_rank_displacement_mean",
        "pairwise_symmetric_regret_mean",
        "pairwise_symmetric_regret_median",
        "pairwise_symmetric_regret_q75",
        "pairwise_symmetric_regret_q95",
        "pairwise_symmetric_regret_max",
        "exact_top_tie_fraction",
        "practical_top_tie_fraction",
    }
    _require_columns(
        decision_null,
        null_required,
        code="decision_null_schema_mismatch",
        label="same-state decision null",
    )
    continuous_value_columns = [
        "position_mean_abs_interaction",
        "position_rms_interaction",
        "position_max_abs_interaction",
    ]
    continuous_required = {"protein_id", "position", *continuous_value_columns}
    _require_columns(
        continuous_position_sensitivity,
        continuous_required,
        code="position_sensitivity_schema_mismatch",
        label="continuous position sensitivity",
    )
    key = ["protein_id", "position"]
    if repeat_effects.duplicated(key + ["repeat_index"]).any():
        raise DecisionSensitivityError(
            "duplicate_repeat_decision_key",
            "A position/repeat decision key repeats",
            outcome="FAIL",
        )
    groups = repeat_effects.groupby(key, sort=False)
    expected_repeats = set(range(repeat_count))
    if not (groups.size() == repeat_count).all() or not groups[
        "repeat_index"
    ].agg(lambda values: {int(value) for value in values}).map(
        lambda values: values == expected_repeats
    ).all():
        raise DecisionSensitivityError(
            "position_repeat_grid_mismatch",
            "Position decision repeat grid is incomplete",
            outcome="FAIL",
        )
    rows: list[dict[str, Any]] = []
    for (protein_id, position), group in groups:
        wt_values = tuple(pd.unique(group["wt_aa"].astype(str)))
        if len(wt_values) != 1:
            raise DecisionSensitivityError(
                "local_wt_identity_mismatch",
                "WT amino acid varies across position repeats",
                outcome="FAIL",
            )
        regrets = group["symmetric_regret"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "protein_id": str(protein_id),
                "position": int(position),
                "wt_aa": wt_values[0],
                "n_repeats": repeat_count,
                "top1_disagreement_fraction": float(
                    group["top1_disagreement"].mean()
                ),
                "top3_instability_mean": float(group["top3_instability"].mean()),
                "top5_instability_mean": float(group["top5_instability"].mean()),
                "spearman_rank_mean": float(group["spearman_rank"].mean()),
                "mean_normalized_rank_displacement_mean": float(
                    group["mean_normalized_rank_displacement"].mean()
                ),
                "max_normalized_rank_displacement_mean": float(
                    group["max_normalized_rank_displacement"].mean()
                ),
                "max_normalized_rank_displacement_max": float(
                    group["max_normalized_rank_displacement"].max()
                ),
                "regret_pdb_to_afdb_mean": float(
                    group["regret_pdb_to_afdb"].mean()
                ),
                "regret_afdb_to_pdb_mean": float(
                    group["regret_afdb_to_pdb"].mean()
                ),
                "symmetric_regret_mean": float(regrets.mean()),
                "symmetric_regret_median": float(np.median(regrets)),
                "symmetric_regret_q75": float(
                    np.quantile(regrets, 0.75, method="linear")
                ),
                "symmetric_regret_q95": float(
                    np.quantile(regrets, 0.95, method="linear")
                ),
                "symmetric_regret_max": float(regrets.max()),
                "wt_top1_changed_fraction": float(group["wt_top1_changed"].mean()),
                "modal_pdb_top1_aa": _canonical_mode(group["top1_pdb"]),
                "modal_afdb_top1_aa": _canonical_mode(group["top1_afdb"]),
                "unique_pdb_top1_count": int(group["top1_pdb"].nunique()),
                "unique_afdb_top1_count": int(group["top1_afdb"].nunique()),
                "pdb_exact_top_tie_fraction": float(
                    group["pdb_exact_top_tie"].mean()
                ),
                "pdb_practical_top_tie_fraction": float(
                    group["pdb_practical_top_tie"].mean()
                ),
                "afdb_exact_top_tie_fraction": float(
                    group["afdb_exact_top_tie"].mean()
                ),
                "afdb_practical_top_tie_fraction": float(
                    group["afdb_practical_top_tie"].mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)

    if decision_null.duplicated(key + ["backbone_condition"]).any():
        raise DecisionSensitivityError(
            "duplicate_position_null_key",
            "A same-state position/condition key repeats",
            outcome="FAIL",
        )
    if set(decision_null["backbone_condition"].astype(str)) != {"PDB", "AFDB"}:
        raise DecisionSensitivityError(
            "position_null_join_mismatch",
            "Same-state null conditions differ from PDB/AFDB",
            outcome="FAIL",
        )
    null_value_columns = [column for column in decision_null.columns if column not in key + ["backbone_condition"]]
    merged = summary
    for condition, prefix in (("PDB", "pdb"), ("AFDB", "afdb")):
        condition_frame = decision_null.loc[
            decision_null["backbone_condition"].astype(str) == condition,
            key + null_value_columns,
        ].rename(
            columns={column: f"{prefix}_same_state_{column}" for column in null_value_columns}
        )
        merged = merged.merge(
            condition_frame,
            on=key,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    required_joined = [
        "pdb_same_state_pairwise_top1_disagreement_rate",
        "afdb_same_state_pairwise_top1_disagreement_rate",
    ]
    if len(merged) != len(summary) or merged[required_joined].isna().any().any():
        raise DecisionSensitivityError(
            "position_null_join_mismatch",
            "Each decision position requires PDB and AFDB null summaries",
            outcome="FAIL",
        )

    continuous = continuous_position_sensitivity[
        key + continuous_value_columns
    ].copy()
    if continuous.duplicated(key).any():
        raise DecisionSensitivityError(
            "duplicate_position_sensitivity_key",
            "A continuous position-sensitivity key repeats",
            outcome="FAIL",
        )
    merged = merged.merge(
        continuous,
        on=key,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if (
        len(merged) != len(summary)
        or merged[continuous_value_columns].isna().any().any()
    ):
        raise DecisionSensitivityError(
            "continuous_position_join_mismatch",
            "Every decision position requires continuous sensitivity evidence",
            outcome="FAIL",
        )
    mean_null_top1 = (
        merged["pdb_same_state_pairwise_top1_disagreement_rate"]
        + merged["afdb_same_state_pairwise_top1_disagreement_rate"]
    ) / 2.0
    mean_null_rank = (
        merged[
            "pdb_same_state_pairwise_mean_normalized_rank_displacement_mean"
        ]
        + merged[
            "afdb_same_state_pairwise_mean_normalized_rank_displacement_mean"
        ]
    ) / 2.0
    mean_null_regret = (
        merged["pdb_same_state_pairwise_symmetric_regret_mean"]
        + merged["afdb_same_state_pairwise_symmetric_regret_mean"]
    ) / 2.0
    merged["structural_minus_mean_same_state_top1_disagreement"] = (
        merged["top1_disagreement_fraction"] - mean_null_top1
    )
    merged["structural_minus_mean_same_state_rank_displacement"] = (
        merged["mean_normalized_rank_displacement_mean"] - mean_null_rank
    )
    merged["structural_minus_mean_same_state_symmetric_regret"] = (
        merged["symmetric_regret_mean"] - mean_null_regret
    )
    order = {protein: index for index, protein in enumerate(protein_order)}
    if set(merged["protein_id"].astype(str)) != set(order):
        raise DecisionSensitivityError(
            "protein_membership_mismatch",
            "Position summaries differ from the frozen protein membership",
            outcome="FAIL",
        )
    return (
        merged.assign(_protein_order=merged["protein_id"].map(order))
        .sort_values(["_protein_order", "position"], kind="stable")
        .drop(columns="_protein_order")
        .reset_index(drop=True)
    )


def summarize_protein_decisions(
    position_summary: pd.DataFrame,
    *,
    protein_order: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate canonical position summaries to the eight biological units."""
    required = {
        "protein_id",
        "position",
        "top1_disagreement_fraction",
        "top3_instability_mean",
        "top5_instability_mean",
        "spearman_rank_mean",
        "mean_normalized_rank_displacement_mean",
        "regret_pdb_to_afdb_mean",
        "regret_afdb_to_pdb_mean",
        "symmetric_regret_mean",
        "wt_top1_changed_fraction",
        "pdb_same_state_pairwise_top1_disagreement_rate",
        "afdb_same_state_pairwise_top1_disagreement_rate",
        "pdb_same_state_pairwise_mean_normalized_rank_displacement_mean",
        "afdb_same_state_pairwise_mean_normalized_rank_displacement_mean",
        "pdb_same_state_pairwise_symmetric_regret_mean",
        "afdb_same_state_pairwise_symmetric_regret_mean",
        "structural_minus_mean_same_state_top1_disagreement",
        "structural_minus_mean_same_state_rank_displacement",
        "structural_minus_mean_same_state_symmetric_regret",
        "pdb_exact_top_tie_fraction",
        "pdb_practical_top_tie_fraction",
        "afdb_exact_top_tie_fraction",
        "afdb_practical_top_tie_fraction",
    }
    _require_columns(
        position_summary,
        required,
        code="position_decision_schema_mismatch",
        label="position decision sensitivity",
    )
    if position_summary.duplicated(["protein_id", "position"]).any():
        raise DecisionSensitivityError(
            "duplicate_position_decision_key",
            "A position decision key repeats",
            outcome="FAIL",
        )
    groups = {
        str(protein): group
        for protein, group in position_summary.groupby("protein_id", sort=False)
    }
    if set(groups) != set(protein_order):
        raise DecisionSensitivityError(
            "protein_membership_mismatch",
            "Protein summary inputs differ from frozen membership",
            outcome="FAIL",
        )
    rows: list[dict[str, Any]] = []
    for protein_id in protein_order:
        group = groups[protein_id]
        regrets = group["symmetric_regret_mean"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "protein_id": protein_id,
                "position_count": len(group),
                "top1_disagreement_fraction_mean": float(
                    group["top1_disagreement_fraction"].mean()
                ),
                "top1_disagreement_fraction_median": float(
                    group["top1_disagreement_fraction"].median()
                ),
                "positions_with_any_top1_disagreement_fraction": float(
                    (group["top1_disagreement_fraction"] > 0.0).mean()
                ),
                "top3_instability_mean": float(group["top3_instability_mean"].mean()),
                "top5_instability_mean": float(group["top5_instability_mean"].mean()),
                "spearman_rank_mean": float(group["spearman_rank_mean"].mean()),
                "mean_normalized_rank_displacement_mean": float(
                    group["mean_normalized_rank_displacement_mean"].mean()
                ),
                "regret_pdb_to_afdb_mean": float(
                    group["regret_pdb_to_afdb_mean"].mean()
                ),
                "regret_afdb_to_pdb_mean": float(
                    group["regret_afdb_to_pdb_mean"].mean()
                ),
                "symmetric_regret_mean": float(regrets.mean()),
                "symmetric_regret_median": float(np.median(regrets)),
                "symmetric_regret_q75": float(
                    np.quantile(regrets, 0.75, method="linear")
                ),
                "symmetric_regret_q95": float(
                    np.quantile(regrets, 0.95, method="linear")
                ),
                "wt_top1_changed_fraction_mean": float(
                    group["wt_top1_changed_fraction"].mean()
                ),
                "pdb_same_state_top1_disagreement_rate_mean": float(
                    group["pdb_same_state_pairwise_top1_disagreement_rate"].mean()
                ),
                "afdb_same_state_top1_disagreement_rate_mean": float(
                    group["afdb_same_state_pairwise_top1_disagreement_rate"].mean()
                ),
                "pdb_same_state_rank_displacement_mean": float(
                    group[
                        "pdb_same_state_pairwise_mean_normalized_rank_displacement_mean"
                    ].mean()
                ),
                "afdb_same_state_rank_displacement_mean": float(
                    group[
                        "afdb_same_state_pairwise_mean_normalized_rank_displacement_mean"
                    ].mean()
                ),
                "pdb_same_state_symmetric_regret_mean": float(
                    group["pdb_same_state_pairwise_symmetric_regret_mean"].mean()
                ),
                "afdb_same_state_symmetric_regret_mean": float(
                    group["afdb_same_state_pairwise_symmetric_regret_mean"].mean()
                ),
                "structural_minus_mean_same_state_top1_disagreement_mean": float(
                    group[
                        "structural_minus_mean_same_state_top1_disagreement"
                    ].mean()
                ),
                "structural_minus_mean_same_state_rank_displacement_mean": float(
                    group[
                        "structural_minus_mean_same_state_rank_displacement"
                    ].mean()
                ),
                "structural_minus_mean_same_state_symmetric_regret_mean": float(
                    group[
                        "structural_minus_mean_same_state_symmetric_regret"
                    ].mean()
                ),
                "pdb_exact_top_tie_fraction_mean": float(
                    group["pdb_exact_top_tie_fraction"].mean()
                ),
                "pdb_practical_top_tie_fraction_mean": float(
                    group["pdb_practical_top_tie_fraction"].mean()
                ),
                "afdb_exact_top_tie_fraction_mean": float(
                    group["afdb_exact_top_tie_fraction"].mean()
                ),
                "afdb_practical_top_tie_fraction_mean": float(
                    group["afdb_practical_top_tie_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _logical_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise DecisionSensitivityError(
            "nonportable_analysis_path",
            f"Scientific artifact is outside the project root: {path}",
            outcome="FAIL",
        ) from exc


def _input_record(
    path: Path,
    sha256: str,
    rows: int | None,
    *,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "path": _logical_path(path, project_root),
        "sha256": sha256,
        "rows": rows,
    }


def build_fixed_probe_decision_result(
    inputs: DecisionInputs,
) -> DecisionSensitivityResult:
    """Run the single canonical local-ranking-to-summary dataflow."""
    local = build_local_amino_acid_scores(inputs, repeat_count=REPEAT_COUNT)
    ranked = rank_local_amino_acid_scores(
        local,
        protein_order=inputs.protein_order,
    )
    repeat_effects = build_repeat_decision_effects(ranked)
    technical_null = summarize_same_state_decision_null(
        ranked,
        repeat_count=REPEAT_COUNT,
    )
    positions = summarize_position_decisions(
        repeat_effects,
        technical_null,
        inputs.position_sensitivity,
        repeat_count=REPEAT_COUNT,
        protein_order=inputs.protein_order,
    )
    proteins = summarize_protein_decisions(
        positions,
        protein_order=inputs.protein_order,
    )
    root = inputs.project_root
    scoring = inputs.scoring
    provenance = {
        "protein_manifest": _input_record(
            scoring.protein_manifest_path,
            scoring.protein_manifest_sha256,
            len(scoring.protein_manifest["proteins"]),
            project_root=root,
        ),
        "fixed_probes": _input_record(
            scoring.fixed_probes_path,
            scoring.fixed_probes_sha256,
            len(scoring.fixed_probes),
            project_root=root,
        ),
        "scoring_manifest": _input_record(
            scoring.scoring_manifest_path,
            scoring.scoring_manifest_sha256,
            None,
            project_root=root,
        ),
        "wt_scores": _input_record(
            scoring.wt_scores_path,
            scoring.wt_scores_sha256,
            len(scoring.wt_scores),
            project_root=root,
        ),
        "raw_scores": _input_record(
            scoring.raw_scores_path,
            scoring.raw_scores_sha256,
            len(scoring.raw_scores),
            project_root=root,
        ),
        "fixed_probe_scoring_null": _input_record(
            scoring.scoring_null_path,
            scoring.scoring_null_sha256,
            len(scoring.scoring_null),
            project_root=root,
        ),
        "sensitivity_manifest": _input_record(
            inputs.sensitivity_manifest_path,
            inputs.sensitivity_manifest_sha256,
            None,
            project_root=root,
        ),
        "structural_effects": _input_record(
            inputs.structural_effects_path,
            inputs.structural_effects_sha256,
            len(inputs.structural_effects),
            project_root=root,
        ),
        "candidate_sensitivity": _input_record(
            inputs.candidate_sensitivity_path,
            inputs.candidate_sensitivity_sha256,
            len(inputs.candidate_sensitivity),
            project_root=root,
        ),
        "position_sensitivity": _input_record(
            inputs.position_sensitivity_path,
            inputs.position_sensitivity_sha256,
            len(inputs.position_sensitivity),
            project_root=root,
        ),
        "protein_sensitivity": _input_record(
            inputs.protein_sensitivity_path,
            inputs.protein_sensitivity_sha256,
            len(inputs.protein_sensitivity),
            project_root=root,
        ),
    }
    result = DecisionSensitivityResult(
        project_root=root,
        input_provenance=provenance,
        repeat_effects=repeat_effects,
        position_summary=positions,
        technical_null=technical_null,
        protein_summary=proteins,
        repeat_count=REPEAT_COUNT,
        protein_order=inputs.protein_order,
    )
    validate_decision_result(result)
    return result


def _require_numeric_bounds(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    lower: float,
    upper: float,
) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < lower - 1.0e-12).any()
        or (values > upper + 1.0e-12).any()
    ):
        raise DecisionSensitivityError(
            "decision_metric_out_of_bounds",
            f"Decision metrics are outside [{lower}, {upper}]: {columns}",
            outcome="FAIL",
        )


def validate_decision_result(
    result: DecisionSensitivityResult,
    *,
    expected_repeat_effect_rows: int = EXPECTED_REPEAT_EFFECT_COUNT,
    expected_position_rows: int = EXPECTED_POSITION_COUNT,
    expected_null_rows: int = EXPECTED_NULL_COUNT,
    expected_protein_rows: int = EXPECTED_PROTEIN_COUNT,
) -> None:
    """Apply the complete offline decision-analysis release gate."""
    frames = (
        result.repeat_effects,
        result.position_summary,
        result.technical_null,
        result.protein_summary,
    )
    if tuple(len(frame) for frame in frames) != (
        expected_repeat_effect_rows,
        expected_position_rows,
        expected_null_rows,
        expected_protein_rows,
    ):
        raise DecisionSensitivityError(
            "decision_output_count_mismatch",
            "Decision-sensitivity output row counts differ",
            outcome="FAIL",
        )
    forbidden_tokens = (
        "p_value",
        "pvalue",
        "fdr",
        "significance",
        "uncertainty_label",
        "classification",
        "recommendation",
    )
    forbidden = sorted(
        {
            column
            for frame in frames
            for column in frame.columns
            if any(token in column.lower() for token in forbidden_tokens)
        }
    )
    if forbidden:
        raise DecisionSensitivityError(
            "forbidden_decision_analysis_field",
            f"Testing/classification fields are forbidden: {forbidden}",
            outcome="FAIL",
        )
    if result.repeat_effects.duplicated(
        ["protein_id", "position", "repeat_index"]
    ).any():
        raise DecisionSensitivityError(
            "duplicate_repeat_decision_key",
            "Repeat-level decision scientific keys repeat",
            outcome="FAIL",
        )
    if result.position_summary.duplicated(["protein_id", "position"]).any():
        raise DecisionSensitivityError(
            "duplicate_position_decision_key",
            "Position-level decision scientific keys repeat",
            outcome="FAIL",
        )
    if result.technical_null.duplicated(
        ["protein_id", "position", "backbone_condition"]
    ).any():
        raise DecisionSensitivityError(
            "duplicate_decision_null_key",
            "Technical-null scientific keys repeat",
            outcome="FAIL",
        )
    if result.protein_summary["protein_id"].astype(str).duplicated().any():
        raise DecisionSensitivityError(
            "duplicate_protein_decision_key",
            "Protein-level decision scientific keys repeat",
            outcome="FAIL",
        )
    if tuple(result.protein_summary["protein_id"].astype(str)) != result.protein_order:
        raise DecisionSensitivityError(
            "protein_membership_mismatch",
            "Protein decision summary does not preserve frozen order",
            outcome="FAIL",
        )
    position_keys = {
        (str(protein_id), int(position))
        for protein_id, position in result.position_summary[
            ["protein_id", "position"]
        ].itertuples(index=False, name=None)
    }
    repeat_position_keys = {
        (str(protein_id), int(position))
        for protein_id, position in result.repeat_effects[
            ["protein_id", "position"]
        ].itertuples(index=False, name=None)
    }
    null_position_keys = {
        (str(protein_id), int(position))
        for protein_id, position in result.technical_null[
            ["protein_id", "position"]
        ].itertuples(index=False, name=None)
    }
    if not (
        position_keys == repeat_position_keys == null_position_keys
    ):
        raise DecisionSensitivityError(
            "decision_key_set_mismatch",
            "Repeat, position, and technical-null key sets differ",
            outcome="FAIL",
        )
    if {protein_id for protein_id, _ in position_keys} != set(result.protein_order):
        raise DecisionSensitivityError(
            "protein_membership_mismatch",
            "Position decision membership differs from frozen proteins",
            outcome="FAIL",
        )
    expected_repeats = set(range(result.repeat_count))
    observed_repeat_sets = result.repeat_effects.groupby(
        ["protein_id", "position"], sort=False
    )["repeat_index"].agg(lambda values: {int(value) for value in values})
    if not observed_repeat_sets.map(
        lambda values: values == expected_repeats
    ).all():
        raise DecisionSensitivityError(
            "decision_repeat_grid_mismatch",
            "Every decision position must contain repeats 0..N-1",
            outcome="FAIL",
        )
    if not (
        result.repeat_effects["seed"].astype(int)
        == result.repeat_effects["repeat_index"].astype(int)
    ).all():
        raise DecisionSensitivityError(
            "decision_seed_provenance_mismatch",
            "Observed seeds differ from the frozen repeat indices",
            outcome="FAIL",
        )
    null_conditions = result.technical_null.groupby(
        ["protein_id", "position"], sort=False
    )["backbone_condition"].agg(lambda values: set(values.astype(str)))
    if not null_conditions.map(lambda values: values == {"PDB", "AFDB"}).all():
        raise DecisionSensitivityError(
            "decision_key_set_mismatch",
            "Every decision position requires PDB and AFDB null conditions",
            outcome="FAIL",
        )
    if set(result.position_summary["n_repeats"].astype(int)) != {
        result.repeat_count
    } or set(result.technical_null["n_repeats"].astype(int)) != {
        result.repeat_count
    }:
        raise DecisionSensitivityError(
            "decision_repeat_grid_mismatch",
            "Decision summary repeat counts differ",
            outcome="FAIL",
        )
    expected_pairs = result.repeat_count * (result.repeat_count - 1) // 2
    if set(result.technical_null["repeat_pair_count"].astype(int)) != {
        expected_pairs
    }:
        raise DecisionSensitivityError(
            "same_state_pair_count_mismatch",
            "Technical-null repeat-pair count differs",
            outcome="FAIL",
        )
    if len(result.repeat_effects) != len(result.position_summary) * result.repeat_count:
        raise DecisionSensitivityError(
            "decision_count_reconciliation_mismatch",
            "Repeat and position decision counts do not reconcile",
            outcome="FAIL",
        )
    if len(result.technical_null) != len(result.position_summary) * 2:
        raise DecisionSensitivityError(
            "decision_count_reconciliation_mismatch",
            "Technical-null and position counts do not reconcile",
            outcome="FAIL",
        )
    if int(result.protein_summary["position_count"].sum()) != len(
        result.position_summary
    ):
        raise DecisionSensitivityError(
            "decision_count_reconciliation_mismatch",
            "Protein and position decision counts do not reconcile",
            outcome="FAIL",
        )
    actual_position_counts = (
        result.position_summary.groupby("protein_id", sort=False)
        .size()
        .astype(int)
    )
    declared_position_counts = result.protein_summary.set_index("protein_id")[
        "position_count"
    ].astype(int)
    if not actual_position_counts.reindex(result.protein_order).equals(
        declared_position_counts.reindex(result.protein_order)
    ):
        raise DecisionSensitivityError(
            "decision_count_reconciliation_mismatch",
            "Per-protein position counts do not reconcile",
            outcome="FAIL",
        )
    alphabet = set(STANDARD_AMINO_ACIDS)
    for column in ("wt_aa", "top1_pdb", "top1_afdb"):
        if not set(result.repeat_effects[column].astype(str)).issubset(alphabet):
            raise DecisionSensitivityError(
                "decision_amino_acid_identity_mismatch",
                f"Repeat decisions contain invalid {column}",
                outcome="FAIL",
            )
    _require_numeric_bounds(
        result.repeat_effects,
        ["top3_instability", "top5_instability"],
        lower=0.0,
        upper=1.0,
    )
    _require_numeric_bounds(
        result.repeat_effects,
        ["spearman_rank"],
        lower=-1.0,
        upper=1.0,
    )
    _require_numeric_bounds(
        result.repeat_effects,
        [
            "mean_normalized_rank_displacement",
            "max_normalized_rank_displacement",
        ],
        lower=0.0,
        upper=1.0,
    )
    regret_columns = [
        "regret_pdb_to_afdb",
        "regret_afdb_to_pdb",
        "symmetric_regret",
    ]
    regrets = result.repeat_effects[regret_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(regrets).all() or (regrets < -1.0e-12).any():
        raise DecisionSensitivityError(
            "negative_local_regret",
            "Local decision regrets must be finite and nonnegative",
            outcome="FAIL",
        )
    _require_numeric_bounds(
        result.position_summary,
        [
            "top1_disagreement_fraction",
            "top3_instability_mean",
            "top5_instability_mean",
            "wt_top1_changed_fraction",
        ],
        lower=0.0,
        upper=1.0,
    )
    _require_numeric_bounds(
        result.technical_null,
        [
            "top1_modal_concentration",
            "pairwise_top1_disagreement_rate",
            "pairwise_top3_instability_mean",
            "pairwise_top5_instability_mean",
            "pairwise_mean_normalized_rank_displacement_mean",
            "exact_top_tie_fraction",
            "practical_top_tie_fraction",
        ],
        lower=0.0,
        upper=1.0,
    )


def _rehash_inputs(result: DecisionSensitivityResult) -> None:
    for label, record in result.input_provenance.items():
        path = result.project_root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise DecisionSensitivityError(
                "upstream_mutated_during_analysis",
                f"Frozen input changed during analysis: {label}",
            )


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    try:
        return write_immutable_parquet(path, frame)
    except FixedProbeSensitivityError as exc:
        raise DecisionSensitivityError(exc.code, str(exc), outcome=exc.outcome) from exc


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    try:
        return write_immutable_json(path, payload)
    except FixedProbeSensitivityError as exc:
        raise DecisionSensitivityError(exc.code, str(exc), outcome=exc.outcome) from exc


def _decision_manifest(
    result: DecisionSensitivityResult,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    frames = {
        "repeat_effects": result.repeat_effects,
        "position_summary": result.position_summary,
        "technical_null": result.technical_null,
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
        "schema_version": "stage0_fixed_probe_decision_sensitivity_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_fixed_probe_local_amino_acid_decision_sensitivity_only",
        "inputs": result.input_provenance,
        "analysis_protocol": {
            "identity": ANALYSIS_PROTOCOL_VERSION,
            "decision_unit": "protein_x_canonical_uniprot_position_x_repeat",
            "local_candidate_set": {
                "definition": "analytical_WT_plus_19_frozen_single_mutants",
                "choice_count": 20,
                "standard_amino_acid_order": STANDARD_AMINO_ACIDS,
            },
            "ranking": {
                "score": "score_mean_logp_mask",
                "direction": "higher_is_better",
                "exact_tie_break": "canonical_standard_amino_acid_order",
                "practical_tie_atol": 1.0e-6,
                "practical_tie_rtol": 1.0e-6,
                "practical_tie_changes_strict_ranking": False,
            },
            "structural_orientation": STRUCTURAL_ORIENTATION,
            "top_k_values": [3, 5],
            "top_k_instability": "1_minus_intersection_size_divided_by_k",
            "rank_metrics": {
                "spearman": "pearson_correlation_of_integer_amino_acid_ranks",
                "normalized_displacement": "absolute_AFDB_minus_PDB_rank_divided_by_19",
            },
            "regret": {
                "pdb_to_afdb": "max_AFDB_score_minus_AFDB_score_of_PDB_argmax",
                "afdb_to_pdb": "max_PDB_score_minus_PDB_score_of_AFDB_argmax",
                "symmetric": "mean_of_two_directional_regrets",
                "semantics": "frozen_internal_model_local_score_regret_only",
            },
            "same_state_technical_null": {
                "definition": "all_unordered_repeat_pairs_within_position_and_backbone",
                "pair_count_per_position_backbone": result.repeat_count
                * (result.repeat_count - 1)
                // 2,
                "pairwise_values_are_independent_observations": False,
            },
            "repeat_count": result.repeat_count,
            "repeat_seeds": list(range(result.repeat_count)),
            "quantile_levels": [0.75, 0.95],
            "quantile_method": "linear",
            "structural_vs_technical_semantics": "descriptive_contrasts_without_thresholds",
            "continuous_association_computed": False,
        },
        "protein_order": list(result.protein_order),
        "outputs": outputs,
        "final_counts": {
            "repeat_effect_rows": len(result.repeat_effects),
            "position_summary_rows": len(result.position_summary),
            "technical_null_rows": len(result.technical_null),
            "protein_summary_rows": len(result.protein_summary),
        },
        "scope_declarations": {
            "hypothesis_testing_performed": False,
            "p_values_computed": False,
            "binary_uncertainty_classification_performed": False,
            "biological_mutation_recommendations_made": False,
            "protein_ranking_performed": False,
            "proteinmpnn_executed": False,
            "sequence_generation_performed": False,
            "generated_candidate_union_started": False,
            "external_evaluator_executed": False,
            "arm_b_started": False,
            "stage0_2b_started": False,
        },
    }


def materialize_fixed_probe_decision_sensitivity(
    result: DecisionSensitivityResult,
    output_root: Path,
    *,
    expected_repeat_effect_rows: int = EXPECTED_REPEAT_EFFECT_COUNT,
    expected_position_rows: int = EXPECTED_POSITION_COUNT,
    expected_null_rows: int = EXPECTED_NULL_COUNT,
    expected_protein_rows: int = EXPECTED_PROTEIN_COUNT,
) -> dict[str, Any]:
    """Validate and immutably render all decision artifacts, manifest last."""
    validate_decision_result(
        result,
        expected_repeat_effect_rows=expected_repeat_effect_rows,
        expected_position_rows=expected_position_rows,
        expected_null_rows=expected_null_rows,
        expected_protein_rows=expected_protein_rows,
    )
    _rehash_inputs(result)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "repeat_effects": output_root / "fixed_probe_local_decision_effects.parquet",
        "position_summary": output_root
        / "fixed_probe_position_decision_sensitivity.parquet",
        "technical_null": output_root / "fixed_probe_local_decision_null.parquet",
        "protein_summary": output_root
        / "fixed_probe_protein_decision_sensitivity.parquet",
    }
    frames = {
        "repeat_effects": result.repeat_effects,
        "position_summary": result.position_summary,
        "technical_null": result.technical_null,
        "protein_summary": result.protein_summary,
    }
    write_status = {
        label: _write_immutable_parquet(output_paths[label], frames[label])
        for label in output_paths
    }
    _rehash_inputs(result)
    manifest = _decision_manifest(result, output_paths)
    manifest_path = output_root / "fixed_probe_decision_sensitivity_manifest.json"
    write_status["manifest"] = _write_immutable_json(manifest_path, manifest)
    return {
        "status": "PASS",
        "manifest_path": manifest_path,
        "write_status": write_status,
        "repeat_effect_rows": len(result.repeat_effects),
        "position_summary_rows": len(result.position_summary),
        "technical_null_rows": len(result.technical_null),
        "protein_summary_rows": len(result.protein_summary),
    }


def run_fixed_probe_decision_sensitivity(
    project_root: Path,
) -> DecisionSensitivityResult:
    """Load frozen inputs and construct the canonical structured result."""
    inputs = load_frozen_decision_inputs(project_root)
    return build_fixed_probe_decision_result(inputs)
