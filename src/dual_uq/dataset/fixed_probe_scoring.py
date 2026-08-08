"""Stage-0 fixed-probe scoring contracts and orchestration.

Torch and the ProteinMPNN submodule are deliberately not imported here. Pure
configuration and provenance gates must remain usable in the standard dataset
environment before any model execution is attempted.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.core.hashing import sha256_canonical, sha256_file
from dual_uq.dataset.services.proteinmpnn_scoring import (
    ProteinMPNNScoringError as FixedProbeScoringError,
)
from dual_uq.dataset.services.proteinmpnn_scoring import (
    implementation_worktree_is_clean,
)
from dual_uq.dataset.storage.proteinmpnn import sequence_sha256

AUTHORIZED_IMPLEMENTATION_PATH = Path("third_party/ProteinMPNN")
AUTHORIZED_IMPLEMENTATION_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
AUTHORIZED_CHECKPOINT_PATH = Path(
    "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
)
AUTHORIZED_CHECKPOINT_SHA256 = (
    "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
)
SCORING_PROTOCOL_VERSION = "stage0_fixed_sequence_autoregressive_mask_logp_v1"
FORMAL_STOCHASTIC_SEEDS = tuple(range(30))
ADMITTED_SUBSET_SHA256 = (
    "0f29fe5309a63ae0e24eda2ebdacfad861c418165bb2552b6f68e40855d6d3f5"
)
PROTEIN_MANIFEST_SHA256 = (
    "fd34ae871c3d5feebba1dbe38bce24141634764882a9d51db1ce79cd7581d30b"
)
FIXED_PROBES_SHA256 = (
    "26dc56745c005c01e78007ad3c3b6dbc59708da23087ac7bd2337acc2ea27ef0"
)
G2_AUDIT_SHA256 = (
    "eb4acae7e4018006156f80f40e50b1c99bebd1af3c0b44f46fcee603152f34ff"
)
ADMITTED_SUBSET_PATH = Path(
    "experiments/p2_design_baseline/stage0/stage0_intervention_admitted_v1.jsonl"
)
PROTEIN_MANIFEST_PATH = Path(
    "experiments/p2_design_baseline/stage0/protein_manifest.json"
)
FIXED_PROBES_PATH = Path(
    "experiments/p2_design_baseline/stage0/fixed_probe_candidates.parquet"
)


@dataclass(frozen=True)
class ScoringModelConfig:
    family: str
    role: str
    implementation_path: Path
    implementation_commit: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class ScoringAuditConfig:
    atol: float
    rtol: float
    audit_seeds: tuple[int, ...]
    formal_stochastic_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Stage0ScoringConfig:
    project_root: Path
    model: ScoringModelConfig
    protocol_version: str
    mode: str
    scoring_domain: str
    residue_index_coordinate_system: str
    backbone_atoms: tuple[str, ...]
    backbone_noise: float
    numerical_dtype: str
    audit: ScoringAuditConfig


@dataclass(frozen=True)
class FixedProbeInputs:
    admitted_subset_path: Path
    admitted_subset_sha256: str
    admitted_records: tuple[dict[str, Any], ...]
    protein_manifest_path: Path
    protein_manifest_sha256: str
    protein_manifest: dict[str, Any]
    fixed_probes_path: Path
    fixed_probes_sha256: str
    fixed_probes: pd.DataFrame


@dataclass(frozen=True)
class VerifiedModelIdentity:
    implementation_path: Path
    implementation_commit: str
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class G2AuditMutant:
    protein_id: str
    position: int
    wt_aa: str
    mut_aa: str
    sequence_hash: str
    full_sequence: str
    projected_sequence: str


@dataclass(frozen=True)
class G2ProteinCandidates:
    protein_id: str
    positions: tuple[int, int, int]
    wt_full_sequence: str
    wt_sequence_hash: str
    wt_projected_sequence: str
    mutants: tuple[G2AuditMutant, ...]


@dataclass(frozen=True)
class G2Classification:
    status: str
    classification: str
    formal_repeat_count: int
    formal_seeds: tuple[int, ...]
    max_same_realization_absolute_deviation: float
    max_same_realization_relative_deviation: float
    max_batch_absolute_deviation: float
    max_batch_relative_deviation: float
    different_realizations_changed_scores: bool


@dataclass(frozen=True)
class FrozenG2Policy:
    audit_path: Path
    audit_sha256: str
    classification: str
    repeat_count: int
    seeds: tuple[int, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class FormalShardSpec:
    protein_id: str
    backbone_condition: str
    backbone_sha256: str
    repeat_index: int
    seed: int
    decoding_realization_sha256: str
    decoding_realization_algorithm: str
    candidate_sequence_hashes: tuple[str, ...]
    candidate_identity_sha256: str
    candidate_count: int
    mask_length: int
    admitted_subset_sha256: str
    protein_manifest_sha256: str
    fixed_probes_sha256: str
    g2_audit_sha256: str
    checkpoint_sha256: str
    implementation_commit: str
    scoring_protocol: str


def _mapping(value: Any, code: str, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FixedProbeScoringError(code, message)
    return value


def _portable_path(value: Any, field: str) -> Path:
    if not isinstance(value, str):
        raise FixedProbeScoringError(
            "nonportable_scoring_path", f"{field} must be repository-relative"
        )
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != value:
        raise FixedProbeScoringError(
            "nonportable_scoring_path", f"{field} must be repository-relative"
        )
    return Path(value)


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise FixedProbeScoringError(
            "invalid_scoring_protocol", f"{field} must be an explicit integer list"
        )
    return tuple(value)


def load_stage0_scoring_config(
    config_path: Path, project_root: Path
) -> Stage0ScoringConfig:
    """Load and strictly validate the human-authorized Stage-0-2A binding."""
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FixedProbeScoringError(
            "invalid_scoring_config", f"Unable to read {config_path}"
        ) from exc
    root = _mapping(payload, "invalid_scoring_config", "Stage-0 config must be a mapping")
    scoring = _mapping(
        root.get("scoring"),
        "missing_scoring_config",
        "Stage-0 config lacks the frozen scoring section",
    )
    model = _mapping(
        scoring.get("model"),
        "missing_scoring_config",
        "Stage-0 scoring config lacks a model binding",
    )
    implementation_path = _portable_path(
        model.get("implementation_path"), "implementation_path"
    )
    checkpoint_path = _portable_path(model.get("checkpoint_path"), "checkpoint_path")
    model_values = (
        model.get("family"),
        model.get("role"),
        implementation_path,
        model.get("implementation_commit"),
        checkpoint_path,
        model.get("checkpoint_sha256"),
    )
    expected_model_values = (
        "ProteinMPNN",
        "stage0_internal_inverse_folding_scorer",
        AUTHORIZED_IMPLEMENTATION_PATH,
        AUTHORIZED_IMPLEMENTATION_COMMIT,
        AUTHORIZED_CHECKPOINT_PATH,
        AUTHORIZED_CHECKPOINT_SHA256,
    )
    if model_values != expected_model_values:
        raise FixedProbeScoringError(
            "unauthorized_scoring_model",
            "Stage-0 scoring model differs from the human-authorized identity",
        )
    audit = _mapping(
        scoring.get("audit"),
        "missing_scoring_config",
        "Stage-0 scoring config lacks audit settings",
    )
    audit_seeds = _integer_tuple(audit.get("audit_seeds"), "audit_seeds")
    formal_seeds = _integer_tuple(
        audit.get("formal_stochastic_seeds"), "formal_stochastic_seeds"
    )
    protocol_values = (
        scoring.get("protocol_version"),
        scoring.get("mode"),
        scoring.get("scoring_domain"),
        scoring.get("residue_index_coordinate_system"),
        tuple(scoring.get("backbone_atoms", ())),
        scoring.get("backbone_noise"),
        scoring.get("numerical_dtype"),
        audit.get("atol"),
        audit.get("rtol"),
        audit_seeds,
        formal_seeds,
    )
    expected_protocol_values = (
        SCORING_PROTOCOL_VERSION,
        "fixed_sequence_autoregressive_mask_logp",
        "frozen_common_mask_projection",
        "uniprot_position_1based",
        ("N", "CA", "C", "O"),
        0.0,
        "float32",
        1.0e-6,
        1.0e-6,
        (0, 1, 2, 3),
        FORMAL_STOCHASTIC_SEEDS,
    )
    if protocol_values != expected_protocol_values:
        raise FixedProbeScoringError(
            "invalid_scoring_protocol", "Stage-0 scoring protocol differs"
        )
    return Stage0ScoringConfig(
        project_root=project_root.resolve(),
        model=ScoringModelConfig(
            family="ProteinMPNN",
            role="stage0_internal_inverse_folding_scorer",
            implementation_path=implementation_path,
            implementation_commit=AUTHORIZED_IMPLEMENTATION_COMMIT,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=AUTHORIZED_CHECKPOINT_SHA256,
        ),
        protocol_version=SCORING_PROTOCOL_VERSION,
        mode="fixed_sequence_autoregressive_mask_logp",
        scoring_domain="frozen_common_mask_projection",
        residue_index_coordinate_system="uniprot_position_1based",
        backbone_atoms=("N", "CA", "C", "O"),
        backbone_noise=0.0,
        numerical_dtype="float32",
        audit=ScoringAuditConfig(
            atol=1.0e-6,
            rtol=1.0e-6,
            audit_seeds=audit_seeds,
            formal_stochastic_seeds=formal_seeds,
        ),
    )


def _input_binding(
    config_path: Path,
    *,
    key: str,
    expected_path: Path,
    expected_sha256: str,
) -> tuple[Path, str]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FixedProbeScoringError(
            "invalid_scoring_config", f"Unable to read {config_path}"
        ) from exc
    root = _mapping(payload, "invalid_scoring_config", "Stage-0 config must be a mapping")
    inputs = _mapping(
        root.get("inputs"), "invalid_scoring_config", "Stage-0 inputs must be a mapping"
    )
    binding = _mapping(
        inputs.get(key), "invalid_scoring_config", f"Missing Stage-0 input {key}"
    )
    logical = _portable_path(binding.get("path"), f"{key}.path")
    digest = binding.get("sha256")
    if logical != expected_path or digest != expected_sha256:
        raise FixedProbeScoringError(
            "invalid_scoring_input_binding", f"Frozen binding differs for {key}"
        )
    return logical, str(digest)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeScoringError(
            "invalid_scoring_input", f"Unable to read {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise FixedProbeScoringError(
            "invalid_scoring_input", f"{path.name} must contain a JSON object"
        )
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        rows = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeScoringError(
            "invalid_scoring_input", f"Unable to read {path.name}"
        ) from exc
    if any(not isinstance(row, dict) for row in rows):
        raise FixedProbeScoringError(
            "invalid_scoring_input", f"{path.name} must contain JSON objects"
        )
    return rows


def load_frozen_scoring_inputs(
    config_path: Path, project_root: Path
) -> FixedProbeInputs:
    """Validate all frozen Stage-0-2A inputs before model execution."""
    load_stage0_scoring_config(config_path, project_root)
    root = project_root.resolve()
    bindings = (
        (
            "stage0_intervention_admitted_subset",
            ADMITTED_SUBSET_PATH,
            ADMITTED_SUBSET_SHA256,
            "admitted_subset_hash_mismatch",
        ),
        (
            "stage0_protein_manifest",
            PROTEIN_MANIFEST_PATH,
            PROTEIN_MANIFEST_SHA256,
            "protein_manifest_hash_mismatch",
        ),
        (
            "stage0_fixed_probe_candidates",
            FIXED_PROBES_PATH,
            FIXED_PROBES_SHA256,
            "fixed_probes_hash_mismatch",
        ),
    )
    paths: dict[str, Path] = {}
    for key, logical, expected_hash, error_code in bindings:
        bound_path, _ = _input_binding(
            config_path, key=key, expected_path=logical, expected_sha256=expected_hash
        )
        path = root / bound_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise FixedProbeScoringError(error_code, f"Frozen SHA differs for {key}")
        paths[key] = path

    admitted = _read_jsonl(paths["stage0_intervention_admitted_subset"])
    manifest = _read_json(paths["stage0_protein_manifest"])
    try:
        probes = pd.read_parquet(paths["stage0_fixed_probe_candidates"])
    except (OSError, ValueError) as exc:
        raise FixedProbeScoringError(
            "invalid_scoring_input", "Unable to read fixed-probe Parquet"
        ) from exc
    admitted_ids = tuple(row.get("protein_id") for row in admitted)
    proteins = manifest.get("proteins")
    if len(admitted) != 8 or len(set(admitted_ids)) != 8:
        raise FixedProbeScoringError(
            "admitted_subset_count_mismatch", "Expected eight unique admitted proteins"
        )
    if not isinstance(proteins, list) or len(proteins) != 8:
        raise FixedProbeScoringError(
            "protein_manifest_count_mismatch", "Expected eight manifest proteins"
        )
    manifest_ids = tuple(record.get("protein_id") for record in proteins)
    if manifest_ids != admitted_ids or len(set(manifest_ids)) != 8:
        raise FixedProbeScoringError(
            "protein_identity_mismatch", "Manifest and admitted identities differ"
        )
    total_positions = sum(
        len(record.get("mask_positions", ()))
        for record in proteins
        if isinstance(record, dict)
    )
    if total_positions != 1790 or manifest.get("summary", {}).get(
        "total_mask_positions"
    ) != 1790:
        raise FixedProbeScoringError(
            "protein_manifest_mask_count_mismatch", "Expected 1,790 mask positions"
        )
    if len(probes) != 34010:
        raise FixedProbeScoringError(
            "fixed_probe_count_mismatch", "Expected 34,010 fixed probes"
        )
    required_columns = {
        "protein_id",
        "position",
        "wt_aa",
        "mut_aa",
        "sequence_hash",
        "full_sequence",
    }
    if required_columns - set(probes.columns):
        raise FixedProbeScoringError(
            "invalid_fixed_probe_schema", "Fixed-probe columns are incomplete"
        )
    if probes.duplicated(["protein_id", "position", "mut_aa"]).any():
        raise FixedProbeScoringError(
            "duplicate_fixed_probe", "Fixed-probe scientific identities repeat"
        )
    if set(probes["protein_id"]) != set(manifest_ids):
        raise FixedProbeScoringError(
            "protein_identity_mismatch", "Probe and manifest identities differ"
        )
    return FixedProbeInputs(
        admitted_subset_path=paths["stage0_intervention_admitted_subset"],
        admitted_subset_sha256=ADMITTED_SUBSET_SHA256,
        admitted_records=admitted,
        protein_manifest_path=paths["stage0_protein_manifest"],
        protein_manifest_sha256=PROTEIN_MANIFEST_SHA256,
        protein_manifest=manifest,
        fixed_probes_path=paths["stage0_fixed_probe_candidates"],
        fixed_probes_sha256=FIXED_PROBES_SHA256,
        fixed_probes=probes,
    )


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FixedProbeScoringError(
            "implementation_commit_unavailable", "Cannot resolve ProteinMPNN commit"
        ) from exc


def verify_authorized_model(
    config: Stage0ScoringConfig,
    *,
    checkpoint_hash_reader: Callable[[Path], str] | None = None,
    implementation_commit_reader: Callable[[Path], str] | None = None,
    implementation_clean_reader: Callable[[Path], bool] | None = None,
) -> VerifiedModelIdentity:
    """Verify the sole authorized model identity without loading Torch."""
    implementation = (config.project_root / config.model.implementation_path).resolve()
    checkpoint = (config.project_root / config.model.checkpoint_path).resolve()
    if not implementation.is_dir():
        raise FixedProbeScoringError(
            "implementation_missing", "Authorized ProteinMPNN implementation is missing"
        )
    if not checkpoint.is_file():
        raise FixedProbeScoringError(
            "checkpoint_missing", "Authorized ProteinMPNN checkpoint is missing"
        )
    hash_reader = checkpoint_hash_reader or sha256_file
    commit_reader = implementation_commit_reader or _git_head
    clean_reader = implementation_clean_reader or implementation_worktree_is_clean
    checkpoint_sha = hash_reader(checkpoint)
    if checkpoint_sha != config.model.checkpoint_sha256:
        raise FixedProbeScoringError(
            "checkpoint_hash_mismatch", "Authorized checkpoint SHA differs"
        )
    implementation_commit = commit_reader(implementation)
    if implementation_commit != config.model.implementation_commit:
        raise FixedProbeScoringError(
            "implementation_commit_mismatch", "ProteinMPNN implementation commit differs"
        )
    if not clean_reader(implementation):
        raise FixedProbeScoringError(
            "implementation_worktree_dirty",
            "ProteinMPNN implementation has tracked-file drift",
        )
    return VerifiedModelIdentity(
        implementation_path=implementation,
        implementation_commit=implementation_commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )


def build_g2_audit_candidates(
    inputs: FixedProbeInputs,
) -> dict[str, G2ProteinCandidates]:
    """Select the exact frozen 24 mutants and two WT audit sequences."""
    fixtures = {
        "5gv8_A__P83686": (1, 136, 272),
        "5mn1_A__P00760": (24, 135, 246),
    }
    proteins = {
        str(record.get("protein_id")): record
        for record in inputs.protein_manifest.get("proteins", [])
        if isinstance(record, dict)
    }
    result: dict[str, G2ProteinCandidates] = {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for protein_id, positions in fixtures.items():
        protein = proteins.get(protein_id)
        if protein is None:
            raise FixedProbeScoringError(
                "missing_g2_fixture", f"G2 protein is absent: {protein_id}"
            )
        mask_positions = tuple(protein.get("mask_positions", ()))
        if any(position not in mask_positions for position in positions):
            raise FixedProbeScoringError(
                "invalid_g2_fixture", f"G2 position is outside the mask: {protein_id}"
            )
        wt_sequence = str(protein.get("canonical_wt_sequence", ""))
        wt_projection = "".join(wt_sequence[position - 1] for position in mask_positions)
        selected: list[G2AuditMutant] = []
        protein_rows = inputs.fixed_probes.loc[
            inputs.fixed_probes["protein_id"] == protein_id
        ]
        for position in positions:
            wt_aa = wt_sequence[position - 1]
            substitutions = [amino_acid for amino_acid in alphabet if amino_acid != wt_aa][
                :4
            ]
            for mut_aa in substitutions:
                rows = protein_rows.loc[
                    (protein_rows["position"] == position)
                    & (protein_rows["mut_aa"] == mut_aa)
                ]
                if len(rows) != 1:
                    raise FixedProbeScoringError(
                        "invalid_g2_fixture",
                        f"Expected one frozen probe for {protein_id}:{position}{mut_aa}",
                    )
                row = rows.iloc[0]
                full_sequence = str(row["full_sequence"])
                expected_hash = sequence_sha256(full_sequence)
                if (
                    str(row["wt_aa"]) != wt_aa
                    or str(row["sequence_hash"]) != expected_hash
                ):
                    raise FixedProbeScoringError(
                        "invalid_g2_fixture", "Frozen G2 candidate identity differs"
                    )
                selected.append(
                    G2AuditMutant(
                        protein_id=protein_id,
                        position=position,
                        wt_aa=wt_aa,
                        mut_aa=mut_aa,
                        sequence_hash=expected_hash,
                        full_sequence=full_sequence,
                        projected_sequence="".join(
                            full_sequence[mask_position - 1]
                            for mask_position in mask_positions
                        ),
                    )
                )
        result[protein_id] = G2ProteinCandidates(
            protein_id=protein_id,
            positions=positions,
            wt_full_sequence=wt_sequence,
            wt_sequence_hash=sequence_sha256(wt_sequence),
            wt_projected_sequence=wt_projection,
            mutants=tuple(selected),
        )
    return result


def _checked_score_vector(value: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != reference_shape or vector.ndim != 1 or not np.isfinite(vector).all():
        raise FixedProbeScoringError(
            "invalid_g2_observations", "G2 score vectors must be finite and aligned"
        )
    return vector


def _max_deviations(reference: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    absolute = np.abs(observed - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(absolute.max(initial=0.0)), float(
        (absolute / denominator).max(initial=0.0)
    )


def _within_frozen_tolerance(
    reference: np.ndarray, observed: np.ndarray, *, atol: float, rtol: float
) -> bool:
    deviation = np.abs(observed - reference)
    threshold = atol + rtol * np.abs(reference)
    return bool(np.all(deviation <= threshold))


def classify_g2_observations(
    *,
    same_realization_runs: tuple[np.ndarray, ...],
    batch_size_one: np.ndarray,
    batch_size_many: np.ndarray,
    different_realization_scores: tuple[np.ndarray, ...],
    realization_fingerprints: tuple[str, ...],
    atol: float,
    rtol: float,
    formal_stochastic_seeds: tuple[int, ...],
) -> G2Classification:
    """Apply the frozen G2 state machine without preferring an outcome."""
    if len(same_realization_runs) < 3 or len(different_realization_scores) != 4:
        raise FixedProbeScoringError(
            "invalid_g2_observations", "G2 requires three repeats and four audit seeds"
        )
    if len(realization_fingerprints) != 4 or len(set(realization_fingerprints)) != 4:
        raise FixedProbeScoringError(
            "duplicate_decoding_realization", "G2 audit realization fingerprints repeat"
        )
    reference = np.asarray(same_realization_runs[0], dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise FixedProbeScoringError(
            "invalid_g2_observations", "G2 reference scores are invalid"
        )
    same = tuple(
        _checked_score_vector(value, reference.shape) for value in same_realization_runs
    )
    batch_one = _checked_score_vector(batch_size_one, reference.shape)
    batch_many = _checked_score_vector(batch_size_many, reference.shape)
    seed_scores = tuple(
        _checked_score_vector(value, reference.shape)
        for value in different_realization_scores
    )
    same_deviations = [_max_deviations(reference, value) for value in same[1:]]
    same_absolute = max((value[0] for value in same_deviations), default=0.0)
    same_relative = max((value[1] for value in same_deviations), default=0.0)
    batch_absolute, batch_relative = _max_deviations(batch_one, batch_many)
    same_stable = all(
        _within_frozen_tolerance(reference, value, atol=atol, rtol=rtol)
        for value in same
    )
    batch_stable = _within_frozen_tolerance(
        batch_one, batch_many, atol=atol, rtol=rtol
    )
    realization_sensitive = any(
        not _within_frozen_tolerance(first, second, atol=atol, rtol=rtol)
        for first, second in combinations(seed_scores, 2)
    )
    if not same_stable:
        status = "BLOCKED"
        classification = "numerically_unstable_same_realization"
        repeat_count = 0
        formal_seeds: tuple[int, ...] = ()
    elif not batch_stable:
        status = "FAIL"
        classification = "batch_size_dependent_scoring"
        repeat_count = 0
        formal_seeds = ()
    elif realization_sensitive:
        if formal_stochastic_seeds != tuple(range(30)):
            raise FixedProbeScoringError(
                "invalid_scoring_protocol", "Stochastic formal seeds must be 0..29"
            )
        status = "PASS"
        classification = "stochastic_due_to_decoding_order"
        repeat_count = 30
        formal_seeds = formal_stochastic_seeds
    else:
        status = "PASS"
        classification = "deterministic_under_frozen_scoring_protocol"
        repeat_count = 1
        formal_seeds = (0,)
    return G2Classification(
        status=status,
        classification=classification,
        formal_repeat_count=repeat_count,
        formal_seeds=formal_seeds,
        max_same_realization_absolute_deviation=same_absolute,
        max_same_realization_relative_deviation=same_relative,
        max_batch_absolute_deviation=batch_absolute,
        max_batch_relative_deviation=batch_relative,
        different_realizations_changed_scores=realization_sensitive,
    )


def render_g2_audit(payload: dict[str, Any]) -> bytes:
    """Render one canonical review artifact with no non-finite JSON values."""
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
        raise FixedProbeScoringError(
            "invalid_g2_audit", "G2 audit payload is not canonical JSON"
        ) from exc


def write_g2_audit(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable G2 audit or reuse byte-identical prior evidence."""
    rendered = render_g2_audit(payload)
    if path.exists():
        if path.read_bytes() != rendered:
            raise FixedProbeScoringError(
                "immutable_g2_audit_conflict", f"Existing G2 audit differs: {path}"
            )
        return "reused_identical"
    atomic_write_new_bytes(path, rendered)
    return "created"


def run_g2_audit(
    *,
    config: Stage0ScoringConfig,
    inputs: FixedProbeInputs,
    model_identity: VerifiedModelIdentity,
    projections: dict[str, Any],
    runtime: Any,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Run only the frozen 24-mutant + two-WT G2 audit matrix."""
    from dual_uq.dataset.services.proteinmpnn_scoring import (
        make_decoding_realization,
        validate_paired_projection,
    )

    candidates = build_g2_audit_candidates(inputs)
    protein_ids = tuple(candidates)
    if set(projections) != set(protein_ids):
        raise FixedProbeScoringError(
            "g2_projection_identity_mismatch",
            "G2 projections must contain exactly the two frozen audit proteins",
        )
    checked_projections: dict[str, Any] = {}
    for protein_id in protein_ids:
        manifest_record = next(
            record
            for record in inputs.protein_manifest["proteins"]
            if record["protein_id"] == protein_id
        )
        paired = projections[protein_id]
        checked_projections[protein_id] = validate_paired_projection(
            paired.pdb,
            paired.afdb,
            expected_positions=tuple(manifest_record["mask_positions"]),
        )

    realizations: dict[str, dict[int, Any]] = {}
    for protein_id in protein_ids:
        length = checked_projections[protein_id].pdb.residue_count
        realizations[protein_id] = {
            seed: make_decoding_realization(
                protein_id=protein_id,
                mask_length=length,
                repeat_index=seed,
                seed=seed,
                protocol_version=config.protocol_version,
            )
            for seed in config.audit.audit_seeds
        }

    score_vector_order: list[dict[str, Any]] = []
    for protein_id in protein_ids:
        fixture = candidates[protein_id]
        identities = [("WT", fixture.wt_sequence_hash)] + [
            (
                f"{mutant.wt_aa}{mutant.position}{mutant.mut_aa}",
                mutant.sequence_hash,
            )
            for mutant in fixture.mutants
        ]
        for backbone in ("PDB", "AFDB"):
            for candidate_id, sequence_hash in identities:
                for metric in ("score_sum_logp_mask", "score_mean_logp_mask"):
                    score_vector_order.append(
                        {
                            "protein_id": protein_id,
                            "backbone_condition": backbone,
                            "candidate_id": candidate_id,
                            "sequence_hash": sequence_hash,
                            "metric": metric,
                        }
                    )

    def score_vector(seed: int, batch_size: int) -> list[float]:
        values: list[float] = []
        for protein_id in protein_ids:
            fixture = candidates[protein_id]
            sequences = (fixture.wt_projected_sequence,) + tuple(
                mutant.projected_sequence for mutant in fixture.mutants
            )
            realization = realizations[protein_id][seed]
            paired = checked_projections[protein_id]
            for projection in (paired.pdb, paired.afdb):
                scores = runtime.score_sequences(
                    projection,
                    sequences,
                    realization,
                    batch_size=batch_size,
                )
                if len(scores) != len(sequences):
                    raise FixedProbeScoringError(
                        "g2_score_count_mismatch", "G2 runtime returned incomplete scores"
                    )
                for score in scores:
                    values.extend(
                        (score.score_sum_logp_mask, score.score_mean_logp_mask)
                    )
        return values

    same_realization_runs = tuple(
        np.asarray(score_vector(0, 8), dtype=np.float64) for _ in range(3)
    )
    batch_size_one = np.asarray(score_vector(0, 1), dtype=np.float64)
    batch_size_eight = np.asarray(score_vector(0, 8), dtype=np.float64)
    seed_scores = tuple(
        np.asarray(score_vector(seed, 8), dtype=np.float64)
        for seed in config.audit.audit_seeds
    )
    aggregate_fingerprints = tuple(
        sha256_canonical(
            {
                protein_id: realizations[protein_id][seed].fingerprint
                for protein_id in protein_ids
            }
        )
        for seed in config.audit.audit_seeds
    )
    classification = classify_g2_observations(
        same_realization_runs=same_realization_runs,
        batch_size_one=batch_size_one,
        batch_size_many=batch_size_eight,
        different_realization_scores=seed_scores,
        realization_fingerprints=aggregate_fingerprints,
        atol=config.audit.atol,
        rtol=config.audit.rtol,
        formal_stochastic_seeds=config.audit.formal_stochastic_seeds,
    )
    audit_proteins = []
    for protein_id in protein_ids:
        fixture = candidates[protein_id]
        audit_proteins.append(
            {
                "protein_id": protein_id,
                "positions": list(fixture.positions),
                "wt_residues": [
                    {
                        "position": position,
                        "wt_aa": fixture.wt_full_sequence[position - 1],
                    }
                    for position in fixture.positions
                ],
                "wt_sequence_hash": fixture.wt_sequence_hash,
                "mutants": [
                    {
                        "candidate_id": (
                            f"{mutant.wt_aa}{mutant.position}{mutant.mut_aa}"
                        ),
                        "position": mutant.position,
                        "wt_aa": mutant.wt_aa,
                        "mut_aa": mutant.mut_aa,
                        "sequence_hash": mutant.sequence_hash,
                    }
                    for mutant in fixture.mutants
                ],
            }
        )
    realization_records = [
        {
            "protein_id": protein_id,
            "repeat_index": realization.repeat_index,
            "seed": realization.seed,
            "algorithm": realization.algorithm,
            "length": len(realization.order),
            "fingerprint": realization.fingerprint,
        }
        for protein_id in protein_ids
        for realization in realizations[protein_id].values()
    ]
    return {
        "schema_version": "stage0_g2_audit_v1",
        "status": classification.status,
        "classification": classification.classification,
        "formal_repeat_count": classification.formal_repeat_count,
        "formal_seeds": list(classification.formal_seeds),
        "scientific_scope": "stage0_internal_fixed_probe_scoring_g2_audit_only",
        "formal_scoring_started": False,
        "upstream_identity": {
            "admitted_subset_sha256": inputs.admitted_subset_sha256,
            "protein_manifest_sha256": inputs.protein_manifest_sha256,
            "fixed_probes_sha256": inputs.fixed_probes_sha256,
        },
        "model_identity": {
            "family": config.model.family,
            "role": config.model.role,
            "implementation_path": config.model.implementation_path.as_posix(),
            "implementation_commit": model_identity.implementation_commit,
            "checkpoint_path": config.model.checkpoint_path.as_posix(),
            "checkpoint_sha256": model_identity.checkpoint_sha256,
            "checkpoint_num_edges": int(runtime.checkpoint_num_edges),
            "checkpoint_noise_level": float(runtime.checkpoint_noise_level),
        },
        "scoring_protocol": {
            "protocol_version": config.protocol_version,
            "mode": config.mode,
            "scoring_domain": config.scoring_domain,
            "residue_index_coordinate_system": config.residue_index_coordinate_system,
            "backbone_atoms": list(config.backbone_atoms),
            "backbone_noise": config.backbone_noise,
            "numerical_dtype": config.numerical_dtype,
            "score_direction": "higher_is_better",
            "sequence_generation": False,
        },
        "audit_fixture": {
            "protein_count": len(audit_proteins),
            "mutant_count": sum(len(value.mutants) for value in candidates.values()),
            "wt_count": len(audit_proteins),
            "proteins": audit_proteins,
        },
        "realizations": realization_records,
        "numerical_audit": {
            "atol": config.audit.atol,
            "rtol": config.audit.rtol,
            "same_realization_repeat_count": len(same_realization_runs),
            "batch_sizes": [1, 8],
            "audit_seeds": list(config.audit.audit_seeds),
            "max_same_realization_absolute_deviation": (
                classification.max_same_realization_absolute_deviation
            ),
            "max_same_realization_relative_deviation": (
                classification.max_same_realization_relative_deviation
            ),
            "max_batch_absolute_deviation": classification.max_batch_absolute_deviation,
            "max_batch_relative_deviation": classification.max_batch_relative_deviation,
            "different_realizations_changed_scores": (
                classification.different_realizations_changed_scores
            ),
        },
        "score_vector_order": score_vector_order,
        "observations": {
            "same_realization_runs": [value.tolist() for value in same_realization_runs],
            "batch_size_one": batch_size_one.tolist(),
            "batch_size_eight": batch_size_eight.tolist(),
            "different_realization_scores": [value.tolist() for value in seed_scores],
        },
        "execution_environment": environment,
    }


def execute_g2_audit(
    *,
    config_path: Path,
    project_root: Path,
    device_name: str,
    output_path: Path,
    model_python: Path | None = None,
    worker_path: Path | None = None,
) -> dict[str, Any]:
    """Verify, execute, and immutably write the pre-formal G2 audit only."""
    from dual_uq.dataset.services.proteinmpnn_scoring import (
        build_paired_scoring_projection,
        load_authorized_proteinmpnn_runtime,
    )

    config = load_stage0_scoring_config(config_path, project_root)
    inputs = load_frozen_scoring_inputs(config_path, project_root)
    model_identity = verify_authorized_model(config)
    candidates = build_g2_audit_candidates(inputs)
    manifest_by_id = {
        record["protein_id"]: record for record in inputs.protein_manifest["proteins"]
    }
    projections = {
        protein_id: build_paired_scoring_projection(
            manifest_by_id[protein_id], project_root
        )
        for protein_id in candidates
    }
    replay_runtime = None
    if model_python is None:
        runtime = load_authorized_proteinmpnn_runtime(
            implementation_path=model_identity.implementation_path,
            checkpoint_path=model_identity.checkpoint_path,
            device_name=device_name,
            backbone_noise=config.backbone_noise,
        )
        torch = runtime.torch
        device_display_name: str | None = None
        if str(runtime.device).startswith("cuda") and torch.cuda.is_available():
            device_display_name = str(torch.cuda.get_device_name(runtime.device))
        environment = {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(runtime.device),
            "device_name": device_display_name,
            "process_boundary": "single_environment",
        }
    else:
        response = _execute_g2_model_worker(
            config=config,
            inputs=inputs,
            model_identity=model_identity,
            candidates=candidates,
            projections=projections,
            model_python=model_python,
            worker_path=(
                worker_path
                if worker_path is not None
                else project_root / "scripts/dataset/proteinmpnn_g2_worker.py"
            ),
            project_root=project_root,
            device_name=device_name,
        )
        replay_runtime = _ReplayedG2Runtime(response)
        runtime = replay_runtime
        environment = {
            **dict(response["execution_environment"]),
            "process_boundary": "data_environment_to_model_environment_json_bridge_v1",
        }
    payload = run_g2_audit(
        config=config,
        inputs=inputs,
        model_identity=model_identity,
        projections=projections,
        runtime=runtime,
        environment=environment,
    )
    if replay_runtime is not None:
        replay_runtime.assert_consumed()
    write_status = write_g2_audit(output_path, payload)
    return {
        "status": payload["status"],
        "classification": payload["classification"],
        "formal_repeat_count": payload["formal_repeat_count"],
        "formal_seeds": payload["formal_seeds"],
        "output_path": output_path,
        "write_status": write_status,
    }


class _ReplayedG2Runtime:
    def __init__(self, response: dict[str, Any]) -> None:
        model = response.get("runtime_model")
        calls = response.get("calls")
        if not isinstance(model, dict) or not isinstance(calls, list):
            raise FixedProbeScoringError(
                "invalid_runtime_response", "Model worker response is incomplete"
            )
        self.checkpoint_num_edges = int(model["checkpoint_num_edges"])
        self.checkpoint_noise_level = float(model["checkpoint_noise_level"])
        self._calls = calls
        self._next = 0

    def score_sequences(
        self,
        projection: Any,
        sequences: tuple[str, ...],
        realization: Any,
        *,
        batch_size: int,
    ) -> tuple[Any, ...]:
        from dual_uq.dataset.services.proteinmpnn_scoring import ProteinMPNNScore

        if self._next >= len(self._calls):
            raise FixedProbeScoringError(
                "runtime_response_call_mismatch", "Model worker returned too few calls"
            )
        record = self._calls[self._next]
        self._next += 1
        expected = {
            "protein_id": projection.protein_id,
            "backbone_condition": projection.backbone_condition,
            "seed": realization.seed,
            "repeat_index": realization.repeat_index,
            "realization_fingerprint": realization.fingerprint,
            "batch_size": batch_size,
            "sequence_count": len(sequences),
        }
        observed = {key: record.get(key) for key in expected}
        if observed != expected:
            raise FixedProbeScoringError(
                "runtime_response_call_mismatch",
                f"Model worker call differs: expected {expected}, observed {observed}",
            )
        scores = record.get("scores")
        if not isinstance(scores, list) or len(scores) != len(sequences):
            raise FixedProbeScoringError(
                "runtime_response_score_mismatch", "Model worker score count differs"
            )
        return tuple(
            ProteinMPNNScore(
                score_sum_logp_mask=float(score["score_sum_logp_mask"]),
                score_mean_logp_mask=float(score["score_mean_logp_mask"]),
            )
            for score in scores
        )

    def assert_consumed(self) -> None:
        if self._next != len(self._calls):
            raise FixedProbeScoringError(
                "runtime_response_call_mismatch", "Model worker returned extra calls"
            )


def _execute_g2_model_worker(
    *,
    config: Stage0ScoringConfig,
    inputs: FixedProbeInputs,
    model_identity: VerifiedModelIdentity,
    candidates: dict[str, G2ProteinCandidates],
    projections: dict[str, Any],
    model_python: Path,
    worker_path: Path,
    project_root: Path,
    device_name: str,
) -> dict[str, Any]:
    from dual_uq.dataset.services.proteinmpnn_scoring import make_decoding_realization

    if not model_python.is_file() or not worker_path.is_file():
        raise FixedProbeScoringError(
            "model_worker_unavailable", "Configured model Python/worker is missing"
        )
    proteins = []
    for protein_id, fixture in candidates.items():
        paired = projections[protein_id]
        sequences = [fixture.wt_projected_sequence] + [
            mutant.projected_sequence for mutant in fixture.mutants
        ]
        realizations = []
        for seed in config.audit.audit_seeds:
            realization = make_decoding_realization(
                protein_id=protein_id,
                mask_length=paired.pdb.residue_count,
                repeat_index=seed,
                seed=seed,
                protocol_version=config.protocol_version,
            )
            realizations.append(
                {
                    "repeat_index": realization.repeat_index,
                    "seed": realization.seed,
                    "algorithm": realization.algorithm,
                    "order": list(realization.order),
                    "fingerprint": realization.fingerprint,
                }
            )
        proteins.append(
            {
                "protein_id": protein_id,
                "uniprot_positions": list(paired.pdb.uniprot_positions),
                "sequences": sequences,
                "pdb_coordinates": paired.pdb.coordinates.tolist(),
                "afdb_coordinates": paired.afdb.coordinates.tolist(),
                "realizations": realizations,
            }
        )
    request = {
        "schema_version": "stage0_g2_runtime_request_v1",
        "upstream_identity": {
            "admitted_subset_sha256": inputs.admitted_subset_sha256,
            "protein_manifest_sha256": inputs.protein_manifest_sha256,
            "fixed_probes_sha256": inputs.fixed_probes_sha256,
        },
        "model_identity": {
            "implementation_path": config.model.implementation_path.as_posix(),
            "implementation_commit": model_identity.implementation_commit,
            "checkpoint_path": config.model.checkpoint_path.as_posix(),
            "checkpoint_sha256": model_identity.checkpoint_sha256,
        },
        "scoring_protocol": {
            "protocol_version": config.protocol_version,
            "backbone_noise": config.backbone_noise,
        },
        "proteins": proteins,
    }
    rendered = (
        json.dumps(request, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    with TemporaryDirectory(prefix="dual-uq-stage0-g2-") as temporary:
        temporary_root = Path(temporary)
        request_path = temporary_root / "request.json"
        response_path = temporary_root / "response.json"
        atomic_write_new_bytes(request_path, rendered)
        environment = os.environ.copy()
        source_root = project_root / "src"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else f"{source_root}{os.pathsep}{existing_pythonpath}"
        )
        completed = subprocess.run(
            [
                str(model_python),
                str(worker_path),
                "--request",
                str(request_path),
                "--response",
                str(response_path),
                "--project-root",
                str(project_root),
                "--device",
                device_name,
            ],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not response_path.is_file():
            raise FixedProbeScoringError(
                "model_worker_failed",
                completed.stderr.strip() or "Model worker failed without a response",
            )
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FixedProbeScoringError(
                "invalid_runtime_response", "Model worker response is unreadable"
            ) from exc
        if (
            not isinstance(response, dict)
            or response.get("schema_version") != "stage0_g2_runtime_response_v1"
            or response.get("request_sha256") != sha256_file(request_path)
            or len(response.get("calls", [])) != 36
        ):
            raise FixedProbeScoringError(
                "invalid_runtime_response", "Model worker response binding differs"
            )
        return response


def load_frozen_g2_policy(
    path: Path, *, expected_sha256: str = G2_AUDIT_SHA256
) -> FrozenG2Policy:
    """Load the authoritative G2 result without reinterpreting its repeat policy."""
    if not path.is_file():
        raise FixedProbeScoringError("g2_audit_missing", f"Missing G2 audit: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FixedProbeScoringError(
            "g2_audit_hash_mismatch", "G2 audit SHA256 differs from the frozen value"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "G2 audit is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "G2 audit must be a JSON object"
        )
    numerical = payload.get("numerical_audit")
    observations = payload.get("observations")
    if not isinstance(numerical, dict) or not isinstance(observations, dict):
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "G2 numerical evidence is incomplete"
        )
    try:
        atol = float(numerical["atol"])
        rtol = float(numerical["rtol"])
        same_runs = tuple(
            np.asarray(value, dtype=np.float64)
            for value in observations["same_realization_runs"]
        )
        batch_one = np.asarray(observations["batch_size_one"], dtype=np.float64)
        batch_many = np.asarray(observations["batch_size_eight"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "G2 numerical evidence is invalid"
        ) from exc
    same_pass = (
        len(same_runs) >= 3
        and all(
            value.shape == same_runs[0].shape
            and np.isfinite(value).all()
            and _within_frozen_tolerance(
                same_runs[0], value, atol=atol, rtol=rtol
            )
            for value in same_runs
        )
    )
    batch_pass = (
        batch_one.shape == batch_many.shape
        and batch_one.size > 0
        and np.isfinite(batch_one).all()
        and np.isfinite(batch_many).all()
        and _within_frozen_tolerance(batch_one, batch_many, atol=atol, rtol=rtol)
    )
    expected_contract = (
        payload.get("status") == "PASS"
        and payload.get("classification") == "stochastic_due_to_decoding_order"
        and payload.get("formal_repeat_count") == 30
        and payload.get("formal_seeds") == list(range(30))
        and payload.get("formal_scoring_started") is False
        and payload.get("upstream_identity")
        == {
            "admitted_subset_sha256": ADMITTED_SUBSET_SHA256,
            "protein_manifest_sha256": PROTEIN_MANIFEST_SHA256,
            "fixed_probes_sha256": FIXED_PROBES_SHA256,
        }
        and isinstance(payload.get("model_identity"), dict)
        and payload["model_identity"].get("implementation_commit")
        == AUTHORIZED_IMPLEMENTATION_COMMIT
        and payload["model_identity"].get("checkpoint_sha256")
        == AUTHORIZED_CHECKPOINT_SHA256
        and isinstance(payload.get("scoring_protocol"), dict)
        and payload["scoring_protocol"].get("protocol_version")
        == SCORING_PROTOCOL_VERSION
        and same_pass
        and batch_pass
    )
    if not expected_contract:
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "G2 audit does not authorize formal scoring"
        )
    return FrozenG2Policy(
        audit_path=path,
        audit_sha256=actual_sha256,
        classification=str(payload["classification"]),
        repeat_count=30,
        seeds=tuple(range(30)),
        payload=payload,
    )


def plan_formal_shards(
    *,
    config: Stage0ScoringConfig,
    inputs: FixedProbeInputs,
    policy: FrozenG2Policy,
) -> tuple[FormalShardSpec, ...]:
    """Build the deterministic 8 × 2 × 30 formal shard plan."""
    from dual_uq.dataset.services.proteinmpnn_scoring import (
        DECODING_REALIZATION_ALGORITHM,
        make_decoding_realization,
    )

    if (
        policy.audit_sha256 != G2_AUDIT_SHA256
        or policy.repeat_count != 30
        or policy.seeds != tuple(range(30))
    ):
        raise FixedProbeScoringError(
            "g2_audit_contract_mismatch", "Formal shard plan requires frozen G2 policy"
        )
    grouped = {
        protein_id: group.reset_index(drop=True)
        for protein_id, group in inputs.fixed_probes.groupby("protein_id", sort=False)
    }
    plan: list[FormalShardSpec] = []
    for protein in inputs.protein_manifest["proteins"]:
        protein_id = str(protein["protein_id"])
        candidates = grouped.get(protein_id)
        if candidates is None or candidates.empty:
            raise FixedProbeScoringError(
                "formal_candidate_identity_mismatch",
                f"No fixed probes found for {protein_id}",
            )
        candidate_hashes = tuple(candidates["sequence_hash"].astype(str))
        candidate_identity_sha = sha256_canonical(
            {"sequence_hashes": list(candidate_hashes)}
        )
        mask_length = int(protein["mask_length"])
        for backbone, hash_field in (
            ("PDB", "pdb_backbone_sha256"),
            ("AFDB", "afdb_backbone_sha256"),
        ):
            for repeat_index, seed in enumerate(policy.seeds):
                realization = make_decoding_realization(
                    protein_id=protein_id,
                    mask_length=mask_length,
                    repeat_index=repeat_index,
                    seed=seed,
                    protocol_version=config.protocol_version,
                )
                plan.append(
                    FormalShardSpec(
                        protein_id=protein_id,
                        backbone_condition=backbone,
                        backbone_sha256=str(protein[hash_field]),
                        repeat_index=repeat_index,
                        seed=seed,
                        decoding_realization_sha256=realization.fingerprint,
                        decoding_realization_algorithm=DECODING_REALIZATION_ALGORITHM,
                        candidate_sequence_hashes=candidate_hashes,
                        candidate_identity_sha256=candidate_identity_sha,
                        candidate_count=len(candidate_hashes),
                        mask_length=mask_length,
                        admitted_subset_sha256=inputs.admitted_subset_sha256,
                        protein_manifest_sha256=inputs.protein_manifest_sha256,
                        fixed_probes_sha256=inputs.fixed_probes_sha256,
                        g2_audit_sha256=policy.audit_sha256,
                        checkpoint_sha256=config.model.checkpoint_sha256,
                        implementation_commit=config.model.implementation_commit,
                        scoring_protocol=config.protocol_version,
                    )
                )
    if len(plan) != 480:
        raise FixedProbeScoringError(
            "formal_shard_count_mismatch", "Expected exactly 480 formal shards"
        )
    return tuple(plan)


def formal_shard_binding(spec: FormalShardSpec) -> dict[str, Any]:
    """Return the complete scientific identity binding for one logical shard."""
    return {
        "protein_id": spec.protein_id,
        "backbone_condition": spec.backbone_condition,
        "backbone_sha256": spec.backbone_sha256,
        "repeat_index": spec.repeat_index,
        "seed": spec.seed,
        "decoding_realization_sha256": spec.decoding_realization_sha256,
        "decoding_realization_algorithm": spec.decoding_realization_algorithm,
        "candidate_identity_sha256": spec.candidate_identity_sha256,
        "candidate_count": spec.candidate_count,
        "mask_length": spec.mask_length,
        "admitted_subset_sha256": spec.admitted_subset_sha256,
        "protein_manifest_sha256": spec.protein_manifest_sha256,
        "fixed_probes_sha256": spec.fixed_probes_sha256,
        "g2_audit_sha256": spec.g2_audit_sha256,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "implementation_commit": spec.implementation_commit,
        "scoring_protocol": spec.scoring_protocol,
    }


def _validate_formal_score_record(
    record: Any, *, mask_length: int, context: str
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise FixedProbeScoringError(
            "invalid_formal_shard", f"{context} score record is not an object"
        )
    try:
        score_sum = float(record["score_sum_logp_mask"])
        score_mean = float(record["score_mean_logp_mask"])
        residue_count = int(record["scored_residue_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FixedProbeScoringError(
            "invalid_formal_shard", f"{context} score record is incomplete"
        ) from exc
    if not np.isfinite((score_sum, score_mean)).all():
        raise FixedProbeScoringError("nonfinite_score", f"{context} score is non-finite")
    if residue_count != mask_length:
        raise FixedProbeScoringError(
            "scored_residue_count_mismatch",
            f"{context} scored residue count differs from frozen mask",
        )
    if not np.isclose(
        score_mean,
        score_sum / mask_length,
        atol=1.0e-7,
        rtol=1.0e-7,
    ):
        raise FixedProbeScoringError(
            "formal_score_arithmetic_mismatch",
            f"{context} mean score differs from sum / mask length",
        )
    return record


def validate_formal_shard_payload(
    payload: Any, spec: FormalShardSpec
) -> dict[str, Any]:
    """Validate a shard by content and every frozen binding, never by filename."""
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "stage0_fixed_probe_scoring_shard_v1"
    ):
        raise FixedProbeScoringError(
            "invalid_formal_shard", "Formal shard schema differs"
        )
    if payload.get("binding") != formal_shard_binding(spec):
        raise FixedProbeScoringError(
            "formal_shard_binding_mismatch", "Formal shard binding differs"
        )
    candidate_scores = payload.get("candidate_scores")
    if not isinstance(candidate_scores, list) or len(candidate_scores) != spec.candidate_count:
        raise FixedProbeScoringError(
            "formal_shard_candidate_count_mismatch",
            "Formal shard candidate count differs",
        )
    observed_hashes = tuple(
        record.get("sequence_hash") if isinstance(record, dict) else None
        for record in candidate_scores
    )
    if observed_hashes != spec.candidate_sequence_hashes:
        raise FixedProbeScoringError(
            "formal_shard_candidate_identity_mismatch",
            "Formal shard candidate identity/order differs",
        )
    if sha256_canonical({"sequence_hashes": list(observed_hashes)}) != (
        spec.candidate_identity_sha256
    ):
        raise FixedProbeScoringError(
            "formal_shard_candidate_identity_mismatch",
            "Formal shard candidate identity digest differs",
        )
    _validate_formal_score_record(
        payload.get("wt_score"), mask_length=spec.mask_length, context="WT"
    )
    for index, record in enumerate(candidate_scores):
        _validate_formal_score_record(
            record, mask_length=spec.mask_length, context=f"candidate[{index}]"
        )
    if not isinstance(payload.get("execution_environment"), dict):
        raise FixedProbeScoringError(
            "invalid_formal_shard", "Formal shard lacks execution provenance"
        )
    return payload


def render_formal_shard(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FixedProbeScoringError(
            "invalid_formal_shard", "Formal shard cannot be serialized"
        ) from exc


def write_formal_shard(
    path: Path, payload: dict[str, Any], spec: FormalShardSpec
) -> str:
    validate_formal_shard_payload(payload, spec)
    rendered = render_formal_shard(payload)
    if path.exists():
        if path.read_bytes() != rendered:
            raise FixedProbeScoringError(
                "immutable_formal_shard_conflict", f"Existing shard differs: {path}"
            )
        return "reused_identical"
    atomic_write_new_bytes(path, rendered)
    return "created"


def formal_shard_filename(spec: FormalShardSpec) -> str:
    backbone = spec.backbone_condition.lower()
    return f"{spec.protein_id}__{backbone}__r{spec.repeat_index:02d}.json"


def load_validated_formal_shard(path: Path, spec: FormalShardSpec) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeScoringError(
            "invalid_formal_shard", f"Unable to read shard: {path}"
        ) from exc
    return validate_formal_shard_payload(payload, spec)


def consolidate_formal_scores(
    *,
    shard_payloads: list[dict[str, Any]],
    fixed_probes: pd.DataFrame,
    protein_manifest: list[dict[str, Any]],
    repeat_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolidate validated shards in frozen protein/candidate/backbone/repeat order."""
    if repeat_count <= 0:
        raise FixedProbeScoringError(
            "formal_repeat_count_mismatch", "Formal repeat count must be positive"
        )
    shard_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    for payload in shard_payloads:
        binding = payload.get("binding")
        if not isinstance(binding, dict):
            raise FixedProbeScoringError(
                "invalid_formal_shard", "Shard binding is absent"
            )
        key = (
            str(binding.get("protein_id")),
            str(binding.get("backbone_condition")),
            int(binding.get("repeat_index", -1)),
        )
        if key in shard_map:
            raise FixedProbeScoringError(
                "duplicate_formal_shard", f"Duplicate shard: {key}"
            )
        shard_map[key] = payload

    wt_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for protein in protein_manifest:
        protein_id = str(protein["protein_id"])
        mask_length = int(protein["mask_length"])
        candidates = fixed_probes.loc[
            fixed_probes["protein_id"].astype(str) == protein_id
        ].reset_index(drop=True)
        fingerprints: dict[int, set[str]] = {repeat: set() for repeat in range(repeat_count)}
        for backbone in ("PDB", "AFDB"):
            for repeat in range(repeat_count):
                key = (protein_id, backbone, repeat)
                payload = shard_map.get(key)
                if payload is None:
                    raise FixedProbeScoringError(
                        "formal_shard_count_mismatch", f"Missing shard: {key}"
                    )
                binding = payload["binding"]
                fingerprints[repeat].add(
                    str(binding["decoding_realization_sha256"])
                )
                wt_score = payload["wt_score"]
                wt_rows.append(
                    {
                        "protein_id": protein_id,
                        "backbone_condition": backbone,
                        "backbone_sha256": str(binding["backbone_sha256"]),
                        "repeat_index": repeat,
                        "seed": int(binding["seed"]),
                        "decoding_realization_sha256": str(
                            binding["decoding_realization_sha256"]
                        ),
                        "score_sum_logp_mask": float(
                            wt_score["score_sum_logp_mask"]
                        ),
                        "score_mean_logp_mask": float(
                            wt_score["score_mean_logp_mask"]
                        ),
                        "scored_residue_count": mask_length,
                        "model_checkpoint_sha256": str(
                            binding["checkpoint_sha256"]
                        ),
                        "scoring_protocol": str(binding["scoring_protocol"]),
                    }
                )
        if any(len(values) != 1 for values in fingerprints.values()):
            raise FixedProbeScoringError(
                "formal_realization_pairing_mismatch",
                f"PDB/AFDB realization differs for {protein_id}",
            )
        for candidate_index, candidate in candidates.iterrows():
            for backbone in ("PDB", "AFDB"):
                for repeat in range(repeat_count):
                    payload = shard_map[(protein_id, backbone, repeat)]
                    binding = payload["binding"]
                    score = payload["candidate_scores"][candidate_index]
                    wt_score = payload["wt_score"]
                    raw_rows.append(
                        {
                            "protein_id": protein_id,
                            "sequence_hash": str(candidate["sequence_hash"]),
                            "position": int(candidate["position"]),
                            "wt_aa": str(candidate["wt_aa"]),
                            "mut_aa": str(candidate["mut_aa"]),
                            "backbone_condition": backbone,
                            "backbone_sha256": str(binding["backbone_sha256"]),
                            "repeat_index": repeat,
                            "seed": int(binding["seed"]),
                            "decoding_realization_sha256": str(
                                binding["decoding_realization_sha256"]
                            ),
                            "score_sum_logp_mask": float(
                                score["score_sum_logp_mask"]
                            ),
                            "score_mean_logp_mask": float(
                                score["score_mean_logp_mask"]
                            ),
                            "delta_score_vs_wt": float(
                                score["score_mean_logp_mask"]
                                - wt_score["score_mean_logp_mask"]
                            ),
                            "scored_residue_count": mask_length,
                            "model_checkpoint_sha256": str(
                                binding["checkpoint_sha256"]
                            ),
                            "scoring_protocol": str(binding["scoring_protocol"]),
                        }
                    )
    if len(shard_map) != len(protein_manifest) * 2 * repeat_count:
        raise FixedProbeScoringError(
            "formal_shard_count_mismatch", "Unexpected extra formal shard"
        )
    return pd.DataFrame(wt_rows), pd.DataFrame(raw_rows)


def build_same_state_scoring_null(
    raw: pd.DataFrame, *, repeat_count: int
) -> pd.DataFrame:
    """Summarize only within-backbone technical repeats using population std."""
    group_keys = [
        "protein_id",
        "sequence_hash",
        "position",
        "wt_aa",
        "mut_aa",
        "backbone_condition",
        "backbone_sha256",
        "model_checkpoint_sha256",
        "scoring_protocol",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in raw.groupby(group_keys, sort=False, dropna=False):
        if len(group) != repeat_count or tuple(group["repeat_index"]) != tuple(
            range(repeat_count)
        ):
            raise FixedProbeScoringError(
                "formal_repeat_grid_mismatch", "Same-state repeat grid is incomplete"
            )
        row = dict(zip(group_keys, key, strict=True))
        row.update(
            {
                "n_repeats": repeat_count,
                "scoring_null_type": "empirical_repeat_distribution",
                "std_convention": "population_ddof0",
            }
        )
        for metric in ("score_mean_logp_mask", "delta_score_vs_wt"):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std_population"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_formal_outputs(
    *,
    wt: pd.DataFrame,
    raw: pd.DataFrame,
    null: pd.DataFrame,
    fixed_probes: pd.DataFrame,
    protein_manifest: list[dict[str, Any]],
    repeat_count: int,
) -> None:
    """Apply the release gate to canonical formal score tables."""
    protein_count = len(protein_manifest)
    probe_count = len(fixed_probes)
    expected_wt = protein_count * 2 * repeat_count
    expected_raw = probe_count * 2 * repeat_count
    expected_null = probe_count * 2
    if (len(wt), len(raw), len(null)) != (
        expected_wt,
        expected_raw,
        expected_null,
    ):
        raise FixedProbeScoringError(
            "formal_output_row_count_mismatch", "Formal output row counts differ"
        )
    wt_key = ["protein_id", "backbone_condition", "repeat_index"]
    raw_key = [
        "protein_id",
        "sequence_hash",
        "backbone_condition",
        "repeat_index",
    ]
    null_key = ["protein_id", "sequence_hash", "backbone_condition"]
    if (
        wt.duplicated(wt_key).any()
        or raw.duplicated(raw_key).any()
        or null.duplicated(null_key).any()
    ):
        raise FixedProbeScoringError(
            "duplicate_formal_scientific_key", "Formal scientific keys repeat"
        )
    numeric_columns = [
        "score_sum_logp_mask",
        "score_mean_logp_mask",
    ]
    if not np.isfinite(wt[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise FixedProbeScoringError("nonfinite_score", "WT output contains non-finite scores")
    raw_numeric = numeric_columns + ["delta_score_vs_wt"]
    if not np.isfinite(raw[raw_numeric].to_numpy(dtype=np.float64)).all():
        raise FixedProbeScoringError(
            "nonfinite_score", "Probe output contains non-finite scores"
        )
    null_numeric = [
        column
        for column in null.columns
        if column.endswith(("_mean", "_std_population", "_min", "_max"))
    ]
    if not np.isfinite(null[null_numeric].to_numpy(dtype=np.float64)).all():
        raise FixedProbeScoringError(
            "nonfinite_score", "Same-state null contains non-finite values"
        )
    mask_lengths = {
        str(record["protein_id"]): int(record["mask_length"])
        for record in protein_manifest
    }
    if any(
        int(count) != mask_lengths[str(protein_id)]
        for protein_id, count in zip(
            raw["protein_id"], raw["scored_residue_count"], strict=True
        )
    ) or any(
        int(count) != mask_lengths[str(protein_id)]
        for protein_id, count in zip(
            wt["protein_id"], wt["scored_residue_count"], strict=True
        )
    ):
        raise FixedProbeScoringError(
            "scored_residue_count_mismatch", "Canonical scored residue count differs"
        )
    expected_hashes = set(fixed_probes["sequence_hash"].astype(str))
    if set(raw["sequence_hash"].astype(str)) != expected_hashes:
        raise FixedProbeScoringError(
            "formal_candidate_identity_mismatch", "Canonical candidate join differs"
        )
    if set(wt["repeat_index"]) != set(range(repeat_count)) or set(
        raw["repeat_index"]
    ) != set(range(repeat_count)):
        raise FixedProbeScoringError(
            "formal_repeat_grid_mismatch", "Canonical repeat grid differs"
        )
    pairing = pd.concat(
        [
            wt[
                [
                    "protein_id",
                    "repeat_index",
                    "decoding_realization_sha256",
                ]
            ],
            raw[
                [
                    "protein_id",
                    "repeat_index",
                    "decoding_realization_sha256",
                ]
            ],
        ],
        ignore_index=True,
    )
    if (
        pairing.groupby(["protein_id", "repeat_index"], sort=False)[
            "decoding_realization_sha256"
        ].nunique()
        != 1
    ).any():
        raise FixedProbeScoringError(
            "formal_realization_pairing_mismatch",
            "Canonical rows do not share one realization per protein/repeat",
        )
    if set(null["n_repeats"]) != {repeat_count} or set(
        null["scoring_null_type"]
    ) != {"empirical_repeat_distribution"}:
        raise FixedProbeScoringError(
            "formal_null_contract_mismatch", "Same-state null semantics differ"
        )


def write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    """Atomically create deterministic Parquet and reject conflicting output."""
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
            raise FixedProbeScoringError(
                "immutable_formal_artifact_conflict",
                f"Existing canonical Parquet differs: {path}",
            )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FixedProbeScoringError(
                "immutable_formal_artifact_conflict",
                f"Canonical Parquet appeared concurrently: {path}",
            ) from exc
        return "created"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_formal_protein_request(
    *,
    config: Stage0ScoringConfig,
    model_identity: VerifiedModelIdentity,
    protein: dict[str, Any],
    candidates: pd.DataFrame,
    projection: Any,
    shard_specs: tuple[FormalShardSpec, ...],
) -> dict[str, Any]:
    """Build one model-worker request for a protein without drawing randomness."""
    from dual_uq.dataset.services.proteinmpnn_scoring import (
        make_decoding_realization,
        project_candidate_sequence,
        validate_paired_projection,
    )

    protein_id = str(protein["protein_id"])
    mask_positions = tuple(int(value) for value in protein["mask_positions"])
    paired = validate_paired_projection(
        projection.pdb,
        projection.afdb,
        expected_positions=mask_positions,
    )
    if not shard_specs or any(spec.protein_id != protein_id for spec in shard_specs):
        raise FixedProbeScoringError(
            "formal_shard_binding_mismatch", "Formal protein shard plan differs"
        )
    expected_hashes = tuple(candidates["sequence_hash"].astype(str))
    if any(spec.candidate_sequence_hashes != expected_hashes for spec in shard_specs):
        raise FixedProbeScoringError(
            "formal_candidate_identity_mismatch", "Formal shard candidate set differs"
        )
    canonical = str(protein["canonical_wt_sequence"])
    projected_sequences = []
    for row in candidates.itertuples(index=False):
        projected_sequences.append(
            project_candidate_sequence(
                full_sequence=str(row.full_sequence),
                sequence_hash=str(row.sequence_hash),
                uniprot_positions=mask_positions,
                canonical_wt_sequence=canonical,
                mutation_position=int(row.position),
                wt_aa=str(row.wt_aa),
                mut_aa=str(row.mut_aa),
            )
        )
    tasks = []
    for spec in shard_specs:
        realization = make_decoding_realization(
            protein_id=protein_id,
            mask_length=spec.mask_length,
            repeat_index=spec.repeat_index,
            seed=spec.seed,
            protocol_version=config.protocol_version,
        )
        if realization.fingerprint != spec.decoding_realization_sha256:
            raise FixedProbeScoringError(
                "decoding_realization_mismatch", "Formal shard realization differs"
            )
        tasks.append(
            {
                "binding": formal_shard_binding(spec),
                "order": list(realization.order),
                "output_filename": formal_shard_filename(spec),
            }
        )
    return {
        "schema_version": "stage0_formal_protein_request_v1",
        "model_identity": {
            "implementation_path": config.model.implementation_path.as_posix(),
            "implementation_commit": model_identity.implementation_commit,
            "checkpoint_path": config.model.checkpoint_path.as_posix(),
            "checkpoint_sha256": model_identity.checkpoint_sha256,
        },
        "scoring_protocol": config.protocol_version,
        "protein_id": protein_id,
        "uniprot_positions": list(mask_positions),
        "wt_sequence": paired.pdb.wt_sequence_projection,
        "candidate_sequence_hashes": list(expected_hashes),
        "candidate_projection_sha256": [
            sequence_sha256(sequence) for sequence in projected_sequences
        ],
        "candidate_sequences": projected_sequences,
        "pdb_coordinates": paired.pdb.coordinates.tolist(),
        "afdb_coordinates": paired.afdb.coordinates.tolist(),
        "tasks": tasks,
    }


def _write_immutable_json_artifact(path: Path, payload: dict[str, Any]) -> str:
    rendered = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != rendered:
            raise FixedProbeScoringError(
                "immutable_formal_artifact_conflict",
                f"Existing canonical JSON differs: {path}",
            )
        return "reused_identical"
    atomic_write_new_bytes(path, rendered)
    return "created"


def _run_formal_model_worker(
    *,
    request: dict[str, Any],
    protein_id: str,
    model_python: Path,
    worker_path: Path,
    project_root: Path,
    device_name: str,
    batch_size: int,
    run_root: Path,
    shard_directory: Path,
) -> dict[str, Any]:
    if not model_python.is_file() or not worker_path.is_file():
        raise FixedProbeScoringError(
            "model_worker_unavailable", "Formal model Python/worker is missing"
        )
    rendered = (
        json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    request_sha = sha256_canonical(request)
    request_directory = run_root / "requests"
    response_directory = run_root / "worker_responses"
    request_path = request_directory / f"{protein_id}__{request_sha[:16]}.json"
    response_path = response_directory / f"{protein_id}__{request_sha[:16]}.json"
    if request_path.exists():
        if request_path.read_bytes() != rendered:
            raise FixedProbeScoringError(
                "formal_request_conflict", "Existing formal worker request differs"
            )
    else:
        atomic_write_new_bytes(request_path, rendered)
    response_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_root = project_root / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else f"{source_root}{os.pathsep}{existing_pythonpath}"
    )
    completed = subprocess.run(
        [
            str(model_python),
            str(worker_path),
            "--formal-request",
            str(request_path),
            "--response",
            str(response_path),
            "--shard-directory",
            str(shard_directory),
            "--project-root",
            str(project_root),
            "--device",
            device_name,
            "--batch-size",
            str(batch_size),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0 or not response_path.is_file():
        raise FixedProbeScoringError(
            "model_worker_failed",
            f"Formal model worker failed for {protein_id}",
        )
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedProbeScoringError(
            "invalid_runtime_response", "Formal worker response is unreadable"
        ) from exc
    if (
        not isinstance(response, dict)
        or response.get("schema_version") != "stage0_formal_runtime_response_v1"
        or response.get("request_sha256") != sha256_file(request_path)
        or response.get("status") != "complete"
    ):
        raise FixedProbeScoringError(
            "invalid_runtime_response", "Formal worker response binding differs"
        )
    return response


def _load_all_formal_shards(
    plan: tuple[FormalShardSpec, ...], shard_directory: Path
) -> list[dict[str, Any]]:
    payloads = []
    for spec in plan:
        path = shard_directory / formal_shard_filename(spec)
        if not path.is_file():
            raise FixedProbeScoringError(
                "formal_shard_count_mismatch", f"Missing formal shard: {path.name}"
            )
        payloads.append(load_validated_formal_shard(path, spec))
    return payloads


def _formal_manifest_payload(
    *,
    config: Stage0ScoringConfig,
    inputs: FixedProbeInputs,
    policy: FrozenG2Policy,
    model_identity: VerifiedModelIdentity,
    protein_manifest: list[dict[str, Any]],
    plan: tuple[FormalShardSpec, ...],
    wt_path: Path,
    raw_path: Path,
    null_path: Path,
    wt: pd.DataFrame,
    raw: pd.DataFrame,
    null: pd.DataFrame,
    project_root: Path,
    execution_environments: list[dict[str, Any]],
) -> dict[str, Any]:
    root = project_root.resolve()

    def logical(path: Path) -> str:
        return path.resolve().relative_to(root).as_posix()

    realization_records = [
        {
            "protein_id": spec.protein_id,
            "repeat_index": spec.repeat_index,
            "seed": spec.seed,
            "algorithm": spec.decoding_realization_algorithm,
            "decoding_realization_sha256": spec.decoding_realization_sha256,
        }
        for spec in plan
        if spec.backbone_condition == "PDB"
    ]
    proteins = [
        {
            "protein_id": record["protein_id"],
            "mask_length": int(record["mask_length"]),
            "pdb_backbone_path": record["pdb_backbone_path"],
            "pdb_backbone_sha256": record["pdb_backbone_sha256"],
            "afdb_backbone_path": record["afdb_backbone_path"],
            "afdb_backbone_sha256": record["afdb_backbone_sha256"],
            "probe_count": int(
                (inputs.fixed_probes["protein_id"] == record["protein_id"]).sum()
            ),
            "logical_shards": 60,
        }
        for record in protein_manifest
    ]
    return {
        "schema_version": "stage0_fixed_probe_scoring_manifest_v1",
        "status": "complete",
        "scientific_scope": "stage0_internal_fixed_probe_scoring_only",
        "upstream": {
            "admitted_subset": {
                "path": logical(inputs.admitted_subset_path),
                "sha256": inputs.admitted_subset_sha256,
                "protein_count": 8,
            },
            "protein_manifest": {
                "path": logical(inputs.protein_manifest_path),
                "sha256": inputs.protein_manifest_sha256,
                "total_mask_positions": 1790,
            },
            "fixed_probes": {
                "path": logical(inputs.fixed_probes_path),
                "sha256": inputs.fixed_probes_sha256,
                "row_count": 34010,
            },
            "g2_audit": {
                "path": logical(policy.audit_path),
                "sha256": policy.audit_sha256,
                "classification": policy.classification,
                "repeat_count": policy.repeat_count,
                "seeds": list(policy.seeds),
            },
        },
        "model": {
            "family": config.model.family,
            "role": config.model.role,
            "implementation_path": config.model.implementation_path.as_posix(),
            "implementation_commit": model_identity.implementation_commit,
            "tracked_worktree_integrity": "clean",
            "checkpoint_path": config.model.checkpoint_path.as_posix(),
            "checkpoint_sha256": model_identity.checkpoint_sha256,
        },
        "scoring_protocol": {
            "identity": config.protocol_version,
            "mode": config.mode,
            "domain": config.scoring_domain,
            "backbone_atoms": list(config.backbone_atoms),
            "backbone_noise": config.backbone_noise,
            "numerical_dtype": config.numerical_dtype,
            "residue_index_coordinate_system": config.residue_index_coordinate_system,
            "score_direction": "higher_is_better",
            "sequence_generation": False,
        },
        "decoding_realizations": {
            "protocol": "sha256_ranked_permutation_v1",
            "count": len(realization_records),
            "records": realization_records,
            "pairing": "one fingerprint per protein and repeat shared across PDB/AFDB/WT/candidates",
        },
        "proteins": proteins,
        "execution_environments": execution_environments,
        "outputs": {
            "wt_scores": {
                "path": logical(wt_path),
                "sha256": sha256_file(wt_path),
                "rows": len(wt),
                "bytes": wt_path.stat().st_size,
            },
            "fixed_probe_scores": {
                "path": logical(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(raw),
                "bytes": raw_path.stat().st_size,
            },
            "same_state_null": {
                "path": logical(null_path),
                "sha256": sha256_file(null_path),
                "rows": len(null),
                "bytes": null_path.stat().st_size,
                "n_repeats": policy.repeat_count,
                "std_convention": "population_ddof0",
                "scoring_null_type": "empirical_repeat_distribution",
            },
        },
        "final_counts": {
            "logical_shards": len(plan),
            "wt_rows": len(wt),
            "raw_score_rows": len(raw),
            "same_state_null_rows": len(null),
        },
        "cross_condition_analysis_performed": False,
    }


def execute_formal_scoring(
    *,
    config_path: Path,
    project_root: Path,
    device_name: str,
    model_python: Path,
    worker_path: Path,
    g2_audit_path: Path,
    run_root: Path,
    output_root: Path,
    batch_size: int,
    resume: bool,
) -> dict[str, Any]:
    """Execute/resume all frozen formal shards and materialize validated artifacts."""
    from dual_uq.dataset.services.proteinmpnn_scoring import (
        build_paired_scoring_projection,
    )

    if not resume:
        raise FixedProbeScoringError(
            "formal_resume_required", "Formal execution requires resume semantics"
        )
    if batch_size <= 0:
        raise FixedProbeScoringError(
            "invalid_batch_size", "Formal batch size must be positive"
        )
    config = load_stage0_scoring_config(config_path, project_root)
    inputs = load_frozen_scoring_inputs(config_path, project_root)
    policy = load_frozen_g2_policy(g2_audit_path)
    model_identity = verify_authorized_model(config)
    plan = plan_formal_shards(config=config, inputs=inputs, policy=policy)
    shard_directory = run_root / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    plan_by_protein = {
        str(record["protein_id"]): tuple(
            spec
            for spec in plan
            if spec.protein_id == str(record["protein_id"])
        )
        for record in inputs.protein_manifest["proteins"]
    }
    execution_environments: list[dict[str, Any]] = []
    reused_before_execution = 0
    recomputed_invalid = 0
    for protein in inputs.protein_manifest["proteins"]:
        protein_id = str(protein["protein_id"])
        specs = plan_by_protein[protein_id]
        missing: list[FormalShardSpec] = []
        for spec in specs:
            path = shard_directory / formal_shard_filename(spec)
            if not path.exists():
                missing.append(spec)
                continue
            try:
                payload = load_validated_formal_shard(path, spec)
            except FixedProbeScoringError:
                path.unlink()
                recomputed_invalid += 1
                missing.append(spec)
            else:
                reused_before_execution += 1
                environment = payload.get("execution_environment")
                if isinstance(environment, dict) and environment not in execution_environments:
                    execution_environments.append(environment)
        if not missing:
            print(
                json.dumps(
                    {
                        "event": "protein_resume_complete",
                        "protein_id": protein_id,
                        "reused_shards": len(specs),
                    }
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "protein_scoring_start",
                    "protein_id": protein_id,
                    "missing_shards": len(missing),
                    "candidate_count": missing[0].candidate_count,
                }
            ),
            flush=True,
        )
        candidates = inputs.fixed_probes.loc[
            inputs.fixed_probes["protein_id"] == protein_id
        ].reset_index(drop=True)
        projection = build_paired_scoring_projection(protein, project_root)
        request = build_formal_protein_request(
            config=config,
            model_identity=model_identity,
            protein=protein,
            candidates=candidates,
            projection=projection,
            shard_specs=tuple(missing),
        )
        response = _run_formal_model_worker(
            request=request,
            protein_id=protein_id,
            model_python=model_python,
            worker_path=worker_path,
            project_root=project_root,
            device_name=device_name,
            batch_size=batch_size,
            run_root=run_root,
            shard_directory=shard_directory,
        )
        environment = response.get("execution_environment")
        if isinstance(environment, dict) and environment not in execution_environments:
            execution_environments.append(environment)
        for spec in specs:
            load_validated_formal_shard(
                shard_directory / formal_shard_filename(spec), spec
            )
        print(
            json.dumps(
                {
                    "event": "protein_scoring_complete",
                    "protein_id": protein_id,
                    "logical_shards": len(specs),
                }
            ),
            flush=True,
        )

    shard_payloads = _load_all_formal_shards(plan, shard_directory)
    wt, raw = consolidate_formal_scores(
        shard_payloads=shard_payloads,
        fixed_probes=inputs.fixed_probes,
        protein_manifest=inputs.protein_manifest["proteins"],
        repeat_count=policy.repeat_count,
    )
    null = build_same_state_scoring_null(raw, repeat_count=policy.repeat_count)
    validate_formal_outputs(
        wt=wt,
        raw=raw,
        null=null,
        fixed_probes=inputs.fixed_probes,
        protein_manifest=inputs.protein_manifest["proteins"],
        repeat_count=policy.repeat_count,
    )
    current_hashes = (
        sha256_file(inputs.admitted_subset_path),
        sha256_file(inputs.protein_manifest_path),
        sha256_file(inputs.fixed_probes_path),
        sha256_file(policy.audit_path),
        sha256_file(model_identity.checkpoint_path),
    )
    expected_hashes = (
        inputs.admitted_subset_sha256,
        inputs.protein_manifest_sha256,
        inputs.fixed_probes_sha256,
        policy.audit_sha256,
        model_identity.checkpoint_sha256,
    )
    if current_hashes != expected_hashes:
        raise FixedProbeScoringError(
            "upstream_mutated_during_scoring",
            "A frozen input changed during formal scoring",
        )

    output_root.mkdir(parents=True, exist_ok=True)
    wt_path = output_root / "fixed_probe_wt_scores.parquet"
    raw_path = output_root / "fixed_probe_scores.parquet"
    null_path = output_root / "fixed_probe_scoring_null.parquet"
    manifest_path = output_root / "fixed_probe_scoring_manifest.json"
    write_status = {
        "wt_scores": write_immutable_parquet(wt_path, wt),
        "fixed_probe_scores": write_immutable_parquet(raw_path, raw),
        "same_state_null": write_immutable_parquet(null_path, null),
    }
    manifest = _formal_manifest_payload(
        config=config,
        inputs=inputs,
        policy=policy,
        model_identity=model_identity,
        protein_manifest=inputs.protein_manifest["proteins"],
        plan=plan,
        wt_path=wt_path,
        raw_path=raw_path,
        null_path=null_path,
        wt=wt,
        raw=raw,
        null=null,
        project_root=project_root,
        execution_environments=execution_environments,
    )
    write_status["manifest"] = _write_immutable_json_artifact(
        manifest_path, manifest
    )
    return {
        "status": "PASS",
        "logical_shards": len(plan),
        "reused_before_execution": reused_before_execution,
        "recomputed_invalid": recomputed_invalid,
        "wt_rows": len(wt),
        "raw_score_rows": len(raw),
        "same_state_null_rows": len(null),
        "manifest_path": manifest_path,
        "write_status": write_status,
    }
