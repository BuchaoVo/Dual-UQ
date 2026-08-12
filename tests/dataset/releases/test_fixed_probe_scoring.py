from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from dual_uq.dataset.fixed_probe_scoring import (
    FIXED_PROBES_SHA256,
    PROTEIN_MANIFEST_SHA256,
    FixedProbeInputs,
    FixedProbeScoringError,
    build_g2_audit_candidates,
    classify_g2_observations,
    load_frozen_scoring_inputs,
    load_stage0_scoring_config,
    render_g2_audit,
    run_g2_audit,
    verify_authorized_model,
    write_g2_audit,
)
from dual_uq.dataset.services.proteinmpnn_scoring import (
    DECODING_REALIZATION_ALGORITHM,
    PairedScoringProjection,
    ProteinMPNNScore,
    ScoringBackboneProjection,
    build_paired_scoring_projection,
    make_decoding_realization,
    project_candidate_sequence,
    score_target_log_probs,
    tile_decoding_order,
    validate_paired_projection,
)
from dual_uq.dataset.storage.proteinmpnn import sequence_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/design_baseline/stage0.yaml"
SCORING_CLI_PATH = REPOSITORY_ROOT / "scripts/dataset/score_fixed_probes.py"
MODEL_WORKER_PATH = REPOSITORY_ROOT / "scripts/dataset/proteinmpnn_g2_worker.py"

EXPECTED_CHECKPOINT_SHA256 = (
    "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
)
EXPECTED_IMPLEMENTATION_COMMIT = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
EXPECTED_FORMAL_SEEDS = tuple(range(30))
EXPECTED_SUBSET_SHA256 = (
    "0f29fe5309a63ae0e24eda2ebdacfad861c418165bb2552b6f68e40855d6d3f5"
)
EXPECTED_MANIFEST_SHA256 = (
    "fd34ae871c3d5feebba1dbe38bce24141634764882a9d51db1ce79cd7581d30b"
)
EXPECTED_PROBES_SHA256 = (
    "26dc56745c005c01e78007ad3c3b6dbc59708da23087ac7bd2337acc2ea27ef0"
)


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "stage0.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_scoring_config_binds_exact_authorized_model_and_protocol() -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)

    assert config.model.family == "ProteinMPNN"
    assert config.model.role == "stage0_internal_inverse_folding_scorer"
    assert config.model.implementation_path == Path("third_party/ProteinMPNN")
    assert config.model.implementation_commit == EXPECTED_IMPLEMENTATION_COMMIT
    assert config.model.checkpoint_path == Path(
        "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    )
    assert config.model.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert config.protocol_version == "stage0_fixed_sequence_autoregressive_mask_logp_v1"
    assert config.mode == "fixed_sequence_autoregressive_mask_logp"
    assert config.scoring_domain == "frozen_common_mask_projection"
    assert config.residue_index_coordinate_system == "uniprot_position_1based"
    assert config.backbone_atoms == ("N", "CA", "C", "O")
    assert config.backbone_noise == 0.0
    assert config.numerical_dtype == "float32"
    assert config.audit.atol == 1.0e-6
    assert config.audit.rtol == 1.0e-6
    assert config.audit.audit_seeds == (0, 1, 2, 3)
    assert config.audit.formal_stochastic_seeds == EXPECTED_FORMAL_SEEDS


@pytest.mark.parametrize(
    ("field_path", "bad_value", "expected_code"),
    [
        (("model", "role"), "other", "unauthorized_scoring_model"),
        (("model", "implementation_commit"), "0" * 40, "unauthorized_scoring_model"),
        (("model", "checkpoint_path"), "other.pt", "unauthorized_scoring_model"),
        (("model", "checkpoint_sha256"), "0" * 64, "unauthorized_scoring_model"),
        (("mode",), "generation", "invalid_scoring_protocol"),
        (("scoring_domain",), "full_length", "invalid_scoring_protocol"),
        (("residue_index_coordinate_system",), "compressed", "invalid_scoring_protocol"),
        (("backbone_atoms",), ["CA"], "invalid_scoring_protocol"),
        (("backbone_noise",), 0.1, "invalid_scoring_protocol"),
        (("numerical_dtype",), "float64", "invalid_scoring_protocol"),
        (("audit", "audit_seeds"), [0, 1], "invalid_scoring_protocol"),
        (("audit", "formal_stochastic_seeds"), [0, 1, 2], "invalid_scoring_protocol"),
    ],
)
def test_scoring_config_rejects_semantic_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    bad_value: object,
    expected_code: str,
) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutated = deepcopy(payload)
    target = mutated["scoring"]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = bad_value

    with pytest.raises(FixedProbeScoringError) as caught:
        load_stage0_scoring_config(_write_config(tmp_path, mutated), REPOSITORY_ROOT)

    assert caught.value.code == expected_code


def test_scoring_config_rejects_missing_scoring_section(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload.pop("scoring", None)

    with pytest.raises(FixedProbeScoringError) as caught:
        load_stage0_scoring_config(_write_config(tmp_path, payload), REPOSITORY_ROOT)

    assert caught.value.code == "missing_scoring_config"


def test_scoring_config_rejects_machine_absolute_paths(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["scoring"]["model"]["checkpoint_path"] = "/home/user/model.pt"

    with pytest.raises(FixedProbeScoringError) as caught:
        load_stage0_scoring_config(_write_config(tmp_path, payload), REPOSITORY_ROOT)

    assert caught.value.code == "nonportable_scoring_path"


def test_upstream_loader_binds_exact_frozen_artifacts_and_counts() -> None:
    inputs = load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)

    assert isinstance(inputs, FixedProbeInputs)
    assert inputs.admitted_subset_sha256 == EXPECTED_SUBSET_SHA256
    assert inputs.protein_manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert inputs.fixed_probes_sha256 == EXPECTED_PROBES_SHA256
    assert PROTEIN_MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    assert FIXED_PROBES_SHA256 == EXPECTED_PROBES_SHA256
    assert len(inputs.admitted_records) == 8
    assert len(inputs.protein_manifest["proteins"]) == 8
    assert inputs.protein_manifest["summary"]["total_mask_positions"] == 1790
    assert len(inputs.fixed_probes) == 34010


@pytest.mark.parametrize(
    ("filename", "expected_code"),
    [
        ("stage0_intervention_admitted_v1.jsonl", "admitted_subset_hash_mismatch"),
        ("protein_manifest.json", "protein_manifest_hash_mismatch"),
        ("fixed_probe_candidates.parquet", "fixed_probes_hash_mismatch"),
    ],
)
def test_upstream_loader_blocks_each_hash_drift(
    monkeypatch: pytest.MonkeyPatch, filename: str, expected_code: str
) -> None:
    from dual_uq.dataset import fixed_probe_scoring as scoring_module

    real_hash = scoring_module.sha256_file

    def drift_one(path: Path) -> str:
        if path.name == filename:
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(scoring_module, "sha256_file", drift_one)
    with pytest.raises(FixedProbeScoringError) as caught:
        load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)

    assert caught.value.code == expected_code


def test_upstream_loader_rejects_invalid_scientific_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dual_uq.dataset import fixed_probe_scoring as scoring_module

    real_read_parquet = scoring_module.pd.read_parquet

    def missing_probe(path: Path):
        return real_read_parquet(path).iloc[:-1].copy()

    monkeypatch.setattr(scoring_module.pd, "read_parquet", missing_probe)
    with pytest.raises(FixedProbeScoringError) as caught:
        load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)

    assert caught.value.code == "fixed_probe_count_mismatch"


def test_authorized_model_identity_is_verified_without_loading_torch() -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)

    verified = verify_authorized_model(config)

    assert verified.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert verified.implementation_commit == EXPECTED_IMPLEMENTATION_COMMIT
    assert verified.checkpoint_path == (
        REPOSITORY_ROOT
        / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    ).resolve()


@pytest.mark.parametrize(
    ("hash_value", "commit_value", "expected_code"),
    [
        ("0" * 64, EXPECTED_IMPLEMENTATION_COMMIT, "checkpoint_hash_mismatch"),
        (EXPECTED_CHECKPOINT_SHA256, "0" * 40, "implementation_commit_mismatch"),
    ],
)
def test_authorized_model_identity_blocks_hash_or_commit_drift(
    hash_value: str, commit_value: str, expected_code: str
) -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)

    with pytest.raises(FixedProbeScoringError) as caught:
        verify_authorized_model(
            config,
            checkpoint_hash_reader=lambda _path: hash_value,
            implementation_commit_reader=lambda _path: commit_value,
        )

    assert caught.value.code == expected_code


def test_authorized_model_identity_blocks_dirty_implementation_worktree() -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)

    with pytest.raises(FixedProbeScoringError) as caught:
        verify_authorized_model(
            config,
            checkpoint_hash_reader=lambda _path: EXPECTED_CHECKPOINT_SHA256,
            implementation_commit_reader=lambda _path: EXPECTED_IMPLEMENTATION_COMMIT,
            implementation_clean_reader=lambda _path: False,
        )

    assert caught.value.code == "implementation_worktree_dirty"


def _projection(
    source: str, positions: tuple[int, ...] = (24, 25, 30)
) -> ScoringBackboneProjection:
    coordinates = np.arange(len(positions) * 4 * 3, dtype=np.float32).reshape(
        len(positions), 4, 3
    )
    return ScoringBackboneProjection(
        protein_id="fixture_A__P00001",
        backbone_condition=source,
        uniprot_positions=positions,
        wt_sequence_projection="ACD",
        coordinates=coordinates,
    )


def test_projection_preserves_true_uniprot_positions_without_compression() -> None:
    paired = validate_paired_projection(
        _projection("PDB"), _projection("AFDB"), expected_positions=(24, 25, 30)
    )

    assert paired.pdb.uniprot_positions == (24, 25, 30)
    assert paired.afdb.uniprot_positions == (24, 25, 30)
    assert paired.pdb.residue_count == 3


@pytest.mark.parametrize(
    ("pdb_positions", "afdb_positions", "expected_positions"),
    [
        ((1, 2, 3), (1, 2, 3), (24, 25, 30)),
        ((24, 30, 25), (24, 30, 25), (24, 25, 30)),
        ((24, 25, 30), (24, 25, 31), (24, 25, 30)),
    ],
)
def test_projection_rejects_compressed_reordered_or_unequal_domains(
    pdb_positions: tuple[int, ...],
    afdb_positions: tuple[int, ...],
    expected_positions: tuple[int, ...],
) -> None:
    with pytest.raises(FixedProbeScoringError) as caught:
        validate_paired_projection(
            _projection("PDB", pdb_positions),
            _projection("AFDB", afdb_positions),
            expected_positions=expected_positions,
        )

    assert caught.value.code == "scoring_projection_domain_mismatch"


def test_projection_rejects_missing_or_nonfinite_backbone_coordinates() -> None:
    invalid = _projection("PDB")
    invalid_coordinates = invalid.coordinates.copy()
    invalid_coordinates[1, 2, 0] = np.nan
    invalid = ScoringBackboneProjection(
        protein_id=invalid.protein_id,
        backbone_condition=invalid.backbone_condition,
        uniprot_positions=invalid.uniprot_positions,
        wt_sequence_projection=invalid.wt_sequence_projection,
        coordinates=invalid_coordinates,
    )

    with pytest.raises(FixedProbeScoringError) as caught:
        validate_paired_projection(
            invalid, _projection("AFDB"), expected_positions=(24, 25, 30)
        )

    assert caught.value.code == "invalid_backbone_coordinates"


def test_candidate_projection_validates_hash_and_declared_single_mutation() -> None:
    full_wt = "M" * 23 + "ACD" + "G" * 4
    full_mutant = full_wt[:24] + "E" + full_wt[25:]
    masked = project_candidate_sequence(
        full_sequence=full_mutant,
        sequence_hash=sequence_sha256(full_mutant),
        uniprot_positions=(24, 25, 26),
        canonical_wt_sequence=full_wt,
        mutation_position=25,
        wt_aa="C",
        mut_aa="E",
    )

    assert masked == "AED"


@pytest.mark.parametrize(
    ("hash_override", "mutation_position", "wt_aa", "mut_aa", "expected_code"),
    [
        ("0" * 64, 25, "C", "E", "candidate_sequence_hash_mismatch"),
        (None, 24, "C", "E", "candidate_mutation_mismatch"),
        (None, 25, "A", "E", "candidate_mutation_mismatch"),
        (None, 25, "C", "D", "candidate_mutation_mismatch"),
    ],
)
def test_candidate_projection_rejects_identity_or_mutation_drift(
    hash_override: str | None,
    mutation_position: int,
    wt_aa: str,
    mut_aa: str,
    expected_code: str,
) -> None:
    full_wt = "M" * 23 + "ACD" + "G" * 4
    full_mutant = full_wt[:24] + "E" + full_wt[25:]

    with pytest.raises(FixedProbeScoringError) as caught:
        project_candidate_sequence(
            full_sequence=full_mutant,
            sequence_hash=hash_override or sequence_sha256(full_mutant),
            uniprot_positions=(24, 25, 26),
            canonical_wt_sequence=full_wt,
            mutation_position=mutation_position,
            wt_aa=wt_aa,
            mut_aa=mut_aa,
        )

    assert caught.value.code == expected_code


@pytest.mark.parametrize("protein_id", ["5gv8_A__P83686", "5mn1_A__P00760"])
def test_real_audit_projection_matches_frozen_manifest_domain(protein_id: str) -> None:
    inputs = load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)
    record = next(
        protein for protein in inputs.protein_manifest["proteins"]
        if protein["protein_id"] == protein_id
    )

    paired = build_paired_scoring_projection(record, REPOSITORY_ROOT)

    expected = tuple(record["mask_positions"])
    assert paired.pdb.uniprot_positions == expected
    assert paired.afdb.uniprot_positions == expected
    assert paired.pdb.wt_sequence_projection == paired.afdb.wt_sequence_projection
    assert paired.pdb.coordinates.shape == (len(expected), 4, 3)
    assert paired.afdb.coordinates.shape == (len(expected), 4, 3)


def test_decoding_realization_is_explicit_deterministic_and_seed_distinct() -> None:
    first = make_decoding_realization(
        protein_id="5gv8_A__P83686",
        mask_length=272,
        repeat_index=0,
        seed=0,
        protocol_version="stage0_fixed_sequence_autoregressive_mask_logp_v1",
    )
    repeated = make_decoding_realization(
        protein_id="5gv8_A__P83686",
        mask_length=272,
        repeat_index=0,
        seed=0,
        protocol_version="stage0_fixed_sequence_autoregressive_mask_logp_v1",
    )
    different = make_decoding_realization(
        protein_id="5gv8_A__P83686",
        mask_length=272,
        repeat_index=1,
        seed=1,
        protocol_version="stage0_fixed_sequence_autoregressive_mask_logp_v1",
    )

    assert first == repeated
    assert first.algorithm == DECODING_REALIZATION_ALGORITHM
    assert first.order != different.order
    assert first.fingerprint != different.fingerprint
    assert sorted(first.order) == list(range(272))


def test_decoding_realization_is_reused_across_arbitrary_batch_sizes() -> None:
    realization = make_decoding_realization(
        protein_id="5mn1_A__P00760",
        mask_length=223,
        repeat_index=3,
        seed=3,
        protocol_version="stage0_fixed_sequence_autoregressive_mask_logp_v1",
    )

    singleton = tile_decoding_order(realization, batch_size=1)
    batch = tile_decoding_order(realization, batch_size=8)

    assert singleton.shape == (1, 223)
    assert batch.shape == (8, 223)
    assert np.array_equal(singleton[0], batch[0])
    assert all(np.array_equal(row, singleton[0]) for row in batch)


def test_direct_target_logp_scoring_preserves_higher_is_better_sign() -> None:
    log_probs = np.asarray(
        [
            [[-4.0, -1.0, -2.0], [-0.5, -3.0, -2.0]],
            [[-4.0, -2.0, -1.0], [-0.25, -3.0, -2.0]],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([[1, 0], [2, 0]], dtype=np.int64)

    scores = score_target_log_probs(log_probs, targets)

    assert scores == (
        ProteinMPNNScore(score_sum_logp_mask=-1.5, score_mean_logp_mask=-0.75),
        ProteinMPNNScore(score_sum_logp_mask=-1.25, score_mean_logp_mask=-0.625),
    )
    assert scores[1].score_mean_logp_mask > scores[0].score_mean_logp_mask


@pytest.mark.parametrize(
    ("log_probs", "targets", "expected_code"),
    [
        (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.int64), "invalid_log_probs"),
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.int64),
            "score_shape_mismatch",
        ),
        (
            np.asarray([[[np.nan, -1.0]]], dtype=np.float32),
            np.asarray([[1]], dtype=np.int64),
            "nonfinite_score",
        ),
        (
            np.zeros((1, 1, 2), dtype=np.float32),
            np.asarray([[2]], dtype=np.int64),
            "target_index_out_of_range",
        ),
    ],
)
def test_direct_target_logp_scoring_rejects_invalid_arrays(
    log_probs: np.ndarray, targets: np.ndarray, expected_code: str
) -> None:
    with pytest.raises(FixedProbeScoringError) as caught:
        score_target_log_probs(log_probs, targets)

    assert caught.value.code == expected_code


def test_g2_candidate_fixture_is_exact_and_deterministic() -> None:
    inputs = load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)

    candidates = build_g2_audit_candidates(inputs)

    assert tuple(candidates) == ("5gv8_A__P83686", "5mn1_A__P00760")
    assert tuple(record.position for record in candidates["5gv8_A__P83686"].mutants) == (
        1,
        1,
        1,
        1,
        136,
        136,
        136,
        136,
        272,
        272,
        272,
        272,
    )
    assert tuple(record.position for record in candidates["5mn1_A__P00760"].mutants) == (
        24,
        24,
        24,
        24,
        135,
        135,
        135,
        135,
        246,
        246,
        246,
        246,
    )
    assert sum(len(value.mutants) for value in candidates.values()) == 24
    for fixture in candidates.values():
        for position in fixture.positions:
            rows = [row for row in fixture.mutants if row.position == position]
            assert len(rows) == 4
            assert [row.mut_aa for row in rows] == [
                amino_acid
                for amino_acid in "ACDEFGHIKLMNPQRSTVWY"
                if amino_acid != rows[0].wt_aa
            ][:4]


def _stable_observations() -> tuple[
    tuple[np.ndarray, ...], np.ndarray, np.ndarray, tuple[np.ndarray, ...]
]:
    reference = np.asarray([-1.0, -2.0, -3.0], dtype=np.float64)
    same = (reference.copy(), reference.copy(), reference.copy())
    seeds = tuple(reference.copy() for _ in range(4))
    return same, reference.copy(), reference.copy(), seeds


def test_g2_state_machine_selects_deterministic_repeat_policy() -> None:
    same, batch_one, batch_many, seeds = _stable_observations()

    result = classify_g2_observations(
        same_realization_runs=same,
        batch_size_one=batch_one,
        batch_size_many=batch_many,
        different_realization_scores=seeds,
        realization_fingerprints=("a", "b", "c", "d"),
        atol=1.0e-6,
        rtol=1.0e-6,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "PASS"
    assert result.classification == "deterministic_under_frozen_scoring_protocol"
    assert result.formal_repeat_count == 1
    assert result.formal_seeds == (0,)


def test_g2_tolerance_uses_frozen_reference_side() -> None:
    reference = np.asarray([1.0], dtype=np.float64)
    observed = np.asarray([1.0000020000015], dtype=np.float64)

    result = classify_g2_observations(
        same_realization_runs=(reference, observed, reference.copy()),
        batch_size_one=reference.copy(),
        batch_size_many=reference.copy(),
        different_realization_scores=tuple(reference.copy() for _ in range(4)),
        realization_fingerprints=("a", "b", "c", "d"),
        atol=1.0e-6,
        rtol=1.0e-6,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "BLOCKED"
    assert result.classification == "numerically_unstable_same_realization"


def test_g2_compares_every_pair_of_distinct_realizations() -> None:
    reference = np.asarray([1.0], dtype=np.float64)
    seed_scores = (
        reference.copy(),
        np.asarray([0.91], dtype=np.float64),
        np.asarray([1.09], dtype=np.float64),
        reference.copy(),
    )

    result = classify_g2_observations(
        same_realization_runs=tuple(reference.copy() for _ in range(3)),
        batch_size_one=reference.copy(),
        batch_size_many=reference.copy(),
        different_realization_scores=seed_scores,
        realization_fingerprints=("a", "b", "c", "d"),
        atol=0.0,
        rtol=0.1,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "PASS"
    assert result.classification == "stochastic_due_to_decoding_order"
    assert result.formal_repeat_count == 30


def test_g2_state_machine_selects_30_repeats_for_order_sensitivity() -> None:
    same, batch_one, batch_many, seeds = _stable_observations()
    changed = list(seeds)
    changed[3] = changed[3].copy()
    changed[3][1] -= 1.0e-3

    result = classify_g2_observations(
        same_realization_runs=same,
        batch_size_one=batch_one,
        batch_size_many=batch_many,
        different_realization_scores=tuple(changed),
        realization_fingerprints=("a", "b", "c", "d"),
        atol=1.0e-6,
        rtol=1.0e-6,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "PASS"
    assert result.classification == "stochastic_due_to_decoding_order"
    assert result.formal_repeat_count == 30
    assert result.formal_seeds == tuple(range(30))


def test_g2_state_machine_blocks_same_realization_instability() -> None:
    same, batch_one, batch_many, seeds = _stable_observations()
    unstable = list(same)
    unstable[2] = unstable[2].copy()
    unstable[2][0] -= 1.0e-3

    result = classify_g2_observations(
        same_realization_runs=tuple(unstable),
        batch_size_one=batch_one,
        batch_size_many=batch_many,
        different_realization_scores=seeds,
        realization_fingerprints=("a", "b", "c", "d"),
        atol=1.0e-6,
        rtol=1.0e-6,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "BLOCKED"
    assert result.classification == "numerically_unstable_same_realization"
    assert result.formal_repeat_count == 0
    assert result.formal_seeds == ()


def test_g2_state_machine_fails_batch_size_dependence() -> None:
    same, batch_one, batch_many, seeds = _stable_observations()
    batch_many[2] -= 1.0e-3

    result = classify_g2_observations(
        same_realization_runs=same,
        batch_size_one=batch_one,
        batch_size_many=batch_many,
        different_realization_scores=seeds,
        realization_fingerprints=("a", "b", "c", "d"),
        atol=1.0e-6,
        rtol=1.0e-6,
        formal_stochastic_seeds=tuple(range(30)),
    )

    assert result.status == "FAIL"
    assert result.classification == "batch_size_dependent_scoring"
    assert result.formal_repeat_count == 0


def test_g2_state_machine_rejects_duplicate_realization_fingerprints() -> None:
    same, batch_one, batch_many, seeds = _stable_observations()

    with pytest.raises(FixedProbeScoringError) as caught:
        classify_g2_observations(
            same_realization_runs=same,
            batch_size_one=batch_one,
            batch_size_many=batch_many,
            different_realization_scores=seeds,
            realization_fingerprints=("same", "same", "c", "d"),
            atol=1.0e-6,
            rtol=1.0e-6,
            formal_stochastic_seeds=tuple(range(30)),
        )

    assert caught.value.code == "duplicate_decoding_realization"


def test_g2_audit_write_is_atomic_immutable_and_deterministic(tmp_path: Path) -> None:
    payload = {
        "schema_version": "stage0_g2_audit_v1",
        "status": "PASS",
        "classification": "stochastic_due_to_decoding_order",
        "formal_repeat_count": 30,
        "formal_seeds": list(range(30)),
    }
    output = tmp_path / "g2_audit.json"

    assert write_g2_audit(output, payload) == "created"
    assert output.read_bytes() == render_g2_audit(payload)
    assert write_g2_audit(output, payload) == "reused_identical"

    with pytest.raises(FixedProbeScoringError) as caught:
        write_g2_audit(output, {**payload, "formal_repeat_count": 1})

    assert caught.value.code == "immutable_g2_audit_conflict"


class _SeedSensitiveFakeRuntime:
    checkpoint_num_edges = 48
    checkpoint_noise_level = 0.2

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score_sequences(
        self,
        projection: ScoringBackboneProjection,
        sequences: tuple[str, ...],
        realization,
        *,
        batch_size: int,
    ) -> tuple[ProteinMPNNScore, ...]:
        self.calls.append(
            {
                "protein_id": projection.protein_id,
                "backbone": projection.backbone_condition,
                "seed": realization.seed,
                "fingerprint": realization.fingerprint,
                "batch_size": batch_size,
                "sequence_count": len(sequences),
            }
        )
        backbone_shift = 0.25 if projection.backbone_condition == "PDB" else -0.25
        seed_shift = realization.seed * 1.0e-3
        return tuple(
            ProteinMPNNScore(
                score_sum_logp_mask=float(
                    -len(sequence) + backbone_shift + seed_shift + index * 1.0e-4
                ),
                score_mean_logp_mask=float(
                    (-len(sequence) + backbone_shift + seed_shift + index * 1.0e-4)
                    / len(sequence)
                ),
            )
            for index, sequence in enumerate(sequences)
        )


def _synthetic_audit_projections(inputs: FixedProbeInputs) -> dict[str, PairedScoringProjection]:
    projections = {}
    for protein in inputs.protein_manifest["proteins"]:
        protein_id = protein["protein_id"]
        if protein_id not in {"5gv8_A__P83686", "5mn1_A__P00760"}:
            continue
        positions = tuple(protein["mask_positions"])
        projected = "".join(
            protein["canonical_wt_sequence"][position - 1] for position in positions
        )
        coordinates = np.zeros((len(positions), 4, 3), dtype=np.float32)
        projections[protein_id] = PairedScoringProjection(
            pdb=ScoringBackboneProjection(
                protein_id=protein_id,
                backbone_condition="PDB",
                uniprot_positions=positions,
                wt_sequence_projection=projected,
                coordinates=coordinates,
            ),
            afdb=ScoringBackboneProjection(
                protein_id=protein_id,
                backbone_condition="AFDB",
                uniprot_positions=positions,
                wt_sequence_projection=projected,
                coordinates=coordinates,
            ),
        )
    return projections


def test_run_g2_audit_executes_only_the_frozen_audit_matrix() -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)
    inputs = load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)
    model = verify_authorized_model(config)
    runtime = _SeedSensitiveFakeRuntime()

    payload = run_g2_audit(
        config=config,
        inputs=inputs,
        model_identity=model,
        projections=_synthetic_audit_projections(inputs),
        runtime=runtime,
        environment={"execution_environment": "test"},
    )

    assert payload["schema_version"] == "stage0_g2_audit_v1"
    assert payload["status"] == "PASS"
    assert payload["classification"] == "stochastic_due_to_decoding_order"
    assert payload["formal_repeat_count"] == 30
    assert payload["formal_seeds"] == list(range(30))
    assert payload["audit_fixture"]["protein_count"] == 2
    assert payload["audit_fixture"]["mutant_count"] == 24
    assert payload["audit_fixture"]["wt_count"] == 2
    assert payload["model_identity"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert payload["upstream_identity"]["fixed_probes_sha256"] == EXPECTED_PROBES_SHA256
    assert len(runtime.calls) == 36
    assert {call["sequence_count"] for call in runtime.calls} == {13}
    assert {call["batch_size"] for call in runtime.calls} == {1, 8}
    assert {call["seed"] for call in runtime.calls} == {0, 1, 2, 3}
    for protein_id in ("5gv8_A__P83686", "5mn1_A__P00760"):
        seed_zero = [
            call for call in runtime.calls
            if call["protein_id"] == protein_id and call["seed"] == 0
        ]
        assert len({call["fingerprint"] for call in seed_zero}) == 1


def test_run_g2_audit_rejects_missing_or_extra_projection_identity() -> None:
    config = load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)
    inputs = load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)
    model = verify_authorized_model(config)
    projections = _synthetic_audit_projections(inputs)
    projections.pop("5mn1_A__P00760")

    with pytest.raises(FixedProbeScoringError) as caught:
        run_g2_audit(
            config=config,
            inputs=inputs,
            model_identity=model,
            projections=projections,
            runtime=_SeedSensitiveFakeRuntime(),
            environment={"execution_environment": "test"},
        )

    assert caught.value.code == "g2_projection_identity_mismatch"


def test_scoring_cli_preserves_explicit_audit_only_mode() -> None:
    help_result = subprocess.run(
        [sys.executable, str(SCORING_CLI_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert help_result.returncode == 0
    assert "--audit-only" in help_result.stdout
    assert "--formal" in help_result.stdout

    conflicting = subprocess.run(
        [
            sys.executable,
            str(SCORING_CLI_PATH),
            "--audit-only",
            "--formal",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert conflicting.returncode != 0


def test_dataset_namespace_import_does_not_require_optional_pyarrow() -> None:
    code = """
import importlib.abc
import sys

class BlockPyArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise ModuleNotFoundError('blocked optional pyarrow')
        return None

sys.meta_path.insert(0, BlockPyArrow())
import dual_uq.dataset as dataset
assert dataset.ProjectPaths is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_scoring_service_imports_without_dataset_data_dependencies() -> None:
    code = """
import importlib.abc
import sys

BLOCKED = {'pandas', 'pyarrow', 'yaml', 'scipy'}

class BlockDataDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in BLOCKED:
            raise ModuleNotFoundError(f'blocked data dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockDataDependencies())
from dual_uq.dataset.services import proteinmpnn_scoring
assert proteinmpnn_scoring.load_authorized_proteinmpnn_runtime is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_worker_is_a_thin_request_response_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(MODEL_WORKER_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--request" in result.stdout
    assert "--response" in result.stdout
    assert "--device" in result.stdout
