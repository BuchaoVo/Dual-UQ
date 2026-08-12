from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.dataset import fixed_probe_scoring as scoring
from dual_uq.dataset.services import proteinmpnn_scoring as model_scoring

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs/experiments/design_baseline/stage0.yaml"
G2_PATH = REPOSITORY_ROOT / "runs/design_baseline/stage0-2a/g2_audit.json"
EXPECTED_G2_SHA256 = (
    "eb4acae7e4018006156f80f40e50b1c99bebd1af3c0b44f46fcee603152f34ff"
)


def test_frozen_g2_policy_requires_exact_hash_and_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = scoring.load_frozen_g2_policy(G2_PATH)

    assert policy.audit_sha256 == EXPECTED_G2_SHA256
    assert policy.classification == "stochastic_due_to_decoding_order"
    assert policy.repeat_count == 30
    assert policy.seeds == tuple(range(30))

    real_hash = scoring.sha256_file
    monkeypatch.setattr(
        scoring,
        "sha256_file",
        lambda path: "0" * 64 if path == G2_PATH else real_hash(path),
    )
    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.load_frozen_g2_policy(G2_PATH)
    assert caught.value.code == "g2_audit_hash_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "deterministic_under_frozen_scoring_protocol"),
        ("formal_repeat_count", 1),
        ("formal_seeds", [0]),
        ("status", "FAIL"),
    ],
)
def test_frozen_g2_policy_rejects_contract_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(G2_PATH.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "g2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.load_frozen_g2_policy(path, expected_sha256=scoring.sha256_file(path))

    assert caught.value.code == "g2_audit_contract_mismatch"


def test_formal_plan_contains_exact_480_bound_shards() -> None:
    config = scoring.load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)
    inputs = scoring.load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)
    policy = scoring.load_frozen_g2_policy(G2_PATH)

    plan = scoring.plan_formal_shards(config=config, inputs=inputs, policy=policy)

    assert len(plan) == 480
    assert {(item.backbone_condition, item.repeat_index) for item in plan} == {
        (backbone, repeat)
        for backbone in ("PDB", "AFDB")
        for repeat in range(30)
    }
    assert all(item.seed == item.repeat_index for item in plan)
    assert all(item.candidate_count > 0 for item in plan)
    assert all(item.fixed_probes_sha256 == scoring.FIXED_PROBES_SHA256 for item in plan)
    per_protein = pd.Series([item.protein_id for item in plan]).value_counts()
    assert set(per_protein.index) == {
        record["protein_id"] for record in inputs.protein_manifest["proteins"]
    }
    assert set(per_protein.index) == set(inputs.fixed_probes["protein_id"])
    assert (per_protein == 60).all()


def _synthetic_spec() -> object:
    candidate_hashes = ("c" * 64, "d" * 64)
    return scoring.FormalShardSpec(
        protein_id="fixture_A__P00001",
        backbone_condition="PDB",
        backbone_sha256="a" * 64,
        repeat_index=0,
        seed=0,
        decoding_realization_sha256="b" * 64,
        decoding_realization_algorithm="sha256_ranked_permutation_v1",
        candidate_sequence_hashes=candidate_hashes,
        candidate_identity_sha256=scoring.sha256_canonical(
            {"sequence_hashes": list(candidate_hashes)}
        ),
        candidate_count=2,
        mask_length=3,
        admitted_subset_sha256=scoring.ADMITTED_SUBSET_SHA256,
        protein_manifest_sha256=scoring.PROTEIN_MANIFEST_SHA256,
        fixed_probes_sha256=scoring.FIXED_PROBES_SHA256,
        g2_audit_sha256=EXPECTED_G2_SHA256,
        checkpoint_sha256=scoring.AUTHORIZED_CHECKPOINT_SHA256,
        implementation_commit=scoring.AUTHORIZED_IMPLEMENTATION_COMMIT,
        scoring_protocol=scoring.SCORING_PROTOCOL_VERSION,
    )


def _synthetic_payload(spec: object) -> dict[str, object]:
    return {
        "schema_version": "stage0_fixed_probe_scoring_shard_v1",
        "binding": scoring.formal_shard_binding(spec),
        "wt_score": {
            "score_sum_logp_mask": -6.0,
            "score_mean_logp_mask": -2.0,
            "scored_residue_count": 3,
        },
        "candidate_scores": [
            {
                "sequence_hash": "c" * 64,
                "score_sum_logp_mask": -5.7,
                "score_mean_logp_mask": -1.9,
                "scored_residue_count": 3,
            },
            {
                "sequence_hash": "d" * 64,
                "score_sum_logp_mask": -6.3,
                "score_mean_logp_mask": -2.1,
                "scored_residue_count": 3,
            },
        ],
        "execution_environment": {"device": "cuda:1"},
    }


def test_shard_validator_accepts_exact_complete_payload() -> None:
    spec = _synthetic_spec()
    payload = _synthetic_payload(spec)

    validated = scoring.validate_formal_shard_payload(payload, spec)

    assert validated is payload


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("binding", "formal_shard_binding_mismatch"),
        ("incomplete", "formal_shard_candidate_count_mismatch"),
        ("fingerprint", "formal_shard_binding_mismatch"),
        ("nonfinite", "nonfinite_score"),
        ("residue_count", "scored_residue_count_mismatch"),
        ("candidate_order", "formal_shard_candidate_identity_mismatch"),
    ],
)
def test_shard_validator_rejects_drift(
    mutation: str, expected_code: str
) -> None:
    spec = _synthetic_spec()
    payload = _synthetic_payload(spec)
    if mutation == "binding":
        payload["binding"]["checkpoint_sha256"] = "f" * 64
    elif mutation == "incomplete":
        payload["candidate_scores"].pop()
    elif mutation == "fingerprint":
        payload["binding"]["decoding_realization_sha256"] = "f" * 64
    elif mutation == "nonfinite":
        payload["candidate_scores"][0]["score_mean_logp_mask"] = float("nan")
    elif mutation == "residue_count":
        payload["candidate_scores"][0]["scored_residue_count"] = 2
    elif mutation == "candidate_order":
        payload["candidate_scores"].reverse()

    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.validate_formal_shard_payload(payload, spec)

    assert caught.value.code == expected_code


def test_shard_write_is_immutable_and_reusable(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    payload = _synthetic_payload(spec)
    path = tmp_path / "shard.json"

    assert scoring.write_formal_shard(path, payload, spec) == "created"
    assert scoring.write_formal_shard(path, payload, spec) == "reused_identical"

    changed = json.loads(json.dumps(payload))
    changed["candidate_scores"][0]["score_sum_logp_mask"] = -4.0
    changed["candidate_scores"][0]["score_mean_logp_mask"] = -4.0 / 3.0
    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.write_formal_shard(path, changed, spec)
    assert caught.value.code == "immutable_formal_shard_conflict"


def _synthetic_consolidation_inputs() -> tuple[
    list[dict[str, object]], pd.DataFrame, list[dict[str, object]]
]:
    specs = []
    payloads = []
    for backbone, backbone_sha in (("PDB", "1" * 64), ("AFDB", "2" * 64)):
        for repeat in range(2):
            spec = replace(
                _synthetic_spec(),
                backbone_condition=backbone,
                backbone_sha256=backbone_sha,
                repeat_index=repeat,
                seed=repeat,
                decoding_realization_sha256=str(repeat + 3) * 64,
            )
            payload = _synthetic_payload(spec)
            payload["binding"] = scoring.formal_shard_binding(spec)
            payload["wt_score"]["score_mean_logp_mask"] = -2.0 - repeat * 0.1
            payload["wt_score"]["score_sum_logp_mask"] = (
                payload["wt_score"]["score_mean_logp_mask"] * 3
            )
            for index, row in enumerate(payload["candidate_scores"]):
                row["score_mean_logp_mask"] = -1.9 - index * 0.2 - repeat * 0.1
                row["score_sum_logp_mask"] = row["score_mean_logp_mask"] * 3
            specs.append(spec)
            payloads.append(payload)
    probes = pd.DataFrame(
        {
            "protein_id": ["fixture_A__P00001", "fixture_A__P00001"],
            "sequence_hash": ["c" * 64, "d" * 64],
            "position": [1, 2],
            "wt_aa": ["A", "C"],
            "mut_aa": ["C", "D"],
        }
    )
    proteins = [
        {
            "protein_id": "fixture_A__P00001",
            "mask_length": 3,
            "pdb_backbone_sha256": "1" * 64,
            "afdb_backbone_sha256": "2" * 64,
        }
    ]
    return payloads, probes, proteins


def test_consolidation_pairs_exact_wt_and_has_deterministic_order() -> None:
    payloads, probes, proteins = _synthetic_consolidation_inputs()

    wt, raw = scoring.consolidate_formal_scores(
        shard_payloads=payloads,
        fixed_probes=probes,
        protein_manifest=proteins,
        repeat_count=2,
    )

    assert len(wt) == 4
    assert len(raw) == 8
    assert list(wt[["backbone_condition", "repeat_index"]].itertuples(index=False, name=None)) == [
        ("PDB", 0),
        ("PDB", 1),
        ("AFDB", 0),
        ("AFDB", 1),
    ]
    assert list(raw[["sequence_hash", "backbone_condition", "repeat_index"]].itertuples(index=False, name=None)) == [
        ("c" * 64, "PDB", 0),
        ("c" * 64, "PDB", 1),
        ("c" * 64, "AFDB", 0),
        ("c" * 64, "AFDB", 1),
        ("d" * 64, "PDB", 0),
        ("d" * 64, "PDB", 1),
        ("d" * 64, "AFDB", 0),
        ("d" * 64, "AFDB", 1),
    ]
    assert np.allclose(
        raw["delta_score_vs_wt"],
        raw["score_mean_logp_mask"]
        - raw.merge(
            wt,
            on=["protein_id", "backbone_condition", "repeat_index"],
            suffixes=("", "_wt"),
        )["score_mean_logp_mask_wt"],
    )


def test_consolidation_rejects_mismatched_realization_pairing() -> None:
    payloads, probes, proteins = _synthetic_consolidation_inputs()
    payloads[2]["binding"]["decoding_realization_sha256"] = "9" * 64

    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.consolidate_formal_scores(
            shard_payloads=payloads,
            fixed_probes=probes,
            protein_manifest=proteins,
            repeat_count=2,
        )

    assert caught.value.code == "formal_realization_pairing_mismatch"


def test_same_state_null_uses_population_standard_deviation() -> None:
    payloads, probes, proteins = _synthetic_consolidation_inputs()
    _wt, raw = scoring.consolidate_formal_scores(
        shard_payloads=payloads,
        fixed_probes=probes,
        protein_manifest=proteins,
        repeat_count=2,
    )

    null = scoring.build_same_state_scoring_null(raw, repeat_count=2)

    assert len(null) == 4
    assert set(null["n_repeats"]) == {2}
    assert set(null["scoring_null_type"]) == {"empirical_repeat_distribution"}
    first = null.iloc[0]
    assert first["score_mean_logp_mask_std_population"] == pytest.approx(0.05)
    assert first["delta_score_vs_wt_std_population"] == pytest.approx(0.0)


def test_final_validator_rejects_duplicate_nonfinite_and_wrong_counts() -> None:
    payloads, probes, proteins = _synthetic_consolidation_inputs()
    wt, raw = scoring.consolidate_formal_scores(
        shard_payloads=payloads,
        fixed_probes=probes,
        protein_manifest=proteins,
        repeat_count=2,
    )
    null = scoring.build_same_state_scoring_null(raw, repeat_count=2)

    scoring.validate_formal_outputs(
        wt=wt,
        raw=raw,
        null=null,
        fixed_probes=probes,
        protein_manifest=proteins,
        repeat_count=2,
    )

    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(scoring.FixedProbeScoringError):
        scoring.validate_formal_outputs(
            wt=wt,
            raw=duplicated,
            null=null,
            fixed_probes=probes,
            protein_manifest=proteins,
            repeat_count=2,
        )


def test_immutable_parquet_write_reuses_identical_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.parquet"
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})

    assert scoring.write_immutable_parquet(path, frame) == "created"
    assert scoring.write_immutable_parquet(path, frame) == "reused_identical"

    with pytest.raises(scoring.FixedProbeScoringError) as caught:
        scoring.write_immutable_parquet(path, frame.iloc[::-1].reset_index(drop=True))
    assert caught.value.code == "immutable_formal_artifact_conflict"


class _FormalFakeRuntime:
    implementation_id = scoring.AUTHORIZED_IMPLEMENTATION_COMMIT
    checkpoint_id = scoring.AUTHORIZED_CHECKPOINT_SHA256

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def score_sequences(self, projection, sequences, realization, *, batch_size):
        self.calls.append(
            (projection.backbone_condition, realization.repeat_index, len(sequences))
        )
        return tuple(
            model_scoring.ProteinMPNNScore(
                score_sum_logp_mask=-3.0 - index * 0.3,
                score_mean_logp_mask=-1.0 - index * 0.1,
            )
            for index, _sequence in enumerate(sequences)
        )


def _formal_runtime_request() -> dict[str, object]:
    canonical_sequence = "AAA"
    projected_sequences = ["CAA", "ACA"]
    candidate_hashes = [
        scoring.sequence_sha256(sequence) for sequence in projected_sequences
    ]
    projection_hashes = [
        scoring.sequence_sha256(sequence) for sequence in projected_sequences
    ]
    candidate_identity = scoring.sha256_canonical(
        {"sequence_hashes": candidate_hashes}
    )
    tasks = []
    realization = model_scoring.make_decoding_realization(
        protein_id="fixture_A__P00001",
        mask_length=3,
        repeat_index=0,
        seed=0,
        protocol_version=scoring.SCORING_PROTOCOL_VERSION,
    )
    order = list(realization.order)
    realization_sha256 = realization.fingerprint
    for backbone, backbone_sha in (("PDB", "1" * 64), ("AFDB", "2" * 64)):
        spec = replace(
            _synthetic_spec(),
            backbone_condition=backbone,
            backbone_sha256=backbone_sha,
            decoding_realization_sha256=realization_sha256,
            candidate_sequence_hashes=tuple(candidate_hashes),
            candidate_identity_sha256=candidate_identity,
        )
        tasks.append(
            {
                "binding": scoring.formal_shard_binding(spec),
                "order": order,
                "output_filename": scoring.formal_shard_filename(spec),
            }
        )
    coordinates = np.arange(36, dtype=np.float32).reshape(3, 4, 3).tolist()
    return {
        "schema_version": "stage0_formal_protein_request_v2",
        "model_identity": {
            "implementation_commit": scoring.AUTHORIZED_IMPLEMENTATION_COMMIT,
            "checkpoint_sha256": scoring.AUTHORIZED_CHECKPOINT_SHA256,
        },
        "scoring_protocol": scoring.SCORING_PROTOCOL_VERSION,
        "protein_id": "fixture_A__P00001",
        "uniprot_positions": [1, 2, 3],
        "wt_sequence": canonical_sequence,
        "canonical_wt_sequence": canonical_sequence,
        "canonical_sequence_sha256": scoring.sequence_sha256(canonical_sequence),
        "common_mask_binding": scoring.sha256_canonical(
            {
                "protein_id": "fixture_A__P00001",
                "canonical_positions": [1, 2, 3],
                "canonical_sequence_sha256": scoring.sequence_sha256(
                    canonical_sequence
                ),
            }
        ),
        "candidate_sequence_hashes": candidate_hashes,
        "candidate_projection_sha256": projection_hashes,
        "candidate_sequences": projected_sequences,
        "candidate_records": [
            {
                "sequence_hash": candidate_hashes[0],
                "full_sequence": projected_sequences[0],
                "position": 1,
                "wt_aa": "A",
                "mut_aa": "C",
            },
            {
                "sequence_hash": candidate_hashes[1],
                "full_sequence": projected_sequences[1],
                "position": 2,
                "wt_aa": "A",
                "mut_aa": "C",
            },
        ],
        "pdb_coordinates": coordinates,
        "afdb_coordinates": coordinates,
        "tasks": tasks,
    }


def _legacy_formal_runtime_request() -> dict[str, object]:
    request = _formal_runtime_request()
    request["schema_version"] = "stage0_formal_protein_request_v1"
    for field in (
        "canonical_wt_sequence",
        "canonical_sequence_sha256",
        "common_mask_binding",
        "candidate_records",
    ):
        request.pop(field)
    return request


def test_legacy_v1_request_requires_explicit_scientific_enrichment() -> None:
    request = _legacy_formal_runtime_request()

    assert set(request) == {
        "schema_version",
        "model_identity",
        "scoring_protocol",
        "protein_id",
        "uniprot_positions",
        "wt_sequence",
        "candidate_sequence_hashes",
        "candidate_projection_sha256",
        "candidate_sequences",
        "pdb_coordinates",
        "afdb_coordinates",
        "tasks",
    }
    with pytest.raises(model_scoring.ProteinMPNNScoringError) as caught:
        model_scoring.adapt_formal_runtime_requests(request)

    assert caught.value.code == "legacy_formal_request_requires_enrichment"


def test_formal_runtime_adapter_builds_two_normalized_requests() -> None:
    request = _formal_runtime_request()

    views = model_scoring.adapt_formal_runtime_requests(request)

    assert len(views) == 2
    assert views[0].request.candidate_collection is (
        views[1].request.candidate_collection
    )
    assert [view.request.condition.condition_id for view in views] == [
        "PDB",
        "AFDB",
    ]
    assert all(
        view.request.result_count == 3
        and view.projection.structure_sha256
        == view.request.condition.structure_sha256
        and view.request.scoring_domain_id == request["common_mask_binding"]
        for view in views
    )
    assert [view.output_filename for view in views] == [
        task["output_filename"] for task in request["tasks"]
    ]


def test_formal_runtime_uses_generic_dispatch_and_legacy_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _formal_runtime_request()
    real_dispatch = model_scoring.execute_score_request
    calls = []

    def checked_dispatch(scorer, normalized_request):
        calls.append(normalized_request)
        return real_dispatch(scorer, normalized_request)

    monkeypatch.setattr(model_scoring, "execute_score_request", checked_dispatch)
    model_scoring.run_formal_runtime_request(
        request=request,
        runtime=_FormalFakeRuntime(),
        shard_directory=tmp_path,
        batch_size=8,
        execution_environment={"device": "test"},
    )

    assert [call.condition.condition_id for call in calls] == ["PDB", "AFDB"]
    for task in request["tasks"]:
        payload = json.loads((tmp_path / task["output_filename"]).read_text())
        assert set(payload) == {
            "schema_version",
            "binding",
            "wt_score",
            "candidate_scores",
            "execution_environment",
        }
        assert payload["binding"] == task["binding"]
        assert payload["wt_score"] == {
            "score_mean_logp_mask": -1.0,
            "score_sum_logp_mask": -3.0,
            "scored_residue_count": 3,
        }
        assert [row["sequence_hash"] for row in payload["candidate_scores"]] == (
            request["candidate_sequence_hashes"]
        )


def test_formal_model_request_writes_one_valid_shard_per_task(tmp_path: Path) -> None:
    runtime = _FormalFakeRuntime()
    request = _formal_runtime_request()

    result = model_scoring.run_formal_runtime_request(
        request=request,
        runtime=runtime,
        shard_directory=tmp_path,
        batch_size=8,
        execution_environment={"device": "test"},
    )

    assert result["completed_shards"] == 2
    assert result["candidate_count"] == 2
    assert runtime.calls == [("PDB", 0, 3), ("AFDB", 0, 3)]
    for task in request["tasks"]:
        payload = json.loads((tmp_path / task["output_filename"]).read_text())
        assert payload["binding"] == task["binding"]
        assert len(payload["candidate_scores"]) == 2
        assert payload["wt_score"]["score_mean_logp_mask"] == -1.0


def test_formal_model_request_rejects_realization_or_candidate_drift(
    tmp_path: Path,
) -> None:
    request = _formal_runtime_request()
    changed_order = [1, 0, 2]
    request["tasks"][1]["order"] = changed_order
    request["tasks"][1]["binding"]["decoding_realization_sha256"] = (
        model_scoring.sha256_bytes(
            np.asarray(changed_order, dtype="<i8").tobytes()
        )
    )

    with pytest.raises(model_scoring.ProteinMPNNScoringError) as caught:
        model_scoring.run_formal_runtime_request(
            request=request,
            runtime=_FormalFakeRuntime(),
            shard_directory=tmp_path,
            batch_size=8,
            execution_environment={"device": "test"},
        )

    assert caught.value.code == "formal_realization_pairing_mismatch"


def test_formal_protein_request_binds_projected_candidates_and_60_tasks() -> None:
    config = scoring.load_stage0_scoring_config(CONFIG_PATH, REPOSITORY_ROOT)
    inputs = scoring.load_frozen_scoring_inputs(CONFIG_PATH, REPOSITORY_ROOT)
    policy = scoring.load_frozen_g2_policy(G2_PATH)
    model = scoring.verify_authorized_model(config)
    plan = scoring.plan_formal_shards(config=config, inputs=inputs, policy=policy)
    protein = inputs.protein_manifest["proteins"][0]
    candidates = inputs.fixed_probes.loc[
        inputs.fixed_probes["protein_id"] == protein["protein_id"]
    ].reset_index(drop=True)
    projection = model_scoring.build_paired_scoring_projection(
        protein, REPOSITORY_ROOT
    )

    request = scoring.build_formal_protein_request(
        config=config,
        model_identity=model,
        protein=protein,
        candidates=candidates,
        projection=projection,
        shard_specs=tuple(
            spec for spec in plan if spec.protein_id == protein["protein_id"]
        ),
    )

    assert request["schema_version"] == "stage0_formal_protein_request_v2"
    assert request["protein_id"] == protein["protein_id"]
    assert len(request["candidate_sequences"]) == 5168
    assert len(request["candidate_sequence_hashes"]) == 5168
    assert len(request["tasks"]) == 60
    assert request["canonical_sequence_sha256"] == protein[
        "canonical_sequence_sha256"
    ]
    assert request["common_mask_binding"] == scoring.sha256_canonical(
        {
            "protein_id": protein["protein_id"],
            "canonical_positions": protein["mask_positions"],
            "canonical_sequence_sha256": protein["canonical_sequence_sha256"],
        }
    )
    assert len(request["candidate_records"]) == 5168
    assert {task["binding"]["backbone_condition"] for task in request["tasks"]} == {
        "PDB",
        "AFDB",
    }
    views = model_scoring.adapt_formal_runtime_requests(request)
    assert len(views) == 60
    assert all(view.request.result_count == 5169 for view in views)


def test_formal_cli_mode_is_available_without_rerunning_g2() -> None:
    cli = REPOSITORY_ROOT / "scripts/dataset/score_fixed_probes.py"
    result = __import__("subprocess").run(
        [__import__("sys").executable, str(cli), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--formal" in result.stdout
    assert "--resume" in result.stdout
    assert "--batch-size" in result.stdout
