from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dual_uq.models.kwdesign import (
    KWDesignAdapter,
    KWDesignContractError,
    KWDesignStructureInput,
    make_kwdesign_record,
    project_kwdesign_log_probabilities,
    select_recycle_log_probabilities,
    validate_kwdesign_load_keys,
    validate_kwdesign_structure,
    validate_kwdesign_weights,
)


def test_kwdesign_structure_requires_complete_n_ca_c_o_coordinates() -> None:
    validate_kwdesign_structure(
        KWDesignStructureInput(
            coordinates=np.zeros((4, 4, 3), dtype=np.float32),
            sequence="ACDE",
            title="protein-a",
        )
    )

    with pytest.raises(KWDesignContractError, match="shape"):
        validate_kwdesign_structure(
            KWDesignStructureInput(
                coordinates=np.zeros((4, 3, 3), dtype=np.float32),
                sequence="ACDE",
                title="protein-a",
            )
        )


def test_kwdesign_projection_renormalizes_the_canonical_amino_acids() -> None:
    token_to_id = {aa: index + 4 for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    log_probabilities = np.full((2, 33), -20.0)
    log_probabilities[:, 0] = 0.0
    log_probabilities[0, token_to_id["A"]] = 2.0
    log_probabilities[1, token_to_id["C"]] = 3.0

    projected = project_kwdesign_log_probabilities(log_probabilities, token_to_id)

    assert projected.shape == (2, 20)
    assert np.allclose(projected.sum(axis=1), 1.0)
    assert projected[0].argmax() == 0
    assert projected[1].argmax() == 1


def test_kwdesign_projection_rejects_missing_canonical_token() -> None:
    with pytest.raises(KWDesignContractError, match="canonical"):
        project_kwdesign_log_probabilities(
            np.zeros((1, 33)), {aa: index for index, aa in enumerate("ACDE")}
        )


def test_kwdesign_record_uses_single_chain_official_fields() -> None:
    coordinates = np.arange(36, dtype=np.float32).reshape(3, 4, 3)

    record = make_kwdesign_record(
        KWDesignStructureInput(coordinates=coordinates, sequence="ACD", title="case-a")
    )

    assert record["title"] == "case-a"
    assert record["seq"] == "ACD"
    assert np.array_equal(record["N"], coordinates[:, 0])
    assert np.array_equal(record["O"], coordinates[:, 3])
    assert np.array_equal(record["chain_mask"], np.ones(3))
    assert np.array_equal(record["chain_encoding"], np.ones(3))


def test_kwdesign_selects_each_residue_from_its_most_confident_recycle() -> None:
    log_probabilities = np.array(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[-5.0, -6.0], [-7.0, -8.0]],
        ]
    )
    confidences = np.array([[0.8, 0.1], [0.2, 0.9]])

    selected = select_recycle_log_probabilities(log_probabilities, confidences)

    assert np.array_equal(selected, np.array([[-1.0, -2.0], [-7.0, -8.0]]))


def test_kwdesign_weight_validation_reports_both_frozen_assets(tmp_path: Path) -> None:
    base = tmp_path / "base.pth"
    tuner = tmp_path / "tuner.pth"
    base.write_bytes(b"base")
    tuner.write_bytes(b"tuner")

    binding = validate_kwdesign_weights(
        base_checkpoint=base,
        tuner_checkpoint=tuner,
        expected_base_sha256="cae662172fd450bb0cd710a769079c05bfc5d8e35efa6576edc7d0377afdd4a2",
        expected_tuner_sha256="fc7a2965fb8ca464e96c1816374803096712abaec3f948e00fdef748d81c5b3d",
    )

    assert binding["base_checkpoint"] == base.as_posix()
    assert binding["tuner_checkpoint"] == tuner.as_posix()


def test_kwdesign_adapter_binding_declares_seeded_geometry_semantics() -> None:
    adapter = KWDesignAdapter(
        model=object(),
        torch_module=object(),
        device="cpu",
        featurizer=lambda records: records,
        token_to_id={aa: index for index, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")},
        implementation_id="revision",
        base_checkpoint_id="base.pth",
        tuner_checkpoint_id="tuner.pth",
        inference_seed=111,
    )

    binding = adapter.binding()

    assert binding["semantic_class"] == "L0_composite"
    assert binding["probe_semantics"] == "geometry_only_seeded_msa"
    assert binding["inference_seed"] == 111
    assert binding["aa_order"] == "ACDEFGHIKLMNPQRSTVWY"


def test_kwdesign_load_audit_allows_only_unused_rotary_position_parameters() -> None:
    audit = validate_kwdesign_load_keys(
        missing_keys=["Design1.design_model.decoder.weight"],
        unexpected_keys=[
            "Design1.GNNTuning.GNNTuning.DesignEmbed.position_ids",
            "Design1.GNNTuning.GNNTuning.ESMEmbed.position_embeddings.weight",
        ],
    )

    assert audit == {
        "missing_pretrained_parameter_count": 1,
        "ignored_rotary_position_parameter_count": 2,
    }

    with pytest.raises(KWDesignContractError, match="unexpected parameters"):
        validate_kwdesign_load_keys([], ["Design1.GNNTuning.GNNTuning.ReadOut.weight"])
    with pytest.raises(KWDesignContractError, match="missing tuning parameters"):
        validate_kwdesign_load_keys(["Design1.GNNTuning.GNNTuning.ReadOut.weight"], [])
