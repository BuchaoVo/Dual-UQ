from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dual_uq.models import proteinmpnn as canonical
from dual_uq.models import proteinmpnn_generation as generation
from dual_uq.models.proteinmpnn import (
    AUTHORIZED_CHECKPOINT_SHA256,
    AUTHORIZED_IMPLEMENTATION_COMMIT,
    ProteinMPNNAdapter,
    ProteinMPNNStructureInput,
)
from dual_uq.models.scoring import ScorerBinding


def fixture_binding() -> ScorerBinding:
    return ScorerBinding(
        scorer_id="ProteinMPNN",
        implementation_id="fixture-implementation",
        checkpoint_id="1" * 64,
        score_contract_id="fixture-generation-v1",
    )


def fixture_generation_request(**changes: object) -> generation.GenerationRequest:
    values: dict[str, object] = {
        "protein_id": "fixture_A__P00001",
        "backbone_condition": "PDB",
        "structure_sha256": "2" * 64,
        "canonical_positions": (2, 5, 8),
        "wt_sequence_projection": "ACD",
        "temperature": 0.1,
        "sample_index": 0,
        "seed": 0,
        "sample_class": "paired",
        "decoding_realization": "3" * 64,
    }
    values.update(changes)
    return generation.GenerationRequest(**values)  # type: ignore[arg-type]


def test_seed_plan_has_128_paired_and_128_condition_disjoint_independent_samples():
    plan = generation.generation_seed_plan()

    assert len(plan) == 256
    assert [row.sample_index for row in plan] == list(range(256))
    assert [row.sample_class for row in plan[:128]] == ["paired"] * 128
    assert [row.sample_class for row in plan[128:]] == ["independent"] * 128
    assert {generation.independent_seed("PDB", i) for i in range(128)}.isdisjoint(
        {generation.independent_seed("AFDB", i) for i in range(128)}
    )


def test_generation_record_rejects_sequence_hash_mismatch():
    request = fixture_generation_request()

    with pytest.raises(generation.GenerationContractError, match="sequence_hash"):
        generation.GeneratedSequenceRecord(
            request,
            "A" * len(request.canonical_positions),
            "0" * 64,
            fixture_binding(),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"canonical_positions": (2, 2, 8)}, "canonical_positions"),
        ({"canonical_positions": (0, 5, 8)}, "canonical_positions"),
        ({"structure_sha256": "2" * 63}, "structure_sha256"),
        ({"decoding_realization": "3" * 63}, "decoding_realization"),
        ({"wt_sequence_projection": "AC"}, "sequence-domain lengths"),
        ({"temperature": 0.2}, "temperature"),
        ({"sample_class": "paired", "sample_index": 128}, "sample_index"),
        ({"sample_class": "independent", "sample_index": 127}, "sample_index"),
    ],
)
def test_generation_request_rejects_invalid_contract_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(generation.GenerationContractError, match=message):
        fixture_generation_request(**changes)


def test_generation_request_rejects_malformed_projected_sequence() -> None:
    with pytest.raises(ValueError, match="standard uppercase"):
        fixture_generation_request(wt_sequence_projection="ACX")


def test_generation_request_rejects_unknown_condition_before_seed_validation() -> None:
    with pytest.raises(
        generation.GenerationContractError,
        match="backbone_condition must be PDB or AFDB",
    ):
        fixture_generation_request(backbone_condition="other", seed=999)


def test_generation_records_are_immutable_and_bind_to_the_request_domain():
    request = fixture_generation_request()
    record = generation.GeneratedSequenceRecord(
        request,
        "ACD",
        generation.sequence_sha256("ACD"),
        fixture_binding(),
    )

    assert record.request is request
    with pytest.raises(FrozenInstanceError):
        record.sequence = "AAA"  # type: ignore[misc]


@pytest.mark.parametrize("sample_index", (-1, 128, True))
def test_seed_functions_reject_out_of_range_indices(sample_index: int) -> None:
    with pytest.raises(generation.GenerationContractError, match="sample_index"):
        generation.paired_seed(sample_index)
    with pytest.raises(generation.GenerationContractError, match="sample_index"):
        generation.independent_seed("PDB", sample_index)


def test_independent_seed_rejects_unknown_condition():
    with pytest.raises(generation.GenerationContractError, match="condition"):
        generation.independent_seed("other", 0)


class _FakeGenerationModel:
    def __init__(self, torch: object) -> None:
        self.torch = torch
        self.calls: list[dict[str, object]] = []

    def sample(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "S": self.torch.tensor([[0, 1, 2, 3]]),
            "decoding_order": self.torch.tensor([[3, 2, 1, 0]]),
        }


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)
        self.device = "cpu"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def detach(self) -> _FakeTensor:
        return self

    def to(self, *_args: object, **_kwargs: object) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def tolist(self) -> list[object]:
        return self.values.tolist()

    def reshape(self, *shape: int) -> _FakeTensor:
        return _FakeTensor(self.values.reshape(*shape))


class _FakeTorch:
    float32 = "float32"
    long = "long"

    @staticmethod
    def tensor(values: object, **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(values)

    @staticmethod
    def zeros(shape: tuple[int, ...], **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(np.zeros(shape))

    @staticmethod
    def ones(shape: tuple[int, ...], **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(np.ones(shape))

    @staticmethod
    def arange(stop: int) -> _FakeTensor:
        return _FakeTensor(np.arange(stop))

    @staticmethod
    def randn(shape: tuple[int, ...], **_kwargs: object) -> _FakeTensor:
        return _FakeTensor(np.zeros(shape))

    @staticmethod
    def inference_mode() -> object:
        return nullcontext()

    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None


def fixture_generation_structure(
    **changes: object,
) -> generation.ProteinMPNNGenerationStructure:
    projection = ProteinMPNNStructureInput(
        protein_id="fixture_A__P00001",
        backbone_condition="PDB",
        uniprot_positions=(2, 5),
        wt_sequence_projection="CE",
        coordinates=np.asarray(
            [
                [[1.0, 1.0, 1.0]] * 4,
                [[3.0, 3.0, 3.0]] * 4,
            ]
        ),
        structure_sha256="2" * 64,
    )
    values: dict[str, object] = {
        "projection": projection,
        "chain_id": "A",
        "chain_sequence": "ACDE",
        "chain_coordinates": np.asarray(
            [
                [[0.0, 0.0, 0.0]] * 4,
                [[1.0, 1.0, 1.0]] * 4,
                [[2.0, 2.0, 2.0]] * 4,
                [[3.0, 3.0, 3.0]] * 4,
            ]
        ),
        "chain_positions": (2, 4),
    }
    values.update(changes)
    return generation.ProteinMPNNGenerationStructure(**values)  # type: ignore[arg-type]


def _unused_tied_featurize(*_args: object) -> tuple[object, ...]:
    raise AssertionError("generation test did not provide its tied_featurize fake")


def fixture_generation_adapter(
    torch: object,
    *,
    tied_featurize: Callable[..., tuple[object, ...]] = _unused_tied_featurize,
) -> tuple[object, ProteinMPNNAdapter]:
    model = _FakeGenerationModel(torch)
    adapter = ProteinMPNNAdapter(
        model=model,
        torch=torch,
        device="cpu",
        checkpoint_num_edges=48,
        checkpoint_noise_level=0.2,
        implementation_id=AUTHORIZED_IMPLEMENTATION_COMMIT,
        checkpoint_id=AUTHORIZED_CHECKPOINT_SHA256,
        _tied_featurize=tied_featurize,
    )
    # Test-only registration of a fake: production registration happens in the loader.
    canonical._AUTHORIZED_PROTEINMPNN_ADAPTERS[id(adapter)] = adapter
    return model, adapter


def test_generation_adapter_rejects_authorized_ids_without_loader_provenance() -> None:
    torch = _FakeTorch()
    adapter = ProteinMPNNAdapter(
        model=_FakeGenerationModel(torch),
        torch=torch,
        device="cpu",
        checkpoint_num_edges=48,
        checkpoint_noise_level=0.2,
        implementation_id=AUTHORIZED_IMPLEMENTATION_COMMIT,
        checkpoint_id=AUTHORIZED_CHECKPOINT_SHA256,
        _tied_featurize=_unused_tied_featurize,
    )

    with pytest.raises(generation.GenerationContractError, match="loader provenance"):
        generation.ProteinMPNNGenerationAdapter(adapter, batch_size=1)


def test_generation_adapter_rejects_loader_provenance_without_featurizer() -> None:
    torch = _FakeTorch()
    _model, adapter = fixture_generation_adapter(torch)
    object.__setattr__(adapter, "_tied_featurize", None)

    with pytest.raises(generation.GenerationContractError, match="tied_featurize"):
        generation.ProteinMPNNGenerationAdapter(adapter, batch_size=1)


def test_generation_adapter_binds_exact_structure_and_common_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    featurize_calls: list[tuple[object, object, object]] = []

    def fake_tied_featurize(
        batch: object, device: object, chain_dict: object, fixed_position_dict: object
    ) -> tuple[object, ...]:
        featurize_calls.append((batch, chain_dict, fixed_position_dict))
        return (
            torch.zeros((1, 4, 4, 3)),
            torch.tensor([[0, 1, 2, 3]]),
            torch.ones((1, 4)),
            np.asarray([4]),
            torch.ones((1, 4)),
            torch.ones((1, 4), dtype=torch.long),
            [["A"]],
            [[]],
            [["A"]],
            [[4]],
            torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            torch.zeros((1, 4, 21)),
            torch.arange(4).reshape(1, 4),
            torch.zeros((1, 4, 3)),
            [[]],
            torch.zeros((1, 4)),
            torch.zeros((1, 4, 21)),
            torch.zeros((1, 4, 21)),
            torch.zeros((1, 4, 21)),
            torch.ones(4),
        )

    model, adapter = fixture_generation_adapter(torch, tied_featurize=fake_tied_featurize)
    request = fixture_generation_request(canonical_positions=(2, 5), wt_sequence_projection="CE")

    records = generation.ProteinMPNNGenerationAdapter(adapter, batch_size=1).generate(
        request, fixture_generation_structure(), n_samples=1
    )

    assert [record.sequence for record in records] == ["CE"]
    assert featurize_calls == [
        (
            [
                {
                    "name": request.protein_id,
                    "seq": "ACDE",
                    "seq_chain_A": "ACDE",
                    "coords_chain_A": {
                        "N_chain_A": [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0],
                        ],
                        "CA_chain_A": [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0],
                        ],
                        "C_chain_A": [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0],
                        ],
                        "O_chain_A": [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0],
                        ],
                    },
                }
            ],
            {request.protein_id: (["A"], [])},
            {request.protein_id: {"A": [1, 3]}},
        )
    ]
    assert model.calls[0]["chain_M_pos"].tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_generation_adapter_uses_the_fixed_seed_plan_for_multiple_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()

    def fake_tied_featurize(*_args: object) -> tuple[object, ...]:
        return (
            torch.zeros((1, 4, 4, 3)),
            torch.tensor([[0, 1, 2, 3]]),
            torch.ones((1, 4)),
            np.asarray([4]),
            torch.ones((1, 4)),
            torch.ones((1, 4), dtype=torch.long),
            [["A"]],
            [[]],
            [["A"]],
            [[4]],
            torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
            torch.zeros((1, 4, 21)),
            torch.arange(4).reshape(1, 4),
            torch.zeros((1, 4, 3)),
            [[]],
            torch.zeros((1, 4)),
            torch.zeros((1, 4, 21)),
            torch.zeros((1, 4, 21)),
            torch.zeros((1, 4, 21)),
            torch.ones(4),
        )

    _model, adapter = fixture_generation_adapter(torch, tied_featurize=fake_tied_featurize)

    records = generation.ProteinMPNNGenerationAdapter(adapter, batch_size=1).generate(
        fixture_generation_request(canonical_positions=(2, 5), wt_sequence_projection="CE"),
        fixture_generation_structure(),
        n_samples=2,
    )

    assert [(record.request.sample_index, record.request.seed) for record in records] == [
        (0, 0),
        (1, 1),
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"chain_positions": (2, 3), "chain_sequence": "ACEE"},
            "projection coordinates",
        ),
        (
            {
                "projection": ProteinMPNNStructureInput(
                    protein_id="other",
                    backbone_condition="PDB",
                    uniprot_positions=(2, 5),
                    wt_sequence_projection="CE",
                    coordinates=np.asarray(
                        [
                            [[1.0, 1.0, 1.0]] * 4,
                            [[3.0, 3.0, 3.0]] * 4,
                        ]
                    ),
                    structure_sha256="2" * 64,
                )
            },
            "protein_id",
        ),
    ],
)
def test_generation_adapter_rejects_structure_mismatched_to_request(
    changes: dict[str, object], message: str
) -> None:
    torch = _FakeTorch()
    _model, adapter = fixture_generation_adapter(torch)

    with pytest.raises(generation.GenerationContractError, match=message):
        generation.ProteinMPNNGenerationAdapter(adapter, batch_size=1).generate(
            fixture_generation_request(canonical_positions=(2, 5), wt_sequence_projection="CE"),
            fixture_generation_structure(**changes),
            n_samples=1,
        )


def test_generation_structure_rejects_sub_float32_projection_coordinate_drift() -> None:
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0]] * 4,
            [[1.0 + 1.0e-9, 1.0, 1.0]] * 4,
            [[2.0, 2.0, 2.0]] * 4,
            [[3.0, 3.0, 3.0]] * 4,
        ]
    )

    with pytest.raises(generation.GenerationContractError, match="projection coordinates"):
        fixture_generation_structure(chain_coordinates=coordinates)


def test_generation_batch_preserves_true_numbering_gaps_as_chain_breaks() -> None:
    structure = fixture_generation_structure(
        chain_sequence="ACDEFG",
        chain_coordinates=np.asarray([[[float(i)] * 3] * 4 for i in range(6)]),
        chain_positions=(2, 5),
        projection=ProteinMPNNStructureInput(
            protein_id="fixture_A__P00001",
            backbone_condition="PDB",
            uniprot_positions=(2, 5),
            wt_sequence_projection="CF",
            coordinates=np.asarray([[[1.0] * 3] * 4, [[4.0] * 3] * 4]),
            structure_sha256="2" * 64,
        ),
        chain_segment_ends=(2, 6),
    )

    batch = generation.ProteinMPNNGenerationAdapter._batch(structure)

    assert batch["seq_chain_A"] == "AC"
    assert batch["seq_chain_A_1"] == "DEFG"
    assert batch["seq"] == "ACDEFG"
