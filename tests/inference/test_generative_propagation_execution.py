from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dual_uq.inference.generative_propagation import (
    GenerationCondition,
    GenerationExecutionError,
    GenerationInputs,
    _binding,
    _full_chain_generation_structure,
    generate_clean_cohort,
    load_generation_records,
)
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput, sequence_sha256
from dual_uq.models.proteinmpnn_generation import (
    GeneratedSequenceRecord,
    GenerationRequest,
    ProteinMPNNGenerationStructure,
)
from dual_uq.models.scoring import ScorerBinding


def _structure(protein_id: str, condition: str) -> ProteinMPNNGenerationStructure:
    projection = ProteinMPNNStructureInput(
        protein_id=protein_id,
        backbone_condition=condition,
        uniprot_positions=(2, 5, 8),
        wt_sequence_projection="ACD",
        coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        structure_sha256=("1" if condition == "PDB" else "2") * 64,
    )
    return ProteinMPNNGenerationStructure(
        projection=projection,
        chain_id="A",
        chain_sequence="ACD",
        chain_coordinates=np.zeros((3, 4, 3), dtype=np.float32),
        chain_positions=(1, 2, 3),
    )


def _inputs(proteins: int = 1) -> GenerationInputs:
    conditions = []
    for index in range(proteins):
        protein_id = f"fixture_{index:03d}"
        for condition in ("PDB", "AFDB"):
            conditions.append(GenerationCondition.from_structure(_structure(protein_id, condition)))
    return GenerationInputs(tuple(conditions), expected_protein_count=proteins)


class _FakeAdapter:
    binding = ScorerBinding(
        scorer_id="ProteinMPNN",
        implementation_id="fixture-implementation",
        checkpoint_id="3" * 64,
        score_contract_id="proteinmpnn_generation_temperature_0.1_v1",
    )

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(
        self, request: GenerationRequest, structure: ProteinMPNNGenerationStructure, *, n_samples: int
    ) -> tuple[GeneratedSequenceRecord, ...]:
        self.calls.append((request.protein_id, request.backbone_condition))
        result = []
        for sample_index in range(n_samples):
            paired = sample_index < 128
            seed = sample_index if paired else (
                1_000_000 + sample_index - 128
                if request.backbone_condition == "PDB"
                else 2_000_000 + sample_index - 128
            )
            sequence = "ACD"
            item = GenerationRequest(
                protein_id=request.protein_id,
                backbone_condition=request.backbone_condition,
                structure_sha256=request.structure_sha256,
                canonical_positions=request.canonical_positions,
                wt_sequence_projection=request.wt_sequence_projection,
                temperature=0.1,
                sample_index=sample_index,
                seed=seed,
                sample_class="paired" if paired else "independent",
                decoding_realization=f"{sample_index + 4:064x}",
            )
            result.append(
                GeneratedSequenceRecord(item, sequence, sequence_sha256(sequence), self.binding)
            )
        return tuple(result)


def test_complete_inputs_require_the_exact_68_by_2_identity_grid() -> None:
    inputs = _inputs(proteins=68)

    assert len(inputs.conditions) == 136
    assert {(row.protein_id, row.backbone_condition) for row in inputs.conditions} == {
        (f"fixture_{index:03d}", condition)
        for index in range(68)
        for condition in ("PDB", "AFDB")
    }

    with pytest.raises(GenerationExecutionError, match="exact PDB/AFDB pair"):
        GenerationInputs(inputs.conditions[:-1], expected_protein_count=68)


def test_fake_execution_materializes_validated_shards_and_reuses_them(tmp_path: Path) -> None:
    inputs = _inputs()
    adapter = _FakeAdapter()

    first = generate_clean_cohort(inputs, adapter, output_root=tmp_path, resume=True)
    second = generate_clean_cohort(inputs, adapter, output_root=tmp_path, resume=True)

    assert len(first.records) == 512
    assert len(second.records) == 512
    assert adapter.calls == [("fixture_000", "PDB"), ("fixture_000", "AFDB")]
    assert first.executed_conditions == 2
    assert second.reused_conditions == 2
    assert {record.request.sample_index for record in first.records} == set(range(256))
    assert {record.request.sample_class for record in first.records} == {"paired", "independent"}
    assert sorted(tmp_path.glob("shards/*/*.json"))
    assert len(load_generation_records(tmp_path)) == 512


def test_malformed_existing_shard_blocks_execution_without_forward_call(tmp_path: Path) -> None:
    inputs = _inputs()
    adapter = _FakeAdapter()
    condition = inputs.conditions[0]
    shard = tmp_path / "shards" / condition.identity[:2] / f"{condition.identity}.json"
    shard.parent.mkdir(parents=True)
    shard.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(GenerationExecutionError, match="malformed generation shard"):
        generate_clean_cohort(inputs, adapter, output_root=tmp_path, resume=True)

    assert adapter.calls == []


def test_binding_retains_full_chain_context_and_projection_mapping() -> None:
    chain_coordinates = np.arange(96, dtype=np.float32).reshape(8, 4, 3)
    projection = ProteinMPNNStructureInput(
        protein_id="full_chain_fixture",
        backbone_condition="PDB",
        uniprot_positions=(11, 14, 18),
        wt_sequence_projection="AEI",
        coordinates=chain_coordinates[[0, 3, 7]],
        structure_sha256="4" * 64,
    )
    structure = ProteinMPNNGenerationStructure(
        projection=projection,
        chain_id="B",
        chain_sequence="ACDEFGHI",
        chain_coordinates=chain_coordinates,
        chain_positions=(1, 4, 8),
    )
    condition = GenerationCondition.from_structure(structure)
    binding = _binding(condition)

    assert binding["chain_id"] == "B"
    assert binding["chain_sequence_sha256"] == sequence_sha256("ACDEFGHI")
    assert binding["chain_positions"] == [1, 4, 8]
    assert binding["chain_coordinates_sha256"]
    assert condition.structure.chain_sequence == "ACDEFGHI"
    assert condition.structure.chain_positions == (1, 4, 8)


def test_generation_binding_reuses_first_model_and_extended_residue_names(tmp_path: Path) -> None:
    rows = []
    atom_names = ("N", "CA", "C", "O")
    residue_names = ("MLY", "ALA", "GLY")
    for model in (1, 2):
        for residue_index, residue_name in enumerate(residue_names, start=1):
            for atom_index, atom_name in enumerate(atom_names):
                coordinate = 100.0 * model + 10.0 * residue_index + atom_index
                rows.append(
                    f"ATOM {len(rows) + 1} {atom_name} {atom_name[0]} . {residue_name} A {residue_index} ? "
                    f"{coordinate:.3f} {coordinate + 0.1:.3f} {coordinate + 0.2:.3f} "
                    f"1.00 10.00 {residue_index} A {model}"
                )
    mmcif = """data_fixture
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
""" + "\n".join(rows) + "\n"
    path = tmp_path / "multi_model_modified.cif"
    path.write_text(mmcif, encoding="utf-8")
    model_one = np.asarray(
        [
            [[110.0 + atom, 110.1 + atom, 110.2 + atom] for atom in range(4)],
            [[130.0 + atom, 130.1 + atom, 130.2 + atom] for atom in range(4)],
        ],
        dtype=np.float32,
    )
    projection = ProteinMPNNStructureInput(
        protein_id="modified_model_fixture",
        backbone_condition="PDB",
        uniprot_positions=(11, 13),
        wt_sequence_projection="KG",
        coordinates=model_one,
        structure_sha256="a" * 64,
    )

    structure = _full_chain_generation_structure(
        projection,
        structure_path=path,
        source_id="modified_model_fixture",
        chain_id="A",
        projection_chain_auth_keys=((1, ""), (3, "")),
        rebind_projection_coordinates=True,
    )

    assert structure.chain_sequence == "KAG"
    assert structure.chain_positions == (1, 3)
    np.testing.assert_array_equal(structure.projection.coordinates, model_one)


def test_reuse_rejects_shard_from_different_scorer_binding(tmp_path: Path) -> None:
    inputs = _inputs()
    adapter = _FakeAdapter()
    generate_clean_cohort(inputs, adapter, output_root=tmp_path, resume=True)
    condition = inputs.conditions[0]
    shard = tmp_path / "shards" / condition.identity[:2] / f"{condition.identity}.json"
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["records"][0]["scorer_binding"]["checkpoint_id"] = "5" * 64
    shard.write_text(json.dumps(payload), encoding="utf-8")
    adapter.calls.clear()

    with pytest.raises(GenerationExecutionError, match="scorer identity"):
        generate_clean_cohort(inputs, adapter, output_root=tmp_path, resume=True)

    assert adapter.calls == []


def test_cli_exposes_only_path_resume_and_smoke_arguments() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/analysis/generate_propagation_sequences.py"
    spec = importlib.util.spec_from_file_location("generate_propagation_sequences", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([
        "--project-root", str(root), "--output-root", "runs/analysis/generative_propagation",
        "--resume", "--smoke-protein-id", "fixture_000", "--smoke-protein-id", "fixture_001",
        "--worker-index", "1", "--worker-count", "4",
    ])

    assert args.resume is True
    assert args.smoke_protein_id == ["fixture_000", "fixture_001"]
    assert args.worker_index == 1
    assert args.worker_count == 4
