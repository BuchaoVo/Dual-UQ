from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dual_uq.evaluation.apo_holo_local_response import (
    build_apo_holo_cases,
    build_decoding_realizations,
    join_structural_associations,
    js_bits,
    materialize_model_local_response,
    run_model_local_response,
    summarize_cross_model,
)
from dual_uq.models.proteinmpnn import ProteinMPNNAdapter, ProteinMPNNStructureInput


def _write_cif(path: Path, *, offset: float) -> None:
    rows = []
    serial = 1
    residues = ((10, "A"), (11, "C"), (13, "D"))
    for seq_id, residue in residues:
        for atom_name, dx, dy, dz in (
            ("N", 0.0, 0.0, 0.0),
            ("CA", 1.0, 0.0, 0.0),
            ("C", 2.0, 0.0, 0.0),
            ("O", 3.0, 0.0, 0.0),
        ):
            rows.append(
                f"ATOM {serial} {atom_name} {residue} A A {seq_id} {seq_id - 9} . "
                f"{offset + dx:.3f} {dy:.3f} {dz:.3f} 1.00 10.0 1"
            )
            serial += 1
    path.write_text(
        "data_fixture\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_PDB_model_num\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def _minimal_release(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    release_root = project_root / "release"
    (project_root / "data/raw/apo_holo/pdb").mkdir(parents=True)
    (project_root / "data/raw/apo_holo/uniprot").mkdir(parents=True)
    _write_cif(project_root / "data/raw/apo_holo/pdb/apo.cif", offset=0.0)
    _write_cif(project_root / "data/raw/apo_holo/pdb/holo.cif", offset=0.5)
    (project_root / "data/raw/apo_holo/uniprot/P00001.json").write_text(
        json.dumps({"primaryAccession": "P00001", "sequence": {"value": "ACGD"}}),
        encoding="utf-8",
    )
    release_root.mkdir()
    pd.DataFrame(
        [
            {
                "protein_id": "P00001",
                "pair_id": "apo_holo__P00001",
                "apo_polymer_entity_id": "apo_1",
                "holo_polymer_entity_id": "holo_1",
                "uniprot_id": "P00001",
                "apo_chain_id": "A",
                "holo_chain_id": "A",
                "apo_mmcif_relative_path": "data/raw/apo_holo/pdb/apo.cif",
                "holo_mmcif_relative_path": "data/raw/apo_holo/pdb/holo.cif",
                "admission_status": "ADMITTED",
                "admitted": True,
            }
        ]
    ).to_parquet(release_root / "primary_pairs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": "apo_holo__P00001",
                "protein_id": "P00001",
                "model_status": "MODEL_EVALUABLE",
                "esm_if1_status": "ESM_IF1_EVALUABLE",
                "model_evaluable": True,
            }
        ]
    ).to_parquet(release_root / "model_evaluability.parquet", index=False)
    mapping_rows = []
    for canonical, apo_auth, holo_auth, aa in ((1, 10, 10, "A"), (2, 11, 11, "C"), (4, 13, 13, "D")):
        mapping_rows.append(
            {
                "pair_id": "apo_holo__P00001",
                "protein_id": "P00001",
                "canonical_position": canonical,
                "apo_auth_seq_id": apo_auth,
                "holo_auth_seq_id": holo_auth,
                "insertion_code_apo": "",
                "insertion_code_holo": "",
                "uniprot_residue_name_apo": aa,
                "uniprot_residue_name_holo": aa,
            }
        )
    pd.DataFrame(mapping_rows).to_parquet(release_root / "residue_mappings.parquet", index=False)
    pd.DataFrame(
        [
            {
                "polymer_entity_id": "apo_1",
                "pdb_id": "apo",
                "uniprot_id": "P00001",
                "pdb_mmcif_relative_path": "data/raw/apo_holo/pdb/apo.cif",
                "uniprot_relative_path": "data/raw/apo_holo/uniprot/P00001.json",
            },
            {
                "polymer_entity_id": "holo_1",
                "pdb_id": "holo",
                "uniprot_id": "P00001",
                "pdb_mmcif_relative_path": "data/raw/apo_holo/pdb/holo.cif",
                "uniprot_relative_path": "data/raw/apo_holo/uniprot/P00001.json",
            },
        ]
    ).to_parquet(release_root / "candidate_structures.parquet", index=False)
    pd.DataFrame(
        [
            {"polymer_entity_id": "apo_1", "asset_type": "pdb_mmcif", "sha256": "a" * 64},
            {"polymer_entity_id": "holo_1", "asset_type": "pdb_mmcif", "sha256": "b" * 64},
        ]
    ).to_parquet(release_root / "asset_ledger.parquet", index=False)
    return project_root, release_root


def test_case_builder_keeps_true_gap_for_proteinmpnn_and_excludes_esm_if1(tmp_path: Path) -> None:
    project_root, release_root = _minimal_release(tmp_path)

    proteinmpnn, esm_if1, exclusions = build_apo_holo_cases(project_root, release_root)

    assert set(proteinmpnn["condition"]) == {"APO", "HOLO"}
    assert sorted(proteinmpnn["canonical_position"].unique()) == [1, 2, 4]
    assert len(esm_if1) == 0
    assert exclusions.to_dict("records") == [
        {
            "protein_id": "P00001",
            "pair_id": "apo_holo__P00001",
            "reason": "true_uniprot_gap",
            "detail": "internal canonical position gap",
        }
    ]
    assert not proteinmpnn["structure_sha256"].isna().any()


def test_decoding_realizations_are_deterministic_and_matched() -> None:
    first = build_decoding_realizations("P00001", 4)
    second = build_decoding_realizations("P00001", 4)

    assert first == second
    assert len(first) == 16
    assert [item.repeat_index for item in first] == list(range(16))
    assert all(len(item.order) == 4 for item in first)


def test_js_bits_is_symmetric_and_zero_safe() -> None:
    assert js_bits([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert js_bits([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)
    assert js_bits([0.2, 0.8], [0.8, 0.2]) == pytest.approx(
        js_bits([0.8, 0.2], [0.2, 0.8])
    )
    with pytest.raises(ValueError):
        js_bits([0, 0], [1, 0])


def _cases_for_model() -> pd.DataFrame:
    rows = []
    for condition, offset in (("APO", 0.0), ("HOLO", 1.0)):
        coordinates = np.full((2, 4, 3), offset, dtype=np.float32)
        for position in (1, 2):
            rows.append(
                {
                    "protein_id": "P00001",
                    "pair_id": "pair__P00001",
                    "condition": condition,
                    "canonical_position": position,
                    "canonical_positions": (1, 2),
                    "wt_sequence": "AC",
                    "wt_sequence_projection": "AC",
                    "coordinates": coordinates,
                    "structure_sha256": "a" * 64,
                }
            )
    return pd.DataFrame(rows)


def _generic_cases_for_model() -> pd.DataFrame:
    cases = _cases_for_model().copy()
    cases["condition"] = cases["condition"].map(
        {"APO": "REFERENCE", "HOLO": "PERTURBED"}
    )
    return cases


def test_model_local_response_accepts_opaque_ordered_conditions() -> None:
    position, protein, metadata = run_model_local_response(
        _FakeProteinMPNN(),
        _generic_cases_for_model(),
        "ProteinMPNN",
        condition_order=("REFERENCE", "PERTURBED"),
    )

    assert set(position["condition"]) == {"REFERENCE", "PERTURBED"}
    assert len(protein) == 1
    assert metadata["model"] == "ProteinMPNN"


def test_model_local_response_accepts_nested_coordinates_after_parquet_roundtrip(tmp_path: Path) -> None:
    cases = _generic_cases_for_model()
    cases["coordinates"] = cases["coordinates"].map(lambda value: value.tolist())
    path = tmp_path / "cases.parquet"
    cases.to_parquet(path, index=False)

    reread = pd.read_parquet(path)
    position, protein, _metadata = run_model_local_response(
        _FakeProteinMPNN(),
        reread,
        "ProteinMPNN",
        condition_order=("REFERENCE", "PERTURBED"),
    )

    assert len(position) == 4
    assert len(protein) == 1


def test_model_local_response_rejects_more_than_two_conditions() -> None:
    cases = pd.concat(
        [
            _generic_cases_for_model(),
            _generic_cases_for_model().iloc[[0]].assign(condition="THIRD"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="exactly the ordered condition pair"):
        run_model_local_response(
            _FakeProteinMPNN(),
            cases,
            "ProteinMPNN",
            condition_order=("REFERENCE", "PERTURBED"),
        )


def test_manifest_free_model_response_materialization_writes_only_outputs(tmp_path: Path) -> None:
    position = pd.DataFrame({"protein_id": ["P00001"], "js_bits_mean": [0.1]})
    protein = pd.DataFrame({"protein_id": ["P00001"], "local_burden_mean": [0.1]})

    result = materialize_model_local_response(
        position,
        protein,
        {"model": "ESM-IF1"},
        tmp_path,
        write_manifest=False,
    )

    assert set(result["write_status"]) == {"position", "protein", "summary"}
    assert (tmp_path / "position_local_response.parquet").is_file()
    assert (tmp_path / "protein_local_response.parquet").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert not (tmp_path / "manifest.json").exists()


class _FakeProteinMPNN:
    implementation_id = "i" * 40
    checkpoint_id = "c" * 64

    def probability_distributions(self, structure, sequences, realization, *, batch_size):
        del sequences, realization, batch_size
        result = np.full((2, 20), 0.05 / 19, dtype=float)
        if structure.backbone_condition == "APO":
            result[:, 0] = 0.95
        else:
            result[:, 1] = 0.95
        result /= result.sum(axis=1, keepdims=True)
        return (result,)


class _FakeESMIF1:
    def score_teacher_forced(self, sequence, coordinates, **_kwargs):
        del sequence, coordinates
        result = np.full((2, 20), 0.05 / 19, dtype=float)
        result[:, 0] = 0.95
        result /= result.sum(axis=1, keepdims=True)
        return result


def test_model_local_response_uses_one_paired_axis_and_nested_realizations() -> None:
    position, protein, metadata = run_model_local_response(
        _FakeProteinMPNN(), _cases_for_model(), "ProteinMPNN"
    )

    assert len(position) == 4
    assert len(protein) == 1
    assert protein.iloc[0]["local_burden_mean"] > 0
    assert metadata["realization_count"] == 16
    assert [row["realization_count"] for row in metadata["convergence"]] == [4, 8, 16]


def test_esm_if1_local_response_is_teacher_forced_without_fake_repeats() -> None:
    position, protein, metadata = run_model_local_response(
        _FakeESMIF1(), _cases_for_model(), "ESM-IF1"
    )

    assert len(position) == 4
    assert len(protein) == 1
    assert metadata["realization_count"] == 1
    assert set(position["realization_count"]) == {1}


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def numpy(self):
        return self.values


class _Torch:
    float32 = "float32"
    long = "long"

    @staticmethod
    def as_tensor(values, **_kwargs):
        return _Tensor(values)

    @staticmethod
    def ones(shape, **_kwargs):
        return _Tensor(np.ones(shape))

    @staticmethod
    def ones_like(value):
        return _Tensor(np.ones_like(value.values))

    @staticmethod
    def zeros_like(value):
        return _Tensor(np.zeros_like(value.values))

    @staticmethod
    def inference_mode():
        from contextlib import nullcontext

        return nullcontext()


class _DistributionModel:
    def __call__(self, x, s, mask, chain_m, residue_idx, chain_encoding, randn, **kwargs):
        del s, mask, chain_m, residue_idx, chain_encoding, randn, kwargs
        values = np.zeros((x.values.shape[0], x.values.shape[1], 20), dtype=float)
        values[..., 0] = 1.0
        return _Tensor(values)


def test_proteinmpnn_native_distribution_is_standard20_normalized() -> None:
    adapter = ProteinMPNNAdapter(
        model=_DistributionModel(),
        torch=_Torch(),
        device="cpu",
        checkpoint_num_edges=48,
        checkpoint_noise_level=0.2,
        implementation_id="i" * 40,
        checkpoint_id="c" * 64,
    )
    structure = ProteinMPNNStructureInput(
        protein_id="P00001",
        backbone_condition="APO",
        uniprot_positions=(1, 2),
        wt_sequence_projection="AC",
        coordinates=np.zeros((2, 4, 3), dtype=np.float32),
    )
    realization = build_decoding_realizations("P00001", 2, 1)[0]

    distributions = adapter.probability_distributions(
        structure, ("AC",), realization, batch_size=1
    )

    assert len(distributions) == 1
    assert distributions[0].shape == (2, 20)
    assert np.allclose(distributions[0].sum(axis=1), 1.0)
    assert np.isfinite(distributions[0]).all()


def test_cross_model_summary_uses_shared_protein_and_position_keys() -> None:
    position = pd.DataFrame(
        [
            {"protein_id": "P00001", "pair_id": "pair", "canonical_position": 1, "condition": "APO", "js_bits_mean": 0.1},
            {"protein_id": "P00001", "pair_id": "pair", "canonical_position": 1, "condition": "HOLO", "js_bits_mean": 0.1},
        ]
    )
    other = position.assign(js_bits_mean=0.2)
    protein = pd.DataFrame({"protein_id": ["P00001"], "local_burden_mean": [0.1]})
    other_protein = pd.DataFrame({"protein_id": ["P00001"], "local_burden_mean": [0.2]})

    merged, metadata = summarize_cross_model(protein, other_protein, position, other)

    assert merged.to_dict("records") == [
        {
            "protein_id": "P00001",
            "proteinmpnn_local_burden": 0.1,
            "esm_if1_local_burden": 0.2,
        }
    ]
    assert metadata["protein_count"] == 1
    assert metadata["shared_position_count"] == 1


def test_model_response_materialization_is_immutable(tmp_path: Path) -> None:
    position = pd.DataFrame({"protein_id": ["P00001"], "js_bits_mean": [0.1]})
    protein = pd.DataFrame({"protein_id": ["P00001"], "local_burden_mean": [0.1]})
    metadata = {"model": "ESM-IF1", "binding": {"checkpoint_sha256": "c" * 64}}

    first = materialize_model_local_response(position, protein, metadata, tmp_path)
    second = materialize_model_local_response(position, protein, metadata, tmp_path)

    assert first["write_status"]["position"] == "created"
    assert second["write_status"]["position"] == "reused_identical"
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable output conflict"):
        materialize_model_local_response(position, protein, metadata, tmp_path)


def test_association_join_retains_explicit_zero_proximal_status(tmp_path: Path) -> None:
    position = pd.DataFrame(
        [
            {
                "protein_id": "P00001",
                "pair_id": "pair",
                "canonical_position": 1,
                "condition": "APO",
                "js_bits_mean": 0.2,
            }
        ]
    )
    pd.DataFrame(
        [
            {
                "protein_id": "P00001",
                "pair_id": "pair",
                "canonical_position": 1,
                "geometry_status": "available",
                "local_pairwise_distance_change": 0.4,
                "ligand_proximal": False,
            }
        ]
    ).to_parquet(tmp_path / "residue_structural_descriptors.parquet", index=False)
    pd.DataFrame(
        [{"pair_id": "pair", "ligand_proximal_count": 0}]
    ).to_parquet(tmp_path / "pair_structural_descriptors.parquet", index=False)

    result = join_structural_associations(position, tmp_path)

    assert result.iloc[0]["ligand_status"] == "explicit_zero_proximal"
