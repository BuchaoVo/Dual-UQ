"""Generate equal-weight true joint ProteinMPNN Apo/Holo ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_uq.core.atomic_io import atomic_write_new_bytes
from dual_uq.inference.apo_holo_generation import (
    ApoHoloGenerationCondition,
    GenerationRequest,
)
from dual_uq.inference.multi_state_generation import generate_equal_weight_joint
from dual_uq.models.proteinmpnn import load_authorized_proteinmpnn_adapter
from dual_uq.models.proteinmpnn import ProteinMPNNStructureInput
from dual_uq.models.proteinmpnn_generation import (
    ProteinMPNNGenerationAdapter,
    ProteinMPNNGenerationStructure,
)
from dual_uq.models.proteinmpnn_generation import sequence_sha256
from dual_uq.models.proteinmpnn_generation import paired_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--cases-pickle", type=Path, required=True)
    parser.add_argument("--single-state-generation-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--protein-id", action="append")
    parser.add_argument("--protein-id-file", type=Path)
    return parser.parse_args(argv)


def _conditions(args: argparse.Namespace) -> dict[str, tuple[object, object]]:
    project_root = Path(args.project_root).resolve()
    release_root = Path(args.release_root).resolve()
    cases = pd.read_pickle(args.cases_pickle)
    pairs = pd.read_parquet(release_root / "primary_pairs.parquet")
    pairs = pairs.loc[pairs["admitted"].eq(True)].set_index(pairs["pair_id"].astype(str), drop=False)
    cases = cases.copy()
    cases["protein_id"] = cases["protein_id"].astype(str)
    case_groups = {str(protein): group for protein, group in cases.groupby("protein_id", sort=False)}
    proteins = sorted(case_groups)
    generation_root = Path(args.single_state_generation_root or project_root / "runs/analysis/apo_holo_generation/proteinmpnn")
    existing_ids: set[str] = set()
    for shard in (generation_root / "shards").glob("*/*.json"):
        payload = json.loads(shard.read_text(encoding="utf-8"))
        existing_ids.add(str(payload["condition"]["protein_id"]))
    proteins = [protein for protein in proteins if protein in existing_ids]
    requested_ids = set(args.protein_id or ())
    if args.protein_id_file is not None:
        id_file = Path(args.protein_id_file)
        if not id_file.is_absolute():
            id_file = project_root / id_file
        requested_ids.update(
            line.strip()
            for line in id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if requested_ids:
        wanted = requested_ids
        missing = wanted - set(proteins)
        if missing:
            raise ValueError(f"requested proteins absent from cases: {sorted(missing)}")
        proteins = sorted(wanted)
    result: dict[str, tuple[object, object]] = {}
    for protein in proteins:
        subset = case_groups[protein]
        pair_id = str(subset["pair_id"].iloc[0])
        pair = pairs.loc[pair_id]
        built = []
        for state in ("APO", "HOLO"):
            group = subset.loc[subset["condition"].astype(str).eq(state)]
            if group.empty:
                raise ValueError(f"missing {state} case for {protein}")
            ordered = group.sort_values("canonical_position", kind="mergesort")
            positions = tuple(int(value) for value in ordered["canonical_position"])
            projection = ProteinMPNNStructureInput(
                protein_id=protein,
                backbone_condition="PDB" if state == "APO" else "AFDB",
                uniprot_positions=positions,
                wt_sequence_projection=str(ordered["wt_sequence_projection"].iloc[0]),
                coordinates=np.asarray(ordered["coordinates"].iloc[0], dtype=np.float32),
                structure_sha256=str(ordered["structure_sha256"].iloc[0]),
            )
            segment_ends = tuple(
                [i + 1 for i, (left, right) in enumerate(zip(positions, positions[1:])) if right != left + 1]
                + [len(positions)]
            )
            structure = ProteinMPNNGenerationStructure(
                projection=projection,
                chain_id=str(pair["apo_chain_id"] if state == "APO" else pair["holo_chain_id"]),
                chain_sequence=projection.wt_sequence_projection,
                chain_coordinates=np.asarray(projection.coordinates, dtype=np.float32),
                chain_positions=tuple(range(1, len(positions) + 1)),
                chain_segment_ends=segment_ends,
            )
            built.append(
                ApoHoloGenerationCondition(
                    protein_id=protein,
                    pair_id=pair_id,
                    state=state,
                    structure_sha256=str(projection.structure_sha256),
                    canonical_positions=positions,
                    wt_sequence_projection=projection.wt_sequence_projection,
                    coordinates=np.asarray(projection.coordinates, dtype=np.float32),
                    structure_path=project_root / str(pair["apo_mmcif_relative_path"] if state == "APO" else pair["holo_mmcif_relative_path"]),
                    chain_id=structure.chain_id,
                    proteinmpnn_structure=structure,
                )
            )
        result[protein] = (built[0], built[1])
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = ProteinMPNNGenerationAdapter(
        load_authorized_proteinmpnn_adapter(
            implementation_path=Path(args.implementation_path or project_root / "third_party/ProteinMPNN"),
            checkpoint_path=Path(args.checkpoint_path or project_root / "third_party/ProteinMPNN/vanilla_model_weights/v_48_020.pt"),
            device_name=args.device,
            backbone_noise=0.0,
        ),
        batch_size=args.batch_size,
    )
    conditions = _conditions(args)
    total = 0
    for index, (protein, (apo, holo)) in enumerate(sorted(conditions.items()), start=1):
        projection = apo.proteinmpnn_structure.projection
        request = GenerationRequest(
            protein_id=protein,
            backbone_condition="PDB",
            structure_sha256=str(projection.structure_sha256),
            canonical_positions=projection.uniprot_positions,
            wt_sequence_projection=projection.wt_sequence_projection,
            temperature=0.1,
            sample_index=0,
            seed=paired_seed(0),
            sample_class="paired",
            decoding_realization="0" * 64,
        )
        def projected(condition):
            projection = condition.proteinmpnn_structure.projection
            positions = projection.uniprot_positions
            segment_ends = tuple(
                [i + 1 for i, (left, right) in enumerate(zip(positions, positions[1:])) if right != left + 1]
                + [len(positions)]
            )
            return ProteinMPNNGenerationStructure(
                projection=projection,
                chain_id=condition.chain_id,
                chain_sequence=projection.wt_sequence_projection,
                chain_coordinates=np.asarray(projection.coordinates, dtype=np.float32),
                chain_positions=tuple(range(1, len(positions) + 1)),
                chain_segment_ends=segment_ends,
            )

        effective_batch_size = args.batch_size
        while True:
            try:
                records = generate_equal_weight_joint(
                    adapter,
                    request,
                    projected(apo),
                    projected(holo),
                    n_samples=args.n_samples,
                    batch_size=effective_batch_size,
                )
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or effective_batch_size <= 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                adapter.adapter.torch.cuda.empty_cache()
                print(
                    f"multi-state {protein}: reducing batch_size to {effective_batch_size} after CUDA OOM",
                    flush=True,
                )
        payload = {
            "schema_version": "apo_holo_multistate_generation_v1",
            "protein_id": protein,
            "pair_id": apo.pair_id,
            "n_samples": args.n_samples,
            "model": "ProteinMPNN",
            "temperature": 0.1,
            "implementation_id": adapter.adapter.implementation_id,
            "checkpoint_id": adapter.adapter.checkpoint_id,
            "records": [
                {
                    "sample_index": r.request.sample_index,
                    "seed": r.request.seed,
                    "decoding_realization": r.request.decoding_realization,
                    "sequence": r.sequence,
                    "sequence_hash": sequence_sha256(r.sequence),
                }
                for r in records
            ],
        }
        rendered = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path = output_root / "shards" / protein[:2] / f"{protein}.json"
        try:
            atomic_write_new_bytes(path, rendered)
        except FileExistsError:
            if path.read_bytes() != rendered:
                raise ValueError(f"immutable multi-state shard conflict: {path}") from None
        total += len(records)
        if index % 10 == 0 or index == len(conditions):
            print(f"multi-state {index}/{len(conditions)} sequences={total}", flush=True)
    manifest = {
        "schema_version": "apo_holo_multistate_generation_manifest_v1",
        "protein_count": len(conditions),
        "sequence_count": total,
        "samples_per_protein": args.n_samples,
        "temperature": 0.1,
        "decoder": "0.5*log_p_APO + 0.5*log_p_HOLO, renormalized at each autoregressive step",
        "checkpoint_id": adapter.adapter.checkpoint_id,
    }
    atomic_write_new_bytes(output_root / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


if __name__ == "__main__":
    run(parse_args())
