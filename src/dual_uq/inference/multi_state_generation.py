"""Shared-context contracts for equal-weight multi-state decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dual_uq.core.hashing import sha256_bytes
from dual_uq.inference.apo_holo_generation import GenerationRequest
from dual_uq.models.proteinmpnn import PROTEINMPNN_ALPHABET, sequence_sha256
from dual_uq.models.proteinmpnn_generation import (
    GENERATION_SCORE_CONTRACT_ID,
    ProteinMPNNGenerationAdapter,
    ProteinMPNNGenerationStructure,
    GeneratedSequenceRecord,
    generation_seed_plan,
)
from dual_uq.models.scoring import ScorerBinding


@dataclass(frozen=True, slots=True)
class SharedDecodeContext:
    apo_positions: tuple[int, ...]
    holo_positions: tuple[int, ...]
    decoding_order: tuple[int, ...]
    prefix: tuple[str, ...]


def validate_shared_decode_context(
    *, apo_positions: tuple[int, ...], holo_positions: tuple[int, ...],
    decoding_order: tuple[int, ...], prefix: tuple[str, ...],
) -> SharedDecodeContext:
    apo = tuple(apo_positions)
    holo = tuple(holo_positions)
    order = tuple(decoding_order)
    prefix_tuple = tuple(prefix)
    if apo != holo:
        raise ValueError("APO/HOLO position axes must match")
    if len(order) != len(apo) or sorted(order) != list(range(len(apo))):
        raise ValueError("decoding order must be a permutation of the shared axis")
    if len(prefix_tuple) != len(apo):
        raise ValueError("shared prefix must match the shared axis")
    return SharedDecodeContext(apo, holo, order, prefix_tuple)


def _prepare(adapter: ProteinMPNNGenerationAdapter, structure: ProteinMPNNGenerationStructure) -> dict[str, Any]:
    """Prepare the vendor model tensors for one immutable structure."""
    layout = adapter._chain_layout(structure)
    projected = set(structure.chain_positions)
    masked_chains = [chain_id for chain_id, _start, _end in layout]
    fixed_positions = {
        chain_id: [
            position - start
            for position in range(start + 1, end + 1)
            if position not in projected
        ]
        for chain_id, start, end in layout
    }
    batch = [adapter._batch(structure)]
    x, s, mask, _lengths, chain_m, chain_encoding, *_rest = adapter.adapter._tied_featurize(
        batch,
        adapter.adapter.device,
        {structure.projection.protein_id: (masked_chains, [])},
        {structure.projection.protein_id: fixed_positions},
    )
    (
        _chain_list,
        _visible_list,
        _masked_list,
        _masked_chain_lengths,
        chain_m_pos,
        omit_aa_mask,
        residue_idx,
        _dihedral_mask,
        _tied_positions,
        pssm_coef,
        pssm_bias,
        _pssm_log_odds,
        bias_by_res,
        _tied_beta,
    ) = _rest
    return {
        "x": x,
        "s_true": s,
        "mask": mask,
        "chain_mask": chain_m * chain_m_pos * mask,
        "chain_encoding": chain_encoding,
        "residue_idx": residue_idx,
        "omit_aa_mask": omit_aa_mask,
        "bias_by_res": bias_by_res,
        "pssm_coef": pssm_coef,
        "pssm_bias": pssm_bias,
    }


def _encode(adapter: ProteinMPNNGenerationAdapter, tensors: dict[str, Any], order: Any) -> dict[str, Any]:
    """Run the unmasked encoder and construct the fixed decoding masks."""
    model = adapter.adapter.model
    torch = adapter.adapter.torch
    gather_nodes = model.sample.__globals__["gather_nodes"]
    cat_neighbors_nodes = model.sample.__globals__["cat_neighbors_nodes"]
    x, mask = tensors["x"], tensors["mask"]
    E, e_idx = model.features(x, mask, tensors["residue_idx"], tensors["chain_encoding"])
    h_v = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=x.device)
    h_e = model.W_e(E)
    mask_attend = gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)
    mask_size = e_idx.shape[1]
    permutation = torch.nn.functional.one_hot(order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        1 - torch.triu(torch.ones(mask_size, mask_size, device=x.device)),
        permutation,
        permutation,
    )
    mask_attend = torch.gather(order_mask_backward, 2, e_idx).unsqueeze(-1)
    mask_1d = mask.view(mask.size(0), mask.size(1), 1, 1)
    mask_bw = mask_1d * mask_attend
    mask_fw = mask_1d * (1.0 - mask_attend)
    h_ex_encoder = cat_neighbors_nodes(torch.zeros_like(h_v), h_e, e_idx)
    h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)
    return {
        **tensors,
        "e_idx": e_idx,
        "h_v": h_v,
        "h_e": h_e,
        "mask_bw": mask_bw,
        "h_exv_encoder_fw": mask_fw * h_exv_encoder,
        "h_s": torch.zeros_like(h_v),
        "S": torch.zeros_like(tensors["s_true"], dtype=torch.long),
        "h_v_stack": [h_v] + [torch.zeros_like(h_v) for _ in range(len(model.decoder_layers))],
    }


def generate_equal_weight_joint(
    adapter: ProteinMPNNGenerationAdapter,
    request: GenerationRequest,
    apo_structure: ProteinMPNNGenerationStructure,
    holo_structure: ProteinMPNNGenerationStructure,
    *,
    n_samples: int = 64,
    batch_size: int = 1,
) -> tuple[GeneratedSequenceRecord, ...]:
    """Generate with a true equal-weight autoregressive APO/HOLO decoder."""
    if not isinstance(adapter, ProteinMPNNGenerationAdapter):
        raise ValueError("adapter must be a ProteinMPNNGenerationAdapter")
    adapter._validate_binding(request, apo_structure)
    adapter._validate_binding(
        GenerationRequest(
            protein_id=request.protein_id,
            backbone_condition="AFDB",
            structure_sha256=holo_structure.projection.structure_sha256,
            canonical_positions=request.canonical_positions,
            wt_sequence_projection=request.wt_sequence_projection,
            temperature=request.temperature,
            sample_index=request.sample_index,
            seed=request.seed,
            sample_class=request.sample_class,
            decoding_realization=request.decoding_realization,
        ),
        holo_structure,
    )
    if (
        apo_structure.chain_sequence != holo_structure.chain_sequence
        or apo_structure.chain_positions != holo_structure.chain_positions
        or apo_structure.chain_segment_ends != holo_structure.chain_segment_ends
    ):
        raise ValueError("APO/HOLO full-chain axes must match for joint decoding")
    if n_samples <= 0 or n_samples > 64:
        raise ValueError("n_samples must be in 1..64")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if request.backbone_condition != "PDB":
        raise ValueError("joint request uses the PDB binding as its canonical request")
    torch = adapter.adapter.torch
    alphabet = np.asarray(tuple(PROTEINMPNN_ALPHABET))
    model = adapter.adapter.model
    sample_specs = generation_seed_plan()[:n_samples]
    outputs: list[GeneratedSequenceRecord] = []
    # Prepare the two immutable structural feature sets once per protein.  The
    # old implementation repeated tied_featurize and the encoder for every
    # sample; batching the sample dimension preserves the per-sample seed
    # streams while removing that dominant overhead.
    base_template = _prepare(adapter, apo_structure)
    other_template = _prepare(adapter, holo_structure)
    if (
        base_template["x"].shape != other_template["x"].shape
        or not torch.equal(base_template["chain_mask"], other_template["chain_mask"])
    ):
        raise ValueError("APO/HOLO model masks differ for joint decoding")
    cat_neighbors_nodes = model.sample.__globals__["cat_neighbors_nodes"]
    F = model.sample.__globals__["F"]

    def _expand(template: dict[str, Any], size: int) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        for key, value in template.items():
            if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0 and value.shape[0] == 1:
                expanded[key] = value.expand((size, *value.shape[1:])).contiguous()
            else:
                expanded[key] = value
        return expanded

    for start in range(0, len(sample_specs), batch_size):
        specs = sample_specs[start : start + batch_size]
        size = len(specs)
        base = _expand(base_template, size)
        other = _expand(other_template, size)
        generators = []
        orders = []
        for spec in specs:
            generator = torch.Generator(device=base["x"].device)
            generator.manual_seed(spec.pdb_seed)
            generators.append(generator)
            random_order = torch.randn(
                base_template["chain_mask"].shape,
                generator=generator,
                device=base["x"].device,
            )
            orders.append(
                torch.argsort(
                    (base_template["chain_mask"] + 0.0001) * torch.abs(random_order),
                    dim=1,
                )[0]
            )
        order = torch.stack(orders, dim=0)
        state_a = _encode(adapter, base, order)
        state_h = _encode(adapter, other, order)
        n_nodes = base["x"].shape[1]
        with torch.inference_mode():
            for step in range(n_nodes):
                t = order[:, step]
                logits = []
                for state in (state_a, state_h):
                    mask_g = torch.gather(state["mask"], 1, t[:, None])
                    if (mask_g == 0).all():
                        continue
                    e_idx_t = torch.gather(state["e_idx"], 1, t[:, None, None].repeat(1, 1, state["e_idx"].shape[-1]))
                    h_e_t = torch.gather(state["h_e"], 1, t[:, None, None, None].repeat(1, 1, state["h_e"].shape[-2], state["h_e"].shape[-1]))
                    h_es_t = cat_neighbors_nodes(state["h_s"], h_e_t, e_idx_t)
                    h_exv_t = torch.gather(state["h_exv_encoder_fw"], 1, t[:, None, None, None].repeat(1, 1, state["h_exv_encoder_fw"].shape[-2], state["h_exv_encoder_fw"].shape[-1]))
                    mask_t = torch.gather(state["mask"], 1, t[:, None])
                    for layer_index, layer in enumerate(model.decoder_layers):
                        h_esv_decoder_t = cat_neighbors_nodes(state["h_v_stack"][layer_index], h_es_t, e_idx_t)
                        h_v_t = torch.gather(state["h_v_stack"][layer_index], 1, t[:, None, None].repeat(1, 1, state["h_v_stack"][layer_index].shape[-1]))
                        mask_t_full = torch.gather(state["mask_bw"], 1, t[:, None, None, None].repeat(1, 1, state["mask_bw"].shape[-2], state["mask_bw"].shape[-1]))
                        updated = layer(h_v_t, mask_t_full * h_esv_decoder_t + h_exv_t, mask_V=mask_t)
                        state["h_v_stack"][layer_index + 1].scatter_(1, t[:, None, None].repeat(1, 1, state["h_v_stack"][layer_index + 1].shape[-1]), updated)
                    h_v_t = torch.gather(state["h_v_stack"][-1], 1, t[:, None, None].repeat(1, 1, state["h_v_stack"][-1].shape[-1]))[:, 0]
                    logits.append(model.W_out(h_v_t) / request.temperature)
                if logits:
                    logp = 0.5 * F.log_softmax(logits[0], dim=-1) + 0.5 * F.log_softmax(logits[1], dim=-1)
                    # Use one generator per sample.  This keeps the fixed seed
                    # domains independent while still batching all model work.
                    sampled = torch.cat(
                        [torch.multinomial(torch.exp(logp[i : i + 1]), 1, generator=generators[i]) for i in range(size)],
                        dim=0,
                    )
                else:
                    sampled = torch.gather(state_a["s_true"], 1, t[:, None])
                for state in (state_a, state_h):
                    fixed = torch.gather(state["s_true"], 1, t[:, None])
                    chain_g = torch.gather(state["chain_mask"], 1, t[:, None])
                    value = (sampled * chain_g + fixed * (1.0 - chain_g)).long()
                    emb = model.W_s(value)
                    state["h_s"].scatter_(1, t[:, None, None].repeat(1, 1, emb.shape[-1]), emb)
                    state["S"].scatter_(1, t[:, None], value)
        indices = state_a["S"].detach().cpu().numpy()
        order_cpu = order.detach().cpu().numpy().astype("<i8", copy=False)
        for row, (spec, realized_order) in enumerate(zip(specs, order_cpu, strict=True)):
            full_sequence = "".join(alphabet[indices[row]].tolist())
            sequence = "".join(full_sequence[pos - 1] for pos in apo_structure.chain_positions)
            realized = sha256_bytes(realized_order.tobytes())
            realized_request = GenerationRequest(
                protein_id=request.protein_id,
                backbone_condition="PDB",
                structure_sha256=request.structure_sha256,
                canonical_positions=request.canonical_positions,
                wt_sequence_projection=request.wt_sequence_projection,
                temperature=request.temperature,
                sample_index=spec.sample_index,
                seed=spec.pdb_seed,
                sample_class=spec.sample_class,
                decoding_realization=realized,
            )
            outputs.append(GeneratedSequenceRecord(
                request=realized_request,
                sequence=sequence,
                sequence_hash=sequence_sha256(sequence),
                scorer_binding=ScorerBinding(
                    scorer_id=adapter.binding.scorer_id,
                    implementation_id=adapter.binding.implementation_id,
                    checkpoint_id=adapter.binding.checkpoint_id,
                    score_contract_id=GENERATION_SCORE_CONTRACT_ID,
                ),
            ))
    return tuple(outputs)
