"""Generation-level consequences of paired structural representations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dual_uq.models.proteinmpnn import PROTEINMPNN_ALPHABET, ProteinMPNNStructureInput

from .cross_model_representation_sensitivity import cluster_bootstrap_summary

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _constant(group: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in group:
        return default
    values = group[column].drop_duplicates()
    if len(values) > 1:
        raise ValueError(f"pair metadata is not constant: {column}")
    return values.iloc[0] if len(values) else default


@dataclass(frozen=True, slots=True)
class GreedyGeneration:
    """One sequence and the exact autoregressive position order used to make it."""

    sequence: str
    decoding_order: tuple[int, ...]


def normalized_hamming(left: str, right: str) -> float:
    """Return sequence mismatch fraction on one already-matched canonical axis."""

    if not left or len(left) != len(right):
        raise ValueError("paired sequences must have equal positive length")
    if set(left + right).difference(STANDARD_AMINO_ACIDS):
        raise ValueError("paired sequences must use the standard 20-AA alphabet")
    return float(sum(a != b for a, b in zip(left, right, strict=True)) / len(left))


@contextmanager
def greedy_multinomial(torch_module: Any) -> Iterator[None]:
    """Run an official sampler with its token draw replaced by exact argmax.

    The model's encoder, decoder, autoregressive state, masks, and position order
    remain untouched. This context is deliberately process-local and must only be
    used by a single-threaded scoring worker.
    """

    original = torch_module.multinomial

    def argmax_draw(probabilities: Any, num_samples: int, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if num_samples != 1:
            raise ValueError("greedy decoding only supports one token draw")
        return probabilities.argmax(dim=-1, keepdim=True)

    torch_module.multinomial = argmax_draw
    try:
        yield
    finally:
        torch_module.multinomial = original


def proteinmpnn_greedy(
    adapter: Any,
    structure: ProteinMPNNStructureInput,
    decoding_order: tuple[int, ...],
) -> GreedyGeneration:
    """Greedily decode through ProteinMPNN's official autoregressive sampler."""

    length = structure.residue_count
    if tuple(sorted(decoding_order)) != tuple(range(length)):
        raise ValueError("ProteinMPNN decoding order must be a position permutation")
    torch = adapter.torch
    alphabet = PROTEINMPNN_ALPHABET
    aa_index = {aa: index for index, aa in enumerate(alphabet)}
    x = torch.as_tensor(
        np.asarray(structure.coordinates, dtype=np.float32)[None],
        dtype=torch.float32,
        device=adapter.device,
    )
    target = torch.as_tensor(
        [[aa_index[aa] for aa in structure.wt_sequence_projection]],
        dtype=torch.long,
        device=adapter.device,
    )
    mask = torch.ones((1, length), dtype=torch.float32, device=adapter.device)
    residue_idx = torch.as_tensor(
        [structure.uniprot_positions], dtype=torch.long, device=adapter.device
    )
    chain_encoding = torch.ones_like(residue_idx)
    # Official sample() obtains its order by argsort(abs(randn)). These ranks
    # encode the frozen order exactly and contain no fresh randomness.
    ranks = np.empty(length, dtype=np.float32)
    ranks[np.asarray(decoding_order)] = np.arange(1, length + 1, dtype=np.float32)
    randn = torch.as_tensor(ranks[None], device=adapter.device)
    zeros_aa = np.zeros(len(alphabet), dtype=np.float32)
    omit_aa = zeros_aa.copy()
    omit_aa[-1] = 1.0  # generation remains on the canonical standard-20 axis
    with greedy_multinomial(torch), torch.inference_mode():
        sampled = adapter.model.sample(
            X=x,
            randn=randn,
            S_true=target,
            chain_mask=mask,
            chain_encoding_all=chain_encoding,
            residue_idx=residue_idx,
            mask=mask,
            temperature=1.0,
            omit_AAs_np=omit_aa,
            bias_AAs_np=zeros_aa,
            chain_M_pos=mask,
            omit_AA_mask=torch.zeros(
                (1, length, len(alphabet)), dtype=torch.float32, device=adapter.device
            ),
            pssm_coef=torch.zeros((1, length), dtype=torch.float32, device=adapter.device),
            pssm_bias=torch.zeros(
                (1, length, len(alphabet)), dtype=torch.float32, device=adapter.device
            ),
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=None,
            pssm_bias_flag=False,
            bias_by_res=torch.zeros(
                (1, length, len(alphabet)), dtype=torch.float32, device=adapter.device
            ),
        )
    indices = np.asarray(sampled["S"].detach().cpu(), dtype=np.int64)[0]
    realized_order = tuple(
        int(value) for value in np.asarray(sampled["decoding_order"].detach().cpu())[0]
    )
    if realized_order != decoding_order:
        raise RuntimeError("ProteinMPNN did not preserve the frozen decoding order")
    sequence = "".join(alphabet[index] for index in indices)
    if set(sequence).difference(STANDARD_AMINO_ACIDS):
        raise RuntimeError("ProteinMPNN greedy output left the standard-AA axis")
    return GreedyGeneration(sequence, realized_order)


def build_generation_response(
    cases: pd.DataFrame,
    generations: dict[tuple[str, str], GreedyGeneration],
    *,
    model_id: str,
    checkpoint_id: str,
    regime: str,
) -> pd.DataFrame:
    """Compare paired generated sequences after strict canonical-axis checks."""

    required = {
        "pair_id",
        "protein_id",
        "condition",
        "condition_label",
        "canonical_position",
        "wt_sequence_projection",
    }
    missing = sorted(required.difference(cases.columns))
    if missing or cases.empty:
        raise ValueError(f"generation cases missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for pair_id, group in cases.groupby("pair_id", sort=True):
        conditions = tuple(sorted(group["condition"].astype(str).unique()))
        if conditions != ("CONDITION_1", "CONDITION_2"):
            raise ValueError(f"pair must contain exactly two ordered conditions: {pair_id}")
        axes: list[tuple[int, ...]] = []
        outputs: list[GreedyGeneration] = []
        labels: list[str] = []
        for condition in conditions:
            selected = group.loc[group["condition"].astype(str).eq(condition)].sort_values(
                "canonical_position", kind="mergesort"
            )
            axes.append(tuple(int(value) for value in selected["canonical_position"]))
            labels.append(str(selected["condition_label"].iloc[0]))
            try:
                outputs.append(generations[(str(pair_id), condition)])
            except KeyError as exc:
                raise ValueError(f"missing greedy generation: {pair_id}/{condition}") from exc
        if axes[0] != axes[1]:
            raise ValueError(f"paired canonical axes differ: {pair_id}")
        if outputs[0].decoding_order != outputs[1].decoding_order:
            raise ValueError(f"paired decoding orders differ: {pair_id}")
        sequence_values = group["wt_sequence_projection"].astype(str).drop_duplicates()
        if len(sequence_values) != 1:
            raise ValueError(f"native projection is not constant: {pair_id}")
        native = str(sequence_values.iloc[0])
        if len(native) != len(axes[0]) or any(len(item.sequence) != len(native) for item in outputs):
            raise ValueError(f"generated sequence length differs from canonical axis: {pair_id}")

        drift = normalized_hamming(outputs[0].sequence, outputs[1].sequence)
        rows.append(
            {
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "regime": regime,
                "pair_id": str(pair_id),
                "protein_id": str(_constant(group, "protein_id", "")),
                "identity_cluster_id": _constant(group, "identity_cluster_id"),
                "perturbation_family": _constant(group, "perturbation_family"),
                "requested_dose": _constant(group, "requested_dose"),
                "reference_label": labels[0],
                "comparison_label": labels[1],
                "n_evaluable_positions": len(native),
                "generated_sequence_left": outputs[0].sequence,
                "generated_sequence_right": outputs[1].sequence,
                "decoding_order": list(outputs[0].decoding_order),
                "r_greedy": drift,
                "sequence_identity": 1.0 - drift,
                "recovery_left": 1.0 - normalized_hamming(outputs[0].sequence, native),
                "recovery_right": 1.0 - normalized_hamming(outputs[1].sequence, native),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_id", "regime", "pair_id"], kind="mergesort", ignore_index=True
    )


def protein_generation_response(response: pd.DataFrame) -> pd.DataFrame:
    """Average repeated structural pairs before any cross-protein inference."""

    keys = ["model_id", "checkpoint_id", "regime", "protein_id", "identity_cluster_id"]
    required = {*keys, "pair_id", "r_greedy", "recovery_left", "recovery_right"}
    missing = sorted(required.difference(response.columns))
    if missing:
        raise ValueError(f"generation response missing columns: {missing}")
    grouped = response.groupby(keys, as_index=False, dropna=False, sort=True)
    result = grouped[["r_greedy", "recovery_left", "recovery_right"]].mean()
    result["n_pairs"] = grouped["pair_id"].nunique()["pair_id"]
    return result


def summarize_generation(response: pd.DataFrame, *, replicates: int = 10_000) -> pd.DataFrame:
    """Return one identity-cluster bootstrap summary per model and regime."""

    proteins = protein_generation_response(response)
    rows = []
    for (model, checkpoint, regime), group in proteins.groupby(
        ["model_id", "checkpoint_id", "regime"], sort=True
    ):
        statistics = cluster_bootstrap_summary(
            group,
            value_column="r_greedy",
            bootstrap_replicates=replicates,
            seed=2_026_09_08,
        )
        rows.append(
            {
                "model_id": model,
                "checkpoint_id": checkpoint,
                "regime": regime,
                **statistics,
                "fraction_nonzero": float(group["r_greedy"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_id", "regime"], kind="mergesort", ignore_index=True
    )


def generation_dose_response(response: pd.DataFrame) -> pd.DataFrame:
    """Build strict high-minus-low controlled contrasts without imputation."""

    controlled = response.loc[response["regime"].eq("controlled")].copy()
    required = {
        "model_id",
        "checkpoint_id",
        "protein_id",
        "identity_cluster_id",
        "pair_id",
        "requested_dose",
        "r_greedy",
    }
    missing = sorted(required.difference(controlled.columns))
    if missing or controlled.empty:
        raise ValueError(f"controlled generation response missing columns: {missing}")
    keys = ["model_id", "checkpoint_id", "protein_id", "identity_cluster_id"]
    means = controlled.groupby(keys + ["requested_dose"], as_index=False, sort=True)[
        "r_greedy"
    ].mean()
    doses = sorted(float(value) for value in means["requested_dose"].dropna().unique())
    if doses != [0.25, 0.5]:
        raise ValueError(f"expected frozen 0.25/0.50 doses, found {doses}")
    low = means.loc[means["requested_dose"].eq(doses[0])]
    high = means.loc[means["requested_dose"].eq(doses[1])]
    joined = low.merge(high, on=keys, how="inner", suffixes=("_low", "_high"), validate="one_to_one")
    result = joined[keys].copy()
    result["requested_dose_low"] = doses[0]
    result["requested_dose_high"] = doses[1]
    result["r_greedy_low"] = joined["r_greedy_low"]
    result["r_greedy_high"] = joined["r_greedy_high"]
    result["r_greedy_high_minus_low"] = joined["r_greedy_high"] - joined["r_greedy_low"]
    return result.sort_values(keys, kind="mergesort", ignore_index=True)
