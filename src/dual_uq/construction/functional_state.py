"""Functional-state-specific policy; generic parsing/mapping stays elsewhere."""

from typing import Any

import pandas as pd

from dual_uq.benchmark.condition_semantics import orientation_for


def normalize_functional_state(state_family: str, state_label: str) -> tuple[str, str]:
    first, second = orientation_for("functional_state", state_family)
    label = str(state_label).strip().upper().replace("-", "_").replace(" ", "_")
    if label not in {first, second}:
        raise ValueError(f"invalid {state_family} state label: {state_label}")
    return state_family, label


def orient_functional_state_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Normalize discovery pair rows to the registry orientation.

    Discovery column names are historical (``*_a``/``*_b``).  Their values are
    swapped as a complete side bundle before any shared mapper sees the row.
    """
    required = {"state_family", "state_a", "state_b", "entity_a", "entity_b"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"functional-state pairs missing orientation fields: {missing}")
    rows: list[dict[str, Any]] = []
    for source in pairs.to_dict(orient="records"):
        family = str(source["state_family"])
        first, second = orientation_for("functional_state", family)
        if {str(source["state_a"]), str(source["state_b"])} != {first, second}:
            raise ValueError(f"invalid functional-state labels for {family}")
        row = dict(source)
        if str(source["state_a"]) != first:
            for key in tuple(source):
                if key.endswith("_a"):
                    twin = f"{key[:-2]}_b"
                    if twin in source:
                        row[key], row[twin] = source[twin], source[key]
        row["structural_condition_semantics"] = "functional_state"
        rows.append(row)
    return pd.DataFrame(rows, columns=pairs.columns).reset_index(drop=True)


def classify_ligand_overlap(
    state_a: str,
    state_b: str,
    *,
    ligand_context_changed: bool | None,
    functional_evidence: bool = True,
) -> str:
    """Classify ligand context without turning it into a state label."""
    del state_a, state_b
    if ligand_context_changed is None:
        return "UNRESOLVED_LIGAND_CONTEXT"
    if not ligand_context_changed:
        return "FUNCTIONAL_STATE_INDEPENDENT_OF_LIGAND_LABEL"
    if not functional_evidence:
        return "PURE_LIGAND_STATE_ONLY"
    return "FUNCTIONAL_STATE_WITH_LIGAND_CONTEXT"


def _tier_rank(value: Any) -> int:
    return {"TIER_1": 3, "TIER_2": 2, "TIER_3": 1}.get(str(value), 0)


def select_primary_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one outcome-blind primary per protein and state family."""
    if pairs.empty:
        return pairs.copy(), pairs.copy()
    oriented = orient_functional_state_pairs(pairs)
    category = oriented["state_contrast_category"] if "state_contrast_category" in oriented else pd.Series(index=oriented.index, dtype=object)
    admission = oriented["admission_status"] if "admission_status" in oriented else pd.Series("ADMITTED", index=oriented.index)
    eligible = oriented.loc[
        ~category.eq("PURE_LIGAND_STATE_ONLY") & ~admission.isin(["EXCLUDED", "UNRESOLVED"])
    ].copy()
    eligible["_evidence_quality"] = eligible.apply(
        lambda row: min(_tier_rank(row.get("evidence_tier_a")), _tier_rank(row.get("evidence_tier_b"))), axis=1
    )
    assembly = eligible["assembly_comparable"] if "assembly_comparable" in eligible else pd.Series(True, index=eligible.index)
    mutation = eligible["engineered_mutation_burden"] if "engineered_mutation_burden" in eligible else pd.Series(0, index=eligible.index)
    for column, default in (("common_mapped_count", 0), ("common_coordinate_visible_count", 0), ("construct_overlap_fraction", 0.0)):
        if column not in eligible:
            eligible[column] = default
    eligible["_assembly_rank"] = assembly.map(lambda value: int(bool(value)))
    eligible["_mutation_burden"] = mutation.fillna(0)
    eligible = eligible.sort_values(
        ["uniprot_id", "state_family", "_evidence_quality", "common_mapped_count", "common_coordinate_visible_count", "construct_overlap_fraction", "_assembly_rank", "_mutation_burden", "pair_id"],
        ascending=[True, True, False, False, False, False, False, True, True],
        kind="mergesort",
    )
    primary_indices = eligible.groupby(["uniprot_id", "state_family"], sort=False).head(1).index
    primary = eligible.loc[primary_indices].copy().sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    alternative = eligible.drop(index=primary_indices).copy().sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    primary["final_pair_state"] = "PRIMARY"
    alternative["final_pair_state"] = "ALTERNATIVE"
    return primary.drop(columns=["_evidence_quality", "_assembly_rank", "_mutation_burden"], errors="ignore"), alternative.drop(columns=["_evidence_quality", "_assembly_rank", "_mutation_burden"], errors="ignore")


def close_candidate_attrition(candidates: pd.DataFrame) -> pd.DataFrame:
    """Require every discovery candidate to have one terminal state."""
    if "pair_id" not in candidates or "final_pair_state" not in candidates:
        raise ValueError("candidate attrition requires pair_id and final_pair_state")
    result = candidates.copy()
    allowed = {"PRIMARY", "ALTERNATIVE", "EXCLUDED", "UNRESOLVED"}
    result["terminal_status"] = result["final_pair_state"].astype("string")
    invalid = sorted(set(result["terminal_status"].dropna()) - allowed)
    if invalid:
        raise ValueError(f"candidate attrition has non-terminal states: {invalid}")
    result["terminal_reason"] = result.get("admission_reasons", pd.Series(index=result.index, dtype="object"))
    result.loc[result["terminal_status"].isin(["EXCLUDED", "UNRESOLVED"]), "terminal_reason"] = result.loc[result["terminal_status"].isin(["EXCLUDED", "UNRESOLVED"]), "terminal_reason"].fillna("other_explicit_reason")
    result["structural_condition_semantics"] = "functional_state"
    result["outcome_blind"] = True
    return result


def validate_attrition_closure(candidates: pd.DataFrame) -> dict[str, int]:
    """Validate and summarize the four-state candidate terminal partition."""

    if "pair_id" not in candidates or "final_pair_state" not in candidates:
        raise ValueError("candidate attrition requires pair_id and final_pair_state")
    if candidates["pair_id"].duplicated().any():
        raise ValueError("candidate attrition contains duplicate pair_id")
    allowed = {"PRIMARY", "ALTERNATIVE", "EXCLUDED", "UNRESOLVED"}
    statuses = candidates["final_pair_state"].astype("string")
    invalid = sorted(set(statuses.dropna()) - allowed)
    if invalid:
        raise ValueError(f"candidate attrition has non-terminal states: {invalid}")
    counts = {
        "candidate_pairs": len(candidates),
        "primary_pairs": int(statuses.eq("PRIMARY").sum()),
        "alternative_pairs": int(statuses.eq("ALTERNATIVE").sum()),
        "excluded_pairs": int(statuses.eq("EXCLUDED").sum()),
        "unresolved_pairs": int(statuses.eq("UNRESOLVED").sum()),
    }
    if counts["candidate_pairs"] != sum(counts[key] for key in counts if key != "candidate_pairs"):
        raise ValueError("candidate attrition closure is incomplete")
    return counts
