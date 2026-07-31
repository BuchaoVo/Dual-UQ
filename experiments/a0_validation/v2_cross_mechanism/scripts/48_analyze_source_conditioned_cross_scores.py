#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def load_score_npz(
    path: Path,
    expected_repeats: int,
    expected_sequence: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"score", "global_score", "S"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}.")
        score = np.asarray(data["score"], dtype=float).reshape(-1)
        global_score = np.asarray(data["global_score"], dtype=float).reshape(-1)
        sequence_indices = np.asarray(data["S"], dtype=int).reshape(-1)

    if len(score) != expected_repeats:
        raise ValueError(
            f"{path}: score repeats={len(score)}, expected {expected_repeats}."
        )
    if len(global_score) != expected_repeats:
        raise ValueError(
            f"{path}: global_score repeats={len(global_score)}, "
            f"expected {expected_repeats}."
        )
    expected_indices = np.asarray(
        [ALPHABET.index(amino_acid) for amino_acid in expected_sequence],
        dtype=int,
    )
    if not np.array_equal(sequence_indices, expected_indices):
        raise ValueError(f"{path}: scored sequence S does not match candidate.")
    if not np.isfinite(score).all() or not np.isfinite(global_score).all():
        raise ValueError(f"{path}: non-finite score values.")
    return {"score": score, "global_score": global_score}


def locate_score_npz(root: Path) -> Path:
    candidates = sorted(
        path for path in root.rglob("*.npz")
        if path.is_file() and "score_only" in path.parts
    )

    # With --score_only 1 and --path_to_fasta, official ProteinMPNN writes
    # both a structure-sequence score (<name>_pdb.npz) and a supplied-FASTA
    # score (<name>_fasta_1.npz). Only the latter belongs to the candidate
    # cross-scoring panel.
    fasta_candidates = [
        path for path in candidates
        if path.stem.endswith("_fasta_1")
    ]
    if len(fasta_candidates) == 1:
        return fasta_candidates[0]

    if len(fasta_candidates) == 0:
        raise ValueError(
            f"{root}: no *_fasta_1.npz score-only output found; "
            f"all NPZ files={len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )
    raise ValueError(
        f"{root}: multiple *_fasta_1.npz outputs found: "
        f"{[str(path) for path in fasta_candidates]}"
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", ascending=True).to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def kendall_tau_a(x: np.ndarray, y: np.ndarray) -> float | None:
    concordant = discordant = usable = 0
    for left, right in itertools.combinations(range(len(x)), 2):
        dx = np.sign(x[left] - x[right])
        dy = np.sign(y[left] - y[right])
        if dx == 0 or dy == 0:
            continue
        usable += 1
        if dx == dy:
            concordant += 1
        else:
            discordant += 1
    if usable == 0:
        return None
    return float((concordant - discordant) / usable)


def pairwise_flip(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    discordant = usable = 0
    for left, right in itertools.combinations(range(len(x)), 2):
        dx = np.sign(x[left] - x[right])
        dy = np.sign(y[left] - y[right])
        if dx == 0 or dy == 0:
            continue
        usable += 1
        discordant += int(dx != dy)
    return {
        "discordant_pairs": discordant,
        "usable_pairs": usable,
        "flip_rate": None if usable == 0 else discordant / usable,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    samples = values[
        rng.integers(0, len(values), size=(reps, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def exact_origin_permutation(
    deltas: np.ndarray,
    pdb_count: int,
) -> dict[str, Any]:
    deltas = np.asarray(deltas, dtype=float)
    observed = float(
        deltas[:pdb_count].mean() - deltas[pdb_count:].mean()
    )
    indices = range(len(deltas))
    null = []
    for pdb_indices in itertools.combinations(indices, pdb_count):
        pdb_set = set(pdb_indices)
        pdb_values = np.asarray(
            [deltas[index] for index in indices if index in pdb_set]
        )
        afdb_values = np.asarray(
            [deltas[index] for index in indices if index not in pdb_set]
        )
        null.append(float(pdb_values.mean() - afdb_values.mean()))
    null_array = np.asarray(null)
    p_value = float(
        np.mean(np.abs(null_array) >= abs(observed) - 1e-15)
    )
    return {
        "observed_group_difference": observed,
        "exact_allocation_count": len(null),
        "two_sided_exact_p": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze source-conditioned cross-backbone ProteinMPNN scores."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--score-repeats", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    panel_path = (
        v2
        / f"manifests/index{args.screening_index}_source_conditioned_candidates.tsv"
    )
    score_root = (
        v2
        / f"outputs/index{args.screening_index}/source_conditioned_cross_scoring"
    )
    panel = pd.read_csv(panel_path, sep="\t").sort_values(
        "candidate_order",
        kind="mergesort",
    )
    rng = np.random.default_rng(args.bootstrap_seed)

    errors: list[str] = []
    score_arrays: dict[tuple[str, str], np.ndarray] = {}
    global_arrays: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for _, candidate in panel.iterrows():
        candidate_id = str(candidate["candidate_id"])
        record = candidate.to_dict()
        for backbone in ("pdb", "afdb"):
            output = score_root / backbone / candidate_id
            try:
                npz_path = locate_score_npz(output)
                loaded = load_score_npz(
                    npz_path,
                    args.score_repeats,
                    str(candidate["sequence"]),
                )
            except Exception as exc:
                errors.append(
                    f"{backbone}/{candidate_id}: {type(exc).__name__}: {exc}"
                )
                continue
            score_arrays[(candidate_id, backbone)] = loaded["score"]
            global_arrays[(candidate_id, backbone)] = loaded["global_score"]
            rows.append({
                **record,
                "scoring_backbone": backbone,
                "score_npz": str(npz_path),
                "score_mean": float(loaded["score"].mean()),
                "score_std": float(loaded["score"].std(ddof=0)),
                "score_min": float(loaded["score"].min()),
                "score_max": float(loaded["score"].max()),
                "global_score_mean": float(loaded["global_score"].mean()),
                "global_score_std": float(loaded["global_score"].std(ddof=0)),
            })

    expected_records = len(panel) * 2
    if len(score_arrays) != expected_records:
        errors.append(
            f"Loaded score arrays={len(score_arrays)}, expected {expected_records}."
        )

    long_frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []

    for _, candidate in panel.iterrows():
        candidate_id = str(candidate["candidate_id"])
        pdb_key = (candidate_id, "pdb")
        afdb_key = (candidate_id, "afdb")
        if pdb_key not in score_arrays or afdb_key not in score_arrays:
            continue
        pdb_scores = score_arrays[pdb_key]
        afdb_scores = score_arrays[afdb_key]
        paired_delta = afdb_scores - pdb_scores
        ci_low, ci_high = bootstrap_mean_ci(
            paired_delta,
            args.bootstrap_reps,
            rng,
        )
        summary = {
            **candidate.to_dict(),
            "score_pdb_mean": float(pdb_scores.mean()),
            "score_pdb_std": float(pdb_scores.std(ddof=0)),
            "score_afdb_mean": float(afdb_scores.mean()),
            "score_afdb_std": float(afdb_scores.std(ddof=0)),
            "delta_score_afdb_minus_pdb": float(paired_delta.mean()),
            "delta_score_std": float(paired_delta.std(ddof=0)),
            "delta_score_ci95_low": ci_low,
            "delta_score_ci95_high": ci_high,
            "abs_delta_score": float(abs(paired_delta.mean())),
            "paired_delta_positive_fraction": float((paired_delta > 0).mean()),
            "robust_delta_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        }

        origin = str(candidate["candidate_origin"])
        if origin == "pdb_generated":
            home = paired_delta
            expected_delta_sign = 1
        elif origin == "afdb_generated":
            home = -paired_delta
            expected_delta_sign = -1
        else:
            home = None
            expected_delta_sign = 0

        if home is not None:
            home_low, home_high = bootstrap_mean_ci(
                home,
                args.bootstrap_reps,
                rng,
            )
            summary.update({
                "home_advantage_away_minus_home": float(home.mean()),
                "home_advantage_std": float(home.std(ddof=0)),
                "home_advantage_ci95_low": home_low,
                "home_advantage_ci95_high": home_high,
                "home_preference_pass": bool(home.mean() > 0),
                "home_preference_ci_excludes_zero": bool(home_low > 0),
                "origin_sign_prediction_pass": bool(
                    np.sign(paired_delta.mean()) == expected_delta_sign
                ),
            })
            generated_rows.append(summary)
        else:
            summary.update({
                "home_advantage_away_minus_home": None,
                "home_advantage_std": None,
                "home_advantage_ci95_low": None,
                "home_advantage_ci95_high": None,
                "home_preference_pass": None,
                "home_preference_ci_excludes_zero": None,
                "origin_sign_prediction_pass": None,
            })
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    generated_frame = pd.DataFrame(generated_rows)

    overall = {}
    if len(summary_frame) == len(panel):
        pdb_values = summary_frame["score_pdb_mean"].to_numpy(float)
        afdb_values = summary_frame["score_afdb_mean"].to_numpy(float)
        flip = pairwise_flip(pdb_values, afdb_values)

        rank_pdb = pd.Series(pdb_values).rank(
            method="average", ascending=True
        ).to_numpy(float)
        rank_afdb = pd.Series(afdb_values).rank(
            method="average", ascending=True
        ).to_numpy(float)
        summary_frame["rank_pdb"] = rank_pdb
        summary_frame["rank_afdb"] = rank_afdb
        summary_frame["absolute_rank_shift"] = np.abs(rank_afdb - rank_pdb)

        top_overlap = {}
        for k in (3, 5):
            pdb_ids = set(
                summary_frame.nsmallest(k, "score_pdb_mean")["candidate_id"]
            )
            afdb_ids = set(
                summary_frame.nsmallest(k, "score_afdb_mean")["candidate_id"]
            )
            top_overlap[str(k)] = {
                "intersection_count": len(pdb_ids & afdb_ids),
                "fraction_of_k": len(pdb_ids & afdb_ids) / k,
                "intersection_candidate_ids": sorted(pdb_ids & afdb_ids),
            }

        overall = {
            "candidate_count": len(summary_frame),
            "spearman": spearman(pdb_values, afdb_values),
            "kendall_tau_a": kendall_tau_a(pdb_values, afdb_values),
            **flip,
            "top_overlap": top_overlap,
            "mean_absolute_rank_shift": float(
                summary_frame["absolute_rank_shift"].mean()
            ),
            "max_absolute_rank_shift": float(
                summary_frame["absolute_rank_shift"].max()
            ),
            "mean_delta_score_afdb_minus_pdb": float(
                summary_frame["delta_score_afdb_minus_pdb"].mean()
            ),
            "median_delta_score_afdb_minus_pdb": float(
                summary_frame["delta_score_afdb_minus_pdb"].median()
            ),
            "mean_abs_delta_score": float(
                summary_frame["abs_delta_score"].mean()
            ),
            "max_abs_delta_score": float(
                summary_frame["abs_delta_score"].max()
            ),
            "robust_delta_ci_excludes_zero_count": int(
                summary_frame["robust_delta_ci_excludes_zero"].sum()
            ),
        }

    origin = {}
    if len(generated_frame) == 8:
        pdb_generated = generated_frame.loc[
            generated_frame["candidate_origin"] == "pdb_generated"
        ].sort_values("generation_sample")
        afdb_generated = generated_frame.loc[
            generated_frame["candidate_origin"] == "afdb_generated"
        ].sort_values("generation_sample")

        deltas = np.concatenate([
            pdb_generated["delta_score_afdb_minus_pdb"].to_numpy(float),
            afdb_generated["delta_score_afdb_minus_pdb"].to_numpy(float),
        ])
        permutation = exact_origin_permutation(
            deltas,
            pdb_count=len(pdb_generated),
        )
        origin = {
            "generated_candidate_count": len(generated_frame),
            "all_generated_prefer_home_backbone": bool(
                generated_frame["home_preference_pass"].all()
            ),
            "home_preference_pass_count": int(
                generated_frame["home_preference_pass"].sum()
            ),
            "home_preference_ci_excludes_zero_count": int(
                generated_frame["home_preference_ci_excludes_zero"].sum()
            ),
            "origin_sign_prediction_count": int(
                generated_frame["origin_sign_prediction_pass"].sum()
            ),
            "origin_sign_prediction_accuracy": float(
                generated_frame["origin_sign_prediction_pass"].mean()
            ),
            "mean_home_advantage": float(
                generated_frame["home_advantage_away_minus_home"].mean()
            ),
            "median_home_advantage": float(
                generated_frame["home_advantage_away_minus_home"].median()
            ),
            "minimum_home_advantage": float(
                generated_frame["home_advantage_away_minus_home"].min()
            ),
            "maximum_home_advantage": float(
                generated_frame["home_advantage_away_minus_home"].max()
            ),
            "pdb_origin_mean_delta": float(
                pdb_generated["delta_score_afdb_minus_pdb"].mean()
            ),
            "pdb_origin_std_delta": float(
                pdb_generated["delta_score_afdb_minus_pdb"].std(ddof=0)
            ),
            "afdb_origin_mean_delta": float(
                afdb_generated["delta_score_afdb_minus_pdb"].mean()
            ),
            "afdb_origin_std_delta": float(
                afdb_generated["delta_score_afdb_minus_pdb"].std(ddof=0)
            ),
            **permutation,
            "source_conditioning_confounding_flag": True,
        }
    elif not errors:
        errors.append(
            f"Generated candidate rows={len(generated_frame)}, expected 8."
        )

    long_path = (
        v2
        / f"manifests/index{args.screening_index}_source_conditioned_score_repeats.tsv"
    )
    summary_path = (
        v2
        / f"manifests/index{args.screening_index}_source_conditioned_cross_scores.tsv"
    )
    metrics_path = (
        v2
        / f"metrics/index{args.screening_index}_source_conditioned_cross_scoring.json"
    )
    markdown_path = (
        v2
        / f"V2C_INDEX{args.screening_index}_SOURCE_CONDITIONED.md"
    )

    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(long_path, sep="\t", index=False)
    summary_frame.to_csv(summary_path, sep="\t", index=False)

    audit = {
        "screening_index": args.screening_index,
        "score_repeats": args.score_repeats,
        "loaded_score_records": len(score_arrays),
        "expected_score_records": expected_records,
        "overall_source_conditioned_landscape": overall,
        "origin_conditioned_specialization": origin,
        "interpretation_boundary": (
            "Candidate origin is confounded with sequence region. Overall ranking "
            "metrics are source-conditioned diagnostics and do not estimate a "
            "source-neutral landscape."
        ),
        "errors": errors,
        "validation_pass": not errors,
        "next_stage": (
            "V2D_source_neutral_landscape"
            if not errors
            else "repair_generation_or_score_outputs"
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# A0-V2C Index {args.screening_index} Source-Conditioned Cross-Scoring",
        "",
        f"- Loaded score records: `{len(score_arrays)}/{expected_records}`",
        f"- Validation pass: `{not errors}`",
        "",
        "## Overall source-conditioned diagnostic",
        "",
        f"- Spearman: `{overall.get('spearman')}`",
        f"- Kendall tau-a: `{overall.get('kendall_tau_a')}`",
        f"- Pairwise flip: `{overall.get('discordant_pairs')}/"
        f"{overall.get('usable_pairs')}`",
        f"- Flip rate: `{overall.get('flip_rate')}`",
        f"- Mean absolute rank shift: `{overall.get('mean_absolute_rank_shift')}`",
        f"- Mean |delta score|: `{overall.get('mean_abs_delta_score')}`",
        "",
        "## Origin-conditioned specialization",
        "",
        f"- All generated prefer home backbone: "
        f"`{origin.get('all_generated_prefer_home_backbone')}`",
        f"- Home-preference pass: "
        f"`{origin.get('home_preference_pass_count')}/"
        f"{origin.get('generated_candidate_count')}`",
        f"- Robust home-preference CI: "
        f"`{origin.get('home_preference_ci_excludes_zero_count')}/"
        f"{origin.get('generated_candidate_count')}`",
        f"- Mean home advantage: `{origin.get('mean_home_advantage')}`",
        f"- Origin sign prediction: "
        f"`{origin.get('origin_sign_prediction_count')}/"
        f"{origin.get('generated_candidate_count')}`",
        f"- Exact two-sided permutation p: "
        f"`{origin.get('two_sided_exact_p')}`",
        "",
        "## Candidate-level results",
        "",
    ]
    for _, row in summary_frame.iterrows():
        lines.append(
            f"- `{row['candidate_id']}` origin=`{row['candidate_origin']}`: "
            f"PDB=`{row['score_pdb_mean']}`, AFDB=`{row['score_afdb_mean']}`, "
            f"delta(AFDB-PDB)=`{row['delta_score_afdb_minus_pdb']}`, "
            f"home advantage=`{row['home_advantage_away_minus_home']}`, "
            f"robust delta=`{row['robust_delta_ci_excludes_zero']}`"
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        audit["interpretation_boundary"],
        "",
    ]
    if errors:
        lines += ["## Errors", ""]
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print("Loaded score records:", len(score_arrays), "/", expected_records)
    print("Overall:", overall)
    print("Origin-conditioned:", origin)
    print("Errors:", errors)
    print("Validation pass:", not errors)
    print("Next stage:", audit["next_stage"])
    print("Wrote:", long_path)
    print("Wrote:", summary_path)
    print("Wrote:", metrics_path)
    print("Wrote:", markdown_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
