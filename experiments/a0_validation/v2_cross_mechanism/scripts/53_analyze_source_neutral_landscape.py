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


def locate_fasta_npz(root: Path) -> Path:
    candidates = sorted(
        path for path in root.rglob("*.npz")
        if path.is_file() and "score_only" in path.parts
    )
    fasta_candidates = [
        path for path in candidates
        if path.stem.endswith("_fasta_1")
    ]
    if len(fasta_candidates) == 1:
        return fasta_candidates[0]
    if len(fasta_candidates) == 0:
        raise ValueError(
            f"{root}: no *_fasta_1.npz found; all NPZ={len(candidates)}."
        )
    raise ValueError(
        f"{root}: multiple *_fasta_1.npz files: "
        f"{[str(path) for path in fasta_candidates]}"
    )


def load_score_npz(
    path: Path,
    expected_repeats: int,
    expected_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"score", "global_score", "S"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}.")
        score = np.asarray(data["score"], dtype=float).reshape(-1)
        global_score = np.asarray(
            data["global_score"], dtype=float
        ).reshape(-1)
        observed_s = np.asarray(data["S"], dtype=int).reshape(-1)

    expected_s = np.asarray(
        [ALPHABET.index(amino_acid) for amino_acid in expected_sequence],
        dtype=int,
    )
    if len(score) != expected_repeats:
        raise ValueError(
            f"{path}: score repeats={len(score)}, expected={expected_repeats}."
        )
    if len(global_score) != expected_repeats:
        raise ValueError(
            f"{path}: global repeats={len(global_score)}, "
            f"expected={expected_repeats}."
        )
    if not np.array_equal(observed_s, expected_s):
        raise ValueError(f"{path}: S does not match candidate sequence.")
    if not np.isfinite(score).all() or not np.isfinite(global_score).all():
        raise ValueError(f"{path}: non-finite score values.")
    return score, global_score


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(
        method="average",
        ascending=True,
    ).to_numpy(float)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    xr, yr = rankdata(x), rankdata(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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
    return (
        None if usable == 0
        else float((concordant - discordant) / usable)
    )


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
    sample_means = values[
        rng.integers(0, len(values), size=(reps, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def landscape_summary(frame: pd.DataFrame) -> dict[str, Any]:
    pdb_values = frame["score_pdb_mean"].to_numpy(float)
    afdb_values = frame["score_afdb_mean"].to_numpy(float)
    flip = pairwise_flip(pdb_values, afdb_values)

    top_overlap = {}
    for k in (5, 10, 20):
        effective_k = min(k, len(frame))
        pdb_ids = set(
            frame.nsmallest(effective_k, "score_pdb_mean")["candidate_id"]
        )
        afdb_ids = set(
            frame.nsmallest(effective_k, "score_afdb_mean")["candidate_id"]
        )
        top_overlap[str(k)] = {
            "effective_k": effective_k,
            "intersection_count": len(pdb_ids & afdb_ids),
            "fraction_of_k": len(pdb_ids & afdb_ids) / effective_k,
            "intersection_candidate_ids": sorted(pdb_ids & afdb_ids),
        }

    return {
        "candidate_count": len(frame),
        "spearman": spearman(pdb_values, afdb_values),
        "kendall_tau_a": kendall_tau_a(pdb_values, afdb_values),
        **flip,
        "top_overlap": top_overlap,
        "mean_absolute_rank_shift": float(
            frame["absolute_rank_shift"].mean()
        ),
        "max_absolute_rank_shift": float(
            frame["absolute_rank_shift"].max()
        ),
        "mean_delta_score_afdb_minus_pdb": float(
            frame["delta_score_afdb_minus_pdb"].mean()
        ),
        "median_delta_score_afdb_minus_pdb": float(
            frame["delta_score_afdb_minus_pdb"].median()
        ),
        "mean_abs_delta_score": float(frame["abs_delta_score"].mean()),
        "median_abs_delta_score": float(frame["abs_delta_score"].median()),
        "max_abs_delta_score": float(frame["abs_delta_score"].max()),
        "robust_delta_ci_excludes_zero_count": int(
            frame["robust_delta_ci_excludes_zero"].sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze source-neutral cross-backbone score stability."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screening-index", type=int, default=36)
    parser.add_argument("--score-repeats", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    args = parser.parse_args()

    project = args.project_root.resolve()
    v2 = project / "experiments/a0_validation/v2_cross_mechanism"
    panel_path = (
        v2 / f"manifests/index{args.screening_index}_source_neutral_candidates.tsv"
    )
    score_root = (
        v2 / f"outputs/index{args.screening_index}/source_neutral_cross_scoring"
    )
    conditioned_path = (
        v2 / f"metrics/index{args.screening_index}_source_conditioned_cross_scoring.json"
    )

    panel = pd.read_csv(panel_path, sep="\t").sort_values(
        "candidate_order",
        kind="mergesort",
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    errors: list[str] = []
    score_arrays: dict[tuple[str, str], np.ndarray] = {}
    long_rows: list[dict[str, Any]] = []

    for _, candidate in panel.iterrows():
        candidate_id = str(candidate["candidate_id"])
        sequence = str(candidate["sequence"])
        for backbone in ("pdb", "afdb"):
            output_dir = score_root / backbone / candidate_id
            try:
                npz_path = locate_fasta_npz(output_dir)
                score, global_score = load_score_npz(
                    npz_path,
                    args.score_repeats,
                    sequence,
                )
            except Exception as exc:
                errors.append(
                    f"{backbone}/{candidate_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            score_arrays[(candidate_id, backbone)] = score
            long_rows.append({
                **candidate.to_dict(),
                "scoring_backbone": backbone,
                "score_npz": str(npz_path),
                "score_mean": float(score.mean()),
                "score_std": float(score.std(ddof=0)),
                "score_min": float(score.min()),
                "score_max": float(score.max()),
                "global_score_mean": float(global_score.mean()),
                "global_score_std": float(global_score.std(ddof=0)),
            })

    expected_records = len(panel) * 2
    if len(score_arrays) != expected_records:
        errors.append(
            f"Loaded score records={len(score_arrays)}, "
            f"expected={expected_records}."
        )

    summary_rows = []
    for _, candidate in panel.iterrows():
        candidate_id = str(candidate["candidate_id"])
        pdb_key = (candidate_id, "pdb")
        afdb_key = (candidate_id, "afdb")
        if pdb_key not in score_arrays or afdb_key not in score_arrays:
            continue

        pdb_score = score_arrays[pdb_key]
        afdb_score = score_arrays[afdb_key]
        paired_delta = afdb_score - pdb_score
        ci_low, ci_high = bootstrap_mean_ci(
            paired_delta,
            args.bootstrap_reps,
            rng,
        )
        summary_rows.append({
            **candidate.to_dict(),
            "score_pdb_mean": float(pdb_score.mean()),
            "score_pdb_std": float(pdb_score.std(ddof=0)),
            "score_afdb_mean": float(afdb_score.mean()),
            "score_afdb_std": float(afdb_score.std(ddof=0)),
            "delta_score_afdb_minus_pdb": float(paired_delta.mean()),
            "delta_score_std": float(paired_delta.std(ddof=0)),
            "delta_score_ci95_low": ci_low,
            "delta_score_ci95_high": ci_high,
            "paired_delta_positive_fraction": float(
                (paired_delta > 0).mean()
            ),
            "abs_delta_score": float(abs(paired_delta.mean())),
            "robust_delta_ci_excludes_zero": bool(
                ci_low > 0 or ci_high < 0
            ),
        })

    summary = pd.DataFrame(summary_rows)
    if len(summary) == len(panel):
        summary["rank_pdb"] = summary["score_pdb_mean"].rank(
            method="average",
            ascending=True,
        )
        summary["rank_afdb"] = summary["score_afdb_mean"].rank(
            method="average",
            ascending=True,
        )
        summary["absolute_rank_shift"] = (
            summary["rank_afdb"] - summary["rank_pdb"]
        ).abs()
    else:
        errors.append(
            f"Candidate summaries={len(summary)}, expected={len(panel)}."
        )

    overall = landscape_summary(summary) if len(summary) == len(panel) else {}

    by_hamming = []
    if not summary.empty:
        for distance, group in summary.loc[
            summary["hamming_distance"] > 0
        ].groupby("hamming_distance", sort=True):
            group = group.copy()
            group_summary = landscape_summary(group)
            group_summary["hamming_distance"] = int(distance)
            by_hamming.append(group_summary)

    variants = summary.loc[
        summary["hamming_distance"] > 0
    ].copy()
    distance_relationship = {}
    if len(variants) >= 2:
        distances = variants["hamming_distance"].to_numpy(float)
        abs_delta = variants["abs_delta_score"].to_numpy(float)
        rank_shift = variants["absolute_rank_shift"].to_numpy(float)
        distance_relationship = {
            "variant_count": len(variants),
            "spearman_hamming_vs_abs_delta": spearman(
                distances,
                abs_delta,
            ),
            "pearson_hamming_vs_abs_delta": pearson(
                distances,
                abs_delta,
            ),
            "spearman_hamming_vs_absolute_rank_shift": spearman(
                distances,
                rank_shift,
            ),
            "pearson_hamming_vs_absolute_rank_shift": pearson(
                distances,
                rank_shift,
            ),
            "mean_abs_delta_by_hamming": {
                str(int(distance)): float(group["abs_delta_score"].mean())
                for distance, group in variants.groupby(
                    "hamming_distance",
                    sort=True,
                )
            },
        }

    conditioned_comparison = None
    if conditioned_path.is_file():
        conditioned = json.loads(
            conditioned_path.read_text(encoding="utf-8")
        )
        conditioned_overall = conditioned.get(
            "overall_source_conditioned_landscape",
            {},
        )
        conditioned_comparison = {
            "source_conditioned": {
                "candidate_count": conditioned_overall.get(
                    "candidate_count"
                ),
                "spearman": conditioned_overall.get("spearman"),
                "kendall_tau_a": conditioned_overall.get(
                    "kendall_tau_a"
                ),
                "flip_rate": conditioned_overall.get("flip_rate"),
                "mean_abs_delta_score": conditioned_overall.get(
                    "mean_abs_delta_score"
                ),
                "mean_absolute_rank_shift": conditioned_overall.get(
                    "mean_absolute_rank_shift"
                ),
            },
            "source_neutral": {
                "candidate_count": overall.get("candidate_count"),
                "spearman": overall.get("spearman"),
                "kendall_tau_a": overall.get("kendall_tau_a"),
                "flip_rate": overall.get("flip_rate"),
                "mean_abs_delta_score": overall.get(
                    "mean_abs_delta_score"
                ),
                "mean_absolute_rank_shift": overall.get(
                    "mean_absolute_rank_shift"
                ),
            },
            "interpretation": (
                "The conditioned panel is generation-origin confounded; "
                "the neutral panel estimates fixed-candidate landscape stability."
            ),
        }

    long_path = (
        v2 / f"manifests/index{args.screening_index}_source_neutral_score_records.tsv"
    )
    summary_path = (
        v2 / f"manifests/index{args.screening_index}_source_neutral_cross_scores.tsv"
    )
    metrics_path = (
        v2 / f"metrics/index{args.screening_index}_source_neutral_landscape.json"
    )
    markdown_path = (
        v2 / f"V2D_INDEX{args.screening_index}_SOURCE_NEUTRAL_LANDSCAPE.md"
    )

    pd.DataFrame(long_rows).to_csv(
        long_path,
        sep="\t",
        index=False,
    )
    summary.to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    audit = {
        "screening_index": args.screening_index,
        "score_repeats": args.score_repeats,
        "loaded_score_records": len(score_arrays),
        "expected_score_records": expected_records,
        "overall_source_neutral_landscape": overall,
        "by_hamming_distance": by_hamming,
        "distance_relationship": distance_relationship,
        "source_conditioned_vs_neutral": conditioned_comparison,
        "interpretation_boundary": (
            "Bootstrap intervals describe ProteinMPNN stochastic repeats within "
            "one protein. Candidates are Monte Carlo probes, not independent "
            "protein-level inference units."
        ),
        "errors": errors,
        "validation_pass": not errors,
        "next_stage": (
            "V2E_residue_probability_localization"
            if not errors
            else "repair_source_neutral_score_outputs"
        ),
    }
    metrics_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# A0-V2D Index {args.screening_index} Source-Neutral Landscape",
        "",
        f"- Loaded score records: `{len(score_arrays)}/{expected_records}`",
        f"- Validation pass: `{not errors}`",
        "",
        "## Overall source-neutral landscape",
        "",
        f"- Candidate count: `{overall.get('candidate_count')}`",
        f"- Spearman: `{overall.get('spearman')}`",
        f"- Kendall tau-a: `{overall.get('kendall_tau_a')}`",
        f"- Flip rate: `{overall.get('flip_rate')}`",
        f"- Mean |delta score|: `{overall.get('mean_abs_delta_score')}`",
        f"- Max |delta score|: `{overall.get('max_abs_delta_score')}`",
        f"- Mean absolute rank shift: "
        f"`{overall.get('mean_absolute_rank_shift')}`",
        f"- Robust delta CIs: "
        f"`{overall.get('robust_delta_ci_excludes_zero_count')}/"
        f"{overall.get('candidate_count')}`",
        "",
        "## By Hamming distance",
        "",
    ]
    for row in by_hamming:
        lines.append(
            f"- H={row['hamming_distance']}: "
            f"n=`{row['candidate_count']}`, "
            f"rho=`{row['spearman']}`, "
            f"tau=`{row['kendall_tau_a']}`, "
            f"flip=`{row['flip_rate']}`, "
            f"mean |delta|=`{row['mean_abs_delta_score']}`"
        )

    lines += [
        "",
        "## Hamming-distance relationship",
        "",
        f"`{distance_relationship}`",
        "",
        "## Source-conditioned versus source-neutral",
        "",
        f"`{conditioned_comparison}`",
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
    print("By Hamming:")
    for row in by_hamming:
        print(row)
    print("Distance relationship:", distance_relationship)
    print("Source-conditioned vs neutral:", conditioned_comparison)
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
