#!/usr/bin/env python3
"""Build the MODAL experiment table used in the paper."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


SELECTED_MORPHOSYNTAX = [
    "lexicon_adverb",
    "lexicon_adverbial",
    "lexicon_modal_verb",
    "lexicon_CTP",
    "lexicon_verb_CTP",
]


def coarse_evidence(label: str) -> str:
    if label == "no_evidence":
        return "no_evidence"
    if label == "indirect_inferential":
        return "inferential"
    if label in {"quotative", "indirect_reportive"}:
        return "reported"
    if label in {"memory", "direct_visual", "direct_auditory", "direct_feeling"}:
        return "situated"
    return "other"


def foil_target(target: str, marker: str, foil: str, scope: str) -> str:
    target = target.strip()
    marker = marker.strip()
    foil = foil.strip()
    scope = scope.strip()
    if target.lower().startswith(marker.lower()):
        return foil + target[len(marker) :]
    if scope:
        return f"{foil} {scope}"
    return foil


def choose_foil(row: pd.Series, candidates: pd.DataFrame, rng: random.Random) -> pd.Series:
    pool = candidates[
        (candidates["row_id"] != row["row_id"])
        & (candidates["language"] == row["language"])
        & (candidates["marker_morphosyntax"] == row["marker_morphosyntax"])
        & (candidates["marker_archi_unit"] != row["marker_archi_unit"])
    ].copy()
    if len(pool) == 0:
        pool = candidates[
            (candidates["row_id"] != row["row_id"])
            & (candidates["language"] == row["language"])
            & (candidates["marker_archi_unit"] != row["marker_archi_unit"])
        ].copy()
    if len(pool) == 0:
        pool = candidates[(candidates["row_id"] != row["row_id"])].copy()

    label_mismatch = (
        (pool["relation_construction_polarity_norm"] != row["relation_construction_polarity_norm"]).astype(int)
        + (pool["evidence_coarse"] != row["evidence_coarse"]).astype(int)
        + (pool["function_coarse"] != row["function_coarse"]).astype(int)
    )
    pool["label_mismatch"] = label_mismatch
    pool["length_gap"] = (pool["marker_text"].str.len() - len(row["marker_text"])).abs()
    best = pool.sort_values(
        ["label_mismatch", "length_gap", "marker_text"],
        ascending=[False, True, True],
    ).head(12)
    return best.iloc[rng.randrange(len(best))]


def build_dataset(input_path: Path, output_path: Path, summary_path: Path, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = pd.read_csv(input_path, sep="\t").fillna("")
    df["row_id"] = [f"modal_{idx:04d}" for idx in range(len(df))]

    keep = df[
        df["marker_morphosyntax"].isin(SELECTED_MORPHOSYNTAX)
        & df["prefix_context"].str.len().gt(0)
        & df["marker_text"].str.len().gt(0)
        & df["target_from_marker"].str.len().gt(0)
        & df["relation_construction_polarity_norm"].isin(["positive", "neutral", "negative"])
        & df["relation_construction_epistemic_type_norm"].str.len().gt(0)
    ].copy()
    keep["evidence_coarse"] = keep["relation_construction_epistemic_type_norm"].map(coarse_evidence)
    keep["function_coarse"] = keep["relation_construction_function_norm"].where(
        keep["relation_construction_function_norm"].eq("qualification"),
        "other",
    )
    keep["gold_continuation"] = keep["target_from_marker"]

    foils = []
    for _, row in keep.iterrows():
        foil = choose_foil(row, keep, rng)
        foils.append(
            {
                "foil_marker_text": foil["marker_text"],
                "foil_marker_lemma": foil["marker_lemma"],
                "foil_marker_archi_unit": foil["marker_archi_unit"],
                "foil_marker_morphosyntax": foil["marker_morphosyntax"],
                "foil_polarity": foil["relation_construction_polarity_norm"],
                "foil_evidence_coarse": foil["evidence_coarse"],
                "foil_function_coarse": foil["function_coarse"],
                "foil_continuation": foil_target(
                    row["target_from_marker"],
                    row["marker_text"],
                    foil["marker_text"],
                    row["scope_text"],
                ),
            }
        )
    keep = pd.concat([keep.reset_index(drop=True), pd.DataFrame(foils)], axis=1)

    columns = [
        "row_id",
        "language",
        "source_file",
        "relation_id",
        "marker_text",
        "marker_lemma",
        "marker_archi_unit",
        "marker_morphosyntax",
        "scope_text",
        "prefix_context",
        "turn_prefix_1",
        "turn_prefix_3",
        "turn_prefix_5",
        "turn_prefix_10",
        "turn_prefix_20",
        "gold_continuation",
        "foil_continuation",
        "foil_marker_text",
        "foil_marker_lemma",
        "foil_marker_archi_unit",
        "foil_marker_morphosyntax",
        "relation_construction_polarity_norm",
        "relation_construction_epistemic_type_norm",
        "relation_construction_function_norm",
        "evidence_coarse",
        "function_coarse",
        "scope_illocution",
        "relation_construction_direction",
        "relation_construction_source_norm",
        "relation_context_environment",
        "relation_context_turn_taking",
        "relation_context_age_group",
        "relation_metadata_file",
        "context",
    ]
    keep = keep[columns].sort_values(["language", "row_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(output_path, sep="\t", index=False)

    summary = {
        "n_rows": int(len(keep)),
        "seed": seed,
        "selected_marker_morphosyntax": SELECTED_MORPHOSYNTAX,
        "by_language": keep["language"].value_counts().to_dict(),
        "polarity": keep["relation_construction_polarity_norm"].value_counts().to_dict(),
        "evidence_coarse": keep["evidence_coarse"].value_counts().to_dict(),
        "function_coarse": keep["function_coarse"].value_counts().to_dict(),
        "marker_morphosyntax": keep["marker_morphosyntax"].value_counts().to_dict(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_relations.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_experiment_dataset.tsv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_experiment_dataset_summary.json"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    dataset = build_dataset(args.input, args.output, args.summary, args.seed)
    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
