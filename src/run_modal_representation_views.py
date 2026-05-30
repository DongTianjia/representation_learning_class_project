#!/usr/bin/env python3
"""Probe modal labels from marker-aware and turn-context representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_modal_experiments import choose_device, embed_texts, model_slug


LABELS = {
    "polarity": "relation_construction_polarity_norm",
    "evidence": "evidence_coarse",
    "function": "function_coarse",
}
TURN_WINDOWS = (1, 3, 5, 10, 20)
KEY_VIEWS = (
    "marker_only",
    "marker_scope",
    "context_only_10",
    "context_marker_10",
    "context_marker_scope_10",
)


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def make_splitter(labels: np.ndarray, groups: np.ndarray | None, split: str):
    if split == "lemma_group":
        codes = pd.Categorical(labels).codes
        group_counts = pd.DataFrame({"label": labels, "group": groups}).drop_duplicates().groupby("label").size()
        n_splits = int(max(2, min(5, group_counts.min())))
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=17), groups, n_splits

    min_class = min(np.bincount(pd.Categorical(labels).codes))
    n_splits = int(max(2, min(5, min_class)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17), None, n_splits


def probe_features(features, labels: np.ndarray, kind: str, split: str, groups: np.ndarray | None) -> dict[str, float]:
    splitter, split_groups, n_splits = make_splitter(labels, groups, split)
    if kind == "majority":
        estimator = DummyClassifier(strategy="most_frequent")
        probe_input = np.zeros((len(labels), 1))
    elif kind == "tfidf":
        estimator = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
        probe_input = features
    else:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
        probe_input = features
    pred = cross_val_predict(estimator, probe_input, labels, cv=splitter, groups=split_groups)
    out = metric_row(labels, pred)
    out["n_splits"] = float(n_splits)
    return out


def build_views(df: pd.DataFrame) -> dict[str, pd.Series]:
    views = {
        "marker_only": df["marker_text"].astype(str),
        "marker_scope": df["gold_continuation"].astype(str),
    }
    for window in TURN_WINDOWS:
        prefix = df[f"turn_prefix_{window}"].astype(str)
        views[f"context_only_{window}"] = prefix
        views[f"context_marker_{window}"] = (prefix + " " + df["marker_text"].astype(str)).str.strip()
        views[f"context_marker_scope_{window}"] = (prefix + " " + df["gold_continuation"].astype(str)).str.strip()
    return views


def within_lemma_subset(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    group = df.groupby(["language", "marker_lemma"])[label_column]
    variable_keys = group.nunique()
    variable_keys = set(variable_keys[variable_keys > 1].index)
    mask = [(row.language, row.marker_lemma) in variable_keys for row in df.itertuples(index=False)]
    subset = df[mask].copy()
    counts = subset[label_column].value_counts()
    keep_labels = set(counts[counts >= 5].index)
    return subset[subset[label_column].isin(keep_labels)].copy()


def run_tfidf_views(df: pd.DataFrame, views: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    groups = (df["language"].astype(str) + "::" + df["marker_lemma"].astype(str)).to_numpy()
    for label_name, label_column in LABELS.items():
        labels = df[label_column].astype(str).to_numpy()
        for split in ["random", "lemma_group"]:
            base = probe_features(None, labels, "majority", split, groups)
            rows.append({"representation": "majority", "view": "none", "split": split, "label": label_name, "n": len(df), **base})
            for view_name, texts in views.items():
                out = probe_features(texts.tolist(), labels, "tfidf", split, groups)
                rows.append({"representation": "tfidf_char_3_5", "view": view_name, "split": split, "label": label_name, "n": len(df), **out})

        subset = within_lemma_subset(df, label_column)
        if len(subset) >= 25 and subset[label_column].nunique() > 1:
            subset_views = build_views(subset)
            subset_labels = subset[label_column].astype(str).to_numpy()
            subset_groups = (subset["language"].astype(str) + "::" + subset["marker_lemma"].astype(str)).to_numpy()
            base = probe_features(None, subset_labels, "majority", "random", subset_groups)
            rows.append({"representation": "majority", "view": "none", "split": "within_lemma", "label": label_name, "n": len(subset), **base})
            for view_name, texts in subset_views.items():
                out = probe_features(texts.tolist(), subset_labels, "tfidf", "random", subset_groups)
                rows.append({"representation": "tfidf_char_3_5", "view": view_name, "split": "within_lemma", "label": label_name, "n": len(subset), **out})
    return pd.DataFrame(rows)


def run_dense_key_views(
    df: pd.DataFrame,
    views: dict[str, pd.Series],
    models: list[str],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    rows = []
    groups = (df["language"].astype(str) + "::" + df["marker_lemma"].astype(str)).to_numpy()
    for model_name in models:
        for view_name in KEY_VIEWS:
            print(f"Embedding {view_name} with {model_name}")
            embeddings = embed_texts(model_name, views[view_name].tolist(), device, batch_size, max_length)
            np.save(output_dir / f"embeddings_{model_slug(model_name)}_{view_name}.npy", embeddings)
            for label_name, label_column in LABELS.items():
                labels = df[label_column].astype(str).to_numpy()
                for split in ["random", "lemma_group"]:
                    out = probe_features(embeddings, labels, "dense", split, groups)
                    rows.append(
                        {
                            "representation": model_name,
                            "view": view_name,
                            "split": split,
                            "label": label_name,
                            "n": len(df),
                            **out,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_experiment_dataset.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/modal_corpus/experiments"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dense-models",
        nargs="*",
        default=["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Qwen/Qwen3-0.6B"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset, sep="\t").fillna("")
    views = build_views(df)
    tfidf = pd.DataFrame()
    if not args.skip_tfidf:
        tfidf = run_tfidf_views(df, views)
        tfidf.to_csv(args.output_dir / "representation_view_tfidf.tsv", sep="\t", index=False)

    dense = pd.DataFrame()
    if not args.skip_dense:
        dense = run_dense_key_views(
            df,
            views,
            args.dense_models,
            args.output_dir,
            choose_device(args.device),
            args.batch_size,
            args.max_length,
        )
        dense.to_csv(args.output_dir / "representation_view_dense.tsv", sep="\t", index=False)

    meta = {
        "n_rows": int(len(df)),
        "turn_windows": TURN_WINDOWS,
        "key_views": KEY_VIEWS,
        "dense_models": [] if args.skip_dense else args.dense_models,
    }
    (args.output_dir / "representation_view_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(tfidf)} TF-IDF rows and {len(dense)} dense rows")


if __name__ == "__main__":
    main()
