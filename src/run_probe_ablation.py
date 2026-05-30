#!/usr/bin/env python3
"""Ablate which text is visible to simple linear probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline


LABELS = {
    "polarity": "relation_construction_polarity_norm",
    "evidence": "evidence_coarse",
    "function": "function_coarse",
}


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def probe(texts, labels, baseline: bool = False) -> dict[str, float]:
    min_class = min(np.bincount(pd.Categorical(labels).codes))
    cv = StratifiedKFold(n_splits=max(2, min(5, int(min_class))), shuffle=True, random_state=17)
    if baseline:
        estimator = DummyClassifier(strategy="most_frequent")
        features = np.zeros((len(labels), 1))
    else:
        estimator = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
        features = texts
    return metrics(labels, cross_val_predict(estimator, features, labels, cv=cv))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/modal_corpus/processed/modal_experiment_dataset.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/modal_corpus/experiments/probe_text_ablation.tsv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, sep="\t").fillna("")
    views = {
        "majority": None,
        "prefix_only": df["prefix_context"].astype(str),
        "marker_only": df["marker_text"].astype(str),
        "marker_plus_scope": (df["marker_text"].astype(str) + " " + df["scope_text"].astype(str)),
        "prefix_plus_marker_scope": (
            df["prefix_context"].astype(str) + " " + df["marker_text"].astype(str) + " " + df["scope_text"].astype(str)
        ),
    }

    rows = []
    for label_name, column in LABELS.items():
        labels = df[column].astype(str).to_numpy()
        for view_name, texts in views.items():
            out = probe(texts, labels, baseline=(view_name == "majority"))
            rows.append({"view": view_name, "label": label_name, **out})

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
