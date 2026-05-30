#!/usr/bin/env python3
"""Run frozen-representation probes and output-distribution tests on MODAL."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


DEFAULT_ENCODERS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "gpt2",
    "Qwen/Qwen3-0.6B",
]
DEFAULT_LMS = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "Qwen/Qwen3-0.6B",
]
LABELS = {
    "polarity": "relation_construction_polarity_norm",
    "evidence": "evidence_coarse",
    "function": "function_coarse",
}
ADVERBIAL_MORPHS = {"lexicon_adverb", "lexicon_adverbial"}


def model_slug(name: str) -> str:
    return name.replace("/", "__")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def cross_validated_probe(features, labels: np.ndarray, model_kind: str) -> dict[str, float]:
    n_classes = len(np.unique(labels))
    min_class = min(np.bincount(pd.Categorical(labels).codes))
    n_splits = max(2, min(5, int(min_class)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)

    if model_kind == "tfidf":
        estimator = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )
    else:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
        )

    pred = cross_val_predict(estimator, features, labels, cv=cv)
    metrics = cv_metrics(labels, pred)
    metrics["n_splits"] = float(n_splits)
    metrics["n_classes"] = float(n_classes)
    return metrics


def dummy_metrics(labels: np.ndarray) -> dict[str, float]:
    dummy = DummyClassifier(strategy="most_frequent")
    pred = cross_val_predict(dummy, np.zeros((len(labels), 1)), labels, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=17))
    out = cv_metrics(labels, pred)
    out["n_splits"] = 5.0
    out["n_classes"] = float(len(np.unique(labels)))
    return out


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def final_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


def embed_texts(
    model_name: str,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    model.to(device)
    model.eval()

    pooled = []
    is_causal = bool(getattr(model.config, "is_decoder", False)) or model.config.model_type in {"gpt2", "gpt_neox", "qwen3"}
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            if is_causal:
                vec = final_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            else:
                vec = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled.append(vec.detach().cpu().numpy())

    del model, tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.vstack(pooled)


def run_representation_probes(
    df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    encoders: list[str],
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    texts = df["prefix_context"].astype(str).tolist()

    for label_name, column in LABELS.items():
        labels = df[column].astype(str).to_numpy()
        metrics = dummy_metrics(labels)
        rows.append({"representation": "majority", "label": label_name, **metrics})

        metrics = cross_validated_probe(texts, labels, model_kind="tfidf")
        rows.append({"representation": "tfidf_char_3_5", "label": label_name, **metrics})

    for encoder_name in encoders:
        print(f"Embedding prefixes with {encoder_name}")
        embeddings = embed_texts(encoder_name, texts, device, batch_size, max_length)
        np.save(output_dir / f"embeddings_{model_slug(encoder_name)}.npy", embeddings)
        for label_name, column in LABELS.items():
            labels = df[column].astype(str).to_numpy()
            metrics = cross_validated_probe(embeddings, labels, model_kind="dense")
            rows.append({"representation": encoder_name, "label": label_name, **metrics})

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "probe_results.tsv", sep="\t", index=False)
    return result


def encode_pair(tokenizer, prefix: str, continuation: str, max_length: int) -> tuple[list[int], int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    room = max_length - len(continuation_ids)
    if room < 1:
        continuation_ids = continuation_ids[-(max_length - 1) :]
        room = 1
    prefix_ids = prefix_ids[-room:]
    return prefix_ids + continuation_ids, len(prefix_ids)


def continuation_logprobs(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, float]]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    pad_id = tokenizer.pad_token_id
    out: list[dict[str, float]] = []

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = [encode_pair(tokenizer, prefix, continuation, max_length) for prefix, continuation in batch_pairs]
            max_len = max(len(ids) for ids, _ in encoded)
            input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
            for idx, (ids, _) in enumerate(encoded):
                input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[idx, : len(ids)] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            for row_idx, (ids, prefix_len) in enumerate(encoded):
                total = 0.0
                count = 0
                for pos in range(max(1, prefix_len), len(ids)):
                    token_id = ids[pos]
                    total += float(log_probs[row_idx, pos - 1, token_id].detach().cpu())
                    count += 1
                out.append(
                    {
                        "logprob_sum": total,
                        "logprob_mean": total / max(count, 1),
                        "n_tokens": float(count),
                    }
                )
    return out


def run_lm_distribution_test(
    df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    lm_names: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_df = df[df["marker_morphosyntax"].isin(ADVERBIAL_MORPHS)].copy()
    score_df = score_df[score_df["gold_continuation"].str.len().gt(0) & score_df["foil_continuation"].str.len().gt(0)]
    score_df = score_df.reset_index(drop=True)
    all_scores = []

    for model_name in lm_names:
        print(f"Scoring gold vs foil continuations with {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
        model.to(device)
        model.eval()

        gold_pairs = [
            (row.prefix_context, " " + row.gold_continuation)
            for row in score_df.itertuples(index=False)
        ]
        foil_pairs = [
            (row.prefix_context, " " + row.foil_continuation)
            for row in score_df.itertuples(index=False)
        ]
        gold_scores = continuation_logprobs(model, tokenizer, gold_pairs, device, batch_size, max_length)
        foil_scores = continuation_logprobs(model, tokenizer, foil_pairs, device, batch_size, max_length)

        for row, gold, foil in zip(score_df.itertuples(index=False), gold_scores, foil_scores):
            all_scores.append(
                {
                    "model": model_name,
                    "row_id": row.row_id,
                    "language": row.language,
                    "marker_text": row.marker_text,
                    "foil_marker_text": row.foil_marker_text,
                    "gold_logprob_mean": gold["logprob_mean"],
                    "foil_logprob_mean": foil["logprob_mean"],
                    "gold_logprob_sum": gold["logprob_sum"],
                    "foil_logprob_sum": foil["logprob_sum"],
                    "gold_tokens": gold["n_tokens"],
                    "foil_tokens": foil["n_tokens"],
                    "delta_mean": gold["logprob_mean"] - foil["logprob_mean"],
                    "gold_preferred": gold["logprob_mean"] > foil["logprob_mean"],
                    "polarity": row.relation_construction_polarity_norm,
                    "evidence": row.evidence_coarse,
                    "marker_morphosyntax": row.marker_morphosyntax,
                }
            )

        del model, tokenizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    scores = pd.DataFrame(all_scores)
    scores.to_csv(output_dir / "lm_distribution_scores.tsv", sep="\t", index=False)
    summary = (
        scores.groupby(["model", "language"], as_index=False)
        .agg(
            n=("row_id", "count"),
            pairwise_accuracy=("gold_preferred", "mean"),
            mean_delta=("delta_mean", "mean"),
            median_delta=("delta_mean", "median"),
        )
        .sort_values(["model", "language"])
    )
    overall = (
        scores.groupby(["model"], as_index=False)
        .agg(
            n=("row_id", "count"),
            pairwise_accuracy=("gold_preferred", "mean"),
            mean_delta=("delta_mean", "mean"),
            median_delta=("delta_mean", "median"),
        )
    )
    summary = pd.concat([summary, overall.assign(language="ALL")], ignore_index=True)
    summary.to_csv(output_dir / "lm_distribution_summary.tsv", sep="\t", index=False)
    return scores, summary


def plot_results(probes: pd.DataFrame, lm_summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    pivot = probes.pivot(index="representation", columns="label", values="balanced_accuracy")
    pivot = pivot.sort_values("evidence", ascending=False)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Balanced accuracy")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "probe_balanced_accuracy.png", dpi=200)
    plt.close()

    lm_all = lm_summary[lm_summary["language"] == "ALL"].copy()
    if len(lm_all):
        plt.figure(figsize=(6, 3.8))
        plt.bar(lm_all["model"], lm_all["pairwise_accuracy"])
        plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
        plt.ylabel("Gold > foil accuracy")
        plt.xlabel("")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "lm_pairwise_accuracy.png", dpi=200)
        plt.close()


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
    parser.add_argument("--encoder-models", nargs="*", default=DEFAULT_ENCODERS)
    parser.add_argument("--lm-models", nargs="*", default=DEFAULT_LMS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset, sep="\t").fillna("")
    device = choose_device(args.device)
    print(f"Using device: {device}")

    probes = pd.DataFrame()
    lm_summary = pd.DataFrame()
    if not args.skip_probes:
        probes = run_representation_probes(
            df,
            args.output_dir,
            device,
            args.encoder_models,
            args.batch_size,
            args.max_length,
        )
    if not args.skip_lm:
        _, lm_summary = run_lm_distribution_test(
            df,
            args.output_dir,
            device,
            args.lm_models,
            args.batch_size,
            args.max_length,
        )
    if args.no_plots:
        pass
    elif len(probes) and len(lm_summary):
        plot_results(probes, lm_summary, args.output_dir)
    elif len(probes):
        plot_results(probes, pd.DataFrame(), args.output_dir)

    meta = {
        "n_rows": int(len(df)),
        "device": str(device),
        "encoder_models": args.encoder_models,
        "lm_models": args.lm_models,
        "labels": LABELS,
        "lm_distribution_subset": sorted(ADVERBIAL_MORPHS),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
