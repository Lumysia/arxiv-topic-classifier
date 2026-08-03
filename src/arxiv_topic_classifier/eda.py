from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from arxiv_topic_classifier.config import CATEGORY_NAMES, RANDOM_SEED, ProjectPaths
from arxiv_topic_classifier.data import dataset_split
from arxiv_topic_classifier.schema import DatasetRow

matplotlib.use("Agg")
from matplotlib import pyplot as plt

TermWeight = dict[str, str | float]


def _word_count(text: str) -> int:
    return len(text.split())


def _count_stats(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": int(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "max": int(np.max(array)),
    }


def _year_coverage(rows: list[DatasetRow]) -> dict[str, Any]:
    years = [row.year for row in rows if row.year is not None]
    counts = Counter(years)
    return {
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "missing_years": sum(1 for row in rows if row.year is None),
        "counts_by_year": {str(year): counts[year] for year in sorted(counts)},
    }


def _plot_abstract_lengths(rows: list[DatasetRow], labels: list[str], destination: Path) -> None:
    plt.figure(figsize=(9, 5))
    for label in labels:
        counts = [_word_count(row.abstract) for row in rows if row.label == label]
        plt.hist(counts, bins=28, alpha=0.48, label=label)
    plt.xlabel("Abstract word count")
    plt.ylabel("Papers")
    plt.title("Bounded sample abstract length distributions")
    plt.legend(ncol=3, fontsize="small")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_year_distribution(rows: list[DatasetRow], destination: Path) -> None:
    counts = Counter(row.year for row in rows if row.year is not None)
    years = sorted(counts)
    plt.figure(figsize=(9, 4.8))
    plt.bar([str(year) for year in years], [counts[year] for year in years], color="#2563eb")
    plt.xlabel("Year")
    plt.ylabel("Papers")
    plt.title("Bounded sample year coverage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def _fit_term_model(train_rows: list[DatasetRow], seed: int) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit([row.text for row in train_rows], [row.label for row in train_rows])
    return pipeline


def _top_terms_by_class(
    pipeline: Pipeline, labels: list[str], top_n: int = 10
) -> dict[str, list[TermWeight]]:
    vectorizer = cast(TfidfVectorizer, pipeline.named_steps["tfidf"])
    classifier = cast(LogisticRegression, pipeline.named_steps["classifier"])
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    terms: dict[str, list[TermWeight]] = {}
    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        top_indexes = np.argsort(coefficients)[-top_n:][::-1]
        terms[str(label)] = [
            {"term": str(feature_names[index]), "coefficient": float(coefficients[index])}
            for index in top_indexes
        ]
    return {label: terms.get(label, []) for label in labels}


def _plot_top_terms(top_terms: dict[str, list[TermWeight]], destination: Path) -> None:
    labels = list(top_terms)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for axis, label in zip(axes.flatten(), labels, strict=True):
        weighted_terms = list(reversed(top_terms[label]))
        terms = [str(item["term"]) for item in weighted_terms]
        coefficients = [float(item["coefficient"]) for item in weighted_terms]
        y = np.arange(len(weighted_terms))
        axis.barh(y, coefficients, color="#10b981")
        axis.set_yticks(y, terms, fontsize="small")
        axis.invert_yaxis()
        axis.set_title(f"{label} · {CATEGORY_NAMES[label]}", fontsize="medium")
        axis.tick_params(axis="x", labelsize="x-small")
    fig.suptitle("Top discriminative training-only TF-IDF/logistic terms")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def generate_eda_artifacts(
    paths: ProjectPaths,
    rows: list[DatasetRow],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("EDA requires at least one dataset row.")
    paths.create_output_dirs()
    labels = sorted({row.label for row in rows})
    train_rows = dataset_split(rows, "train")
    if not train_rows:
        raise ValueError("EDA term extraction requires non-empty training rows.")
    word_counts = [_word_count(row.abstract) for row in rows]
    top_terms = _top_terms_by_class(_fit_term_model(train_rows, seed), labels)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_contract": "Abstract text only; title is metadata for display and error analysis.",
        "row_count": len(rows),
        "abstract_word_counts": {
            "overall": _count_stats(word_counts),
            "by_class": {
                label: _count_stats(
                    [_word_count(row.abstract) for row in rows if row.label == label]
                )
                for label in labels
            },
        },
        "year_coverage": {
            "overall": _year_coverage(rows),
            "by_class": {
                label: _year_coverage([row for row in rows if row.label == label])
                for label in labels
            },
        },
        "top_discriminative_terms": {
            "fit_split": "train",
            "method": (
                "TF-IDF word/phrase features plus multinomial logistic regression coefficients."
            ),
            "terms_by_class": top_terms,
        },
    }
    (paths.artifacts / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_abstract_lengths(rows, labels, paths.figures / "abstract_length_distributions.png")
    _plot_year_distribution(rows, paths.figures / "year_distribution.png")
    _plot_top_terms(top_terms, paths.figures / "top_tfidf_logistic_terms.png")
    return summary
