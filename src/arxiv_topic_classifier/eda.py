from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from arxiv_topic_classifier.config import (
    CATEGORY_NAMES,
    RANDOM_SEED,
    TARGET_CATEGORIES,
    ProjectPaths,
)
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


def _plot_abstract_lengths(rows: list[DatasetRow], destination: str) -> None:
    counts = [_word_count(row.abstract) for row in rows]
    plt.figure(figsize=(9, 5))
    plt.hist(counts, bins=32, color="#2563eb", alpha=0.78)
    plt.xlabel("Abstract word count")
    plt.ylabel("Papers")
    plt.title("Bounded sample abstract length distribution")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_year_distribution(rows: list[DatasetRow], destination: str) -> None:
    counts = Counter(row.year for row in rows if row.year is not None)
    years = sorted(counts)
    plt.figure(figsize=(9, 4.8))
    plt.bar([str(year) for year in years], [counts[year] for year in years], color="#0f766e")
    plt.xlabel("Year")
    plt.ylabel("Papers")
    plt.title("Bounded sample year coverage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def _fit_term_model(train_rows: list[DatasetRow], seed: int) -> Pipeline:
    labels = list(TARGET_CATEGORIES)
    y_train = MultiLabelBinarizer(classes=labels).fit_transform([row.labels for row in train_rows])
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
                OneVsRestClassifier(
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit([row.text for row in train_rows], y_train)
    return pipeline


def _top_terms_by_class(
    pipeline: Pipeline, labels: list[str], top_n: int = 10
) -> dict[str, list[TermWeight]]:
    vectorizer = cast(TfidfVectorizer, pipeline.named_steps["tfidf"])
    classifier = cast(OneVsRestClassifier, pipeline.named_steps["classifier"])
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    terms: dict[str, list[TermWeight]] = {}
    estimators = cast(list[LogisticRegression], classifier.estimators_)
    for label, estimator in zip(labels, estimators, strict=True):
        coefficients = estimator.coef_[0]
        top_indexes = np.argsort(coefficients)[-top_n:][::-1]
        terms[label] = [
            {"term": str(feature_names[index]), "coefficient": float(coefficients[index])}
            for index in top_indexes
        ]
    return terms


def _plot_top_terms(top_terms: dict[str, list[TermWeight]], destination: str) -> None:
    labels = list(top_terms)[:12]
    columns = 3
    rows = int(np.ceil(len(labels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, max(4, rows * 2.6)))
    flattened = np.atleast_1d(axes).flatten()
    for axis, label in zip(flattened, labels, strict=False):
        weighted_terms = list(reversed(top_terms[label]))
        terms = [str(item["term"]) for item in weighted_terms]
        coefficients = [float(item["coefficient"]) for item in weighted_terms]
        y = np.arange(len(weighted_terms))
        axis.barh(y, coefficients, color="#10b981")
        axis.set_yticks(y, terms, fontsize="x-small")
        axis.invert_yaxis()
        axis.set_title(f"{label} · {CATEGORY_NAMES[label]}", fontsize="small")
    for axis in flattened[len(labels) :]:
        axis.axis("off")
    fig.suptitle("Top training-only TF-IDF/logistic terms for frequent labels")
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
    labels = list(TARGET_CATEGORIES)
    train_rows = dataset_split(rows, "train")
    if not train_rows:
        raise ValueError("EDA term extraction requires non-empty training rows.")
    word_counts = [_word_count(row.abstract) for row in rows]
    label_counts = Counter(label for row in rows for label in row.labels)
    top_term_labels = [label for label, _ in label_counts.most_common(12)]
    all_terms = _top_terms_by_class(_fit_term_model(train_rows, seed), labels)
    top_terms = {label: all_terms[label] for label in top_term_labels}
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_contract": "Abstract text only; title is metadata for display and error analysis.",
        "row_count": len(rows),
        "official_label_count": len(labels),
        "label_assignments": {
            "total": sum(label_counts.values()),
            "by_label": {label: label_counts[label] for label in labels},
            "labels_per_row": _count_stats([len(row.labels) for row in rows]),
        },
        "abstract_word_counts": {"overall": _count_stats(word_counts)},
        "year_coverage": {"overall": _year_coverage(rows)},
        "top_discriminative_terms": {
            "fit_split": "train",
            "method": (
                "Multilabel TF-IDF features plus one-vs-rest logistic regression "
                "coefficients."
            ),
            "terms_by_class": top_terms,
        },
    }
    (paths.artifacts / "eda_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_abstract_lengths(rows, str(paths.figures / "abstract_length_distributions.png"))
    _plot_year_distribution(rows, str(paths.figures / "year_distribution.png"))
    _plot_top_terms(top_terms, str(paths.figures / "top_tfidf_logistic_terms.png"))
    return summary
