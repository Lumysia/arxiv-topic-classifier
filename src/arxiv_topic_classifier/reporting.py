from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np

from arxiv_topic_classifier.config import CATEGORY_NAMES, ProjectPaths
from arxiv_topic_classifier.schema import DatasetRow

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_class_distribution(rows: list[DatasetRow], destination: Path) -> None:
    counts = Counter(label for row in rows for label in row.labels)
    plotted_labels = [label for label, _ in counts.most_common(40)]
    split_counts = {
        split: Counter(label for row in rows if row.split == split for label in row.labels)
        for split in ("train", "validation", "test")
    }
    x = np.arange(len(plotted_labels))
    bottom = np.zeros(len(plotted_labels))
    colors = ("#2563eb", "#f59e0b", "#10b981")
    plt.figure(figsize=(14, 5.5))
    for split, color in zip(("train", "validation", "test"), colors, strict=True):
        values = np.array([split_counts[split][label] for label in plotted_labels])
        plt.bar(x, values, bottom=bottom, label=split.title(), color=color)
        bottom += values
    plt.xticks(x, plotted_labels, rotation=60, ha="right", fontsize="x-small")
    plt.ylabel("Label assignments")
    plt.title(f"Top label coverage in bounded multilabel dataset (rows={len(rows):,})")
    plt.legend()
    _save_figure(destination)


def _plot_label_f1(selected_metrics: dict[str, Any], labels: list[str], destination: Path) -> None:
    report = cast(dict[str, Any], selected_metrics["classification_report"])
    scored = [
        (label, cast(float, cast(dict[str, Any], report[label])["f1-score"])) for label in labels
    ]
    plotted = sorted(scored, key=lambda item: item[1])[:40]
    x = np.arange(len(plotted))
    plt.figure(figsize=(14, 5.5))
    plt.bar(x, [score for _, score in plotted], color="#0f766e")
    plt.xticks(x, [label for label, _ in plotted], rotation=60, ha="right", fontsize="x-small")
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title("Lowest per-label F1 scores for selected multilabel model")
    _save_figure(destination)


def _plot_model_comparison(metrics: dict[str, Any], destination: Path) -> None:
    models = list(cast(dict[str, Any], metrics["test"]))
    validation_scores = [metrics["validation"][name]["macro_f1"] for name in models]
    test_scores = [metrics["test"][name]["macro_f1"] for name in models]
    x = np.arange(len(models))
    width = 0.36
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width / 2, validation_scores, width, label="Validation", color="#f59e0b")
    plt.bar(x + width / 2, test_scores, width, label="Test", color="#2563eb")
    plt.xticks(x, [name.replace("_", "\n") for name in models])
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.title("Multilabel model comparison")
    plt.legend()
    _save_figure(destination)


def _write_class_metrics(
    selected_metrics: dict[str, Any], labels: list[str], destination: Path
) -> None:
    report = cast(dict[str, Any], selected_metrics["classification_report"])
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "name", "precision", "recall", "f1", "support"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in labels:
            values = cast(dict[str, Any], report[label])
            writer.writerow(
                {
                    "label": label,
                    "name": CATEGORY_NAMES[label],
                    "precision": f"{values['precision']:.6f}",
                    "recall": f"{values['recall']:.6f}",
                    "f1": f"{values['f1-score']:.6f}",
                    "support": int(cast(float, values["support"])),
                }
            )


def _example_rows(
    test_rows: list[DatasetRow],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row, prediction_row, probability_row in zip(
        test_rows, predictions, probabilities, strict=True
    ):
        predicted_labels = [
            label for label, value in zip(labels, prediction_row, strict=True) if value
        ]
        actual = set(row.labels)
        predicted = set(predicted_labels)
        jaccard = len(actual & predicted) / len(actual | predicted) if actual or predicted else 1.0
        examples.append(
            {
                "arxiv_id": row.arxiv_id,
                "actual_labels": list(row.labels),
                "predicted_labels": predicted_labels,
                "exact_match": actual == predicted,
                "jaccard": jaccard,
                "confidence": float(np.max(probability_row)),
                "title": row.title,
                "abstract_excerpt": row.abstract[:400],
                "probabilities": {
                    label: float(value)
                    for label, value in zip(labels, probability_row, strict=True)
                },
            }
        )
    exact = sorted(
        (item for item in examples if item["exact_match"]),
        key=lambda item: cast(float, item["confidence"]),
        reverse=True,
    )[:12]
    misses = sorted(
        (item for item in examples if not item["exact_match"]),
        key=lambda item: cast(float, item["jaccard"]),
    )[:24]
    return [*exact, *misses]


def _error_patterns(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_positives: Counter[str] = Counter()
    false_negatives: Counter[str] = Counter()
    for item in examples:
        actual = set(cast(list[str], item["actual_labels"]))
        predicted = set(cast(list[str], item["predicted_labels"]))
        false_positives.update(predicted - actual)
        false_negatives.update(actual - predicted)
    return [
        {"pattern": "false_positive", "label": label, "count": count}
        for label, count in false_positives.most_common(12)
    ] + [
        {"pattern": "false_negative", "label": label, "count": count}
        for label, count in false_negatives.most_common(12)
    ]


def generate_evaluation_artifacts(
    paths: ProjectPaths,
    all_rows: list[DatasetRow],
    test_rows: list[DatasetRow],
    labels: list[str],
    selected_name: str,
    metrics: dict[str, Any],
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    selected_metrics = cast(dict[str, Any], metrics["test"][selected_name])
    _write_class_metrics(selected_metrics, labels, paths.artifacts / "class_metrics.csv")
    examples = _example_rows(test_rows, predictions, probabilities, labels)
    (paths.artifacts / "prediction_examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    (paths.artifacts / "error_patterns.json").write_text(
        json.dumps(_error_patterns(examples), indent=2) + "\n", encoding="utf-8"
    )
    _plot_class_distribution(all_rows, paths.figures / "class_distribution.png")
    _plot_label_f1(selected_metrics, labels, paths.figures / "label_f1.png")
    _plot_model_comparison(metrics, paths.figures / "model_comparison.png")
    test = selected_metrics
    report_lines = [
        "# Evaluation report",
        "",
        f"Selected model: **{selected_name.replace('_', ' ')}**.",
        "",
        "Selection used validation multilabel macro-F1. The held-out test split was not "
        "used to fit or select the model.",
        "",
        "## Held-out test results",
        "",
        f"- Exact-match ratio: {test['exact_match_ratio']:.3f}",
        f"- Micro-F1: {test['micro_f1']:.3f}",
        f"- Macro-F1: {test['macro_f1']:.3f}",
        f"- Samples-F1: {test['samples_f1']:.3f}",
        "",
        "## Error analysis",
        "",
        "The most frequent false-positive and false-negative labels are generated in "
        "`error_patterns.json`.",
        "",
        "## Interpretation boundary",
        "",
        "Model features are normalized abstracts only. Titles are retained solely as "
        "metadata for display and error analysis.",
        "",
        "Metrics describe a bounded multilabel sample over the official 155-label arXiv "
        "taxonomy, not all arXiv papers.",
    ]
    (paths.artifacts / "evaluation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
