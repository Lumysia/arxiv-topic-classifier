from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from arxiv_topic_classifier.config import CATEGORY_NAMES, ProjectPaths
from arxiv_topic_classifier.schema import DatasetRow

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_class_distribution(rows: list[DatasetRow], labels: list[str], destination: Path) -> None:
    counts = Counter(row.label for row in rows)
    split_counts = {
        split: Counter(row.label for row in rows if row.split == split)
        for split in ("train", "validation", "test")
    }
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = ("#2563eb", "#f59e0b", "#10b981")
    for split, color in zip(("train", "validation", "test"), colors, strict=True):
        values = np.array([split_counts[split][label] for label in labels])
        plt.bar(x, values, bottom=bottom, label=split.title(), color=color)
        bottom += values
    plt.xticks(x, labels)
    plt.ylabel("Papers")
    plt.title(f"Balanced bounded dataset (n={sum(counts.values()):,})")
    plt.legend()
    _save_figure(destination)


def _plot_confusion_matrix(matrix: list[list[int]], labels: list[str], destination: Path) -> None:
    values = np.asarray(matrix)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=float),
        where=row_sums != 0,
    )
    display = ConfusionMatrixDisplay(normalized, display_labels=labels)
    display.plot(cmap="Blues", values_format=".2f", colorbar=False)
    plt.title("Selected model: normalized test confusion matrix")
    plt.xticks(rotation=30, ha="right")
    _save_figure(destination)


def _plot_model_comparison(metrics: dict[str, Any], destination: Path) -> None:
    models = list(cast(dict[str, Any], metrics["test"]))
    validation_scores = [metrics["validation"][name]["macro_f1"] for name in models]
    test_scores = [metrics["test"][name]["macro_f1"] for name in models]
    x = np.arange(len(models))
    width = 0.36
    plt.bar(x - width / 2, validation_scores, width, label="Validation", color="#f59e0b")
    plt.bar(x + width / 2, test_scores, width, label="Test", color="#2563eb")
    plt.xticks(x, [name.replace("_", "\n") for name in models])
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.title("Model comparison")
    plt.legend()
    _save_figure(destination)


def _write_class_metrics(
    selected_metrics: dict[str, Any], labels: list[str], destination: Path
) -> None:
    report = selected_metrics["classification_report"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "name", "precision", "recall", "f1", "support"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in labels:
            values = report[label]
            writer.writerow(
                {
                    "label": label,
                    "name": CATEGORY_NAMES[label],
                    "precision": f"{values['precision']:.6f}",
                    "recall": f"{values['recall']:.6f}",
                    "f1": f"{values['f1-score']:.6f}",
                    "support": int(values["support"]),
                }
            )


def _write_confusion_csv(matrix: list[list[int]], labels: list[str], destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _example_rows(
    test_rows: list[DatasetRow],
    predictions: list[str],
    probabilities: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row, prediction, probability_row in zip(test_rows, predictions, probabilities, strict=True):
        confidence = float(np.max(probability_row))
        examples.append(
            {
                "arxiv_id": row.arxiv_id,
                "actual": row.label,
                "predicted": prediction,
                "correct": row.label == prediction,
                "confidence": confidence,
                "title": row.title,
                "abstract_excerpt": row.abstract[:400],
                "secondary_categories": row.categories.split()[1:],
                "probabilities": {
                    label: float(value)
                    for label, value in zip(labels, probability_row, strict=True)
                },
            }
        )
    correct = sorted(
        (item for item in examples if item["correct"]),
        key=lambda item: cast(float, item["confidence"]),
        reverse=True,
    )[:12]
    incorrect = sorted(
        (item for item in examples if not item["correct"]),
        key=lambda item: cast(float, item["confidence"]),
        reverse=True,
    )[:24]
    return [*correct, *incorrect]


def _error_patterns(
    matrix: list[list[int]], labels: list[str], examples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            if actual == predicted:
                continue
            patterns.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": matrix[actual_index][predicted_index],
                }
            )
    patterns.sort(key=lambda item: cast(int, item["count"]), reverse=True)
    cross_listed_errors = sum(
        1
        for item in examples
        if not item["correct"] and item["predicted"] in item["secondary_categories"]
    )
    return [
        *patterns[:8],
        {
            "pattern": "prediction_matches_source_secondary_category",
            "count_in_representative_errors": cross_listed_errors,
            "note": (
                "This is evidence of label ambiguity, not proof that the source primary label "
                "is wrong."
            ),
        },
    ]


def generate_evaluation_artifacts(
    paths: ProjectPaths,
    all_rows: list[DatasetRow],
    test_rows: list[DatasetRow],
    labels: list[str],
    selected_name: str,
    metrics: dict[str, Any],
    predictions: list[str],
    probabilities: np.ndarray,
) -> None:
    selected_metrics = cast(dict[str, Any], metrics["test"][selected_name])
    matrix = cast(list[list[int]], selected_metrics["confusion_matrix"])
    _write_class_metrics(selected_metrics, labels, paths.artifacts / "class_metrics.csv")
    _write_confusion_csv(matrix, labels, paths.artifacts / "confusion_matrix.csv")
    examples = _example_rows(test_rows, predictions, probabilities, labels)
    (paths.artifacts / "prediction_examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    patterns = _error_patterns(matrix, labels, examples)
    (paths.artifacts / "error_patterns.json").write_text(
        json.dumps(patterns, indent=2) + "\n", encoding="utf-8"
    )
    _plot_class_distribution(all_rows, labels, paths.figures / "class_distribution.png")
    _plot_confusion_matrix(matrix, labels, paths.figures / "confusion_matrix.png")
    _plot_model_comparison(metrics, paths.figures / "model_comparison.png")
    test = selected_metrics
    report_lines = [
        "# Evaluation report",
        "",
        f"Selected model: **{selected_name.replace('_', ' ')}**.",
        "",
        "Selection used validation macro-F1. The held-out test split was not used to fit "
        "or select the model.",
        "",
        "## Held-out test results",
        "",
        f"- Accuracy: {test['accuracy']:.3f}",
        f"- Balanced accuracy: {test['balanced_accuracy']:.3f}",
        f"- Macro-F1: {test['macro_f1']:.3f}",
        f"- Weighted-F1: {test['weighted_f1']:.3f}",
        "",
        "## Error analysis",
        "",
        "The most frequent directional confusions are generated in `error_patterns.json`. "
        "Cross-listed papers are retained so errors can be checked against secondary source "
        "categories. A match with a secondary category indicates plausible ambiguity, not a "
        "corrected ground truth.",
        "",
        "## Interpretation boundary",
        "",
        "Model features are normalized abstracts only. Titles are retained solely as metadata for "
        "display and error analysis.",
        "",
        "Metrics describe a balanced six-topic sample, not all arXiv papers. Source labels "
        "are author-assigned "
        "and multi-category overlap imposes a ceiling on unambiguous single-label evaluation.",
    ]
    (paths.artifacts / "evaluation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
