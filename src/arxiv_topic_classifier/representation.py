from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from arxiv_topic_classifier.config import DEFAULT_EMBEDDING_MODEL, RANDOM_SEED, ProjectPaths
from arxiv_topic_classifier.data import dataset_split, read_dataset
from arxiv_topic_classifier.embeddings import EmbeddingHelper, default_embedding_helper
from arxiv_topic_classifier.schema import DatasetRow

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _evaluate_arrays(
    truth: list[str], predictions: list[str], labels: list[str]
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
    }


def _embed_rows(
    helper: EmbeddingHelper, rows: list[DatasetRow], batch_size: int
) -> np.ndarray:
    embeddings = helper.embed([row.text for row in rows], batch_size=batch_size)
    if embeddings.shape[0] != len(rows):
        raise ValueError(
            f"Embedding helper returned {embeddings.shape[0]} rows for {len(rows)} inputs."
        )
    return embeddings


def _load_classical_metrics(paths: ProjectPaths) -> dict[str, Any] | None:
    metrics_path = paths.artifacts / "metrics.json"
    if not metrics_path.is_file():
        return None
    return cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))


def _plot_representation_comparison(
    summary: dict[str, Any], classical_metrics: dict[str, Any] | None, paths: ProjectPaths
) -> None:
    names = ["BGE embeddings\n+ logistic"]
    validation = [summary["validation"]["macro_f1"]]
    test = [summary["test"]["macro_f1"]]
    if classical_metrics is not None:
        selected = classical_metrics["selected_model"]
        names.insert(0, f"Selected TF-IDF\n({selected.replace('_', ' ')})")
        validation.insert(0, classical_metrics["validation"][selected]["macro_f1"])
        test.insert(0, classical_metrics["test"][selected]["macro_f1"])
    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width / 2, validation, width, label="Validation", color="#f59e0b")
    plt.bar(x + width / 2, test, width, label="Test", color="#2563eb")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.title("Representation benchmark vs selected deployable pipeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        paths.figures / "representation_benchmark_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def run_representation_benchmark(
    paths: ProjectPaths,
    seed: int = RANDOM_SEED,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_helper: EmbeddingHelper | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    paths.create_output_dirs()
    rows = read_dataset(paths.dataset)
    train_rows = dataset_split(rows, "train")
    validation_rows = dataset_split(rows, "validation")
    test_rows = dataset_split(rows, "test")
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Dataset must contain non-empty train, validation, and test splits.")
    labels = sorted({row.label for row in rows})
    helper = embedding_helper or default_embedding_helper(paths, model_name)
    train_embeddings = _embed_rows(helper, train_rows, batch_size)
    validation_embeddings = _embed_rows(helper, validation_rows, batch_size)
    test_embeddings = _embed_rows(helper, test_rows, batch_size)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=seed,
    )
    classifier.fit(train_embeddings, [row.label for row in train_rows])
    validation_predictions = cast(
        list[str], classifier.predict(validation_embeddings).tolist()
    )
    test_predictions = cast(list[str], classifier.predict(test_embeddings).tolist())
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "representation": f"{helper.model_name} dense embeddings",
        "model": helper.model_name,
        "classifier": "multinomial logistic regression",
        "input_contract": "Abstract text only; title is metadata and is not embedded.",
        "split_policy": (
            "Embeddings and logistic classifier are fit on train rows only; validation is reported "
            "for comparison; held-out test is evaluated once."
        ),
        "deployment_tradeoff": (
            "The offline demo remains the selected TF-IDF pipeline because this semantic "
            "benchmark depends on embedding-model network/cache availability and has a larger "
            "runtime footprint."
        ),
        "selected_for_deployment": False,
        "seed": seed,
        "row_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "embedding_cache_dir": str(helper.cache_dir),
        "labels": labels,
        "validation": _evaluate_arrays(
            [row.label for row in validation_rows], validation_predictions, labels
        ),
        "test": _evaluate_arrays([row.label for row in test_rows], test_predictions, labels),
    }
    (paths.artifacts / "representation_benchmark.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_representation_comparison(summary, _load_classical_metrics(paths), paths)
    return summary
