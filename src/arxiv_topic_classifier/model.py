from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from arxiv_topic_classifier.config import CATEGORY_NAMES, RANDOM_SEED, ProjectPaths
from arxiv_topic_classifier.data import dataset_split, read_dataset
from arxiv_topic_classifier.eda import generate_eda_artifacts
from arxiv_topic_classifier.reporting import generate_evaluation_artifacts
from arxiv_topic_classifier.schema import DatasetRow, PredictionResult
from arxiv_topic_classifier.text import validate_prediction_text

Estimator = Any


class ModelArtifactError(RuntimeError):
    pass


def _vectorizer(max_features: int = 30_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
    )


def candidate_models(seed: int = RANDOM_SEED) -> dict[str, Estimator]:
    return {
        "majority_baseline": Pipeline(
            [
                ("tfidf", _vectorizer(max_features=100)),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("classifier", MultinomialNB(alpha=0.3)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", _vectorizer()),
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
        ),
    }


def _probabilities(estimator: Estimator, texts: list[str]) -> np.ndarray:
    raw = estimator.predict_proba(texts)
    return cast(np.ndarray, raw)


def evaluate_model(
    estimator: Estimator,
    rows: list[DatasetRow],
    labels: list[str],
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    texts = [row.text for row in rows]
    truth = [row.label for row in rows]
    predictions = cast(list[str], estimator.predict(texts).tolist())
    probabilities = _probabilities(estimator, texts)
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
    }
    return metrics, predictions, probabilities


def train_and_evaluate(paths: ProjectPaths, seed: int = RANDOM_SEED) -> dict[str, Any]:
    paths.create_output_dirs()
    rows = read_dataset(paths.dataset)
    train_rows = dataset_split(rows, "train")
    validation_rows = dataset_split(rows, "validation")
    test_rows = dataset_split(rows, "test")
    labels = sorted({row.label for row in rows})
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Dataset must contain non-empty train, validation, and test splits.")

    trained: dict[str, Estimator] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    for name, estimator in candidate_models(seed).items():
        estimator.fit([row.text for row in train_rows], [row.label for row in train_rows])
        trained[name] = estimator
        validation_metrics, _, _ = evaluate_model(estimator, validation_rows, labels)
        validation_results[name] = validation_metrics

    selected_name = max(
        validation_results,
        key=lambda name: cast(float, validation_results[name]["macro_f1"]),
    )
    test_results: dict[str, dict[str, Any]] = {}
    selected_predictions: list[str] = []
    selected_probabilities = np.empty((0, len(labels)))
    for name, estimator in trained.items():
        metrics, predictions, probabilities = evaluate_model(estimator, test_rows, labels)
        test_results[name] = metrics
        if name == selected_name:
            selected_predictions = predictions
            selected_probabilities = probabilities

    artifact = {
        "format_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected_name,
        "labels": labels,
        "category_names": CATEGORY_NAMES,
        "pipeline": trained[selected_name],
        "input_contract": (
            "Normalized abstract text only; title is metadata for display/error analysis."
        ),
        "training_rows": len(train_rows),
        "seed": seed,
    }
    joblib.dump(artifact, paths.model, compress=3)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "Highest macro-F1 on the validation split; test used once for final comparison."
        ),
        "selected_model": selected_name,
        "labels": labels,
        "row_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "validation": validation_results,
        "test": test_results,
    }
    (paths.artifacts / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    generate_evaluation_artifacts(
        paths=paths,
        all_rows=rows,
        test_rows=test_rows,
        labels=labels,
        selected_name=selected_name,
        metrics=summary,
        predictions=selected_predictions,
        probabilities=selected_probabilities,
    )
    generate_eda_artifacts(paths, rows, seed)
    return summary


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelArtifactError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ModelArtifactError("Unsupported or invalid model artifact.")
    required = {"pipeline", "labels", "category_names", "selected_model"}
    if not required.issubset(artifact):
        raise ModelArtifactError("Model artifact is missing required fields.")
    return cast(dict[str, Any], artifact)


def predict_validated_abstract(artifact: dict[str, Any], abstract: str) -> PredictionResult:
    estimator = artifact["pipeline"]
    labels = cast(list[str], estimator.classes_.tolist())
    probability_values = _probabilities(estimator, [abstract])[0]
    probabilities = {
        label: float(probability)
        for label, probability in zip(labels, probability_values, strict=True)
    }
    label = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
    return PredictionResult(
        label=label,
        category_name=cast(dict[str, str], artifact["category_names"])[label],
        confidence=probabilities[label],
        probabilities=dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True)),
    )


def predict_abstract(artifact: dict[str, Any], abstract: object) -> PredictionResult:
    return predict_validated_abstract(artifact, validate_prediction_text(abstract))
