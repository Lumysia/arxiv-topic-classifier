from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from arxiv_topic_classifier.config import (
    CATEGORY_NAMES,
    DEFAULT_PREDICTION_THRESHOLD,
    RANDOM_SEED,
    TARGET_CATEGORIES,
    ProjectPaths,
)
from arxiv_topic_classifier.data import dataset_split, read_dataset
from arxiv_topic_classifier.eda import generate_eda_artifacts
from arxiv_topic_classifier.environment import dependency_versions
from arxiv_topic_classifier.files import atomic_joblib_dump, atomic_write_json, sha256_file
from arxiv_topic_classifier.reporting import generate_evaluation_artifacts
from arxiv_topic_classifier.schema import DatasetRow, LabelProbability, PredictionResult
from arxiv_topic_classifier.text import validate_prediction_text

Estimator = Any
ARTIFACT_FORMAT_VERSION = 2
MODEL_MANIFEST_FORMAT_VERSION = 1


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
        "multinomial_nb": Pipeline(
            [
                ("tfidf", _vectorizer()),
                ("classifier", OneVsRestClassifier(MultinomialNB(alpha=0.3), n_jobs=1)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", _vectorizer()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=4.0,
                            class_weight="balanced",
                            max_iter=1_000,
                            random_state=seed,
                        ),
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _probabilities(estimator: Estimator, texts: list[str]) -> np.ndarray:
    raw = estimator.predict_proba(texts)
    return cast(np.ndarray, raw)


def _binarizer(labels: list[str]) -> MultiLabelBinarizer:
    return MultiLabelBinarizer(classes=labels)


def _prediction_matrix(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predicted = probabilities >= threshold
    for row_index, row in enumerate(predicted):
        if not np.any(row):
            row[int(np.argmax(probabilities[row_index]))] = True
    return predicted.astype(int)


def _validate_complete_label_coverage(rows: list[DatasetRow], labels: list[str]) -> None:
    label_set = set(labels)
    unknown = sorted({label for row in rows for label in row.labels if label not in label_set})
    if unknown:
        raise ValueError(f"Dataset contains labels outside the official taxonomy: {unknown[:5]}")
    train_counts = {label: 0 for label in labels}
    eligible_counts = {label: 0 for label in labels}
    for row in rows:
        for label in row.labels:
            eligible_counts[label] += 1
            if row.split == "train":
                train_counts[label] += 1
    zero_eligible = [label for label, count in eligible_counts.items() if count == 0]
    zero_train = [label for label, count in train_counts.items() if count == 0]
    if zero_eligible or zero_train:
        raise ValueError(
            "Dataset must include every official label with at least one eligible and train row: "
            f"zero_eligible={zero_eligible}, zero_train={zero_train}"
        )


def evaluate_model(
    estimator: Estimator,
    rows: list[DatasetRow],
    labels: list[str],
    threshold: float = DEFAULT_PREDICTION_THRESHOLD,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    texts = [row.text for row in rows]
    mlb = _binarizer(labels)
    truth = mlb.fit_transform([row.labels for row in rows])
    probabilities = _probabilities(estimator, texts)
    predictions = _prediction_matrix(probabilities, threshold)
    report = classification_report(
        truth,
        predictions,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "exact_match_ratio": float(np.mean(np.all(truth == predictions, axis=1)))
        if len(rows)
        else 0.0,
        "hamming_loss": float(hamming_loss(truth, predictions)) if len(rows) else 0.0,
        "jaccard_samples": float(
            jaccard_score(truth, predictions, average="samples", zero_division=0)
        )
        if len(rows)
        else 0.0,
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(truth, predictions, average="samples", zero_division=0)),
        "macro_precision": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "classification_report": report,
    }
    return metrics, predictions, probabilities


def train_and_evaluate(
    paths: ProjectPaths,
    seed: int = RANDOM_SEED,
    *,
    processed_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths.create_output_dirs()
    rows = read_dataset(paths.dataset)
    train_rows = dataset_split(rows, "train")
    validation_rows = dataset_split(rows, "validation")
    test_rows = dataset_split(rows, "test")
    labels = list(TARGET_CATEGORIES)
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Dataset must contain non-empty train, validation, and test splits.")
    _validate_complete_label_coverage(rows, labels)
    mlb = _binarizer(labels)
    y_train = mlb.fit_transform([row.labels for row in train_rows])

    trained: dict[str, Estimator] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    for name, estimator in candidate_models(seed).items():
        estimator.fit([row.text for row in train_rows], y_train)
        trained[name] = estimator
        validation_metrics, _, _ = evaluate_model(estimator, validation_rows, labels)
        validation_results[name] = validation_metrics

    selected_name = max(
        validation_results,
        key=lambda name: cast(float, validation_results[name]["macro_f1"]),
    )
    test_results: dict[str, dict[str, Any]] = {}
    selected_predictions = np.empty((0, len(labels)), dtype=int)
    selected_probabilities = np.empty((0, len(labels)))
    for name, estimator in trained.items():
        metrics, predictions, probabilities = evaluate_model(estimator, test_rows, labels)
        test_results[name] = metrics
        if name == selected_name:
            selected_predictions = predictions
            selected_probabilities = probabilities

    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "task": "official_arxiv_taxonomy_multilabel",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected_name,
        "labels": labels,
        "category_names": CATEGORY_NAMES,
        "threshold": DEFAULT_PREDICTION_THRESHOLD,
        "pipeline": trained[selected_name],
        "input_contract": (
            "Normalized abstract text only; title is metadata for display/error analysis."
        ),
        "training_rows": len(train_rows),
        "seed": seed,
        "dependency_versions": dependency_versions(),
        "processed_manifest": processed_manifest or {},
    }
    atomic_joblib_dump(artifact, paths.model, compress=3)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "Highest multilabel macro-F1 on the validation split; test used once for "
            "final comparison."
        ),
        "task": "official_arxiv_taxonomy_multilabel",
        "selected_model": selected_name,
        "labels": labels,
        "row_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "validation": validation_results,
        "test": test_results,
        "dependency_versions": dependency_versions(),
        "processed_manifest": processed_manifest or {},
    }
    atomic_write_json(paths.artifacts / "metrics.json", summary)
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
    manifest = {
        "format_version": MODEL_MANIFEST_FORMAT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "task": "official_arxiv_taxonomy_multilabel",
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_path": paths.relative_path(paths.model),
        "model_sha256": sha256_file(paths.model),
        "metrics_path": paths.relative_path(paths.artifacts / "metrics.json"),
        "metrics_sha256": sha256_file(paths.artifacts / "metrics.json"),
        "selected_model": selected_name,
        "labels": labels,
        "label_count": len(labels),
        "row_counts": summary["row_counts"],
        "dependency_versions": dependency_versions(),
        "processed_manifest": processed_manifest or {},
    }
    atomic_write_json(paths.model_manifest, manifest)
    return summary


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelArtifactError(f"Model artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ModelArtifactError("Unsupported or invalid model artifact.")
    required = {"pipeline", "labels", "category_names", "selected_model", "threshold", "task"}
    if not required.issubset(artifact):
        raise ModelArtifactError("Model artifact is missing required fields.")
    if artifact.get("task") != "official_arxiv_taxonomy_multilabel":
        raise ModelArtifactError("Model artifact is not the official multilabel taxonomy task.")
    labels = artifact.get("labels")
    if labels != list(TARGET_CATEGORIES):
        raise ModelArtifactError("Model artifact labels do not match the 155-label taxonomy.")
    return cast(dict[str, Any], artifact)


def predict_validated_abstract(artifact: dict[str, Any], abstract: str) -> PredictionResult:
    estimator = artifact["pipeline"]
    labels = cast(list[str], artifact["labels"])
    threshold = cast(float, artifact["threshold"])
    probability_values = _probabilities(estimator, [abstract])[0]
    probabilities = {
        label: float(probability)
        for label, probability in zip(labels, probability_values, strict=True)
    }
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_label = ranked[0][0]
    selected = [label for label, probability in ranked if probability >= threshold]
    if not selected:
        selected = [top_label]
    return PredictionResult(
        labels=selected,
        top_label=top_label,
        top_category_name=cast(dict[str, str], artifact["category_names"])[top_label],
        top_probability=probabilities[top_label],
        threshold=threshold,
        probabilities=dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True)),
        ranked=[
            LabelProbability(
                label=label,
                category_name=cast(dict[str, str], artifact["category_names"])[label],
                probability=probability,
                selected=label in selected,
            )
            for label, probability in ranked
        ],
    )


def predict_abstract(artifact: dict[str, Any], abstract: object) -> PredictionResult:
    return predict_validated_abstract(artifact, validate_prediction_text(abstract))
