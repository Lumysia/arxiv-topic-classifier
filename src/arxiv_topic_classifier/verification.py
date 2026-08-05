from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from arxiv_topic_classifier.config import TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.environment import dependency_versions, write_environment
from arxiv_topic_classifier.files import atomic_write_json, sha256_file
from arxiv_topic_classifier.model import ARTIFACT_FORMAT_VERSION, load_artifact, predict_abstract
from arxiv_topic_classifier.taxonomy import EXPECTED_TAXONOMY_SIZE, taxonomy_fingerprint

SMOKE_ABSTRACT = (
    "We introduce a multilingual neural language model for document translation and text "
    "classification. The method uses attention-based representations and evaluates language "
    "understanding performance on reproducible natural language processing benchmarks."
)


class VerificationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def verify_model(paths: ProjectPaths, processed_manifest: dict[str, Any]) -> dict[str, Any]:
    environment = write_environment(paths.environment)
    artifact = load_artifact(paths.model)
    labels = cast(list[str], artifact["labels"])
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise VerificationError("Model artifact format version mismatch.")
    if artifact.get("task") != "official_arxiv_taxonomy_multilabel":
        raise VerificationError("Model artifact task mismatch.")
    if labels != list(TARGET_CATEGORIES) or len(labels) != EXPECTED_TAXONOMY_SIZE:
        raise VerificationError(
            "Model labels do not exactly match the 155 official taxonomy labels."
        )
    if artifact.get("dependency_versions") != dependency_versions():
        raise VerificationError("Model dependency versions do not match the current environment.")
    if artifact.get("processed_manifest") != processed_manifest:
        raise VerificationError("Model processed manifest does not match the prepared dataset.")

    classifier = artifact["pipeline"].named_steps.get("classifier")
    classes = getattr(classifier, "classes_", None)
    if classes is None or list(cast(np.ndarray, classes)) != list(range(EXPECTED_TAXONOMY_SIZE)):
        raise VerificationError("Classifier class indices are not aligned to label order.")

    probability_matrix = artifact["pipeline"].predict_proba([SMOKE_ABSTRACT])
    probabilities_array = cast(np.ndarray, probability_matrix)
    if probabilities_array.shape != (1, EXPECTED_TAXONOMY_SIZE):
        raise VerificationError("Classifier probability output width is not 155.")
    if not np.all(np.isfinite(probabilities_array)):
        raise VerificationError("Classifier produced non-finite probabilities.")

    prediction = predict_abstract(artifact, SMOKE_ABSTRACT)
    if not prediction.labels:
        raise VerificationError("Smoke prediction selected no labels.")
    if set(prediction.probabilities) != set(TARGET_CATEGORIES):
        raise VerificationError("Smoke prediction does not include all 155 probability keys.")
    if any(not math.isfinite(value) for value in prediction.probabilities.values()):
        raise VerificationError("Smoke prediction contains non-finite probability values.")

    model_manifest = load_json(paths.model_manifest)
    if model_manifest.get("model_sha256") != sha256_file(paths.model):
        raise VerificationError("Model manifest SHA-256 does not match the joblib file.")
    if model_manifest.get("metrics_sha256") != sha256_file(paths.artifacts / "metrics.json"):
        raise VerificationError("Model manifest metrics SHA-256 does not match metrics.json.")
    if model_manifest.get("labels") != list(TARGET_CATEGORIES):
        raise VerificationError("Model manifest labels do not match the official taxonomy.")
    if model_manifest.get("dependency_versions") != dependency_versions():
        raise VerificationError("Model manifest dependency versions do not match the environment.")
    if model_manifest.get("processed_manifest") != processed_manifest:
        raise VerificationError("Model manifest processed fingerprint does not match.")

    metrics = load_json(paths.artifacts / "metrics.json")
    if metrics.get("labels") != list(TARGET_CATEGORIES):
        raise VerificationError("Metrics labels do not match the official taxonomy.")
    if metrics.get("processed_manifest") != processed_manifest:
        raise VerificationError("Metrics processed fingerprint does not match.")
    if processed_manifest.get("taxonomy_sha256") != taxonomy_fingerprint():
        raise VerificationError(
            "Processed taxonomy fingerprint does not match the bundled taxonomy."
        )

    report = {
        "format_version": 1,
        "status": "passed",
        "task": "official_arxiv_taxonomy_multilabel",
        "label_count": EXPECTED_TAXONOMY_SIZE,
        "model_sha256": sha256_file(paths.model),
        "metrics_sha256": sha256_file(paths.artifacts / "metrics.json"),
        "processed_manifest": processed_manifest,
        "environment": environment,
        "smoke_prediction": {
            "selected_labels": prediction.labels,
            "top_label": prediction.top_label,
            "probability_key_count": len(prediction.probabilities),
        },
    }
    atomic_write_json(paths.verification_report, report)
    return report
