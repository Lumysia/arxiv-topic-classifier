from __future__ import annotations

from typing import Any

from arxiv_topic_classifier.config import TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.model import load_artifact


class FinalReportValidationError(ValueError):
    pass


def run_final_report(paths: ProjectPaths, expected_test_rows: int | None = None) -> dict[str, Any]:
    del expected_test_rows
    artifact = load_artifact(paths.model)
    if artifact.get("labels") != list(TARGET_CATEGORIES):
        raise FinalReportValidationError(
            "Classical model labels do not match the official taxonomy."
        )
    raise FinalReportValidationError(
        "Paired classical-vs-LLM report generation is disabled until the LLM test artifacts are "
        "regenerated for the 155-label multilabel task. Existing six-class artifacts are stale "
        "and must not be presented as valid."
    )
