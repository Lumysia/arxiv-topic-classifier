from __future__ import annotations

import pytest

from arxiv_topic_classifier.text import (
    TextValidationError,
    base_arxiv_id,
    document_text,
    normalize_text,
    validate_prediction_text,
)


def test_preprocessing_normalizes_without_removing_scientific_tokens() -> None:
    assert normalize_text("  Graph\n\t neural   $x^2$ ") == "Graph neural $x^2$"
    assert document_text("  Title\n", " Abstract   body ") == "Abstract body"
    assert base_arxiv_id("2501.12345v3") == "2501.12345"


def test_prediction_text_validation() -> None:
    valid = (
        "Neural language processing and translation methods are evaluated on several benchmark "
        "datasets with detailed analysis."
    )
    assert validate_prediction_text(valid) == valid
    with pytest.raises(TextValidationError, match="string"):
        validate_prediction_text(None)
    with pytest.raises(TextValidationError, match="at least"):
        validate_prediction_text("too short")
