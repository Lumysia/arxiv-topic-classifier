from __future__ import annotations

import hashlib
import re

from arxiv_topic_classifier.config import MAX_INPUT_LENGTH, TEXT_MIN_LENGTH

_WHITESPACE = re.compile(r"\s+")
_VERSION_SUFFIX = re.compile(r"v\d+$")


class TextValidationError(ValueError):
    pass


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def base_arxiv_id(value: str) -> str:
    return _VERSION_SUFFIX.sub("", value.strip())


def document_text(title: str, abstract: str) -> str:
    return normalize_text(abstract)


def content_fingerprint(abstract: str) -> str:
    normalized = normalize_text(abstract).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_prediction_text(value: object) -> str:
    if not isinstance(value, str):
        raise TextValidationError("Abstract must be a string.")
    normalized = normalize_text(value)
    if len(normalized) < TEXT_MIN_LENGTH:
        raise TextValidationError(
            f"Abstract must contain at least {TEXT_MIN_LENGTH} non-whitespace characters."
        )
    if len(normalized) > MAX_INPUT_LENGTH:
        raise TextValidationError(f"Abstract must not exceed {MAX_INPUT_LENGTH} characters.")
    return normalized
