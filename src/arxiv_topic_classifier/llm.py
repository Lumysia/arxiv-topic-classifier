from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

from arxiv_topic_classifier.config import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_NAMES,
    DEFAULT_LLM_MAX_CONCURRENCY,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_LLM_USER_AGENT,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MAX_CONCURRENCY_ENV,
    LLM_MAX_RETRIES_ENV,
    LLM_MODEL_ENV,
    LLM_TIMEOUT_SECONDS_ENV,
    LLM_USER_AGENT_ENV,
    TARGET_CATEGORIES,
)
from arxiv_topic_classifier.schema import LLMAuditRecord, LLMParsedPrediction, LLMUsage

TAXONOMY_PROMPT_VERSION = "taxonomy_multilabel_json_v2"
AUDIT_PROMPT_VERSION = "taxonomy_audit_json_v1"
INVALID_LABEL = "__invalid__"


class LLMConfigError(ValueError):
    pass


class LLMProviderError(RuntimeError):
    pass


class LLMRetryableProviderError(LLMProviderError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class LLMRuntimeConfig:
    base_url: str
    api_key: str
    model: str = DEFAULT_LLM_MODEL
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    max_concurrency: int = DEFAULT_LLM_MAX_CONCURRENCY
    max_retries: int = DEFAULT_LLM_MAX_RETRIES
    user_agent: str = DEFAULT_LLM_USER_AGENT


@dataclass(frozen=True)
class LLMRunFailure:
    arxiv_id: str
    error_type: str


class LLMRunError(RuntimeError):
    failures: tuple[LLMRunFailure, ...]

    def __init__(self, operation: str, failures: list[LLMRunFailure]) -> None:
        self.failures = tuple(failures)
        failed_ids = ", ".join(failure.arxiv_id for failure in self.failures[:5])
        suffix = "" if len(self.failures) <= 5 else f", and {len(self.failures) - 5} more"
        super().__init__(
            f"{operation} failed for {len(self.failures)} row(s): {failed_ids}{suffix}. "
            "Successful rows were cached; rerun with --resume to retry failed rows."
        )


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMChatPrompt:
    version: str
    prompt_hash: str
    messages: tuple[LLMMessage, ...]
    max_output_tokens: int


@dataclass(frozen=True)
class LLMClientResponse:
    content: str
    usage: LLMUsage
    latency_seconds: float


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: LLMChatPrompt) -> LLMClientResponse:
        pass


def _required_string(env: Mapping[str, str], name: str, require_credentials: bool) -> str:
    value = env.get(name, "").strip()
    if require_credentials and not value:
        raise LLMConfigError(f"{name} is required for live LLM calls.")
    return value


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise LLMConfigError(f"{name} must be a number of seconds.") from error
    if value <= 0:
        raise LLMConfigError(f"{name} must be greater than zero.")
    return value


def _int_env(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise LLMConfigError(f"{name} must be an integer.") from error
    if value < minimum:
        raise LLMConfigError(f"{name} must be at least {minimum}.")
    return value


def parse_llm_runtime_config(
    env: Mapping[str, str] | None = None, require_credentials: bool = True
) -> LLMRuntimeConfig:
    source = os.environ if env is None else env
    base_url = _required_string(source, LLM_BASE_URL_ENV, require_credentials)
    api_key = _required_string(source, LLM_API_KEY_ENV, require_credentials)
    model = source.get(LLM_MODEL_ENV, DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
    user_agent = source.get(LLM_USER_AGENT_ENV, DEFAULT_LLM_USER_AGENT).strip()
    return LLMRuntimeConfig(
        base_url=base_url.rstrip("/") + "/" if base_url else "",
        api_key=api_key,
        model=model,
        timeout_seconds=_float_env(source, LLM_TIMEOUT_SECONDS_ENV, DEFAULT_LLM_TIMEOUT_SECONDS),
        max_concurrency=_int_env(
            source, LLM_MAX_CONCURRENCY_ENV, DEFAULT_LLM_MAX_CONCURRENCY, minimum=1
        ),
        max_retries=_int_env(source, LLM_MAX_RETRIES_ENV, DEFAULT_LLM_MAX_RETRIES, minimum=0),
        user_agent=user_agent or DEFAULT_LLM_USER_AGENT,
    )


def label_definition_payload() -> list[dict[str, str]]:
    return [
        {
            "label": label,
            "name": CATEGORY_NAMES[label],
            "definition": CATEGORY_DESCRIPTIONS[label],
        }
        for label in TARGET_CATEGORIES
    ]


def _prompt_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def taxonomy_prompt_hash() -> str:
    return _prompt_hash(
        {
            "version": TAXONOMY_PROMPT_VERSION,
            "labels": label_definition_payload(),
            "output_schema": {"labels": list(TARGET_CATEGORIES), "reason": "short string"},
        }
    )


def audit_prompt_hash() -> str:
    return _prompt_hash(
        {
            "version": AUDIT_PROMPT_VERSION,
            "labels": label_definition_payload(),
            "output_schema": {
                "suggested_label": list(TARGET_CATEGORIES),
                "alternative_labels": list(TARGET_CATEGORIES),
                "ambiguity_flag": "boolean",
                "short_reason": "short string",
            },
        }
    )


def taxonomy_json_v1_prompt(abstract: str) -> LLMChatPrompt:
    system = (
        "You are an arXiv abstract topic classifier. Use only the supplied abstract text. "
        "Return exactly one JSON object and no surrounding prose."
    )
    user = json.dumps(
        {
            "task": (
                "Choose every applicable current official arXiv category for this abstract. "
                "Use only labels from the supplied taxonomy and do not invent labels."
            ),
            "labels": label_definition_payload(),
            "abstract": abstract,
            "output_json_schema": {
                "labels": "array of one or more labels from: " + ", ".join(TARGET_CATEGORIES),
                "reason": "short reason based only on the abstract",
            },
        },
        indent=2,
    )
    return LLMChatPrompt(
        version=TAXONOMY_PROMPT_VERSION,
        prompt_hash=taxonomy_prompt_hash(),
        messages=(LLMMessage("system", system), LLMMessage("user", user)),
        max_output_tokens=120,
    )


def taxonomy_audit_json_v1_prompt(abstract: str) -> LLMChatPrompt:
    system = (
        "You are auditing possible arXiv label ambiguity from abstract text only. "
        "Return exactly one JSON object and no surrounding prose."
    )
    user = json.dumps(
        {
            "task": (
                "Suggest the best primary category and any plausible alternatives. "
                "Do not assume the source label is correct because it is not supplied."
            ),
            "labels": label_definition_payload(),
            "abstract": abstract,
            "output_json_schema": {
                "suggested_label": "one of: " + ", ".join(TARGET_CATEGORIES),
                "alternative_labels": "array of zero or more other valid labels",
                "ambiguity_flag": "true when multiple labels are plausible",
                "short_reason": "short reason based only on the abstract",
            },
        },
        indent=2,
    )
    return LLMChatPrompt(
        version=AUDIT_PROMPT_VERSION,
        prompt_hash=audit_prompt_hash(),
        messages=(LLMMessage("system", system), LLMMessage("user", user)),
        max_output_tokens=180,
    )


def parse_taxonomy_output(content: str) -> LLMParsedPrediction:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        return LLMParsedPrediction(
            labels=(), valid_output=False, error=f"Malformed JSON: {error.msg}"
        )
    if not isinstance(payload, dict):
        return LLMParsedPrediction(
            labels=(), valid_output=False, error="Output is not a JSON object."
        )
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        return LLMParsedPrediction(
            labels=(), valid_output=False, error="Unknown or missing labels."
        )
    if any(not isinstance(label, str) or label not in TARGET_CATEGORIES for label in labels):
        return LLMParsedPrediction(
            labels=(), valid_output=False, error="labels must contain only known taxonomy IDs."
        )
    reason = payload.get("reason")
    ordered = tuple(label for label in TARGET_CATEGORIES if label in set(labels))
    return LLMParsedPrediction(
        labels=ordered,
        valid_output=True,
        reason=reason if isinstance(reason, str) else None,
    )


def parse_audit_output(
    content: str,
) -> tuple[str | None, list[str], bool | None, str | None, str | None]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        return None, [], None, None, f"Malformed JSON: {error.msg}"
    if not isinstance(payload, dict):
        return None, [], None, None, "Output is not a JSON object."
    suggested = payload.get("suggested_label")
    alternatives = payload.get("alternative_labels")
    ambiguity = payload.get("ambiguity_flag")
    reason = payload.get("short_reason")
    if not isinstance(suggested, str) or suggested not in TARGET_CATEGORIES:
        return None, [], None, None, "Unknown or missing suggested_label."
    if not isinstance(alternatives, list) or any(
        not isinstance(label, str) or label not in TARGET_CATEGORIES for label in alternatives
    ):
        return None, [], None, None, "alternative_labels must contain only valid labels."
    if not isinstance(ambiguity, bool):
        return None, [], None, None, "ambiguity_flag must be boolean."
    return (
        suggested,
        [label for label in alternatives if label != suggested],
        ambiguity,
        reason if isinstance(reason, str) else None,
        None,
    )


def audit_valid_output(record: LLMAuditRecord) -> bool:
    return record.valid_output and record.suggested_label in TARGET_CATEGORIES


class OpenAICompatibleChatClient:
    def __init__(
        self,
        config: LLMRuntimeConfig,
        transport: httpx.BaseTransport | None = None,
        retry_base_delay_seconds: float = 0.5,
    ) -> None:
        self.model = config.model
        self._config = config
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "User-Agent": config.user_agent,
                "Content-Type": "application/json",
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def complete(self, prompt: LLMChatPrompt) -> LLMClientResponse:
        request_body = {
            "model": self.model,
            "messages": [message.__dict__ for message in prompt.messages],
            "temperature": 0,
            "max_tokens": prompt.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.post("chat/completions", json=request_body)
                if response.status_code in {429, 500, 502, 503, 504}:
                    self._raise_or_retry(response, attempt)
                    continue
                if response.status_code >= 400:
                    raise LLMProviderError(
                        f"LLM provider returned HTTP {response.status_code}."
                    )
                payload = cast(dict[str, Any], response.json())
                return self._parse_response_payload(payload, time.perf_counter() - started)
            except (httpx.TimeoutException, httpx.NetworkError, LLMRetryableProviderError) as error:
                if attempt >= self._config.max_retries:
                    raise LLMProviderError("LLM provider request failed after retries.") from error
                retry_after = (
                    error.retry_after_seconds
                    if isinstance(error, LLMRetryableProviderError)
                    else None
                )
                self._sleep_before_retry(attempt, retry_after)
        raise LLMProviderError("LLM provider request failed without a response.")

    def _raise_or_retry(self, response: httpx.Response, attempt: int) -> None:
        if attempt >= self._config.max_retries:
            raise LLMProviderError(f"LLM provider returned HTTP {response.status_code}.")
        raise LLMRetryableProviderError(
            f"Retryable LLM HTTP {response.status_code}.",
            retry_after_seconds=_retry_after_seconds(response),
        )

    def _sleep_before_retry(self, attempt: int, retry_after_seconds: float | None = None) -> None:
        delay = (
            min(30.0, retry_after_seconds)
            if retry_after_seconds is not None
            else min(30.0, self._retry_base_delay_seconds * (2**attempt))
        )
        time.sleep(delay)

    def _parse_response_payload(
        self, payload: dict[str, Any], latency_seconds: float
    ) -> LLMClientResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("LLM provider response has no choices.")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMProviderError("LLM provider choice is malformed.")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError("LLM provider message content is missing.")
        usage_payload = payload.get("usage", {})
        usage = usage_payload if isinstance(usage_payload, dict) else {}
        return LLMClientResponse(
            content=cast(str, message["content"]),
            usage=LLMUsage(
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
                total_tokens=_optional_int(usage.get("total_tokens")),
                reasoning_tokens=_optional_int(
                    _completion_token_details(usage).get("reasoning_tokens")
                ),
                cache_read_tokens=_optional_int(
                    _prompt_token_details(usage).get("cache_read_tokens")
                ),
            ),
            latency_seconds=latency_seconds,
        )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _prompt_token_details(usage: dict[str, Any]) -> dict[str, Any]:
    details = usage.get("prompt_tokens_details")
    return details if isinstance(details, dict) else {}


def _completion_token_details(usage: dict[str, Any]) -> dict[str, Any]:
    details = usage.get("completion_tokens_details")
    return details if isinstance(details, dict) else {}


def _retry_after_seconds(response: httpx.Response) -> float | None:
    if response.status_code not in {429, 503}:
        return None
    raw_value = response.headers.get("Retry-After")
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None
