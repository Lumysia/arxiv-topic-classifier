from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from flask import Flask, Response, jsonify, render_template, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from arxiv_topic_classifier.config import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_NAMES,
    MAX_INPUT_LENGTH,
    TARGET_CATEGORIES,
    ProjectPaths,
)
from arxiv_topic_classifier.llm import (
    LLMChatPrompt,
    LLMClient,
    parse_taxonomy_output,
    taxonomy_json_v1_prompt,
)
from arxiv_topic_classifier.model import load_artifact, predict_abstract, predict_validated_abstract
from arxiv_topic_classifier.schema import LLMUsage, PredictionResult
from arxiv_topic_classifier.text import TextValidationError, validate_prediction_text

LLMMode = Literal["offline", "live", "live-with-cache-fallback"]
LLMTransport = Literal["openai", "opencode"]
LLMSourceMode = Literal["live", "cached"]

DEMO_LLM_CACHE_FILENAME = "demo_llm_cache.json"
DEMO_SELECTION_FILENAME = "demo_selection.json"
DEMO_LLM_CACHE_FORMAT_VERSION = 1

_LLM_MODES = {"offline", "live", "live-with-cache-fallback"}
_LLM_TRANSPORTS = {"openai", "opencode"}


def _typed_route[ViewFunction: Callable[..., Any]](
    decorator: Callable[[ViewFunction], ViewFunction],
) -> Callable[[ViewFunction], ViewFunction]:
    return decorator


class DemoLLMCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_abstract: str
    abstract_hash: str
    label: str | None
    reason: str | None = None
    valid_output: bool
    parse_error: str | None = None
    model: str
    transport: str
    latency_seconds: float
    usage: LLMUsage = Field(default_factory=LLMUsage)
    response_id: str | None = None


class DemoLLMCacheFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1]
    prompt_version: str
    prompt_hash: str
    entries: list[DemoLLMCacheEntry]


@dataclass(frozen=True)
class LoadedDemoLLMCache:
    path: Path
    available: bool
    entries_by_hash: dict[str, DemoLLMCacheEntry]
    warnings: tuple[str, ...] = ()
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.entries_by_hash)


class DemoSelectionExample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    display_label: str
    arxiv_id: str
    actual_label: str
    classical_label: str
    llm_label: str | None
    outcome: str
    normalized_abstract: str
    abstract_hash: str


class DemoSelectionFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    policy: str
    examples: list[DemoSelectionExample]


@dataclass(frozen=True)
class LoadedDemoSelection:
    path: Path
    available: bool
    examples: tuple[DemoSelectionExample, ...]
    warnings: tuple[str, ...] = ()
    error: str | None = None


def demo_abstract_hash(normalized_abstract: str) -> str:
    return hashlib.sha256(normalized_abstract.encode("utf-8")).hexdigest()


def load_demo_llm_cache(path: Path) -> LoadedDemoLLMCache:
    if not path.is_file():
        return LoadedDemoLLMCache(
            path=path,
            available=False,
            entries_by_hash={},
            error="Demo LLM cache artifact is not present.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache_file = DemoLLMCacheFile.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return LoadedDemoLLMCache(
            path=path,
            available=False,
            entries_by_hash={},
            error="Demo LLM cache artifact is malformed.",
        )

    probe_prompt: LLMChatPrompt = taxonomy_json_v1_prompt("x" * 80)
    if (
        cache_file.prompt_version != probe_prompt.version
        or cache_file.prompt_hash != probe_prompt.prompt_hash
    ):
        return LoadedDemoLLMCache(
            path=path,
            available=False,
            entries_by_hash={},
            error="Demo LLM cache prompt metadata does not match the current taxonomy prompt.",
        )

    entries_by_hash: dict[str, DemoLLMCacheEntry] = {}
    warnings: list[str] = []
    for index, entry in enumerate(cache_file.entries):
        expected_hash = demo_abstract_hash(entry.normalized_abstract)
        if entry.abstract_hash != expected_hash:
            warnings.append(
                f"entry {index} skipped: abstract_hash does not match normalized_abstract"
            )
            continue
        if entry.valid_output and entry.label not in TARGET_CATEGORIES:
            warnings.append(f"entry {index} skipped: valid output has an unknown label")
            continue
        if entry.abstract_hash in entries_by_hash:
            warnings.append(f"entry {index} skipped: duplicate abstract_hash")
            continue
        entries_by_hash[entry.abstract_hash] = entry
    return LoadedDemoLLMCache(
        path=path,
        available=True,
        entries_by_hash=entries_by_hash,
        warnings=tuple(warnings),
    )


def load_demo_selection(path: Path, limit: int = 3) -> LoadedDemoSelection:
    if not path.is_file():
        return LoadedDemoSelection(
            path=path,
            available=False,
            examples=(),
            error="Demo selection artifact is not present.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selection_file = DemoSelectionFile.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return LoadedDemoSelection(
            path=path,
            available=False,
            examples=(),
            error="Demo selection artifact is malformed.",
        )

    examples: list[DemoSelectionExample] = []
    warnings: list[str] = []
    seen_hashes: set[str] = set()
    for index, example in enumerate(selection_file.examples):
        expected_hash = demo_abstract_hash(example.normalized_abstract)
        if example.abstract_hash != expected_hash:
            warnings.append(
                f"example {index} skipped: abstract_hash does not match normalized_abstract"
            )
            continue
        if example.actual_label not in TARGET_CATEGORIES:
            warnings.append(f"example {index} skipped: unknown actual_label")
            continue
        if example.abstract_hash in seen_hashes:
            warnings.append(f"example {index} skipped: duplicate abstract_hash")
            continue
        seen_hashes.add(example.abstract_hash)
        examples.append(example)
        if len(examples) >= limit:
            break
    return LoadedDemoSelection(
        path=path,
        available=True,
        examples=tuple(examples),
        warnings=tuple(warnings),
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(
    root: Path,
    *,
    llm_mode: LLMMode = "offline",
    llm_transport: LLMTransport = "openai",
    llm_client: LLMClient | None = None,
    llm_cache_path: Path | None = None,
    demo_selection_path: Path | None = None,
) -> Flask:
    if llm_mode not in _LLM_MODES:
        raise ValueError(f"Unsupported LLM mode: {llm_mode}")
    if llm_transport not in _LLM_TRANSPORTS:
        raise ValueError(f"Unsupported LLM transport: {llm_transport}")
    if llm_mode in {"live", "live-with-cache-fallback"} and llm_client is None:
        raise ValueError("A configured LLM client is required for live LLM modes.")

    paths = ProjectPaths.from_root(root)
    artifact = load_artifact(paths.model)
    cache = load_demo_llm_cache(llm_cache_path or (paths.artifacts / DEMO_LLM_CACHE_FILENAME))
    selection = load_demo_selection(
        demo_selection_path or (paths.artifacts / DEMO_SELECTION_FILENAME)
    )
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    @_typed_route(app.get("/"))
    def index() -> str:
        metrics = _load_json(paths.artifacts / "metrics.json", {})
        profile = _load_json(paths.profile, {})
        audit = _load_json(paths.artifacts / "ai_audit_summary.json", {})
        return render_template(
            "index.html",
            metrics=metrics,
            profile=profile,
            audit=audit,
            max_length=MAX_INPUT_LENGTH,
            taxonomy=_taxonomy_payload(),
            llm_mode=llm_mode,
            llm_transport=llm_transport,
            cache_available=cache.available,
            cache_count=cache.count,
            demo_examples=[example.model_dump() for example in selection.examples],
            demo_selection_available=selection.available,
        )

    @_typed_route(app.get("/health"))
    def health() -> Response:
        return cast(
            Response,
            jsonify(
                {
                    "status": "ok",
                    "ready": True,
                    "model": artifact["selected_model"],
                    "labels": artifact["labels"],
                    "classical": {
                        "available": True,
                        "selected_model": artifact["selected_model"],
                        "labels": artifact["labels"],
                    },
                    "llm": {
                        "mode": llm_mode,
                        "transport": llm_transport,
                        "client_configured": llm_client is not None,
                    },
                    "cache": _cache_health(cache),
                    "demo_selection": _selection_health(selection),
                    "artifacts": {
                        "llm_validation_metrics": paths.llm_metrics("validation").is_file(),
                        "llm_test_metrics": paths.llm_metrics("test").is_file(),
                    },
                }
            ),
        )

    @_typed_route(app.post("/api/predict"))
    def predict() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "abstract" not in payload:
            return jsonify({"error": "JSON body must contain an abstract field."}), 400
        try:
            result = predict_abstract(artifact, payload["abstract"])
        except TextValidationError as error:
            return jsonify({"error": str(error)}), 400
        return cast(Response, jsonify(result.model_dump()))

    @_typed_route(app.post("/api/compare"))
    def compare() -> tuple[Response, int] | Response:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "abstract" not in payload:
            return jsonify({"error": "JSON body must contain an abstract field."}), 400
        try:
            abstract = validate_prediction_text(payload["abstract"])
        except TextValidationError as error:
            return jsonify({"error": str(error)}), 400

        total_started = time.perf_counter()
        classical_started = time.perf_counter()
        classical = predict_validated_abstract(artifact, abstract)
        classical_latency = time.perf_counter() - classical_started
        llm = _llm_compare_result(
            abstract=abstract,
            mode=llm_mode,
            configured_transport=llm_transport,
            client=llm_client,
            cache=cache,
        )
        agreement = _agreement_status(classical.label, llm)
        return cast(
            Response,
            jsonify(
                {
                    "input": _input_metadata(abstract),
                    "classical": _classical_result(
                        classical, artifact["selected_model"], classical_latency
                    ),
                    "llm": llm,
                    "agreement": agreement[0],
                    "agreement_status": agreement[1],
                    "total_latency_seconds": time.perf_counter() - total_started,
                }
            ),
        )

    return app


def _taxonomy_payload() -> list[dict[str, str]]:
    return [
        {"label": label, "name": CATEGORY_NAMES[label], "description": CATEGORY_DESCRIPTIONS[label]}
        for label in TARGET_CATEGORIES
    ]


def _cache_health(cache: LoadedDemoLLMCache) -> dict[str, Any]:
    return {
        "path": str(cache.path),
        "available": cache.available,
        "count": cache.count,
        "error": cache.error,
        "warnings": list(cache.warnings),
    }


def _selection_health(selection: LoadedDemoSelection) -> dict[str, Any]:
    return {
        "path": str(selection.path),
        "available": selection.available,
        "count": len(selection.examples),
        "error": selection.error,
        "warnings": list(selection.warnings),
    }


def _input_metadata(abstract: str) -> dict[str, Any]:
    return {
        "abstract_hash": demo_abstract_hash(abstract),
        "characters": len(abstract),
        "words": len(abstract.split()),
        "max_characters": MAX_INPUT_LENGTH,
    }


def _classical_result(
    classical: PredictionResult, selected_model: str, latency_seconds: float
) -> dict[str, Any]:
    return {
        "label": classical.label,
        "category_name": classical.category_name,
        "confidence": classical.confidence,
        "probabilities": classical.probabilities,
        "probability_type": "logistic_regression_predict_proba",
        "model": selected_model,
        "latency_seconds": latency_seconds,
    }


def _llm_compare_result(
    *,
    abstract: str,
    mode: LLMMode,
    configured_transport: LLMTransport,
    client: LLMClient | None,
    cache: LoadedDemoLLMCache,
) -> dict[str, Any]:
    if mode == "offline":
        return _cached_or_unavailable(abstract, cache, configured_transport)
    if client is None:
        return _llm_unavailable(configured_transport, "Live LLM client is not configured.")

    try:
        return _live_llm_result(abstract, client)
    except Exception:
        if mode == "live-with-cache-fallback":
            cached = _cache_lookup(abstract, cache)
            if cached is not None:
                return _cache_entry_result(
                    cached,
                    fallback_message=(
                        "Live LLM transport was unavailable; an exact cached demo response "
                        "is shown."
                    ),
                )
        return _llm_unavailable(configured_transport, "Live LLM transport is unavailable.")


def _live_llm_result(abstract: str, client: LLMClient) -> dict[str, Any]:
    response = client.complete(taxonomy_json_v1_prompt(abstract))
    parsed = parse_taxonomy_output(response.content)
    return _llm_result(
        label=parsed.label,
        reason=parsed.reason,
        valid_output=parsed.valid_output,
        parse_error=parsed.error,
        model=client.model,
        transport=_client_transport(client),
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        response_id=response.response_id,
        source_mode="live",
        fallback_message=None,
        unavailable_reason=None,
    )


def _cached_or_unavailable(
    abstract: str, cache: LoadedDemoLLMCache, configured_transport: LLMTransport
) -> dict[str, Any]:
    cached = _cache_lookup(abstract, cache)
    if cached is not None:
        return _cache_entry_result(cached, fallback_message=None)
    if cache.available:
        reason = "No exact cached LLM response exists for this normalized abstract."
    else:
        reason = cache.error or "Demo LLM cache is unavailable."
    return _llm_unavailable(configured_transport, reason)


def _cache_lookup(abstract: str, cache: LoadedDemoLLMCache) -> DemoLLMCacheEntry | None:
    return cache.entries_by_hash.get(demo_abstract_hash(abstract))


def _cache_entry_result(entry: DemoLLMCacheEntry, fallback_message: str | None) -> dict[str, Any]:
    return _llm_result(
        label=entry.label,
        reason=entry.reason,
        valid_output=entry.valid_output,
        parse_error=entry.parse_error,
        model=entry.model,
        transport=entry.transport,
        latency_seconds=entry.latency_seconds,
        usage=entry.usage,
        response_id=entry.response_id,
        source_mode="cached",
        fallback_message=fallback_message,
        unavailable_reason=None,
    )


def _llm_result(
    *,
    label: str | None,
    reason: str | None,
    valid_output: bool,
    parse_error: str | None,
    model: str | None,
    transport: str,
    latency_seconds: float | None,
    usage: LLMUsage,
    response_id: str | None,
    source_mode: LLMSourceMode,
    fallback_message: str | None,
    unavailable_reason: str | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "category_name": CATEGORY_NAMES.get(label) if label is not None else None,
        "reason": reason,
        "valid_output": valid_output,
        "parse_error": parse_error,
        "model": model,
        "transport": transport,
        "latency_seconds": latency_seconds,
        "usage": usage.model_dump(),
        "response_id": response_id,
        "source_mode": source_mode,
        "status": "ok" if valid_output else "invalid_output",
        "fallback_message": fallback_message,
        "unavailable_reason": unavailable_reason,
    }


def _llm_unavailable(transport: str, reason: str) -> dict[str, Any]:
    return {
        "label": None,
        "category_name": None,
        "reason": None,
        "valid_output": False,
        "parse_error": None,
        "model": None,
        "transport": transport,
        "latency_seconds": None,
        "usage": LLMUsage().model_dump(),
        "response_id": None,
        "source_mode": None,
        "status": "unavailable",
        "fallback_message": None,
        "unavailable_reason": reason,
    }


def _client_transport(client: LLMClient) -> str:
    raw_value = getattr(client, "transport", None)
    return raw_value if isinstance(raw_value, str) else "injected"


def _agreement_status(classical_label: str, llm: dict[str, Any]) -> tuple[bool | None, str]:
    llm_label = llm.get("label")
    if llm.get("valid_output") is True and isinstance(llm_label, str):
        agreement = classical_label == llm_label
        return agreement, "agree" if agreement else "disagree"
    if llm.get("status") == "unavailable":
        return None, "llm_unavailable"
    return None, "llm_invalid_output"
