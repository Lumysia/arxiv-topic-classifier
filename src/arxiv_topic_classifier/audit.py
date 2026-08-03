from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from arxiv_topic_classifier.config import (
    CATEGORY_DESCRIPTIONS,
    DEFAULT_AUDIT_SIZE,
    RANDOM_SEED,
    ProjectPaths,
)
from arxiv_topic_classifier.data import read_dataset
from arxiv_topic_classifier.llm import (
    LLMChatPrompt,
    LLMClient,
    LLMRunError,
    LLMRunFailure,
    LLMRuntimeConfig,
    OpenAICompatibleChatClient,
    parse_audit_output,
    parse_llm_runtime_config,
    taxonomy_audit_json_v1_prompt,
)
from arxiv_topic_classifier.schema import DatasetRow, LLMAuditRecord

DevelopmentSplit = Literal["train", "validation"]


@dataclass(frozen=True)
class _CompletedAuditRow:
    record: LLMAuditRecord
    raw_response: str


def _stratified_audit_sample(
    rows: list[DatasetRow], sample_size: int, seed: int
) -> list[DatasetRow]:
    development_rows = [row for row in rows if row.split != "test"]
    labels = sorted(CATEGORY_DESCRIPTIONS)
    per_class = sample_size // len(labels)
    remainder = sample_size % len(labels)
    rng = random.Random(seed)
    sampled: list[DatasetRow] = []
    for index, label in enumerate(labels):
        candidates = [row for row in development_rows if row.label == label]
        wanted = per_class + int(index < remainder)
        if len(candidates) < wanted:
            raise ValueError(f"Not enough development rows for LLM audit label {label}.")
        sampled.extend(rng.sample(candidates, wanted))
    return sampled


def _hash_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _generation_contract(prompt: LLMChatPrompt) -> str:
    return _hash_key(
        prompt.version,
        prompt.prompt_hash,
        str(prompt.max_output_tokens),
        "temperature=0",
        "response_format=json_object",
    )


def _cache_key(row: DatasetRow, model: str, prompt: LLMChatPrompt) -> str:
    return _hash_key(
        row.arxiv_id,
        row.content_hash,
        model,
        prompt.version,
        prompt.prompt_hash,
        _generation_contract(prompt),
    )


def _append_jsonl_atomic(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _write_jsonl(destination: Path, records: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(destination)


def _load_audit_cache(path: Path) -> dict[str, LLMAuditRecord]:
    if not path.is_file():
        return {}
    records: dict[str, LLMAuditRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            raw_record = payload.get("record", payload) if isinstance(payload, dict) else payload
            record = LLMAuditRecord.model_validate(raw_record)
            records[record.cache_key] = record
    return records


def _prepare_audit_cache(path: Path, resume: bool) -> dict[str, LLMAuditRecord]:
    if resume:
        return _load_audit_cache(path)
    if path.exists():
        path.unlink()
    return {}


def _development_split(split: str) -> DevelopmentSplit:
    if split not in {"train", "validation"}:
        raise ValueError("LLM audit sample must exclude the held-out test split.")
    return "train" if split == "train" else "validation"


def _audit_outcome(
    row: DatasetRow,
    suggested_label: str | None,
    alternative_labels: list[str],
    ambiguity_flag: bool | None,
    valid_output: bool,
) -> str:
    if not valid_output or suggested_label is None:
        return "invalid_rejected"
    if suggested_label == row.label:
        return "agreement"
    secondary = set(row.categories.split()[1:])
    if suggested_label in secondary or row.label in alternative_labels or ambiguity_flag is True:
        return "plausible_alternative_cross_list_ambiguity"
    return "human_review_candidate"


def _classify_audit_row(row: DatasetRow, client: LLMClient) -> _CompletedAuditRow:
    prompt = taxonomy_audit_json_v1_prompt(row.text)
    response = client.complete(prompt)
    suggested, alternatives, ambiguity, reason, error = parse_audit_output(response.content)
    valid_output = error is None
    record = LLMAuditRecord(
        cache_key=_cache_key(row, client.model, prompt),
        arxiv_id=row.arxiv_id,
        split=_development_split(row.split),
        abstract_hash=row.content_hash,
        model=client.model,
        prompt_version=prompt.version,
        prompt_hash=prompt.prompt_hash,
        source_primary_label=row.label,
        source_all_categories=row.categories.split(),
        suggested_label=suggested,
        alternative_labels=alternatives,
        ambiguity_flag=ambiguity,
        short_reason=reason,
        valid_output=valid_output,
        parse_error=error,
        outcome=_audit_outcome(row, suggested, alternatives, ambiguity, valid_output),
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        response_id=response.response_id,
    )
    return _CompletedAuditRow(record=record, raw_response=response.content)


def _owned_client(
    client: LLMClient | None, runtime_config: LLMRuntimeConfig | None, max_concurrency: int | None
) -> tuple[LLMClient, OpenAICompatibleChatClient | None, int]:
    if client is not None:
        if max_concurrency is not None:
            return client, None, max_concurrency
        return client, None, runtime_config.max_concurrency if runtime_config is not None else 1
    config = runtime_config or parse_llm_runtime_config(require_credentials=True)
    owned = OpenAICompatibleChatClient(config)
    return owned, owned, config.max_concurrency


def run_llm_audit(
    paths: ProjectPaths,
    sample_size: int = DEFAULT_AUDIT_SIZE,
    seed: int = RANDOM_SEED,
    resume: bool = True,
    client: LLMClient | None = None,
    runtime_config: LLMRuntimeConfig | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    paths.create_output_dirs()
    sample = _stratified_audit_sample(read_dataset(paths.dataset), sample_size, seed)
    active_client, owned_client, max_workers = _owned_client(
        client, runtime_config, max_concurrency
    )
    prompt = taxonomy_audit_json_v1_prompt(sample[0].text)
    cache = _prepare_audit_cache(paths.llm_audit_cache, resume)
    records_by_key: dict[str, LLMAuditRecord] = {}
    pending: list[DatasetRow] = []
    cache_hits = 0
    try:
        for row in sample:
            key = _cache_key(row, active_client.model, prompt)
            cached = cache.get(key)
            if cached is not None:
                records_by_key[key] = cached
                cache_hits += 1
            else:
                pending.append(row)
        if pending:
            failures: list[LLMRunFailure] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future[_CompletedAuditRow], DatasetRow] = {
                    executor.submit(_classify_audit_row, row, active_client): row for row in pending
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        completed = future.result()
                        _append_jsonl_atomic(
                            paths.llm_audit_cache,
                            {
                                "record": completed.record.model_dump(),
                                "raw_response": completed.raw_response,
                            },
                        )
                    except Exception as error:
                        failures.append(
                            LLMRunFailure(arxiv_id=row.arxiv_id, error_type=type(error).__name__)
                        )
                    else:
                        records_by_key[completed.record.cache_key] = completed.record
            if failures:
                raise LLMRunError("LLM label audit", failures)
    finally:
        if owned_client is not None:
            owned_client.close()
    ordered_records = [
        records_by_key[_cache_key(row, active_client.model, prompt)] for row in sample
    ]
    outcome_counts = Counter(record.outcome for record in ordered_records)
    source_counts = Counter(record.source_primary_label for record in ordered_records)
    suggestion_counts = Counter(
        record.suggested_label for record in ordered_records if record.suggested_label is not None
    )
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Development-only prompt-based label audit; suggestions are never applied to labels "
            "and never enter held-out test evaluation."
        ),
        "model": active_client.model,
        "transport": _transport_name(active_client),
        "prompt_version": prompt.version,
        "prompt_hash": prompt.prompt_hash,
        "input_contract": "Abstract text only; no titles, authors, source labels, or test rows.",
        "sample": {
            "size": len(sample),
            "seed": seed,
            "eligible_splits": ["train", "validation"],
            "source_class_counts": dict(source_counts),
            "cache_hit_count": cache_hits,
        },
        "counts": {
            "valid_outputs": sum(record.valid_output for record in ordered_records),
            "invalid_outputs": sum(not record.valid_output for record in ordered_records),
            "outcomes": dict(outcome_counts),
            "suggested_labels": dict(suggestion_counts),
        },
        "tokens": _token_totals(ordered_records),
        "reported_cost_usd": _reported_cost_total(ordered_records),
        "outcome_definitions": {
            "agreement": "Prompt suggestion matches the source primary label.",
            "plausible_alternative_cross_list_ambiguity": (
                "Prompt suggests another plausible label, often matching a secondary category "
                "or explicit ambiguity."
            ),
            "human_review_candidate": (
                "Prompt disagrees without a recorded ambiguity signal; not auto-corrected."
            ),
            "invalid_rejected": "Malformed or invalid prompt output rejected by strict parser.",
        },
        "limitations": [
            "The prompt audit is not ground truth and is not used to change source labels.",
            "Only train/validation rows are sampled; test rows are excluded by construction.",
            "Short reasons are model-generated explanations, not verified evidence.",
        ],
        "risk_controls": [
            "The prompt never receives the source label, title, authors, or held-out answer.",
            "Row-level outputs retain valid-output flags and rejected parse errors.",
            "The audit writes separate artifacts from model evaluation outputs.",
        ],
    }
    _write_json(paths.llm_audit_summary, summary)
    _write_jsonl(paths.llm_label_audit, [record.model_dump() for record in ordered_records])
    return summary


def _transport_name(client: LLMClient) -> str:
    value = getattr(client, "transport", None)
    return value if isinstance(value, str) else "openai-compatible"


def _token_totals(records: list[LLMAuditRecord]) -> dict[str, int | None]:
    return {
        "input_tokens": _sum_optional([record.usage.input_tokens for record in records]),
        "output_tokens": _sum_optional([record.usage.output_tokens for record in records]),
        "total_tokens": _sum_optional([record.usage.total_tokens for record in records]),
        "reasoning_tokens": _sum_optional([record.usage.reasoning_tokens for record in records]),
        "cache_read_tokens": _sum_optional([record.usage.cache_read_tokens for record in records]),
        "cache_write_tokens": _sum_optional(
            [record.usage.cache_write_tokens for record in records]
        ),
    }


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _reported_cost_total(records: list[LLMAuditRecord]) -> float | None:
    values = [record.usage.reported_cost_usd for record in records]
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))
