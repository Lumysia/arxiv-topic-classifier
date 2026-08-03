from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, cast

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from arxiv_topic_classifier.config import CATEGORY_NAMES, TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.data import dataset_split, read_dataset
from arxiv_topic_classifier.llm import (
    INVALID_LABEL,
    LLMChatPrompt,
    LLMClient,
    LLMClientResponse,
    LLMRunError,
    LLMRunFailure,
    LLMRuntimeConfig,
    OpenAICompatibleChatClient,
    parse_llm_runtime_config,
    parse_taxonomy_output,
    taxonomy_json_v1_prompt,
)
from arxiv_topic_classifier.schema import (
    DatasetRow,
    EvaluationSplit,
    LLMEvaluationRecord,
)
from arxiv_topic_classifier.text import validate_prediction_text

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class _CompletedLLMRow:
    record: LLMEvaluationRecord
    raw_response: str


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


def _load_evaluation_cache(path: Path) -> dict[str, LLMEvaluationRecord]:
    if not path.is_file():
        return {}
    records: dict[str, LLMEvaluationRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            raw_record = payload.get("record", payload) if isinstance(payload, dict) else payload
            record = LLMEvaluationRecord.model_validate(raw_record)
            records[record.cache_key] = record
    return records


def _prepare_evaluation_cache(path: Path, resume: bool) -> dict[str, LLMEvaluationRecord]:
    if resume:
        return _load_evaluation_cache(path)
    if path.exists():
        path.unlink()
    return {}


def _classify_row(row: DatasetRow, client: LLMClient, split: EvaluationSplit) -> _CompletedLLMRow:
    prompt = taxonomy_json_v1_prompt(row.text)
    response = client.complete(prompt)
    parsed = parse_taxonomy_output(response.content)
    correct = parsed.valid_output and parsed.label == row.label
    record = LLMEvaluationRecord(
        cache_key=_cache_key(row, client.model, prompt),
        arxiv_id=row.arxiv_id,
        split=split,
        abstract_hash=row.content_hash,
        model=client.model,
        prompt_version=prompt.version,
        prompt_hash=prompt.prompt_hash,
        actual_label=row.label,
        predicted_label=parsed.label,
        valid_output=parsed.valid_output,
        correct=correct,
        reason=parsed.reason,
        parse_error=parsed.error,
        latency_seconds=response.latency_seconds,
        usage=response.usage,
        response_id=response.response_id,
    )
    return _CompletedLLMRow(record=record, raw_response=response.content)


def _metrics(
    records: list[LLMEvaluationRecord], labels: list[str], cache_hits: int
) -> dict[str, Any]:
    truth = [record.actual_label for record in records]
    predictions = [
        record.predicted_label if record.valid_output and record.predicted_label else INVALID_LABEL
        for record in records
    ]
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    invalid_count = sum(1 for prediction in predictions if prediction == INVALID_LABEL)
    latencies = [record.latency_seconds for record in records]
    total_rows = len(records)
    recalls = [cast(float, cast(dict[str, Any], report[label])["recall"]) for label in labels]
    return {
        "accuracy": float(accuracy_score(truth, predictions)) if records else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        )
        if records
        else 0.0,
        "weighted_f1": float(
            f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)
        )
        if records
        else 0.0,
        "classification_report": report,
        "invalid_output_count": invalid_count,
        "invalid_output_rate": invalid_count / total_rows if total_rows else 0.0,
        "latency_seconds": {
            "average": float(sum(latencies) / total_rows) if total_rows else 0.0,
            "median": float(median(latencies)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "tokens": _token_totals(records),
        "reported_cost_usd": _reported_cost_total(records),
        "cache_hit_count": cache_hits,
    }


def _token_totals(records: list[LLMEvaluationRecord]) -> dict[str, int | None]:
    input_values = [record.usage.input_tokens for record in records]
    output_values = [record.usage.output_tokens for record in records]
    total_values = [record.usage.total_tokens for record in records]
    return {
        "input_tokens": _sum_optional(input_values),
        "output_tokens": _sum_optional(output_values),
        "total_tokens": _sum_optional(total_values),
        "reasoning_tokens": _sum_optional([record.usage.reasoning_tokens for record in records]),
        "cache_read_tokens": _sum_optional([record.usage.cache_read_tokens for record in records]),
        "cache_write_tokens": _sum_optional(
            [record.usage.cache_write_tokens for record in records]
        ),
    }


def _sum_optional(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _reported_cost_total(records: list[LLMEvaluationRecord]) -> float | None:
    values = [record.usage.reported_cost_usd for record in records]
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _transport_name(client: LLMClient) -> str:
    value = getattr(client, "transport", None)
    return value if isinstance(value, str) else "openai-compatible"


def _write_class_metrics(
    report: dict[str, Any], labels: list[str], destination: Path
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "name", "precision", "recall", "f1", "support"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in labels:
            values = cast(dict[str, Any], report[label])
            writer.writerow(
                {
                    "label": label,
                    "name": CATEGORY_NAMES[label],
                    "precision": f"{values['precision']:.6f}",
                    "recall": f"{values['recall']:.6f}",
                    "f1": f"{values['f1-score']:.6f}",
                    "support": int(cast(float, values["support"])),
                }
            )


def _confusion_counts(
    records: list[LLMEvaluationRecord], labels: list[str]
) -> tuple[list[str], list[list[int]]]:
    include_invalid = any(not record.valid_output for record in records)
    columns = [*labels, INVALID_LABEL] if include_invalid else labels
    matrix: list[list[int]] = []
    for actual in labels:
        row_counts: list[int] = []
        actual_records = [record for record in records if record.actual_label == actual]
        for predicted in columns:
            row_counts.append(
                sum(
                    1
                    for record in actual_records
                    if (
                        record.predicted_label
                        if record.valid_output and record.predicted_label is not None
                        else INVALID_LABEL
                    )
                    == predicted
                )
            )
        matrix.append(row_counts)
    return columns, matrix


def _write_confusion_csv(
    columns: list[str], matrix: list[list[int]], labels: list[str], destination: Path
) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual/predicted", *columns])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _plot_confusion(
    columns: list[str], matrix: list[list[int]], labels: list[str], destination: Path
) -> None:
    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    plt.figure(figsize=(7.2, 5.2))
    plt.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(np.arange(len(columns)), columns, rotation=30, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Prompt LLM normalized confusion matrix")
    for row_index, row in enumerate(normalized):
        for column_index, value in enumerate(row):
            plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()


def _error_patterns(records: list[LLMEvaluationRecord]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        predicted = (
            record.predicted_label
            if record.valid_output and record.predicted_label is not None
            else INVALID_LABEL
        )
        if predicted != record.actual_label:
            counts[(record.actual_label, predicted)] += 1
    return [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in counts.most_common(12)
    ]


def _prediction_examples(
    rows_by_id: dict[str, DatasetRow], records: list[LLMEvaluationRecord]
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        row = rows_by_id[record.arxiv_id]
        predicted = record.predicted_label if record.valid_output else INVALID_LABEL
        examples.append(
            {
                "arxiv_id": record.arxiv_id,
                "actual": record.actual_label,
                "predicted": predicted,
                "correct": record.correct,
                "valid_output": record.valid_output,
                "abstract_excerpt": row.abstract[:400],
                "secondary_categories": row.categories.split()[1:],
                "reason": record.reason,
                "parse_error": record.parse_error,
            }
        )
    correct = [item for item in examples if item["correct"]][:12]
    incorrect = [item for item in examples if not item["correct"]][:24]
    return [*correct, *incorrect]


def _write_artifacts(
    paths: ProjectPaths,
    split: EvaluationSplit,
    rows: list[DatasetRow],
    records: list[LLMEvaluationRecord],
    summary: dict[str, Any],
    cache_hit_keys: set[str],
) -> None:
    labels = list(TARGET_CATEGORIES)
    report = cast(dict[str, Any], summary["classification_report"])
    _write_class_metrics(report, labels, paths.llm_class_metrics(split))
    columns, matrix = _confusion_counts(records, labels)
    _write_confusion_csv(columns, matrix, labels, paths.llm_confusion_matrix(split))
    _plot_confusion(columns, matrix, labels, paths.llm_confusion_figure(split))
    _write_json(paths.llm_error_patterns(split), _error_patterns(records))
    _write_json(
        paths.llm_prediction_examples(split),
        _prediction_examples({row.arxiv_id: row for row in rows}, records),
    )
    trace = [
        {
            **record.model_dump(),
            "cache_hit": record.cache_key in cache_hit_keys,
        }
        for record in records
    ]
    _write_jsonl(paths.llm_prediction_trace(split), trace)


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


def evaluate_llm_split(
    paths: ProjectPaths,
    split: EvaluationSplit,
    resume: bool = True,
    client: LLMClient | None = None,
    runtime_config: LLMRuntimeConfig | None = None,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    paths.create_output_dirs()
    rows = dataset_split(read_dataset(paths.dataset), split)
    if not rows:
        raise ValueError(f"Dataset split is empty: {split}")
    active_client, owned_client, max_workers = _owned_client(
        client, runtime_config, max_concurrency
    )
    prompt = taxonomy_json_v1_prompt(rows[0].text)
    cache = _prepare_evaluation_cache(paths.llm_cache(split), resume)
    records_by_key: dict[str, LLMEvaluationRecord] = {}
    pending: list[DatasetRow] = []
    cache_hit_keys: set[str] = set()
    cache_hits = 0
    try:
        for row in rows:
            key = _cache_key(row, active_client.model, prompt)
            cached = cache.get(key)
            if cached is not None:
                records_by_key[key] = cached
                cache_hit_keys.add(key)
                cache_hits += 1
            else:
                pending.append(row)
        if pending:
            failures: list[LLMRunFailure] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: dict[Future[_CompletedLLMRow], DatasetRow] = {
                    executor.submit(_classify_row, row, active_client, split): row
                    for row in pending
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        completed = future.result()
                        _append_jsonl_atomic(
                            paths.llm_cache(split),
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
                raise LLMRunError(
                    f"LLM evaluation for split {split}",
                    failures,
                )
    finally:
        if owned_client is not None:
            owned_client.close()
    ordered_records = [
        records_by_key[_cache_key(row, active_client.model, prompt)] for row in rows
    ]
    labels = list(TARGET_CATEGORIES)
    metric_values = _metrics(ordered_records, labels, cache_hits)
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "split": split,
        "row_count": len(rows),
        "model": active_client.model,
        "transport": _transport_name(active_client),
        "prompt_version": prompt.version,
        "prompt_hash": prompt.prompt_hash,
        "input_contract": "Abstract text only; no titles, authors, source labels, or answers.",
        "selection_policy": (
            "taxonomy_json_v1 is frozen before held-out test evaluation; test is not used for "
            "prompt tuning."
        ),
        "labels": labels,
        **metric_values,
    }
    _write_json(paths.llm_metrics(split), summary)
    _write_artifacts(paths, split, rows, ordered_records, summary, cache_hit_keys)
    return summary


def predict_llm_text(
    text: object,
    client: LLMClient | None = None,
    runtime_config: LLMRuntimeConfig | None = None,
) -> dict[str, Any]:
    abstract = validate_prediction_text(text)
    active_client, owned_client, _ = _owned_client(client, runtime_config, max_concurrency=None)
    prompt = taxonomy_json_v1_prompt(abstract)
    try:
        response: LLMClientResponse = active_client.complete(prompt)
        parsed = parse_taxonomy_output(response.content)
        return {
            "model": active_client.model,
            "transport": _transport_name(active_client),
            "prompt_version": prompt.version,
            "prompt_hash": prompt.prompt_hash,
            "label": parsed.label,
            "category_name": CATEGORY_NAMES[parsed.label] if parsed.label else None,
            "valid_output": parsed.valid_output,
            "reason": parsed.reason,
            "error": parsed.error,
            "latency_seconds": response.latency_seconds,
            "usage": response.usage.model_dump(),
            "response_id": response.response_id,
        }
    finally:
        if owned_client is not None:
            owned_client.close()
