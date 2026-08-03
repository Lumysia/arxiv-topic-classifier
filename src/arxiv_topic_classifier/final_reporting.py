from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, TypedDict, cast

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from arxiv_topic_classifier.config import (
    CATEGORY_NAMES,
    RANDOM_SEED,
    TARGET_CATEGORIES,
    ProjectPaths,
)
from arxiv_topic_classifier.data import dataset_split, read_dataset
from arxiv_topic_classifier.llm import INVALID_LABEL, taxonomy_json_v1_prompt
from arxiv_topic_classifier.model import load_artifact
from arxiv_topic_classifier.schema import DatasetRow, LLMEvaluationRecord, LLMUsage
from arxiv_topic_classifier.text import content_fingerprint
from arxiv_topic_classifier.web import (
    DEMO_LLM_CACHE_FILENAME,
    DEMO_SELECTION_FILENAME,
    DemoLLMCacheEntry,
    DemoLLMCacheFile,
    demo_abstract_hash,
    load_demo_llm_cache,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FLOAT_TOLERANCE = 1e-9


class FinalReportValidationError(ValueError):
    pass


def run_final_report(paths: ProjectPaths, expected_test_rows: int = 900) -> dict[str, Any]:
    paths.create_output_dirs()
    labels = list(TARGET_CATEGORIES)
    rows = _validated_test_rows(paths.dataset, expected_test_rows, labels)
    metrics = _load_json_object(paths.artifacts / "metrics.json")
    llm_metrics = _load_json_object(paths.llm_metrics("test"))
    trace_payloads, llm_records = _load_trace(paths.llm_prediction_trace("test"))
    _validate_llm_inputs(rows, llm_records, trace_payloads, llm_metrics, labels, expected_test_rows)

    artifact = load_artifact(paths.model)
    classical_predictions = _classical_predictions(artifact, rows, labels)
    classical_metrics = _classification_metrics(
        [row.label for row in rows],
        [row["predicted_label"] for row in classical_predictions],
        labels,
    )
    _validate_classical_metrics(metrics, artifact, classical_metrics)

    pairwise_records = _pairwise_records(rows, classical_predictions, llm_records)
    comparison = _comparison_payload(
        rows=rows,
        labels=labels,
        metrics=metrics,
        llm_metrics=llm_metrics,
        llm_records=llm_records,
        classical_metrics=classical_metrics,
        classical_predictions=classical_predictions,
        pairwise_records=pairwise_records,
    )
    demo_selection = _demo_selection(pairwise_records, llm_metrics)
    demo_cache = _demo_cache_payload(demo_selection, pairwise_records, llm_metrics)

    comparison_path = paths.artifacts / "classical_vs_llm_comparison.json"
    pairwise_path = paths.artifacts / "pairwise_test_records.jsonl"
    selection_path = paths.artifacts / DEMO_SELECTION_FILENAME
    cache_path = paths.artifacts / DEMO_LLM_CACHE_FILENAME

    _write_json(comparison_path, comparison)
    _write_jsonl(pairwise_path, pairwise_records)
    _write_json(selection_path, demo_selection)
    _write_json(cache_path, demo_cache)
    _validate_written_demo_cache(cache_path)
    _write_figures(paths.figures, labels, comparison)
    return comparison


def _validated_test_rows(
    path: Path, expected_test_rows: int, labels: list[str]
) -> list[DatasetRow]:
    rows = dataset_split(read_dataset(path), "test")
    if len(rows) != expected_test_rows:
        raise FinalReportValidationError(
            f"Expected {expected_test_rows} test rows, found {len(rows)} in {path}."
        )
    ids = [row.arxiv_id for row in rows]
    duplicate_ids = [arxiv_id for arxiv_id, count in Counter(ids).items() if count != 1]
    if duplicate_ids:
        raise FinalReportValidationError(f"Test split has duplicate arXiv IDs: {duplicate_ids[:5]}")
    actual_labels = sorted({row.label for row in rows})
    if actual_labels != labels:
        raise FinalReportValidationError(
            f"Expected exactly six target labels, found {actual_labels}."
        )
    for row in rows:
        if row.split != "test":
            raise FinalReportValidationError(f"Non-test row reached test report: {row.arxiv_id}")
        if row.text != row.abstract:
            raise FinalReportValidationError(
                "Dataset text column is not abstract-only; reporting refuses to use titles."
            )
        if row.content_hash != content_fingerprint(row.abstract):
            raise FinalReportValidationError(
                f"Abstract hash mismatch in dataset row {row.arxiv_id}."
            )
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalReportValidationError(f"Required artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalReportValidationError(f"Required artifact is not a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _load_trace(path: Path) -> tuple[list[dict[str, Any]], list[LLMEvaluationRecord]]:
    if not path.is_file():
        raise FinalReportValidationError(f"Required trace artifact is missing: {path}")
    payloads: list[dict[str, Any]] = []
    records: list[LLMEvaluationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or "record" in payload:
                raise FinalReportValidationError(
                    f"Trace line {line_number} is not a finalized prediction-trace row."
                )
            payloads.append(cast(dict[str, Any], payload))
            records.append(LLMEvaluationRecord.model_validate(payload))
    return payloads, records


def _validate_llm_inputs(
    rows: list[DatasetRow],
    records: list[LLMEvaluationRecord],
    trace_payloads: list[dict[str, Any]],
    llm_metrics: dict[str, Any],
    labels: list[str],
    expected_test_rows: int,
) -> None:
    if len(records) != expected_test_rows:
        raise FinalReportValidationError(
            f"Expected {expected_test_rows} LLM trace rows, found {len(records)}."
        )
    trace_ids = [record.arxiv_id for record in records]
    duplicate_trace_ids = [arxiv_id for arxiv_id, count in Counter(trace_ids).items() if count != 1]
    if duplicate_trace_ids:
        raise FinalReportValidationError(
            f"LLM trace has duplicate arXiv IDs: {duplicate_trace_ids[:5]}"
        )
    if llm_metrics.get("split") != "test":
        raise FinalReportValidationError("LLM metrics are not for the test split.")
    if llm_metrics.get("row_count") != expected_test_rows:
        raise FinalReportValidationError("LLM metrics row_count does not match expected test rows.")
    if llm_metrics.get("labels") != labels:
        raise FinalReportValidationError("LLM metrics labels do not match the six target labels.")

    prompt = taxonomy_json_v1_prompt(rows[0].text)
    expected_prompt = {"prompt_version": prompt.version, "prompt_hash": prompt.prompt_hash}
    for key, value in expected_prompt.items():
        if llm_metrics.get(key) != value:
            raise FinalReportValidationError(f"LLM metrics {key} is stale or mixed.")
    model = _required_string(llm_metrics, "model")
    transport = _required_string(llm_metrics, "transport")
    for row, record, payload in zip(rows, records, trace_payloads, strict=True):
        if record.arxiv_id != row.arxiv_id:
            raise FinalReportValidationError(
                f"LLM trace order/id mismatch: expected {row.arxiv_id}, found {record.arxiv_id}."
            )
        if record.split != "test" or record.actual_label != row.label:
            raise FinalReportValidationError(f"LLM trace metadata mismatch for {row.arxiv_id}.")
        if record.abstract_hash != row.content_hash:
            raise FinalReportValidationError(
                f"LLM trace abstract hash mismatch for {row.arxiv_id}."
            )
        if record.model != model:
            raise FinalReportValidationError(f"LLM trace has mixed model values at {row.arxiv_id}.")
        if record.prompt_version != prompt.version or record.prompt_hash != prompt.prompt_hash:
            raise FinalReportValidationError(
                f"LLM trace has stale prompt metadata at {row.arxiv_id}."
            )
        if "transport" in payload and payload["transport"] != transport:
            raise FinalReportValidationError(
                f"LLM trace has mixed transport values at {row.arxiv_id}."
            )
        if record.actual_label not in labels:
            raise FinalReportValidationError(
                f"Unknown actual label in LLM trace at {row.arxiv_id}."
            )
        if record.valid_output and record.predicted_label not in labels:
            raise FinalReportValidationError(f"Unknown LLM predicted label at {row.arxiv_id}.")
        if record.correct != (
            record.valid_output and record.predicted_label == record.actual_label
        ):
            raise FinalReportValidationError(
                f"LLM trace correctness flag mismatch at {row.arxiv_id}."
            )

    recomputed = _classification_metrics(
        [record.actual_label for record in records],
        [_llm_label(record) for record in records],
        labels,
    )
    _assert_metric_close(
        recomputed, llm_metrics, ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    )
    invalid_count = sum(1 for record in records if not record.valid_output)
    if llm_metrics.get("invalid_output_count") != invalid_count:
        raise FinalReportValidationError("LLM invalid_output_count does not match the trace.")
    expected_rate = invalid_count / len(records) if records else 0.0
    if not math.isclose(
        float(llm_metrics.get("invalid_output_rate", -1.0)),
        expected_rate,
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise FinalReportValidationError("LLM invalid_output_rate does not match the trace.")


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FinalReportValidationError(f"Required string is missing from LLM metrics: {key}")
    return value


class _ClassicalRow(TypedDict):
    arxiv_id: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    latency_seconds: float


def _classical_predictions(
    artifact: dict[str, Any], rows: list[DatasetRow], labels: list[str]
) -> list[_ClassicalRow]:
    if artifact.get("labels") != labels:
        raise FinalReportValidationError(
            "Classical model labels do not match the six target labels."
        )
    input_contract = artifact.get("input_contract")
    if not isinstance(input_contract, str) or "abstract" not in input_contract.lower():
        raise FinalReportValidationError(
            "Classical model artifact lacks an abstract-only input contract."
        )
    estimator = artifact["pipeline"]
    estimator_labels = cast(list[str], estimator.classes_.tolist())
    if estimator_labels != labels:
        raise FinalReportValidationError("Classical estimator class order does not match labels.")

    predictions: list[_ClassicalRow] = []
    for row in rows:
        started = perf_counter()
        raw_prediction = cast(list[str], estimator.predict([row.text]).tolist())[0]
        raw_probabilities = cast(np.ndarray, estimator.predict_proba([row.text]))[0]
        latency = perf_counter() - started
        probabilities = {
            label: float(value) for label, value in zip(labels, raw_probabilities, strict=True)
        }
        if raw_prediction not in labels:
            raise FinalReportValidationError(f"Unknown classical prediction for {row.arxiv_id}.")
        predictions.append(
            {
                "arxiv_id": row.arxiv_id,
                "predicted_label": raw_prediction,
                "confidence": probabilities[raw_prediction],
                "probabilities": dict(
                    sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
                ),
                "latency_seconds": latency,
            }
        )
    return predictions


def _classification_metrics(
    truth: list[str], predictions: list[str], labels: list[str]
) -> dict[str, Any]:
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    recalls = [cast(float, cast(dict[str, Any], report[label])["recall"]) for label in labels]
    return {
        "accuracy": float(accuracy_score(truth, predictions)) if truth else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        )
        if truth
        else 0.0,
        "weighted_f1": float(
            f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)
        )
        if truth
        else 0.0,
        "classification_report": report,
    }


def _validate_classical_metrics(
    metrics: dict[str, Any], artifact: dict[str, Any], recomputed: dict[str, Any]
) -> None:
    selected = artifact.get("selected_model")
    if not isinstance(selected, str):
        raise FinalReportValidationError("Classical artifact is missing selected_model.")
    if metrics.get("selected_model") != selected:
        raise FinalReportValidationError(
            "metrics.json selected_model does not match model artifact."
        )
    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, dict) or selected not in test_metrics:
        raise FinalReportValidationError("metrics.json is missing selected model test metrics.")
    saved = cast(dict[str, Any], test_metrics[selected])
    _assert_metric_close(recomputed, saved, ("accuracy", "macro_f1"))


def _assert_metric_close(
    recomputed: dict[str, Any], saved: dict[str, Any], names: tuple[str, ...]
) -> None:
    for name in names:
        if name not in saved:
            raise FinalReportValidationError(f"Saved metrics missing {name}.")
        if not math.isclose(
            float(recomputed[name]),
            float(saved[name]),
            rel_tol=FLOAT_TOLERANCE,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise FinalReportValidationError(
                f"Metric mismatch for {name}: recomputed={recomputed[name]} saved={saved[name]}."
            )


def _llm_label(record: LLMEvaluationRecord) -> str:
    if record.valid_output and record.predicted_label is not None:
        return record.predicted_label
    return INVALID_LABEL


def _pairwise_records(
    rows: list[DatasetRow],
    classical_predictions: list[_ClassicalRow],
    llm_records: list[LLMEvaluationRecord],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row, classical, llm in zip(rows, classical_predictions, llm_records, strict=True):
        llm_label = _llm_label(llm)
        classical_correct = classical["predicted_label"] == row.label
        llm_correct = llm.valid_output and llm_label == row.label
        outcome = _outcome(classical_correct, llm_correct, llm.valid_output)
        agreement = (
            classical["predicted_label"] == llm.predicted_label if llm.valid_output else None
        )
        records.append(
            {
                "arxiv_id": row.arxiv_id,
                "split": row.split,
                "actual_label": row.label,
                "category_name": CATEGORY_NAMES[row.label],
                "abstract_hash": row.content_hash,
                "demo_abstract_hash": demo_abstract_hash(row.text),
                "normalized_abstract": row.text,
                "secondary_categories": row.categories.split()[1:],
                "classical": {
                    "label": classical["predicted_label"],
                    "correct": classical_correct,
                    "confidence": classical["confidence"],
                    "probabilities": classical["probabilities"],
                    "latency_seconds": classical["latency_seconds"],
                },
                "llm": {
                    "label": llm.predicted_label,
                    "effective_label": llm_label,
                    "correct": llm_correct,
                    "valid_output": llm.valid_output,
                    "reason": llm.reason,
                    "parse_error": llm.parse_error,
                    "model": llm.model,
                    "prompt_version": llm.prompt_version,
                    "prompt_hash": llm.prompt_hash,
                    "latency_seconds": llm.latency_seconds,
                    "usage": llm.usage.model_dump(),
                    "response_id": llm.response_id,
                },
                "agreement": agreement,
                "outcome": outcome,
            }
        )
    return records


def _outcome(classical_correct: bool, llm_correct: bool, llm_valid: bool) -> str:
    if not llm_valid:
        return "llm_invalid"
    if classical_correct and llm_correct:
        return "both_correct"
    if classical_correct:
        return "classical_only_correct"
    if llm_correct:
        return "llm_only_correct"
    return "both_wrong"


def _comparison_payload(
    *,
    rows: list[DatasetRow],
    labels: list[str],
    metrics: dict[str, Any],
    llm_metrics: dict[str, Any],
    llm_records: list[LLMEvaluationRecord],
    classical_metrics: dict[str, Any],
    classical_predictions: list[_ClassicalRow],
    pairwise_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = cast(str, metrics["selected_model"])
    llm_summary = _metric_summary(llm_metrics, labels)
    classical_summary = _metric_summary(classical_metrics, labels)
    agreements = sum(1 for record in pairwise_records if record["agreement"] is True)
    valid_llm_rows = sum(1 for record in pairwise_records if record["llm"]["valid_output"] is True)
    outcome_counts = Counter(cast(str, record["outcome"]) for record in pairwise_records)
    classical_latencies = [item["latency_seconds"] for item in classical_predictions]
    llm_latencies = [record.latency_seconds for record in llm_records]
    classical_input_contract = (
        "Observed local inference over normalized abstract text only; "
        "title metadata is not used."
    )
    llm_input_contract = (
        "Held-out prompt trace over normalized abstract text only; "
        "no titles or source labels are supplied."
    )
    latency_note = (
        "Measured with perf_counter while recomputing per-row local predictions and probabilities; "
        "this is inference latency, not training time, and not identical infrastructure to "
        "remote/model runtime latency."
    )
    notes = [
        "OpenCode runtime token counts include agent/runtime overhead, not only prompt-visible "
        "taxonomy text.",
        "Provider-reported zero cost is not zero compute cost; it only means the provider "
        "reported 0.0 for every row.",
        "Classical local latency and LLM recorded runtime latency are reported on different "
        "infrastructure and are not direct system benchmarks.",
    ]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_split": "test",
        "row_count": len(rows),
        "seed": _seed(metrics),
        "labels": labels,
        "category_names": {label: CATEGORY_NAMES[label] for label in labels},
        "input_contract": {
            "classical": classical_input_contract,
            "llm": llm_input_contract,
        },
        "classical": {
            "selected_model": selected,
            "metrics": classical_summary,
            "observed_local_inference_latency_seconds": _latency_summary(classical_latencies),
            "latency_note": latency_note,
            "top_directional_error_patterns": _error_patterns(pairwise_records, "classical"),
        },
        "llm": {
            "model": llm_metrics["model"],
            "transport": llm_metrics["transport"],
            "prompt_version": llm_metrics["prompt_version"],
            "prompt_hash": llm_metrics["prompt_hash"],
            "metrics": llm_summary,
            "recorded_latency_seconds": _latency_summary(llm_latencies),
            "tokens": _token_totals(llm_records),
            "provider_reported_cost_usd": _reported_cost_total(llm_records),
            "invalid_output_count": llm_metrics["invalid_output_count"],
            "invalid_output_rate": llm_metrics["invalid_output_rate"],
            "top_directional_error_patterns": _error_patterns(pairwise_records, "llm"),
        },
        "absolute_differences_llm_minus_classical": {
            name: float(llm_summary[name] - classical_summary[name])
            for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "per_class_f1_differences_llm_minus_classical": {
            label: float(
                llm_summary["per_class_f1"][label] - classical_summary["per_class_f1"][label]
            )
            for label in labels
        },
        "agreement": {
            "agree_count": agreements,
            "disagree_count": valid_llm_rows - agreements,
            "valid_llm_row_count": valid_llm_rows,
            "rate_among_valid_llm_outputs": agreements / valid_llm_rows if valid_llm_rows else 0.0,
        },
        "pairwise_outcomes": {key: outcome_counts.get(key, 0) for key in _outcome_order()},
        "notes": notes,
    }


def _seed(metrics: dict[str, Any]) -> int | None:
    value = metrics.get("seed")
    return value if isinstance(value, int) else RANDOM_SEED


def _metric_summary(source: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    report = cast(dict[str, Any], source["classification_report"])
    return {
        "accuracy": float(source["accuracy"]),
        "balanced_accuracy": float(source["balanced_accuracy"]),
        "macro_f1": float(source["macro_f1"]),
        "weighted_f1": float(source["weighted_f1"]),
        "per_class_f1": {
            label: float(cast(dict[str, Any], report[label])["f1-score"]) for label in labels
        },
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "average": float(sum(values) / len(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p95": float(np.percentile(values, 95)) if values else 0.0,
    }


def _token_totals(records: list[LLMEvaluationRecord]) -> dict[str, int | None]:
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


def _reported_cost_total(records: list[LLMEvaluationRecord]) -> float | None:
    values = [record.usage.reported_cost_usd for record in records]
    if not values or any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _error_patterns(pairwise_records: list[dict[str, Any]], model_key: str) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in pairwise_records:
        actual = cast(str, record["actual_label"])
        if model_key == "classical":
            predicted = cast(str, record["classical"]["label"])
        else:
            predicted = cast(str, record["llm"]["effective_label"])
        if predicted != actual:
            counts[(actual, predicted)] += 1
    return [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in counts.most_common(12)
    ]


def _demo_selection(
    pairwise_records: list[dict[str, Any]], llm_metrics: dict[str, Any]
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    agreement = _choose_agreement(pairwise_records)
    if agreement is not None:
        selected.append(_selection_entry("clear_agreement", "Clear agreement", agreement))
    disagreement = _choose_disagreement(pairwise_records)
    if disagreement is not None:
        selected.append(
            _selection_entry("informative_disagreement", "Informative disagreement", disagreement)
        )
    classical_only = _choose_classical_only(pairwise_records, selected)
    if classical_only is not None:
        selected.append(
            _selection_entry(
                "classical_correct_llm_wrong", "Classical correct, LLM wrong", classical_only
            )
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy": (
            "Deterministic evidence-driven sample: prefer one both-correct cs.RO high-confidence "
            "agreement, one LLM-correct/classical-wrong cs.AI<->cs.LG or cs.AI<->cs.CL "
            "disagreement when present, and optionally one classical-correct/LLM-wrong example. "
            "These examples are for demo coverage and must not be read as aggregate performance."
        ),
        "aggregate_context": {
            "test_row_count": llm_metrics["row_count"],
            "llm_accuracy": llm_metrics["accuracy"],
            "llm_macro_f1": llm_metrics["macro_f1"],
        },
        "examples": selected,
    }


def _choose_agreement(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if record["outcome"] == "both_correct"]
    preferred = [record for record in candidates if record["actual_label"] == "cs.RO"]
    return _highest_confidence(preferred or candidates)


def _choose_disagreement(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if record["outcome"] == "llm_only_correct"]
    preferred_pairs = ({"cs.AI", "cs.LG"}, {"cs.AI", "cs.CL"})
    preferred = [
        record
        for record in candidates
        if {record["actual_label"], record["classical"]["label"]} in preferred_pairs
    ]
    fallback = [
        record
        for record in records
        if record["agreement"] is False and record["llm"]["valid_output"] is True
    ]
    return _deterministic_record(preferred or candidates or fallback)


def _choose_classical_only(
    records: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, Any] | None:
    selected_ids = {item["arxiv_id"] for item in selected}
    candidates = [
        record
        for record in records
        if record["outcome"] == "classical_only_correct" and record["arxiv_id"] not in selected_ids
    ]
    return _highest_confidence(candidates)


def _highest_confidence(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(
        records,
        key=lambda record: (
            -cast(float, record["classical"]["confidence"]),
            cast(str, record["arxiv_id"]),
        ),
    )[0]


def _deterministic_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return sorted(records, key=lambda record: cast(str, record["arxiv_id"]))[0]


def _selection_entry(key: str, display_label: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "display_label": display_label,
        "arxiv_id": record["arxiv_id"],
        "actual_label": record["actual_label"],
        "classical_label": record["classical"]["label"],
        "llm_label": record["llm"]["label"],
        "outcome": record["outcome"],
        "classical_confidence": record["classical"]["confidence"],
        "normalized_abstract": record["normalized_abstract"],
        "abstract_hash": record["demo_abstract_hash"],
        "dataset_abstract_hash": record["abstract_hash"],
    }


def _demo_cache_payload(
    selection: dict[str, Any], pairwise_records: list[dict[str, Any]], llm_metrics: dict[str, Any]
) -> dict[str, Any]:
    records_by_id = {record["arxiv_id"]: record for record in pairwise_records}
    prompt = taxonomy_json_v1_prompt("x" * 80)
    entries: list[DemoLLMCacheEntry] = []
    for example in cast(list[dict[str, Any]], selection["examples"]):
        record = records_by_id[example["arxiv_id"]]
        llm = cast(dict[str, Any], record["llm"])
        entries.append(
            DemoLLMCacheEntry(
                normalized_abstract=cast(str, record["normalized_abstract"]),
                abstract_hash=cast(str, record["demo_abstract_hash"]),
                label=cast(str | None, llm["label"]),
                reason=cast(str | None, llm["reason"]),
                valid_output=cast(bool, llm["valid_output"]),
                parse_error=cast(str | None, llm["parse_error"]),
                model=cast(str, llm["model"]),
                transport=cast(str, llm_metrics["transport"]),
                latency_seconds=cast(float, llm["latency_seconds"]),
                usage=LLMUsage.model_validate(llm["usage"]),
                response_id=cast(str | None, llm["response_id"]),
            )
        )
    cache = DemoLLMCacheFile(
        format_version=1,
        prompt_version=prompt.version,
        prompt_hash=prompt.prompt_hash,
        entries=entries,
    )
    return cache.model_dump()


def _validate_written_demo_cache(path: Path) -> None:
    loaded = load_demo_llm_cache(path)
    if not loaded.available or loaded.warnings:
        reason = loaded.error or loaded.warnings
        raise FinalReportValidationError(
            f"Generated demo cache did not pass the web cache loader: {reason}"
        )


def _write_figures(figures: Path, labels: list[str], comparison: dict[str, Any]) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    _plot_metrics(comparison, figures / "classical_vs_llm_metrics.png")
    _plot_per_class_f1(labels, comparison, figures / "classical_vs_llm_per_class_f1.png")
    _plot_outcomes(comparison, figures / "classical_vs_llm_outcomes.png")
    _plot_latency(comparison, figures / "classical_vs_llm_latency.png")


def _plot_metrics(comparison: dict[str, Any], destination: Path) -> None:
    names = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    classical = [comparison["classical"]["metrics"][name] for name in names]
    llm = [comparison["llm"]["metrics"][name] for name in names]
    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(7.8, 4.5))
    plt.bar(x - width / 2, classical, width, label="Classical TF-IDF", color="#0f766e")
    plt.bar(x + width / 2, llm, width, label="Prompt LLM", color="#6d28d9")
    plt.xticks(x, [name.replace("_", "\n") for name in names])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Held-out test metrics")
    plt.legend()
    _save_figure(destination)


def _plot_per_class_f1(labels: list[str], comparison: dict[str, Any], destination: Path) -> None:
    classical = [comparison["classical"]["metrics"]["per_class_f1"][label] for label in labels]
    llm = [comparison["llm"]["metrics"]["per_class_f1"][label] for label in labels]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8.2, 4.5))
    plt.bar(x - width / 2, classical, width, label="Classical TF-IDF", color="#0f766e")
    plt.bar(x + width / 2, llm, width, label="Prompt LLM", color="#6d28d9")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title("Per-class held-out F1")
    plt.legend()
    _save_figure(destination)


def _plot_outcomes(comparison: dict[str, Any], destination: Path) -> None:
    outcomes = _outcome_order()
    values = [comparison["pairwise_outcomes"][name] for name in outcomes]
    plt.figure(figsize=(8.2, 4.5))
    plt.bar(
        np.arange(len(outcomes)),
        values,
        color=["#15803d", "#0f766e", "#6d28d9", "#c2410c", "#b91c1c"],
    )
    plt.xticks(np.arange(len(outcomes)), [name.replace("_", "\n") for name in outcomes])
    plt.ylabel("Rows")
    plt.title("Paired prediction outcomes")
    _save_figure(destination)


def _plot_latency(comparison: dict[str, Any], destination: Path) -> None:
    names = ["average", "median", "p95"]
    classical = [
        comparison["classical"]["observed_local_inference_latency_seconds"][name] for name in names
    ]
    llm = [comparison["llm"]["recorded_latency_seconds"][name] for name in names]
    x = np.arange(len(names))
    width = 0.36
    positive = [value for value in [*classical, *llm] if value > 0]
    plt.figure(figsize=(7.6, 4.5))
    plt.bar(x - width / 2, classical, width, label="Classical local observed", color="#0f766e")
    plt.bar(x + width / 2, llm, width, label="LLM recorded runtime", color="#6d28d9")
    if positive and max(positive) / min(positive) >= 20:
        plt.yscale("log")
        plt.ylabel("Seconds (log scale)")
    else:
        plt.ylabel("Seconds")
    plt.xticks(x, names)
    plt.title("Inference/runtime latency summaries")
    plt.legend()
    _save_figure(destination)


def _outcome_order() -> tuple[str, ...]:
    return (
        "both_correct",
        "classical_only_correct",
        "llm_only_correct",
        "both_wrong",
        "llm_invalid",
    )


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


def _save_figure(destination: Path) -> None:
    plt.tight_layout()
    plt.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close()
