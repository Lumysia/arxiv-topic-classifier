from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arxiv_topic_classifier.config import CATEGORY_DESCRIPTIONS, ProjectPaths
from arxiv_topic_classifier.data import dataset_split, read_dataset, write_dataset
from arxiv_topic_classifier.final_reporting import FinalReportValidationError, run_final_report
from arxiv_topic_classifier.llm import taxonomy_json_v1_prompt
from arxiv_topic_classifier.llm_evaluation import _metrics
from arxiv_topic_classifier.model import train_and_evaluate
from arxiv_topic_classifier.schema import DatasetRow, LLMEvaluationRecord, LLMUsage
from arxiv_topic_classifier.text import content_fingerprint
from arxiv_topic_classifier.web import create_app, load_demo_llm_cache

KEYWORDS = {
    "cs.AI": "agents reasoning planning knowledge",
    "cs.CL": "language translation text linguistic",
    "cs.CV": "image vision object segmentation",
    "cs.LG": "learning neural optimization generalization",
    "cs.RO": "robot navigation manipulation control",
    "cs.SE": "software testing debugging maintenance",
}


def _reporting_project(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths.from_root(tmp_path)
    paths.create_output_dirs()
    rows: list[DatasetRow] = []
    for label in CATEGORY_DESCRIPTIONS:
        for index in range(20):
            terms = KEYWORDS[label]
            if label == "cs.AI" and index == 17:
                terms = KEYWORDS["cs.LG"]
            title = f"Synthetic title {label} {index}"
            abstract = (
                f"This paper studies {terms} with controlled experiments and reproducible "
                f"benchmark evaluation number {index}. The abstract intentionally contains "
                "enough domain-specific evidence for classification without relying on title text."
            )
            split = "train" if index < 14 else "validation" if index < 17 else "test"
            rows.append(
                DatasetRow(
                    arxiv_id=f"{label}-{index}",
                    title=title,
                    abstract=abstract,
                    text=abstract,
                    label=label,
                    categories=label,
                    year=2026,
                    split=split,
                    content_hash=content_fingerprint(abstract),
                )
            )
    write_dataset(rows, paths.dataset)
    train_and_evaluate(paths)
    _write_llm_outputs(paths)
    return paths


def _write_llm_outputs(paths: ProjectPaths) -> None:
    rows = dataset_split(read_dataset(paths.dataset), "test")
    labels = list(CATEGORY_DESCRIPTIONS)
    prompt = taxonomy_json_v1_prompt(rows[0].text)
    records: list[LLMEvaluationRecord] = []
    for index, row in enumerate(rows):
        predicted = row.label
        if row.arxiv_id == "cs.CL-18":
            predicted = "cs.AI"
        records.append(
            LLMEvaluationRecord(
                cache_key=f"synthetic-key-{row.arxiv_id}",
                arxiv_id=row.arxiv_id,
                split="test",
                abstract_hash=row.content_hash,
                model="synthetic-llm",
                prompt_version=prompt.version,
                prompt_hash=prompt.prompt_hash,
                actual_label=row.label,
                predicted_label=predicted,
                valid_output=True,
                correct=predicted == row.label,
                reason=f"Synthetic reason {index}.",
                parse_error=None,
                latency_seconds=0.4 + index / 100,
                usage=LLMUsage(
                    input_tokens=100 + index,
                    output_tokens=8,
                    total_tokens=108 + index,
                    reasoning_tokens=2,
                    cache_read_tokens=0,
                    cache_write_tokens=0,
                    reported_cost_usd=0.0,
                ),
                response_id=f"response-{index}",
            )
        )
    summary: dict[str, Any] = {
        "generated_at_utc": "2026-08-02T00:00:00+00:00",
        "split": "test",
        "row_count": len(rows),
        "model": "synthetic-llm",
        "transport": "synthetic-transport",
        "prompt_version": prompt.version,
        "prompt_hash": prompt.prompt_hash,
        "input_contract": "Abstract text only; no titles, authors, source labels, or answers.",
        "selection_policy": "Synthetic fixed prompt for tests.",
        "labels": labels,
        **_metrics(records, labels, cache_hits=0),
    }
    paths.llm_metrics("test").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with paths.llm_prediction_trace("test").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record.model_dump(), "cache_hit": False}) + "\n")


def test_final_report_generates_aligned_artifacts_and_cache(tmp_path: Path) -> None:
    paths = _reporting_project(tmp_path)

    comparison = run_final_report(paths, expected_test_rows=18)

    assert comparison["row_count"] == 18
    assert comparison["llm"]["provider_reported_cost_usd"] == 0.0
    expected = [
        "classical_vs_llm_comparison.json",
        "pairwise_test_records.jsonl",
        "demo_selection.json",
        "demo_llm_cache.json",
        "figures/classical_vs_llm_metrics.png",
        "figures/classical_vs_llm_per_class_f1.png",
        "figures/classical_vs_llm_outcomes.png",
        "figures/classical_vs_llm_latency.png",
    ]
    assert all((paths.artifacts / relative).stat().st_size > 0 for relative in expected)
    assert len((paths.artifacts / "pairwise_test_records.jsonl").read_text().splitlines()) == 18
    loaded_cache = load_demo_llm_cache(paths.artifacts / "demo_llm_cache.json")
    assert loaded_cache.available is True
    assert loaded_cache.count >= 2


def test_final_report_rejects_partial_and_mixed_trace(tmp_path: Path) -> None:
    partial_paths = _reporting_project(tmp_path / "partial")
    lines = partial_paths.llm_prediction_trace("test").read_text(encoding="utf-8").splitlines()
    partial_paths.llm_prediction_trace("test").write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8"
    )
    with pytest.raises(FinalReportValidationError, match="trace rows"):
        run_final_report(partial_paths, expected_test_rows=18)

    mixed_paths = _reporting_project(tmp_path / "mixed")
    mixed_lines = mixed_paths.llm_prediction_trace("test").read_text(encoding="utf-8").splitlines()
    first = json.loads(mixed_lines[0])
    first["model"] = "other-model"
    mixed_lines[0] = json.dumps(first)
    mixed_paths.llm_prediction_trace("test").write_text(
        "\n".join(mixed_lines) + "\n", encoding="utf-8"
    )
    with pytest.raises(FinalReportValidationError, match="mixed model"):
        run_final_report(mixed_paths, expected_test_rows=18)


def test_final_report_rejects_metric_mismatch(tmp_path: Path) -> None:
    paths = _reporting_project(tmp_path)
    metrics = json.loads(paths.llm_metrics("test").read_text(encoding="utf-8"))
    metrics["accuracy"] = 0.0
    paths.llm_metrics("test").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(FinalReportValidationError, match="Metric mismatch for accuracy"):
        run_final_report(paths, expected_test_rows=18)


def test_demo_selection_policy_is_deterministic(tmp_path: Path) -> None:
    paths = _reporting_project(tmp_path)
    run_final_report(paths, expected_test_rows=18)

    selection = json.loads((paths.artifacts / "demo_selection.json").read_text(encoding="utf-8"))
    examples = selection["examples"]
    assert examples[0]["key"] == "clear_agreement"
    assert examples[0]["actual_label"] == "cs.RO"
    assert examples[1]["key"] == "informative_disagreement"
    assert examples[1]["outcome"] == "llm_only_correct"
    assert {examples[1]["actual_label"], examples[1]["classical_label"]} == {"cs.AI", "cs.LG"}
    assert "aggregate performance" in selection["policy"]


def test_web_page_handles_missing_and_present_demo_selection(
    trained_project: ProjectPaths, tmp_path: Path
) -> None:
    missing_client = create_app(trained_project.root).test_client()
    missing_page = missing_client.get("/")
    assert missing_page.status_code == 200
    missing_html = missing_page.get_data(as_text=True)
    assert "Language preset" not in missing_html
    assert "Manual input is enabled" in missing_html

    paths = _reporting_project(tmp_path)
    run_final_report(paths, expected_test_rows=18)
    present_client = create_app(paths.root).test_client()
    present_page = present_client.get("/")
    assert present_page.status_code == 200
    present_html = present_page.get_data(as_text=True)
    assert "Clear agreement" in present_html
    assert "Language preset" not in present_html
