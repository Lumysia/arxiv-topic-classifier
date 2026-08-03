from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arxiv_topic_classifier.config import CATEGORY_DESCRIPTIONS, ProjectPaths
from arxiv_topic_classifier.data import parse_raw_paper, prepare_dataset, read_dataset


def _record(label: str, index: int, abstract: str | None = None) -> dict[str, object]:
    return {
        "id": f"2501.{label.replace('.', '')}{index:04d}",
        "title": f"A {label} research title {index}",
        "abstract": abstract
        or (
            f"This is a sufficiently long abstract about {label} methods, experiments, analysis, "
            f"results and limitations for synthetic parser fixture number {index}."
        ),
        "categories": f"{label} stat.ML" if index % 2 else label,
        "versions": [{"version": "v1", "created": "Mon, 1 Jan 2025 00:00:00 GMT"}],
        "update_date": "2025-01-01",
    }


def test_parse_raw_paper_boundary() -> None:
    paper = parse_raw_paper(json.dumps(_record("cs.AI", 1)).encode())
    assert paper.id.startswith("2501")
    assert paper.categories.startswith("cs.AI")
    with pytest.raises(ValidationError):
        parse_raw_paper(b'{"id": "missing-fields"}')


def test_prepare_is_balanced_and_leakage_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    records = [_record(label, index) for label in CATEGORY_DESCRIPTIONS for index in range(14)]
    duplicate = _record("cs.AI", 999, abstract=str(records[0]["abstract"]))
    duplicate["title"] = records[0]["title"]
    records.append(duplicate)
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path / "project")
    profile = prepare_dataset(source, paths, per_class=10)
    rows = read_dataset(paths.dataset)
    assert len(rows) == len(CATEGORY_DESCRIPTIONS) * 10
    assert profile["records"]["duplicate_target_content"] == 1
    assert {row.split for row in rows} == {"train", "validation", "test"}
    assert all(row.text == row.abstract for row in rows)
    split_by_hash: dict[str, set[str]] = {}
    split_by_id: dict[str, set[str]] = {}
    for row in rows:
        split_by_hash.setdefault(row.content_hash, set()).add(row.split)
        split_by_id.setdefault(row.arxiv_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in split_by_hash.values())
    assert all(len(splits) == 1 for splits in split_by_id.values())
