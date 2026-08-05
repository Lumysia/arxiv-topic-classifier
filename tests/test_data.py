from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arxiv_topic_classifier.config import TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.data import (
    _official_labels,
    _ReservoirSelection,
    parse_raw_paper,
    prepare_dataset,
    read_dataset,
)
from arxiv_topic_classifier.schema import RawPaper
from arxiv_topic_classifier.taxonomy import EXPECTED_TAXONOMY_SIZE, load_taxonomy


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


def _paper(label: str, index: int, categories: str | None = None) -> RawPaper:
    record = _record(label, index)
    if categories is not None:
        record["categories"] = categories
    return parse_raw_paper(json.dumps(record).encode())


def test_parse_raw_paper_boundary() -> None:
    paper = parse_raw_paper(json.dumps(_record("cs.AI", 1)).encode())
    assert paper.id.startswith("2501")
    assert paper.categories.startswith("cs.AI")
    with pytest.raises(ValidationError):
        parse_raw_paper(b'{"id": "missing-fields"}')


def test_official_labels_keep_source_order_and_deduplicate() -> None:
    assert _official_labels(["stat.ML", "cs.AI", "stat.ML", "old.XX", "cs.CL"]) == (
        "stat.ML",
        "cs.AI",
        "cs.CL",
    )


def test_reservoir_replacement_removes_unreferenced_row() -> None:
    selection = _ReservoirSelection(per_class=1, seed=1)
    first = _paper("cs.AI", 1)
    second = _paper("cs.AI", 2)

    selection.add_candidate(first, first.id, ("cs.AI",))
    selection.add_candidate(second, second.id, ("cs.AI",))

    assert selection.reservoirs["cs.AI"] == [second.id]
    assert selection.eligible_counts["cs.AI"] == 2
    assert set(selection.rows) == {second.id}


def test_reservoir_keeps_overlapping_row_referenced_by_another_label() -> None:
    selection = _ReservoirSelection(per_class=1, seed=1)
    overlap = _paper("cs.AI", 1, categories="cs.AI cs.CL")
    ai_only = _paper("cs.AI", 2)

    selection.add_candidate(overlap, overlap.id, ("cs.AI", "cs.CL"))
    selection.add_candidate(ai_only, ai_only.id, ("cs.AI",))

    assert selection.reservoirs["cs.AI"] == [ai_only.id]
    assert selection.reservoirs["cs.CL"] == [overlap.id]
    assert set(selection.rows) == {overlap.id, ai_only.id}


def test_reservoir_does_not_store_unselected_candidate() -> None:
    selection = _ReservoirSelection(per_class=1, seed=0)
    selected = _paper("cs.AI", 1)
    unselected = _paper("cs.AI", 2)

    selection.add_candidate(selected, selected.id, ("cs.AI",))
    selection.add_candidate(unselected, unselected.id, ("cs.AI",))

    assert selection.reservoirs["cs.AI"] == [selected.id]
    assert selection.eligible_counts["cs.AI"] == 2
    assert set(selection.rows) == {selected.id}


def test_official_taxonomy_snapshot_is_complete() -> None:
    taxonomy = load_taxonomy()
    assert len(taxonomy.entries) == EXPECTED_TAXONOMY_SIZE
    assert taxonomy.ids == TARGET_CATEGORIES
    assert set(taxonomy.names) == set(TARGET_CATEGORIES)
    assert taxonomy.names["cs.CL"] == "Computation and Language"


def test_prepare_is_balanced_and_leakage_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    records = [_record(label, index) for label in TARGET_CATEGORIES for index in range(14)]
    overlap = _record("cs.AI", 10_000)
    overlap["categories"] = "cs.AI cs.CL old.XX"
    records.append(overlap)
    duplicate = _record("cs.AI", 999, abstract=str(records[0]["abstract"]))
    duplicate["title"] = records[0]["title"]
    records.append(duplicate)
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path / "project")
    profile = prepare_dataset(source, paths, per_class=10)
    rows = read_dataset(paths.dataset)
    assert len(rows) >= len(TARGET_CATEGORIES)
    assert profile["records"]["duplicate_eligible_content"] == 1
    assert profile["sample"]["official_labels_with_zero_eligible_rows"] == []
    assert profile["sample"]["official_labels_with_zero_train_rows"] == []
    assert profile["legacy_category_counts"] == {"old.XX": 1}
    assert {row.split for row in rows} == {"train", "validation", "test"}
    assert all(row.text == row.abstract for row in rows)
    assert any(set(row.labels) == {"cs.AI", "cs.CL"} for row in rows)
    split_by_hash: dict[str, set[str]] = {}
    split_by_id: dict[str, set[str]] = {}
    for row in rows:
        split_by_hash.setdefault(row.content_hash, set()).add(row.split)
        split_by_id.setdefault(row.arxiv_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in split_by_hash.values())
    assert all(len(splits) == 1 for splits in split_by_id.values())


def test_prepare_per_label_cap_keeps_rare_labels(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    records = []
    for label in TARGET_CATEGORIES:
        count = 14 if label != "cs.CL" else 4
        records.extend(_record(label, index) for index in range(count))
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path / "project")

    profile = prepare_dataset(source, paths, per_class=10)
    rows = read_dataset(paths.dataset)

    assert profile["official_label_eligible_counts"]["cs.CL"] == 4
    assert profile["sample"]["label_counts"]["cs.CL"] == 4
    assert sum("cs.CL" in row.labels for row in rows) == 4
