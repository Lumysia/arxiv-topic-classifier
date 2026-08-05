from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from arxiv_topic_classifier.config import RANDOM_SEED, TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.schema import DatasetRow, RawPaper, Split
from arxiv_topic_classifier.taxonomy import taxonomy_fingerprint
from arxiv_topic_classifier.text import (
    base_arxiv_id,
    content_fingerprint,
    document_text,
    normalize_text,
)

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_SPLIT_RATIOS: tuple[tuple[Split, float], ...] = (
    ("train", 0.70),
    ("validation", 0.15),
    ("test", 0.15),
)
_DATASET_FIELDS = tuple(DatasetRow.model_fields)
_TARGET_SET = frozenset(TARGET_CATEGORIES)


class SourceChangedError(RuntimeError):
    pass


class InsufficientDataError(RuntimeError):
    pass


def parse_raw_paper(raw_line: bytes) -> RawPaper:
    return RawPaper.model_validate_json(raw_line)


def _stable_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest)


def _paper_year(paper: RawPaper) -> int | None:
    if paper.versions:
        matches = _YEAR_PATTERN.findall(paper.versions[0].created)
        if matches:
            full_match = re.search(r"\b(?:19|20)\d{2}\b", paper.versions[0].created)
            if full_match is not None:
                return int(full_match.group())
    if paper.update_date and len(paper.update_date) >= 4:
        try:
            return int(paper.update_date[:4])
        except ValueError:
            return None
    return None


def _official_labels(categories: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    labels: list[str] = []
    for category in categories:
        if category in _TARGET_SET and category not in seen:
            labels.append(category)
            seen.add(category)
    return tuple(labels)


def _row_from_paper(paper: RawPaper, labels: tuple[str, ...]) -> DatasetRow:
    abstract = normalize_text(paper.abstract)
    return DatasetRow(
        arxiv_id=base_arxiv_id(paper.id),
        title=normalize_text(paper.title),
        abstract=abstract,
        text=document_text(paper.title, paper.abstract),
        labels=labels,
        categories=normalize_text(paper.categories),
        year=_paper_year(paper),
        split="train",
        content_hash=content_fingerprint(paper.abstract),
    )


class _ReservoirSelection:
    def __init__(self, per_class: int, seed: int) -> None:
        self._per_class = per_class
        self._rng = random.Random(seed)
        self.reservoirs: dict[str, list[str]] = {label: [] for label in TARGET_CATEGORIES}
        self.eligible_counts: Counter[str] = Counter()
        self.rows: dict[str, DatasetRow] = {}
        self._reference_counts: Counter[str] = Counter()

    def add_candidate(self, paper: RawPaper, row_key: str, labels: tuple[str, ...]) -> None:
        selected_count = 0
        for label in labels:
            selected, replaced_key = self._add_reservoir_key(row_key, label)
            if replaced_key is not None:
                self._remove_reference(replaced_key)
            if selected:
                self._reference_counts[row_key] += 1
                selected_count += 1
        if selected_count > 0 and row_key not in self.rows:
            self.rows[row_key] = _row_from_paper(paper, labels)

    def _add_reservoir_key(self, row_key: str, label: str) -> tuple[bool, str | None]:
        self.eligible_counts[label] += 1
        reservoir = self.reservoirs[label]
        if len(reservoir) < self._per_class:
            reservoir.append(row_key)
            return True, None
        replacement_index = self._rng.randrange(self.eligible_counts[label])
        if replacement_index < self._per_class:
            replaced_key = reservoir[replacement_index]
            reservoir[replacement_index] = row_key
            return True, replaced_key
        return False, None

    def _remove_reference(self, row_key: str) -> None:
        self._reference_counts[row_key] -= 1
        if self._reference_counts[row_key] <= 0:
            del self._reference_counts[row_key]
            self.rows.pop(row_key, None)


def _assign_splits(rows: list[DatasetRow], seed: int) -> list[DatasetRow]:
    ordered = sorted(rows, key=lambda row: _stable_u64(f"{seed}:{row.arxiv_id}"))
    count = len(ordered)
    train_end = round(count * _SPLIT_RATIOS[0][1])
    validation_end = train_end + round(count * _SPLIT_RATIOS[1][1])
    assigned: list[DatasetRow] = []
    for index, row in enumerate(ordered):
        split: Split
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        assigned.append(row.model_copy(update={"split": split}))
    return assigned


def _assert_leakage_safe(rows: list[DatasetRow]) -> None:
    id_splits: dict[str, set[Split]] = {}
    content_splits: dict[str, set[Split]] = {}
    for row in rows:
        id_splits.setdefault(row.arxiv_id, set()).add(row.split)
        content_splits.setdefault(row.content_hash, set()).add(row.split)
    leaking_ids = [key for key, splits in id_splits.items() if len(splits) > 1]
    leaking_content = [key for key, splits in content_splits.items() if len(splits) > 1]
    if leaking_ids or leaking_content:
        raise RuntimeError(
            f"Leakage check failed: {len(leaking_ids)} IDs and "
            f"{len(leaking_content)} contents cross splits."
        )


def write_dataset(rows: list[DatasetRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_DATASET_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.split, item.arxiv_id)):
            payload = row.model_dump(mode="json")
            payload["labels"] = json.dumps(payload["labels"], separators=(",", ":"))
            writer.writerow(payload)
    temporary.replace(destination)


def read_dataset(path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            raw_row["year"] = raw_row["year"] or None
            raw_labels = raw_row.get("labels")
            if raw_labels is None:
                raise ValueError("Dataset is missing multilabel 'labels' column.")
            labels = json.loads(raw_labels)
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise ValueError("Dataset labels column must contain a JSON string array.")
            raw_row["labels"] = labels
            rows.append(DatasetRow.model_validate(raw_row))
    return rows


def dataset_split(rows: list[DatasetRow], split: Split) -> list[DatasetRow]:
    return [row for row in rows if row.split == split]


def prepare_dataset(
    source: Path,
    paths: ProjectPaths,
    per_class: int,
    seed: int = RANDOM_SEED,
    allow_growing_source: bool = False,
) -> dict[str, Any]:
    if per_class < 10:
        raise ValueError("per_class must be at least 10.")
    if not source.is_file():
        raise FileNotFoundError(f"Source metadata not found: {source}")
    paths.create_output_dirs()
    source_size_start = source.stat().st_size
    scan_started = datetime.now(UTC)
    counters: Counter[str] = Counter()
    broad_archives: Counter[str] = Counter()
    primary_categories: Counter[str] = Counter()
    complete_categories: Counter[str] = Counter()
    legacy_categories: Counter[str] = Counter()
    selection = _ReservoirSelection(per_class=per_class, seed=seed)
    seen_ids: set[int] = set()
    seen_eligible_content: set[str] = set()
    source_digest = hashlib.sha256()
    scanned_bytes = 0

    with source.open("rb") as handle:
        for raw_line in handle:
            scanned_bytes += len(raw_line)
            source_digest.update(raw_line)
            counters["physical_lines"] += 1
            try:
                paper = parse_raw_paper(raw_line)
            except (ValidationError, ValueError):
                counters["malformed_or_incomplete_lines"] += 1
                continue
            counters["valid_records"] += 1
            normalized_id = base_arxiv_id(paper.id)
            id_hash = _stable_u64(normalized_id)
            if id_hash in seen_ids:
                counters["duplicate_base_ids"] += 1
                continue
            seen_ids.add(id_hash)
            categories = paper.categories.split()
            if not normalize_text(paper.title):
                counters["missing_title"] += 1
            if not normalize_text(paper.abstract):
                counters["missing_abstract"] += 1
            if len(categories) > 1:
                counters["multi_category_records"] += 1
            if len(paper.versions) > 1:
                counters["records_with_multiple_versions"] += 1
            counters["total_embedded_versions"] += len(paper.versions)
            if not categories:
                counters["missing_categories"] += 1
                continue
            primary = categories[0]
            primary_categories[primary] += 1
            broad_archives[primary.split(".", maxsplit=1)[0]] += 1
            for category in categories:
                complete_categories[category] += 1
                if category not in _TARGET_SET:
                    legacy_categories[category] += 1
            labels = _official_labels(categories)
            if not labels:
                continue
            counters["target_complete_category_records"] += 1
            if not normalize_text(paper.abstract):
                counters["target_missing_text"] += 1
                continue
            fingerprint = content_fingerprint(paper.abstract)
            if fingerprint in seen_eligible_content:
                counters["duplicate_eligible_content"] += 1
                continue
            seen_eligible_content.add(fingerprint)
            if len(labels) > 1:
                counters["multilabel_eligible_records"] += 1
            selection.add_candidate(paper, normalized_id, labels)

    source_size_end = source.stat().st_size
    if source_size_start != source_size_end and not allow_growing_source:
        raise SourceChangedError(
            f"Source changed during scan ({source_size_start} to {source_size_end} bytes). "
            "Wait for the download to finish or pass --allow-growing-source."
        )
    zero_eligible = [label for label in TARGET_CATEGORIES if selection.eligible_counts[label] == 0]
    if zero_eligible:
        raise InsufficientDataError(f"Official labels with zero eligible records: {zero_eligible}")

    rows = _assign_splits(list(selection.rows.values()), seed)
    _assert_leakage_safe(rows)
    split_counts = Counter(row.split for row in rows)
    sampled_label_counts: Counter[str] = Counter(label for row in rows for label in row.labels)
    train_label_counts: Counter[str] = Counter(
        label for row in rows if row.split == "train" for label in row.labels
    )
    zero_train = [label for label in TARGET_CATEGORIES if train_label_counts[label] == 0]
    if zero_train:
        raise InsufficientDataError(f"Official labels with zero train rows: {zero_train}")
    write_dataset(rows, paths.dataset)
    sampled_years = [row.year for row in rows if row.year is not None]
    primary_zero_official = [label for label in TARGET_CATEGORIES if primary_categories[label] == 0]
    profile: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scan_started_utc": scan_started.isoformat(),
        "source": {
            "dataset": "Cornell University arXiv metadata on Kaggle",
            "kaggle_slug": "Cornell-University/arxiv",
            "license": "CC0: Public Domain",
            "path": paths.artifact_path(source),
            "bytes_at_scan_start": source_size_start,
            "bytes_at_scan_end": source_size_end,
            "scanned_bytes": scanned_bytes,
            "sha256_of_scanned_bytes": source_digest.hexdigest(),
            "changed_during_scan": source_size_start != source_size_end,
        },
        "records": dict(sorted(counters.items())),
        "broad_primary_archive_counts": dict(broad_archives.most_common()),
        "top_primary_category_counts": dict(primary_categories.most_common(30)),
        "top_complete_category_counts": dict(complete_categories.most_common(30)),
        "legacy_category_counts": dict(legacy_categories.most_common(50)),
        "official_label_eligible_counts": {
            label: selection.eligible_counts[label] for label in TARGET_CATEGORIES
        },
        "official_primary_zero_labels": primary_zero_official,
        "category_design": {
            "formulation": "multilabel classification over the official arXiv taxonomy",
            "taxonomy_sha256": taxonomy_fingerprint(),
            "label_rule": (
                "A row's labels are the intersection of its complete source categories field with "
                "the 155 official taxonomy IDs. Legacy IDs outside the current taxonomy are "
                "profiled and excluded."
            ),
            "categories": list(TARGET_CATEGORIES),
            "official_label_count": len(TARGET_CATEGORIES),
        },
        "sample": {
            "seed": seed,
            "per_label_cap": per_class,
            "total_rows": len(rows),
            "label_counts": {label: sampled_label_counts[label] for label in TARGET_CATEGORIES},
            "train_label_counts": {label: train_label_counts[label] for label in TARGET_CATEGORIES},
            "labels_with_full_cap": sum(
                1 for label in TARGET_CATEGORIES if len(selection.reservoirs[label]) == per_class
            ),
            "split_counts": dict(split_counts),
            "year_min": min(sampled_years) if sampled_years else None,
            "year_max": max(sampled_years) if sampled_years else None,
            "duplicate_ids_across_splits": 0,
            "duplicate_content_across_splits": 0,
            "official_labels_with_zero_eligible_rows": zero_eligible,
            "official_labels_with_zero_train_rows": zero_train,
        },
    }
    temporary_profile = paths.profile.with_suffix(paths.profile.suffix + ".tmp")
    temporary_profile.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary_profile.replace(paths.profile)
    return profile


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
