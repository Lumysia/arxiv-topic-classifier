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


def _primary_category(paper: RawPaper) -> str:
    categories = paper.categories.split()
    return categories[0] if categories else ""


def _reservoir_add(
    reservoirs: dict[str, list[RawPaper]],
    eligible_counts: Counter[str],
    paper: RawPaper,
    label: str,
    per_class: int,
    rng: random.Random,
) -> None:
    eligible_counts[label] += 1
    reservoir = reservoirs[label]
    if len(reservoir) < per_class:
        reservoir.append(paper)
        return
    replacement_index = rng.randrange(eligible_counts[label])
    if replacement_index < per_class:
        reservoir[replacement_index] = paper


def _assign_splits(papers: list[RawPaper], label: str, seed: int) -> list[DatasetRow]:
    ordered = sorted(papers, key=lambda paper: _stable_u64(f"{seed}:{paper.id}"))
    count = len(ordered)
    train_end = round(count * _SPLIT_RATIOS[0][1])
    validation_end = train_end + round(count * _SPLIT_RATIOS[1][1])
    rows: list[DatasetRow] = []
    for index, paper in enumerate(ordered):
        split: Split
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        rows.append(
            DatasetRow(
                arxiv_id=base_arxiv_id(paper.id),
                title=normalize_text(paper.title),
                abstract=normalize_text(paper.abstract),
                text=document_text(paper.title, paper.abstract),
                label=label,
                categories=normalize_text(paper.categories),
                year=_paper_year(paper),
                split=split,
                content_hash=content_fingerprint(paper.abstract),
            )
        )
    return rows


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
        for row in sorted(rows, key=lambda item: (item.split, item.label, item.arxiv_id)):
            writer.writerow(row.model_dump())
    temporary.replace(destination)


def read_dataset(path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            raw_row["year"] = raw_row["year"] or None
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
    eligible_counts: Counter[str] = Counter()
    reservoirs: dict[str, list[RawPaper]] = {label: [] for label in TARGET_CATEGORIES}
    seen_ids: set[int] = set()
    seen_target_content: set[str] = set()
    source_digest = hashlib.sha256()
    rng = random.Random(seed)
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
            if primary not in reservoirs:
                continue
            counters["target_primary_records"] += 1
            if not normalize_text(paper.abstract):
                counters["target_missing_text"] += 1
                continue
            fingerprint = content_fingerprint(paper.abstract)
            if fingerprint in seen_target_content:
                counters["duplicate_target_content"] += 1
                continue
            seen_target_content.add(fingerprint)
            _reservoir_add(reservoirs, eligible_counts, paper, primary, per_class, rng)

    source_size_end = source.stat().st_size
    if source_size_start != source_size_end and not allow_growing_source:
        raise SourceChangedError(
            f"Source changed during scan ({source_size_start} to {source_size_end} bytes). "
            "Wait for the download to finish or pass --allow-growing-source."
        )
    insufficient = {
        label: eligible_counts[label]
        for label in TARGET_CATEGORIES
        if eligible_counts[label] < per_class
    }
    if insufficient:
        raise InsufficientDataError(f"Insufficient eligible records: {insufficient}")

    rows = [
        row for label in TARGET_CATEGORIES for row in _assign_splits(reservoirs[label], label, seed)
    ]
    _assert_leakage_safe(rows)
    write_dataset(rows, paths.dataset)
    split_counts = Counter(row.split for row in rows)
    sampled_class_counts = Counter(row.label for row in rows)
    sampled_years = [row.year for row in rows if row.year is not None]
    profile: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scan_started_utc": scan_started.isoformat(),
        "source": {
            "dataset": "Cornell University arXiv metadata on Kaggle",
            "kaggle_slug": "Cornell-University/arxiv",
            "license": "CC0: Public Domain",
            "path": str(source),
            "bytes_at_scan_start": source_size_start,
            "bytes_at_scan_end": source_size_end,
            "scanned_bytes": scanned_bytes,
            "sha256_of_scanned_bytes": source_digest.hexdigest(),
            "changed_during_scan": source_size_start != source_size_end,
        },
        "records": dict(sorted(counters.items())),
        "broad_primary_archive_counts": dict(broad_archives.most_common()),
        "top_primary_category_counts": dict(primary_categories.most_common(30)),
        "target_eligible_counts": dict(eligible_counts),
        "category_design": {
            "formulation": "single-label six-class classification",
            "label_rule": (
                "The first source category is the target; other categories are retained only "
                "for audit."
            ),
            "categories": list(TARGET_CATEGORIES),
            "reason": (
                "Six related computer-science topics create a bounded, interpretable task with "
                "meaningful semantic overlap. Reservoir sampling balances class size, while source "
                "primary labels preserve a reproducible target without inventing labels."
            ),
        },
        "sample": {
            "seed": seed,
            "per_class": per_class,
            "total_rows": len(rows),
            "class_counts": dict(sampled_class_counts),
            "split_counts": dict(split_counts),
            "year_min": min(sampled_years),
            "year_max": max(sampled_years),
            "duplicate_ids_across_splits": 0,
            "duplicate_content_across_splits": 0,
        },
    }
    paths.profile.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return profile


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
