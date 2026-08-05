from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

EXPECTED_TAXONOMY_SIZE = 155
TAXONOMY_RESOURCE = "taxonomy.json"


class TaxonomyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TaxonomyEntry:
    id: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Taxonomy:
    entries: tuple[TaxonomyEntry, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    @property
    def names(self) -> dict[str, str]:
        return {entry.id: entry.name for entry in self.entries}

    @property
    def descriptions(self) -> dict[str, str]:
        return {entry.id: entry.description for entry in self.entries}

    def ordered_intersection(self, category_ids: list[str]) -> tuple[str, ...]:
        present = set(category_ids)
        return tuple(label for label in self.ids if label in present)

    def records(self) -> list[dict[str, str]]:
        return [
            {"id": entry.id, "name": entry.name, "description": entry.description}
            for entry in self.entries
        ]


def _entry_from_payload(index: int, payload: object) -> TaxonomyEntry:
    if not isinstance(payload, dict):
        raise TaxonomyError(f"Taxonomy entry {index} is not an object.")
    raw = dict[str, Any](payload)
    identifier = raw.get("id")
    name = raw.get("name")
    description = raw.get("description")
    if not isinstance(identifier, str) or not identifier.strip():
        raise TaxonomyError(f"Taxonomy entry {index} has an invalid id.")
    if not isinstance(name, str) or not name.strip():
        raise TaxonomyError(f"Taxonomy entry {identifier} has an invalid name.")
    if not isinstance(description, str) or not description.strip():
        raise TaxonomyError(f"Taxonomy entry {identifier} has an invalid description.")
    return TaxonomyEntry(id=identifier, name=name, description=description)


def _validate(entries: tuple[TaxonomyEntry, ...]) -> Taxonomy:
    if len(entries) != EXPECTED_TAXONOMY_SIZE:
        raise TaxonomyError(
            f"Expected {EXPECTED_TAXONOMY_SIZE} official taxonomy entries, found {len(entries)}."
        )
    ids = [entry.id for entry in entries]
    duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
    if duplicates:
        raise TaxonomyError(f"Duplicate taxonomy IDs: {duplicates[:5]}")
    return Taxonomy(entries=entries)


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    resource = resources.files("arxiv_topic_classifier").joinpath(TAXONOMY_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TaxonomyError("Taxonomy resource must contain a JSON array.")
    entries = tuple(_entry_from_payload(index, item) for index, item in enumerate(payload))
    return _validate(entries)


def taxonomy_fingerprint() -> str:
    payload = json.dumps(load_taxonomy().records(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
