from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from arxiv_topic_classifier.artifact_package import create_artifact_package
from arxiv_topic_classifier.config import RANDOM_SEED, TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.data import prepare_dataset
from arxiv_topic_classifier.environment import dependency_versions
from arxiv_topic_classifier.files import atomic_write_json, sha256_file
from arxiv_topic_classifier.model import train_and_evaluate
from arxiv_topic_classifier.source import (
    DEFAULT_SOURCE_SPEC,
    SourceSpec,
    SourceStatus,
    acquire_source,
    use_local_source,
)
from arxiv_topic_classifier.taxonomy import EXPECTED_TAXONOMY_SIZE, taxonomy_fingerprint
from arxiv_topic_classifier.verification import VerificationError, load_json, verify_model

PROCESSED_MANIFEST_FORMAT_VERSION = 1


class ResumeCheckError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    source_reused: bool
    prepared: bool
    trained: bool
    artifact_zip: Path

    def summary(self) -> dict[str, Any]:
        return {
            "source_reused": self.source_reused,
            "prepared": self.prepared,
            "trained": self.trained,
            "artifact_zip": str(self.artifact_zip),
        }


def processed_manifest_path(paths: ProjectPaths) -> Path:
    return paths.artifacts / "processed_manifest.json"


def _profile_matches(
    paths: ProjectPaths, source: SourceStatus, per_class: int, seed: int
) -> dict[str, Any]:
    if not paths.dataset.is_file() or not paths.profile.is_file():
        raise ResumeCheckError("Processed dataset/profile is absent.")
    profile = load_json(paths.profile)
    profile_source = cast(dict[str, Any], profile.get("source", {}))
    category_design = cast(dict[str, Any], profile.get("category_design", {}))
    sample = cast(dict[str, Any], profile.get("sample", {}))
    if profile_source.get("sha256_of_scanned_bytes") != source.sha256:
        raise ResumeCheckError("Data profile source SHA-256 does not match the pinned source.")
    if profile_source.get("bytes_at_scan_end") != source.bytes:
        raise ResumeCheckError("Data profile source size does not match the pinned source.")
    if category_design.get("taxonomy_sha256") != taxonomy_fingerprint():
        raise ResumeCheckError("Data profile taxonomy fingerprint does not match.")
    if category_design.get("categories") != list(TARGET_CATEGORIES):
        raise ResumeCheckError("Data profile categories do not match the official taxonomy.")
    if sample.get("per_label_cap") != per_class or sample.get("seed") != seed:
        raise ResumeCheckError("Data profile cap or seed does not match this run.")
    return profile


def compute_processed_manifest(
    paths: ProjectPaths, source: SourceStatus, per_class: int, seed: int
) -> dict[str, Any]:
    profile = _profile_matches(paths, source, per_class, seed)
    return {
        "format_version": PROCESSED_MANIFEST_FORMAT_VERSION,
        "task": "official_arxiv_taxonomy_multilabel",
        "source_sha256": source.sha256,
        "source_bytes": source.bytes,
        "taxonomy_sha256": taxonomy_fingerprint(),
        "labels": list(TARGET_CATEGORIES),
        "label_count": EXPECTED_TAXONOMY_SIZE,
        "per_label_cap": per_class,
        "seed": seed,
        "dataset_path": paths.relative_path(paths.dataset),
        "dataset_sha256": sha256_file(paths.dataset),
        "profile_path": paths.relative_path(paths.profile),
        "profile_sha256": sha256_file(paths.profile),
        "row_count": cast(dict[str, Any], profile["sample"])["total_rows"],
    }


def write_processed_manifest(
    paths: ProjectPaths, source: SourceStatus, per_class: int, seed: int
) -> dict[str, Any]:
    manifest = compute_processed_manifest(paths, source, per_class, seed)
    atomic_write_json(processed_manifest_path(paths), manifest)
    return manifest


def processed_dataset_is_current(
    paths: ProjectPaths, source: SourceStatus, per_class: int, seed: int
) -> dict[str, Any]:
    try:
        existing = load_json(processed_manifest_path(paths))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ResumeCheckError("Processed manifest is absent or invalid.") from error
    current = compute_processed_manifest(paths, source, per_class, seed)
    if existing != current:
        raise ResumeCheckError("Processed dataset/profile does not match the existing manifest.")
    return current


def model_is_current(paths: ProjectPaths, processed_manifest: dict[str, Any]) -> bool:
    try:
        manifest = load_json(paths.model_manifest)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if manifest.get("task") != "official_arxiv_taxonomy_multilabel":
        return False
    if manifest.get("labels") != list(TARGET_CATEGORIES):
        return False
    if manifest.get("label_count") != EXPECTED_TAXONOMY_SIZE:
        return False
    if manifest.get("dependency_versions") != dependency_versions():
        return False
    if manifest.get("processed_manifest") != processed_manifest:
        return False
    if not paths.model.is_file() or not (paths.artifacts / "metrics.json").is_file():
        return False
    if manifest.get("model_sha256") != sha256_file(paths.model):
        return False
    if manifest.get("metrics_sha256") != sha256_file(paths.artifacts / "metrics.json"):
        return False
    try:
        verify_model(paths, processed_manifest)
    except VerificationError:
        return False
    return True


def run_full_train_package(
    paths: ProjectPaths,
    *,
    per_class: int = 1_000,
    seed: int = RANDOM_SEED,
    source: Path | None = None,
    source_spec: SourceSpec | None = None,
    force_download: bool = False,
    force_prepare: bool = False,
    force_train: bool = False,
    include_processed_csv: bool = False,
) -> WorkflowResult:
    paths.create_output_dirs()
    source_spec = DEFAULT_SOURCE_SPEC if source_spec is None else source_spec
    source_status = (
        use_local_source(paths, source, source_spec)
        if source is not None
        else acquire_source(paths, source_spec, force_download=force_download)
    )

    prepared = False
    if force_prepare:
        prepare_dataset(source_status.path, paths, per_class, seed)
        prepared = True
        processed_manifest = write_processed_manifest(paths, source_status, per_class, seed)
    else:
        try:
            processed_manifest = processed_dataset_is_current(paths, source_status, per_class, seed)
        except ResumeCheckError:
            prepare_dataset(source_status.path, paths, per_class, seed)
            prepared = True
            processed_manifest = write_processed_manifest(paths, source_status, per_class, seed)

    trained = False
    if force_train or not model_is_current(paths, processed_manifest):
        train_and_evaluate(paths, seed, processed_manifest=processed_manifest)
        trained = True
    verification = verify_model(paths, processed_manifest)
    zip_path = create_artifact_package(
        paths,
        processed_manifest=processed_manifest,
        verification_report=verification,
        include_processed_csv=include_processed_csv,
    )
    return WorkflowResult(
        source_reused=source_status.reused_existing and not source_status.downloaded,
        prepared=prepared,
        trained=trained,
        artifact_zip=zip_path,
    )
