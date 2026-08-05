from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from arxiv_topic_classifier.config import ProjectPaths
from arxiv_topic_classifier.files import sha256_file

PACKAGE_NAME = "arxiv-topic-classifier-artifacts.zip"
MANIFEST_NAME = "ARTIFACT_PACKAGE_MANIFEST.json"
SHA256_MANIFEST_NAME = "MANIFEST_SHA256.txt"

ROOT_FILES = ("README.md", "pyproject.toml", "uv.lock")
SOURCE_FILES = (
    "src/arxiv_topic_classifier/__init__.py",
    "src/arxiv_topic_classifier/config.py",
    "src/arxiv_topic_classifier/data.py",
    "src/arxiv_topic_classifier/eda.py",
    "src/arxiv_topic_classifier/environment.py",
    "src/arxiv_topic_classifier/files.py",
    "src/arxiv_topic_classifier/model.py",
    "src/arxiv_topic_classifier/reporting.py",
    "src/arxiv_topic_classifier/schema.py",
    "src/arxiv_topic_classifier/taxonomy.py",
    "src/arxiv_topic_classifier/taxonomy.json",
    "src/arxiv_topic_classifier/text.py",
)
REQUIRED_ARTIFACTS = (
    "models/arxiv_topic_pipeline.joblib",
    "models/arxiv_topic_pipeline.manifest.json",
    "artifacts/environment.json",
    "artifacts/verification_report.json",
    "artifacts/processed_manifest.json",
    "artifacts/data_profile.json",
    "artifacts/metrics.json",
    "artifacts/class_metrics.csv",
    "artifacts/evaluation_report.md",
    "artifacts/source/source_metadata.json",
    "src/arxiv_topic_classifier/taxonomy.json",
)
OPTIONAL_ARTIFACTS = (
    "artifacts/eda_summary.json",
    "artifacts/prediction_examples.json",
    "artifacts/error_patterns.json",
    "artifacts/figures/abstract_length_distributions.png",
    "artifacts/figures/class_distribution.png",
    "artifacts/figures/label_f1.png",
    "artifacts/figures/model_comparison.png",
    "artifacts/figures/top_tfidf_logistic_terms.png",
    "artifacts/figures/year_distribution.png",
    "artifacts/source/arxiv-metadata-oai-snapshot.json.sha256",
)
EXCLUDED_PREFIXES = (
    "data/raw/",
    "artifacts/cache/",
    "artifacts/logs/",
    ".venv/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "slides/",
)
EXCLUDED_NAMES = (
    "demo_llm_cache.json",
    "llm_",
    "FINAL_RESULTS.md",
    "Q_AND_A_GUIDE.md",
    "rubric-checklist.md",
    "classical_vs_llm",
    "confusion_matrix",
)


class ArtifactPackageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PayloadFile:
    source: Path
    archive_name: str


def _existing_payloads(paths: ProjectPaths, include_processed_csv: bool) -> list[PayloadFile]:
    relatives = [*ROOT_FILES, *SOURCE_FILES, *REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS]
    if include_processed_csv:
        relatives.append("data/processed/arxiv_topics.csv")
    payloads: dict[str, PayloadFile] = {}
    missing_required: list[str] = []
    for relative in relatives:
        source = paths.root / relative
        if source.is_file():
            payloads[relative] = PayloadFile(source=source, archive_name=relative)
        elif relative in REQUIRED_ARTIFACTS or relative in ROOT_FILES:
            missing_required.append(relative)
    if missing_required:
        raise ArtifactPackageError(
            f"Required artifact-package files are missing: {missing_required}"
        )
    return [payloads[name] for name in sorted(payloads)]


def _assert_allowed_archive_name(name: str) -> None:
    if name.startswith(EXCLUDED_PREFIXES):
        raise ArtifactPackageError(f"Forbidden path selected for artifact package: {name}")
    if any(token in name for token in EXCLUDED_NAMES):
        raise ArtifactPackageError(
            f"Obsolete or unrelated artifact selected for artifact package: {name}"
        )
    if name.endswith((".pyc", ".pyo")) or "__pycache__" in name:
        raise ArtifactPackageError(f"Cache file selected for artifact package: {name}")


def _manifest(payloads: list[PayloadFile], processed_manifest: dict[str, Any]) -> dict[str, Any]:
    files = [
        {
            "path": payload.archive_name,
            "bytes": payload.source.stat().st_size,
            "sha256": sha256_file(payload.source),
        }
        for payload in payloads
    ]
    files.append({"path": MANIFEST_NAME, "bytes": None, "sha256": None})
    files.append({"path": SHA256_MANIFEST_NAME, "bytes": None, "sha256": None})
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "package": PACKAGE_NAME,
        "processed_manifest": processed_manifest,
        "files": files,
        "exclusions": {
            "raw_source_excluded": True,
            "processed_csv_included": any(
                payload.archive_name == "data/processed/arxiv_topics.csv" for payload in payloads
            ),
            "slides_excluded": True,
            "llm_artifacts_excluded": True,
            "caches_logs_secrets_excluded": True,
        },
    }


def _sha256_manifest(payloads: list[PayloadFile]) -> str:
    lines = [f"{sha256_file(payload.source)}  {payload.archive_name}" for payload in payloads]
    return "\n".join(lines) + "\n"


def create_artifact_package(
    paths: ProjectPaths,
    *,
    processed_manifest: dict[str, Any],
    verification_report: dict[str, Any],
    include_processed_csv: bool = False,
) -> Path:
    del verification_report
    paths.create_output_dirs()
    payloads = _existing_payloads(paths, include_processed_csv)
    for payload in payloads:
        _assert_allowed_archive_name(payload.archive_name)
    manifest = _manifest(payloads, processed_manifest)
    destination = paths.returns / PACKAGE_NAME
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for payload in payloads:
                archive.write(payload.source, payload.archive_name)
            archive.writestr(SHA256_MANIFEST_NAME, _sha256_manifest(payloads))
            archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify_artifact_zip(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def verify_artifact_zip(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ArtifactPackageError(f"ZIP CRC verification failed for {corrupt}.")
        names = sorted(archive.namelist())
        for name in names:
            _assert_allowed_archive_name(name)
        if MANIFEST_NAME not in names:
            raise ArtifactPackageError("Artifact package manifest is missing.")
        if SHA256_MANIFEST_NAME not in names:
            raise ArtifactPackageError("Artifact package SHA-256 manifest is missing.")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        entries = {item["path"]: item for item in manifest["files"]}
        if set(entries) != set(names):
            raise ArtifactPackageError("Artifact package manifest does not cover every ZIP member.")
        for name in names:
            if name in {MANIFEST_NAME, SHA256_MANIFEST_NAME}:
                continue
            payload = archive.read(name)
            digest = __import__("hashlib").sha256(payload).hexdigest()
            if entries[name]["sha256"] != digest or entries[name]["bytes"] != len(payload):
                raise ArtifactPackageError(f"Artifact package manifest hash mismatch for {name}.")
        with tempfile.TemporaryDirectory() as temporary_dir:
            extract_root = Path(temporary_dir)
            archive.extractall(extract_root)
            expected_sha_entries = set(names) - {MANIFEST_NAME, SHA256_MANIFEST_NAME}
            actual_sha_entries: dict[str, str] = {}
            sha_manifest_text = (extract_root / SHA256_MANIFEST_NAME).read_text(
                encoding="utf-8"
            )
            for line in sha_manifest_text.splitlines():
                digest, relative = line.split("  ", maxsplit=1)
                actual_sha_entries[relative] = digest
            if set(actual_sha_entries) != expected_sha_entries:
                raise ArtifactPackageError("SHA-256 manifest does not cover every payload file.")
            for relative, expected_digest in actual_sha_entries.items():
                if sha256_file(extract_root / relative) != expected_digest:
                    raise ArtifactPackageError(f"SHA-256 manifest hash mismatch for {relative}.")
            model_path = extract_root / "models" / "arxiv_topic_pipeline.joblib"
            artifact = joblib.load(model_path)
            if (
                not isinstance(artifact, dict)
                or artifact.get("task") != "official_arxiv_taxonomy_multilabel"
            ):
                raise ArtifactPackageError("Extracted model does not load as the expected task.")
            shutil.rmtree(extract_root, ignore_errors=True)
    return {
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "member_count": len(names),
        "crc": "passed",
    }
