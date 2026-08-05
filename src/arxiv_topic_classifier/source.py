from __future__ import annotations

import json
import os
import re
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import ValidationError

from arxiv_topic_classifier.config import ProjectPaths
from arxiv_topic_classifier.files import atomic_write_json, atomic_write_text, sha256_file
from arxiv_topic_classifier.schema import RawPaper

KAGGLE_DATASET_BASE = "Cornell-University/arxiv"
KAGGLE_DATASET = f"{KAGGLE_DATASET_BASE}/versions/297"
KAGGLE_FILE = "arxiv-metadata-oai-snapshot.json"
EXPECTED_SOURCE_BYTES = 5_437_318_298
EXPECTED_SOURCE_SHA256 = "fabd00ed721608c57d0e3b9b9cf798062328771c6e6cfcc08cb63d9eb434d43f"
SOURCE_METADATA_VERSION = 1

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(token|key|secret|password)\s*[:=]\s*[^\s,;]+"),
)


class SourceDownloadError(RuntimeError):
    pass


class SourceValidationError(RuntimeError):
    pass


class KaggleDownloader(Protocol):
    def dataset_download(
        self,
        handle: str,
        *,
        path: str | None = None,
        force_download: bool = False,
        output_dir: str | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    kaggle_dataset: str = KAGGLE_DATASET
    kaggle_base_slug: str = KAGGLE_DATASET_BASE
    kaggle_file: str = KAGGLE_FILE
    expected_bytes: int = EXPECTED_SOURCE_BYTES
    expected_sha256: str = EXPECTED_SOURCE_SHA256


DEFAULT_SOURCE_SPEC = SourceSpec()


@dataclass(frozen=True, slots=True)
class SourceStatus:
    path: Path
    bytes: int
    sha256: str
    reused_existing: bool
    downloaded: bool

    def metadata(self, spec: SourceSpec) -> dict[str, Any]:
        return {
            "format_version": SOURCE_METADATA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "kaggle_dataset": spec.kaggle_dataset,
            "kaggle_base_slug": spec.kaggle_base_slug,
            "kaggle_file": spec.kaggle_file,
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "expected_bytes": spec.expected_bytes,
            "expected_sha256": spec.expected_sha256,
            "reused_existing": self.reused_existing,
            "downloaded": self.downloaded,
        }


def default_source_path(paths: ProjectPaths, spec: SourceSpec | None = None) -> Path:
    spec = DEFAULT_SOURCE_SPEC if spec is None else spec
    return paths.raw / spec.kaggle_file


def source_metadata_path(paths: ProjectPaths) -> Path:
    return paths.source_artifacts / "source_metadata.json"


def _sanitize_error(error: BaseException) -> str:
    message = str(error) or error.__class__.__name__
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub("[redacted]", message)
    lowered = message.lower()
    if any(token in lowered for token in ("401", "403", "unauthorized", "forbidden", "consent")):
        return (
            "Kaggle denied access to the arXiv dataset. Sign in with Kaggle credentials and "
            f"accept any dataset terms/consent prompt for {KAGGLE_DATASET_BASE}, then rerun. "
            f"KaggleHub reported: {message}"
        )
    return f"KaggleHub could not download the pinned arXiv metadata file: {message}"


def _validate_jsonl_edge(path: Path, first: bool) -> None:
    line = _first_line(path) if first else _last_line(path)
    if not line:
        edge = "first" if first else "last"
        raise SourceValidationError(f"Source file has no {edge} JSONL record.")
    try:
        RawPaper.model_validate_json(line)
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        edge = "first" if first else "last"
        raise SourceValidationError(f"Source file has an invalid {edge} JSONL record.") from error


def _first_line(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.readline().rstrip(b"\r\n")


def _last_line(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        if position == 0:
            return b""
        buffer = bytearray()
        position -= 1
        while position >= 0:
            handle.seek(position)
            byte = handle.read(1)
            if byte == b"\n" and buffer:
                break
            if byte not in {b"\n", b"\r"}:
                buffer.extend(byte)
            position -= 1
        buffer.reverse()
        return bytes(buffer)


def validate_source_file(path: Path, spec: SourceSpec | None = None) -> SourceStatus:
    spec = DEFAULT_SOURCE_SPEC if spec is None else spec
    if not path.exists():
        raise SourceValidationError(f"Source metadata file is absent: {path}")
    if not path.is_file():
        raise SourceValidationError(f"Source metadata path is not a file: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise SourceValidationError(f"Source metadata file is not readable: {path}") from error
    if size != spec.expected_bytes:
        raise SourceValidationError(
            f"Source metadata size mismatch for {path}: expected {spec.expected_bytes}, "
            f"found {size}."
        )
    _validate_jsonl_edge(path, first=True)
    _validate_jsonl_edge(path, first=False)
    digest = sha256_file(path)
    if digest != spec.expected_sha256:
        raise SourceValidationError(
            f"Source metadata SHA-256 mismatch for {path}: expected {spec.expected_sha256}, "
            f"found {digest}."
        )
    return SourceStatus(
        path=path,
        bytes=size,
        sha256=digest,
        reused_existing=True,
        downloaded=False,
    )


def _quarantine_invalid_owned_file(path: Path) -> Path:
    try:
        stat = path.stat()
    except OSError as error:
        raise SourceValidationError(f"Invalid source path cannot be inspected: {path}") from error
    current_uid = getattr(os, "getuid", lambda: stat.st_uid)()
    if stat.st_uid != current_uid:
        raise SourceValidationError(
            f"Invalid source file is not owned by the current user and will not be moved: {path}"
        )
    quarantine = path.with_name(f"{path.name}.invalid-{int(time.time())}")
    path.replace(quarantine)
    return quarantine


def _import_kagglehub() -> KaggleDownloader:
    import kagglehub  # type: ignore[import-untyped]

    return cast(KaggleDownloader, kagglehub)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _cleanup_download_root(download_root: Path) -> None:
    shutil.rmtree(download_root, ignore_errors=True)
    with suppress(OSError):
        download_root.parent.rmdir()


def acquire_source(
    paths: ProjectPaths,
    spec: SourceSpec | None = None,
    *,
    downloader: KaggleDownloader | None = None,
    force_download: bool = False,
) -> SourceStatus:
    spec = DEFAULT_SOURCE_SPEC if spec is None else spec
    paths.create_output_dirs()
    destination = default_source_path(paths, spec)
    destination_needs_quarantine = False
    if destination.exists() and not force_download:
        try:
            status = validate_source_file(destination, spec)
            _write_source_metadata(paths, status, spec)
            return status
        except SourceValidationError:
            destination_needs_quarantine = True
    elif destination.exists() and force_download:
        destination_needs_quarantine = True

    download_root = paths.raw / ".kagglehub-downloads" / str(int(time.time()))
    download_root.mkdir(parents=True, exist_ok=False)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    client = downloader if downloader is not None else _import_kagglehub()
    try:
        downloaded = Path(
            client.dataset_download(
                spec.kaggle_dataset,
                path=spec.kaggle_file,
                force_download=force_download,
                output_dir=str(download_root),
            )
        )
        if downloaded.is_dir():
            downloaded = downloaded / spec.kaggle_file
        status = validate_source_file(downloaded, spec)
        if _is_relative_to(downloaded, download_root):
            downloaded.replace(temporary)
        else:
            shutil.copy2(downloaded, temporary)
        if destination_needs_quarantine and destination.exists():
            _quarantine_invalid_owned_file(destination)
        temporary.replace(destination)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        _cleanup_download_root(download_root)
        if isinstance(error, SourceValidationError):
            raise
        raise SourceDownloadError(_sanitize_error(error)) from error
    _cleanup_download_root(download_root)
    final_status = SourceStatus(
        path=destination,
        bytes=status.bytes,
        sha256=status.sha256,
        reused_existing=False,
        downloaded=True,
    )
    _write_source_metadata(paths, final_status, spec)
    return final_status


def use_local_source(paths: ProjectPaths, source: Path, spec: SourceSpec) -> SourceStatus:
    paths.create_output_dirs()
    status = validate_source_file(source, spec)
    local_status = SourceStatus(
        path=source.resolve(),
        bytes=status.bytes,
        sha256=status.sha256,
        reused_existing=True,
        downloaded=False,
    )
    _write_source_metadata(paths, local_status, spec)
    return local_status


def _write_source_metadata(paths: ProjectPaths, status: SourceStatus, spec: SourceSpec) -> None:
    metadata = status.metadata(spec)
    metadata["path"] = paths.artifact_path(status.path)
    atomic_write_json(source_metadata_path(paths), metadata)
    atomic_write_text(
        paths.source_artifacts / f"{spec.kaggle_file}.sha256",
        f"{status.sha256}  {paths.artifact_path(status.path)}\n",
    )
