from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import joblib
import pytest

import arxiv_topic_classifier.files as file_helpers
import arxiv_topic_classifier.workflow as workflow
from arxiv_topic_classifier.artifact_package import (
    MANIFEST_NAME,
    ROOT_FILES,
    SHA256_MANIFEST_NAME,
    SOURCE_FILES,
    create_artifact_package,
    verify_artifact_zip,
)
from arxiv_topic_classifier.config import TARGET_CATEGORIES, ProjectPaths
from arxiv_topic_classifier.files import atomic_joblib_dump, atomic_write_json, sha256_file
from arxiv_topic_classifier.source import (
    KAGGLE_DATASET,
    SourceSpec,
    acquire_source,
    validate_source_file,
)
from arxiv_topic_classifier.taxonomy import taxonomy_fingerprint
from arxiv_topic_classifier.verification import VerificationError, verify_model
from arxiv_topic_classifier.workflow import (
    ResumeCheckError,
    model_is_current,
    processed_dataset_is_current,
    run_full_train_package,
    write_processed_manifest,
)


def _raw_record(label: str, index: int) -> dict[str, object]:
    return {
        "id": f"2501.{index:05d}",
        "title": f"Synthetic {label} paper",
        "abstract": (
            f"This reproducible abstract discusses {label} methods, experiments, results, "
            "benchmarks, and limitations for source downloader validation."
        ),
        "categories": label,
        "versions": [{"version": "v1", "created": "Mon, 1 Jan 2025 00:00:00 GMT"}],
        "update_date": "2025-01-01",
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> SourceSpec:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return SourceSpec(expected_bytes=path.stat().st_size, expected_sha256=sha256_file(path))


class FakeDownloader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls: list[dict[str, object]] = []

    def dataset_download(
        self,
        handle: str,
        *,
        path: str | None = None,
        force_download: bool = False,
        output_dir: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "handle": handle,
                "path": path,
                "force_download": force_download,
                "output_dir": output_dir,
            }
        )
        assert output_dir is not None
        destination = Path(output_dir) / str(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.source, destination)
        return str(destination)


def test_downloader_cold_warm_and_invalid_owned_source(tmp_path: Path) -> None:
    project = ProjectPaths.from_root(tmp_path / "project")
    remote = tmp_path / "remote.jsonl"
    spec = _write_jsonl(remote, [_raw_record("cs.CL", 1), _raw_record("cs.LG", 2)])
    downloader = FakeDownloader(remote)

    cold = acquire_source(project, spec, downloader=downloader)
    assert cold.downloaded is True
    assert downloader.calls[0]["handle"] == KAGGLE_DATASET
    assert downloader.calls[0]["path"] == spec.kaggle_file
    assert cold.path == project.raw / spec.kaggle_file
    assert not (project.raw / ".kagglehub-downloads").exists()

    warm = acquire_source(project, spec, downloader=downloader)
    assert warm.reused_existing is True
    assert len(downloader.calls) == 1

    cold.path.write_text("partial\n", encoding="utf-8")
    repaired = acquire_source(project, spec, downloader=downloader)
    assert repaired.downloaded is True
    assert len(downloader.calls) == 2
    assert list(project.raw.glob(f"{spec.kaggle_file}.invalid-*"))


def test_validate_source_rejects_bad_last_record(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(_raw_record("cs.CL", 1)) + "\n{}\n", encoding="utf-8")
    spec = SourceSpec(expected_bytes=source.stat().st_size, expected_sha256=sha256_file(source))

    with pytest.raises(RuntimeError, match="last JSONL record"):
        validate_source_file(source, spec)


def test_acquire_source_preserves_invalid_existing_file_when_download_fails(
    tmp_path: Path,
) -> None:
    project = ProjectPaths.from_root(tmp_path / "project")
    project.create_output_dirs()
    destination = project.raw / "arxiv-metadata-oai-snapshot.json"
    destination.write_text("partial\n", encoding="utf-8")
    spec = SourceSpec(expected_bytes=10, expected_sha256="0" * 64)

    class FailingDownloader:
        def dataset_download(
            self,
            handle: str,
            *,
            path: str | None = None,
            force_download: bool = False,
            output_dir: str | None = None,
        ) -> str:
            del handle, path, force_download, output_dir
            raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        acquire_source(project, spec, downloader=FailingDownloader())

    assert destination.read_text(encoding="utf-8") == "partial\n"


def test_atomic_joblib_write_preserves_prior_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "model.joblib"
    atomic_joblib_dump({"version": 1}, destination)

    def fail_dump(payload: object, filename: object, compress: int = 0) -> None:
        del payload, compress
        Path(str(filename)).write_bytes(b"partial")
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(file_helpers.joblib, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="simulated crash"):
        atomic_joblib_dump({"version": 2}, destination)

    assert joblib.load(destination) == {"version": 1}


def _prepare_return_payloads(paths: ProjectPaths, processed_manifest: dict[str, Any]) -> None:
    repo = Path(__file__).resolve().parents[1]
    for relative in (*ROOT_FILES, *SOURCE_FILES):
        source = repo / relative
        destination = paths.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    atomic_write_json(paths.source_artifacts / "source_metadata.json", {"sha256": "synthetic"})
    atomic_write_json(paths.artifacts / "processed_manifest.json", processed_manifest)
    atomic_write_json(paths.profile, {"sample": {"total_rows": processed_manifest["row_count"]}})
    verify_model(paths, processed_manifest)


def test_strict_verification_failure_on_fingerprint_mismatch(
    trained_project: ProjectPaths,
) -> None:
    with pytest.raises(VerificationError, match="processed manifest"):
        verify_model(trained_project, {"bad": True})


def test_model_resume_check_uses_strict_manifest(trained_project: ProjectPaths) -> None:
    manifest = json.loads(trained_project.model_manifest.read_text(encoding="utf-8"))[
        "processed_manifest"
    ]
    assert model_is_current(trained_project, manifest) is True
    assert model_is_current(trained_project, {"bad": True}) is False


def _write_resume_profile(paths: ProjectPaths, source_spec: SourceSpec, row_count: int) -> None:
    atomic_write_json(
        paths.profile,
        {
            "source": {
                "sha256_of_scanned_bytes": source_spec.expected_sha256,
                "bytes_at_scan_end": source_spec.expected_bytes,
            },
            "category_design": {
                "taxonomy_sha256": taxonomy_fingerprint(),
                "categories": list(TARGET_CATEGORIES),
            },
            "sample": {"per_label_cap": 10, "seed": 5775, "total_rows": row_count},
        },
    )


def test_missing_processed_manifest_raises_resume_check_error(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path / "project")
    paths.create_output_dirs()
    source = tmp_path / "source.jsonl"
    source_spec = _write_jsonl(source, [_raw_record("cs.CL", 1), _raw_record("cs.LG", 2)])
    paths.dataset.write_text("version,content\n1,valid\n", encoding="utf-8")
    _write_resume_profile(paths, source_spec, row_count=1)

    with pytest.raises(ResumeCheckError, match="absent or invalid"):
        processed_dataset_is_current(
            paths,
            validate_source_file(source, source_spec),
            per_class=10,
            seed=5775,
        )


def test_corrupt_processed_csv_forces_prepare_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ProjectPaths.from_root(tmp_path / "project")
    paths.create_output_dirs()
    source = tmp_path / "source.jsonl"
    source_spec = _write_jsonl(source, [_raw_record("cs.CL", 1), _raw_record("cs.LG", 2)])
    paths.dataset.write_text("version,content\n1,valid\n", encoding="utf-8")
    _write_resume_profile(paths, source_spec, row_count=1)
    write_processed_manifest(
        paths,
        validate_source_file(source, source_spec),
        per_class=10,
        seed=5775,
    )
    paths.dataset.write_text("version,content\n2,corrupt\n", encoding="utf-8")
    prepare_calls = 0

    def fake_prepare(
        source_path: Path,
        project_paths: ProjectPaths,
        per_class: int,
        seed: int,
    ) -> dict[str, Any]:
        nonlocal prepare_calls
        assert source_path == source.resolve()
        assert per_class == 10
        assert seed == 5775
        prepare_calls += 1
        project_paths.dataset.write_text("version,content\n3,repaired\n", encoding="utf-8")
        _write_resume_profile(project_paths, source_spec, row_count=1)
        return {}

    monkeypatch.setattr(workflow, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(workflow, "train_and_evaluate", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "verify_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        workflow,
        "create_artifact_package",
        lambda project_paths, **kwargs: project_paths.returns / "synthetic.zip",
    )

    result = run_full_train_package(
        paths,
        per_class=10,
        source=source,
        source_spec=source_spec,
    )

    assert result.prepared is True
    assert prepare_calls == 1
    assert "repaired" in paths.dataset.read_text(encoding="utf-8")


def test_artifact_zip_manifest_exclusions_and_model_load(trained_project: ProjectPaths) -> None:
    processed_manifest = json.loads(trained_project.model_manifest.read_text(encoding="utf-8"))[
        "processed_manifest"
    ]
    _prepare_return_payloads(trained_project, processed_manifest)
    raw_source = trained_project.raw / "arxiv-metadata-oai-snapshot.json"
    raw_source.write_text("must not enter artifact archives", encoding="utf-8")

    zip_path = create_artifact_package(
        trained_project,
        processed_manifest=processed_manifest,
        verification_report={},
    )
    report = verify_artifact_zip(zip_path)
    assert report["crc"] == "passed"

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        sha_manifest = archive.read(SHA256_MANIFEST_NAME).decode("utf-8")
    assert "data/raw/arxiv-metadata-oai-snapshot.json" not in names
    assert "data/processed/arxiv_topics.csv" not in names
    assert "models/arxiv_topic_pipeline.joblib" in names
    assert "src/arxiv_topic_classifier/data.py" in names
    assert "src/arxiv_topic_classifier/eda.py" in names
    assert "src/arxiv_topic_classifier/reporting.py" in names
    assert {entry["path"] for entry in manifest["files"]} == names
    assert SHA256_MANIFEST_NAME in names
    assert MANIFEST_NAME not in sha_manifest
    assert SHA256_MANIFEST_NAME not in sha_manifest
    assert "models/arxiv_topic_pipeline.joblib" in sha_manifest
    assert not any(name.startswith("slides/") or "llm_" in name for name in names)
    assert str(trained_project.root) not in json.dumps(manifest)


def test_processed_and_model_manifests_use_project_relative_paths(
    trained_project: ProjectPaths,
) -> None:
    processed_manifest = json.loads(
        (trained_project.artifacts / "processed_manifest.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(trained_project.model_manifest.read_text(encoding="utf-8"))

    assert processed_manifest["dataset_path"] == "data/processed/arxiv_topics.csv"
    assert processed_manifest["profile_path"] == "artifacts/data_profile.json"
    assert str(trained_project.root) not in json.dumps(processed_manifest)
    assert model_manifest["model_path"] == "models/arxiv_topic_pipeline.joblib"
    assert model_manifest["metrics_path"] == "artifacts/metrics.json"
    assert str(trained_project.root) not in json.dumps(model_manifest)
