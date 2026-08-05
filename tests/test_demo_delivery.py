from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from arxiv_topic_classifier.files import sha256_file
from arxiv_topic_classifier.llm import taxonomy_json_v1_prompt
from arxiv_topic_classifier.web import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / ".gitignore",
    REPOSITORY_ROOT / ".gitattributes",
)
PACKAGED_ARTIFACTS = (
    "models/arxiv_topic_pipeline.joblib",
    "models/arxiv_topic_pipeline.manifest.json",
    "artifacts/data_profile.json",
    "artifacts/demo_llm_cache.json",
    "artifacts/demo_selection.json",
    "artifacts/metrics.json",
    "run_demo.command",
)
PRIVATE_TEXT_MARKERS = (
    "Group " + "11",
    "E" + "NGR",
    "Ai" + "ju",
    "Li" + "via",
    "run_full_" + "workflow.command",
    "school",
    "provider/router",
    "assistant-production",
)


@pytest.fixture(scope="module")
def local_demo_client() -> FlaskClient:
    app = create_app(REPOSITORY_ROOT, llm_mode="offline")
    return app.test_client()


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))


def test_create_app_at_repository_root_renders_packaged_demo(
    local_demo_client: FlaskClient,
) -> None:
    response = local_demo_client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Official arXiv taxonomy" in html
    assert "149,685" in html
    assert "22,452" in html
    assert "Clear agreement" in html
    assert "Classical complete, LLM partial" in html
    assert "LLM partial recovery" in html
    assert "6 real candidate calls · 3 representative cached excerpts" in html
    unavailable_message = (
        "The local metrics, data profile, and comparison-scope artifacts are unavailable"
    )
    assert unavailable_message not in html
    assert html.count('class="taxon"') == 155
    assert html.count("data-example-index=") == 3
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker not in html


def test_health_and_predict_use_packaged_model(
    local_demo_client: FlaskClient,
) -> None:
    health = local_demo_client.get("/health")
    assert health.status_code == 200
    health_payload = health.get_json()
    assert health_payload["status"] == "ok"
    assert health_payload["ready"] is True
    assert len(health_payload["labels"]) == 155
    assert health_payload["llm"]["mode"] == "offline"
    assert health_payload["llm"]["client_configured"] is False
    assert health_payload["cache"]["available"] is True
    assert health_payload["cache"]["count"] == 3
    assert health_payload["cache"]["warnings"] == []
    assert health_payload["demo_selection"]["available"] is True
    assert health_payload["demo_selection"]["count"] == 3
    assert health_payload["demo_selection"]["warnings"] == []
    assert health_payload["demo_selection"]["llm_scope"]["packaged_examples"] == 3

    abstract = (
        "We introduce a multilingual transformer for document translation and natural language "
        "understanding, evaluated on reproducible text classification benchmarks."
    )
    prediction = local_demo_client.post("/api/predict", json={"abstract": abstract})
    assert prediction.status_code == 200
    prediction_payload = prediction.get_json()
    assert prediction_payload["labels"]
    assert len(prediction_payload["probabilities"]) == 155
    assert prediction_payload["top_label"] in health_payload["labels"]

    assert local_demo_client.post("/api/predict", json={"abstract": 7}).status_code == 400


def test_each_packaged_example_receives_matching_offline_cached_response(
    local_demo_client: FlaskClient,
) -> None:
    selection = _read_json("artifacts/demo_selection.json")
    cache = _read_json("artifacts/demo_llm_cache.json")
    cache_entries = {
        entry["abstract_hash"]: entry
        for entry in cache["entries"]
    }

    assert len(selection["examples"]) == 3
    assert len(cache_entries) == 3
    for example in selection["examples"]:
        response = local_demo_client.post(
            "/api/compare",
            json={"abstract": example["normalized_abstract"]},
        )
        assert response.status_code == 200
        payload = response.get_json()
        cached = cache_entries[example["abstract_hash"]]

        assert payload["input"]["abstract_hash"] == example["abstract_hash"]
        assert payload["classical"]["labels"]
        assert payload["llm"]["status"] == "ok"
        assert payload["llm"]["source_mode"] == "cached"
        assert payload["llm"]["labels"] == cached["labels"]
        assert payload["llm"]["reason"] == cached["reason"]


def test_arbitrary_offline_comparison_reports_exact_cache_unavailable(
    local_demo_client: FlaskClient,
) -> None:
    abstract = (
        "We present an original study of resilient distributed storage with erasure coding, "
        "replication, and recovery under network partitions, evaluated on a new local benchmark."
    )

    comparison = local_demo_client.post("/api/compare", json={"abstract": abstract})
    assert comparison.status_code == 200
    comparison_payload = comparison.get_json()
    assert comparison_payload["classical"]["labels"]
    assert comparison_payload["llm"]["status"] == "unavailable"
    assert comparison_payload["llm"]["labels"] == []
    assert comparison_payload["llm"]["source_mode"] is None
    assert (
        comparison_payload["llm"]["unavailable_reason"]
        == "No exact cached LLM response exists for this normalized abstract."
    )
    assert comparison_payload["llm"]["reason"] is None
    assert local_demo_client.post("/api/compare", json={}).status_code == 400


def test_packaged_model_sha256_matches_manifest() -> None:
    manifest = _read_json("models/arxiv_topic_pipeline.manifest.json")
    metrics = _read_json("artifacts/metrics.json")
    profile_path = REPOSITORY_ROOT / manifest["processed_manifest"]["profile_path"]
    profile_sha256 = sha256_file(profile_path)

    assert manifest["model_path"] == "models/arxiv_topic_pipeline.joblib"
    assert manifest["label_count"] == 155
    assert sha256_file(REPOSITORY_ROOT / manifest["model_path"]) == manifest["model_sha256"]
    assert sha256_file(REPOSITORY_ROOT / manifest["metrics_path"]) == manifest["metrics_sha256"]
    assert metrics["processed_manifest"]["profile_sha256"] == profile_sha256
    assert manifest["processed_manifest"]["profile_sha256"] == profile_sha256


def test_launcher_invokes_offline_smoke_and_serve_and_is_executable() -> None:
    launcher = REPOSITORY_ROOT / "run_demo.command"
    text = launcher.read_text(encoding="utf-8")
    mode = launcher.stat().st_mode

    assert mode & stat.S_IXUSR
    assert os.access(launcher, os.X_OK)
    assert "uv run arxiv-classifier --root \"$PROJECT_ROOT\" smoke" in text
    assert "uv run arxiv-classifier --root \"$PROJECT_ROOT\" serve" in text
    assert "--llm-mode offline" in text


def test_public_docs_state_narrow_packaged_demo_without_private_framing() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
    lowered = combined.lower()

    assert "Git LFS" in combined
    assert "uv sync --locked" in combined
    assert "git lfs install" in combined
    assert "git lfs pull" in combined
    assert "run_demo.command" in combined
    assert "uv run arxiv-classifier smoke" in combined
    assert "uv run arxiv-classifier serve --llm-mode offline" in combined
    assert "149,685" in combined
    assert "104,780" in combined
    assert "22,453" in combined
    assert "22,452" in combined
    assert "155-label" in combined
    assert "0.4504" in combined
    assert "0.4470" in combined
    assert "raw arxiv data" in lowered
    assert "processed training datasets" in lowered
    assert "private materials" in lowered
    assert "slides" in lowered
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker not in combined

    assert re.search(r"\bsix[- ]class\b", lowered) is None
    assert re.search(r"\b900(?:-row|\s+rows?|\s+test)\b", lowered) is None
    assert re.search(r"\bprimary[- ]only\b", lowered) is None
    assert "both models are evaluated" not in lowered
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker.lower() not in lowered


def test_ignore_and_lfs_rules_enforce_only_narrow_public_payload() -> None:
    gitattributes = (REPOSITORY_ROOT / ".gitattributes").read_text(encoding="utf-8")
    gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "models/*.joblib filter=lfs diff=lfs merge=lfs -text" in gitattributes
    for relative_path in PACKAGED_ARTIFACTS:
        assert relative_path in gitignore or relative_path == "run_demo.command"

    check_attr = subprocess.run(
        [
            "git",
            "check-attr",
            "filter",
            "diff",
            "merge",
            "text",
            "--",
            "models/arxiv_topic_pipeline.joblib",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "filter: lfs" in check_attr.stdout
    assert "diff: lfs" in check_attr.stdout
    assert "merge: lfs" in check_attr.stdout
    assert "text: unset" in check_attr.stdout

    for relative_path in PACKAGED_ARTIFACTS:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative_path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 1, relative_path
    for relative_path in (
        "models/other.joblib",
        "models/other.manifest.json",
        "artifacts/llm_prediction_trace_test.jsonl",
        "artifacts/cache/llm_test_responses.jsonl",
        "artifacts/report.html",
        "data/processed/arxiv_topics.csv",
        "slides/final.pdf",
        "other.command",
        "screenshots/demo.png",
        "tools/browser_check.py",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative_path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, relative_path


def test_packaged_cache_matches_prompt_metadata() -> None:
    prompt = taxonomy_json_v1_prompt("x" * 80)
    cache = _read_json("artifacts/demo_llm_cache.json")

    assert prompt.version == "taxonomy_multilabel_json_v2"
    assert len(prompt.prompt_hash) == 16
    assert cache["prompt_version"] == prompt.version
    assert cache["prompt_hash"] == prompt.prompt_hash
