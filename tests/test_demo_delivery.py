from __future__ import annotations

import re
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from arxiv_topic_classifier.config import ProjectPaths
from arxiv_topic_classifier.llm import taxonomy_json_v1_prompt
from arxiv_topic_classifier.web import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "pyproject.toml",
)
PRIVATE_TEXT_MARKERS = (
    "Group " + "11",
    "E" + "NGR",
    "Ai" + "ju",
    "Li" + "via",
    "run_" + "demo.command",
    "run_full_" + "workflow.command",
)


@pytest.fixture(scope="module")
def local_demo_client(trained_project: ProjectPaths) -> FlaskClient:
    app = create_app(trained_project.root, llm_mode="offline")
    return app.test_client()


def test_page_renders_code_only_state_without_committed_artifacts(
    local_demo_client: FlaskClient,
) -> None:
    response = local_demo_client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Official arXiv taxonomy" in html
    assert "Manual input is enabled" in html
    assert "local metrics" in html
    assert html.count('class="taxon"') == 155
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker not in html


def test_arbitrary_offline_comparison_reports_missing_cache(
    local_demo_client: FlaskClient,
) -> None:
    arbitrary_abstract = (
        "We present an original study of resilient distributed storage with erasure coding, "
        "replication, and recovery under network partitions, evaluated on a new local benchmark."
    )
    response = local_demo_client.post("/api/compare", json={"abstract": arbitrary_abstract})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["classical"]["labels"]
    assert payload["llm"]["status"] == "unavailable"
    assert payload["llm"]["source_mode"] is None
    assert payload["llm"]["unavailable_reason"] == "Demo LLM cache artifact is not present."


def test_local_health_predict_and_compare_contracts(
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
    assert health_payload["cache"]["available"] is False
    assert health_payload["demo_selection"]["count"] == 0

    abstract = (
        "We introduce a multilingual transformer for document translation and natural language "
        "understanding, evaluated on reproducible text classification benchmarks."
    )
    prediction = local_demo_client.post("/api/predict", json={"abstract": abstract})
    assert prediction.status_code == 200
    prediction_payload = prediction.get_json()
    assert prediction_payload["labels"]
    assert len(prediction_payload["probabilities"]) == 155

    comparison = local_demo_client.post("/api/compare", json={"abstract": abstract})
    assert comparison.status_code == 200
    comparison_payload = comparison.get_json()
    assert comparison_payload["classical"]["labels"]
    assert comparison_payload["llm"]["status"] == "unavailable"
    assert local_demo_client.post("/api/compare", json={}).status_code == 400
    assert local_demo_client.post("/api/predict", json={"abstract": 7}).status_code == 400


def test_public_docs_state_code_only_scope_without_private_framing() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)
    lowered = combined.lower()

    assert "149,685" in combined
    assert "104,780" in combined
    assert "22,453" in combined
    assert "22,452" in combined
    assert "155-label" in combined
    assert "0.4504" in combined
    assert "0.4470" in combined
    assert "AGPL-3.0-only" in combined
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker not in combined

    assert re.search(r"\bsix[- ]class\b", lowered) is None
    assert re.search(r"\b900(?:-row|\s+rows?|\s+test)\b", lowered) is None
    assert re.search(r"\bprimary[- ]only\b", lowered) is None
    assert "both models are evaluated" not in lowered
    for marker in PRIVATE_TEXT_MARKERS:
        assert marker.lower() not in lowered


def test_prompt_metadata_is_available_without_committed_cache() -> None:
    prompt = taxonomy_json_v1_prompt("x" * 80)

    assert prompt.version == "taxonomy_multilabel_json_v2"
    assert len(prompt.prompt_hash) == 16
