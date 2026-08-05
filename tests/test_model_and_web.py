from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arxiv_topic_classifier.config import ProjectPaths
from arxiv_topic_classifier.llm import LLMChatPrompt, LLMClientResponse, taxonomy_json_v1_prompt
from arxiv_topic_classifier.model import load_artifact, predict_abstract
from arxiv_topic_classifier.representation import run_representation_benchmark
from arxiv_topic_classifier.schema import LLMUsage
from arxiv_topic_classifier.text import TextValidationError
from arxiv_topic_classifier.web import create_app, demo_abstract_hash, load_demo_llm_cache


class FakeWebLLMClient:
    model = "fake-web-llm"

    def __init__(self, label: str = "cs.CL", invalid: bool = False, fail: bool = False) -> None:
        self.label = label
        self.invalid = invalid
        self.fail = fail
        self.calls = 0

    def complete(self, prompt: LLMChatPrompt) -> LLMClientResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider failed with SECRET_API_KEY and raw prompt body")
        del prompt
        content = '{"labels":["cs.UNKNOWN"]}' if self.invalid else json.dumps(
            {"labels": [self.label], "reason": "Synthetic web response."}
        )
        return LLMClientResponse(
            content=content,
            usage=LLMUsage(input_tokens=11, output_tokens=7, total_tokens=18),
            latency_seconds=0.02,
        )


class FakeEmbeddingHelper:
    model_name = "fake-bge-small"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        del batch_size
        keywords = (
            "agents reasoning planning knowledge",
            "language translation text linguistic",
            "image vision object segmentation",
            "learning neural optimization generalization",
            "robot navigation manipulation control",
            "software testing debugging maintenance",
        )
        values: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            row = [
                float(sum(term in lowered for term in keywords_text.split()))
                for keywords_text in keywords
            ]
            values.append(row)
        return np.asarray(values, dtype=float)


def _language_abstract() -> str:
    return (
        "We introduce a transformer language model for multilingual document translation. "
        "The model uses attention and contrastive representation learning, and improves "
        "translation quality on several natural language processing benchmarks."
    )


def _write_demo_cache(path: Path, abstract: str, label: str = "cs.CL") -> None:
    prompt = taxonomy_json_v1_prompt("x" * 80)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "prompt_version": prompt.version,
                "prompt_hash": prompt.prompt_hash,
                "entries": [
                    {
                        "normalized_abstract": abstract,
                        "abstract_hash": demo_abstract_hash(abstract),
                        "labels": [label],
                        "reason": "Cached synthetic response.",
                        "valid_output": True,
                        "parse_error": None,
                        "model": "cached-fake-model",
                        "source": "Offline cache",
                        "latency_seconds": 1.23,
                        "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_training_generates_real_artifacts(trained_project: ProjectPaths) -> None:
    metrics = json.loads((trained_project.artifacts / "metrics.json").read_text())
    assert metrics["selected_model"] in metrics["test"]
    assert metrics["task"] == "official_arxiv_taxonomy_multilabel"
    assert len(metrics["labels"]) == 155
    assert 0.0 <= metrics["test"][metrics["selected_model"]]["macro_f1"] <= 1.0
    assert trained_project.model.is_file()
    expected = [
        "class_metrics.csv",
        "prediction_examples.json",
        "error_patterns.json",
        "evaluation_report.md",
        "figures/class_distribution.png",
        "figures/label_f1.png",
        "figures/model_comparison.png",
        "eda_summary.json",
        "figures/abstract_length_distributions.png",
        "figures/year_distribution.png",
        "figures/top_tfidf_logistic_terms.png",
    ]
    assert all((trained_project.artifacts / relative).stat().st_size > 0 for relative in expected)
    eda = json.loads((trained_project.artifacts / "eda_summary.json").read_text())
    assert eda["input_contract"].startswith("Abstract text only")
    assert eda["top_discriminative_terms"]["fit_split"] == "train"
    assert eda["official_label_count"] == 155


def test_serialized_prediction_and_invalid_input(trained_project: ProjectPaths) -> None:
    artifact = load_artifact(trained_project.model)
    result = predict_abstract(
        artifact,
        "We study source code testing and debugging with automated software maintenance tools. "
        "The evaluation uses developer projects and reports reproducible software quality results.",
    )
    assert result.top_label in artifact["labels"]
    assert result.labels
    assert all(0.0 <= value <= 1.0 for value in result.probabilities.values())
    with pytest.raises(TextValidationError):
        predict_abstract(artifact, "short")


def test_web_demo_smoke_and_negative_boundary(trained_project: ProjectPaths) -> None:
    client = create_app(trained_project.root).test_client()
    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    response = client.post(
        "/api/predict",
        json={
            "abstract": (
                "We present image segmentation and visual object recognition experiments using "
                "a convolutional model evaluated on multiple computer vision benchmarks."
            )
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["top_label"] in payload["probabilities"]
    assert client.post("/api/predict", json={"abstract": 7}).status_code == 400
    assert client.post("/api/predict", json={}).status_code == 400


def test_compare_live_agreement_and_disagreement(trained_project: ProjectPaths) -> None:
    abstract = _language_abstract()
    agreeing_client = create_app(
        trained_project.root,
        llm_mode="live",
        llm_client=FakeWebLLMClient(label="cs.CL"),
    ).test_client()
    agreement_response = agreeing_client.post("/api/compare", json={"abstract": abstract})
    assert agreement_response.status_code == 200
    agreement_payload = agreement_response.get_json()
    assert agreement_payload["classical"]["top_label"] == "cs.CL"
    assert agreement_payload["llm"]["labels"] == ["cs.CL"]
    assert agreement_payload["llm"]["source_mode"] == "live"
    assert agreement_payload["agreement"] is True
    assert agreement_payload["agreement_status"] == "agree"

    disagreeing_client = create_app(
        trained_project.root,
        llm_mode="live",
        llm_client=FakeWebLLMClient(label="cs.RO"),
    ).test_client()
    disagreement_response = disagreeing_client.post("/api/compare", json={"abstract": abstract})
    assert disagreement_response.status_code == 200
    disagreement_payload = disagreement_response.get_json()
    assert disagreement_payload["classical"]["top_label"] == "cs.CL"
    assert disagreement_payload["llm"]["labels"] == ["cs.RO"]
    assert disagreement_payload["agreement"] is False
    assert disagreement_payload["agreement_status"] == "disagree"


def test_compare_live_invalid_llm_output_is_reported(trained_project: ProjectPaths) -> None:
    client = create_app(
        trained_project.root,
        llm_mode="live",
        llm_client=FakeWebLLMClient(invalid=True),
    ).test_client()
    response = client.post("/api/compare", json={"abstract": _language_abstract()})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["classical"]["top_label"] == "cs.CL"
    assert payload["llm"]["valid_output"] is False
    assert payload["llm"]["status"] == "invalid_output"
    assert payload["llm"]["parse_error"] == "labels must contain only known taxonomy IDs."
    assert payload["agreement"] is None
    assert payload["agreement_status"] == "llm_invalid_output"


def test_compare_live_failure_uses_exact_cache_fallback(
    trained_project: ProjectPaths, tmp_path: Path
) -> None:
    abstract = _language_abstract()
    cache_path = tmp_path / "demo_llm_cache.json"
    _write_demo_cache(cache_path, abstract)
    fake_client = FakeWebLLMClient(fail=True)
    client = create_app(
        trained_project.root,
        llm_mode="live-with-cache-fallback",
        llm_client=fake_client,
        llm_cache_path=cache_path,
    ).test_client()
    response = client.post("/api/compare", json={"abstract": abstract})
    assert response.status_code == 200
    payload = response.get_json()
    assert fake_client.calls == 1
    assert payload["classical"]["top_label"] == "cs.CL"
    assert payload["llm"]["source_mode"] == "cached"
    assert payload["llm"]["model"] == "cached-fake-model"
    assert payload["llm"]["fallback_message"].startswith("The live LLM service")
    assert "SECRET" not in json.dumps(payload)
    assert "raw prompt" not in json.dumps(payload)


def test_compare_offline_unknown_keeps_classical_result(trained_project: ProjectPaths) -> None:
    client = create_app(trained_project.root, llm_mode="offline").test_client()
    response = client.post("/api/compare", json={"abstract": _language_abstract()})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["classical"]["top_label"] == "cs.CL"
    assert payload["llm"]["status"] == "unavailable"
    assert payload["llm"]["source_mode"] is None
    assert payload["agreement"] is None
    assert payload["agreement_status"] == "llm_unavailable"


def test_health_reports_modes_artifacts_without_llm_call(
    trained_project: ProjectPaths, tmp_path: Path
) -> None:
    (trained_project.artifacts / "llm_validation_metrics.json").write_text("{}\n", encoding="utf-8")
    (trained_project.artifacts / "llm_test_metrics.json").write_text("{}\n", encoding="utf-8")
    cache_path = tmp_path / "demo_llm_cache.json"
    _write_demo_cache(cache_path, _language_abstract())
    fake_client = FakeWebLLMClient()
    client = create_app(
        trained_project.root,
        llm_mode="live-with-cache-fallback",
        llm_client=fake_client,
        llm_cache_path=cache_path,
    ).test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert fake_client.calls == 0
    assert payload["classical"]["available"] is True
    assert payload["llm"] == {
        "mode": "live-with-cache-fallback",
        "client_configured": True,
    }
    assert payload["cache"]["available"] is True
    assert payload["cache"]["count"] == 1
    assert payload["artifacts"]["llm_validation_metrics"] is True
    assert payload["artifacts"]["llm_test_metrics"] is True


def test_demo_cache_loader_rejects_old_single_label_cache(tmp_path: Path) -> None:
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("[]", encoding="utf-8")
    malformed = load_demo_llm_cache(malformed_path)
    assert malformed.available is False
    assert malformed.error == "Demo LLM cache artifact is malformed."

    stale_path = tmp_path / "stale.json"
    prompt = taxonomy_json_v1_prompt("x" * 80)
    stale_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "prompt_version": prompt.version,
                "prompt_hash": prompt.prompt_hash,
                "entries": [
                    {
                        "normalized_abstract": _language_abstract(),
                        "abstract_hash": "stale-hash",
                        "label": "cs.CL",
                        "reason": "Stale cached response.",
                        "valid_output": True,
                        "parse_error": None,
                        "model": "cached-fake-model",
                        "source": "Offline cache",
                        "latency_seconds": 1.0,
                        "usage": {"total_tokens": 10},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stale = load_demo_llm_cache(stale_path)
    assert stale.available is False
    assert stale.error == "Demo LLM cache artifact is malformed."


def test_semantic_benchmark_schema_with_mocked_embeddings(trained_project: ProjectPaths) -> None:
    benchmark = run_representation_benchmark(
        trained_project,
        embedding_helper=FakeEmbeddingHelper(trained_project.root / ".cache" / "fake"),
    )
    assert benchmark["selected_for_deployment"] is False
    assert benchmark["input_contract"].startswith("Abstract text only")
    assert benchmark["row_counts"] == {"train": 465, "validation": 155, "test": 155}
    assert 0.0 <= benchmark["validation"]["macro_f1"] <= 1.0
    assert 0.0 <= benchmark["test"]["macro_f1"] <= 1.0
    assert (trained_project.artifacts / "representation_benchmark.json").is_file()
    assert (trained_project.figures / "representation_benchmark_comparison.png").stat().st_size > 0
