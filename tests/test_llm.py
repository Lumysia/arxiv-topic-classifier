from __future__ import annotations

import json
import time
from threading import Lock

import httpx
import pytest

from arxiv_topic_classifier.audit import run_llm_audit
from arxiv_topic_classifier.cli import build_parser
from arxiv_topic_classifier.config import CATEGORY_DESCRIPTIONS, ProjectPaths
from arxiv_topic_classifier.llm import (
    LLMChatPrompt,
    LLMClientResponse,
    LLMConfigError,
    LLMRunError,
    LLMRuntimeConfig,
    OpenAICompatibleChatClient,
    parse_llm_runtime_config,
    parse_taxonomy_output,
    taxonomy_json_v1_prompt,
)
from arxiv_topic_classifier.llm_evaluation import evaluate_llm_split
from arxiv_topic_classifier.schema import LLMUsage


class FakeLLMClient:
    model = "fake-prompt-model"

    def __init__(
        self,
        invalid_first: bool = False,
        fail_first_request: bool = False,
        success_delay_seconds: float = 0.0,
    ) -> None:
        self.invalid_first = invalid_first
        self.fail_first_request = fail_first_request
        self.success_delay_seconds = success_delay_seconds
        self.calls = 0
        self._lock = Lock()

    def complete(self, prompt: LLMChatPrompt) -> LLMClientResponse:
        with self._lock:
            self.calls += 1
            call_number = self.calls
        if self.fail_first_request and call_number == 1:
            raise RuntimeError("provider failure with SECRET_API_KEY and raw response body")
        if self.success_delay_seconds > 0:
            time.sleep(self.success_delay_seconds)
        if self.invalid_first and call_number == 1:
            content = '{"labels":["cs.UNKNOWN"]}'
        elif prompt.version.startswith("taxonomy_audit"):
            content = json.dumps(
                {
                    "suggested_label": self._label_for_prompt(prompt),
                    "alternative_labels": [],
                    "ambiguity_flag": False,
                    "short_reason": "Synthetic audit response.",
                }
            )
        else:
            content = json.dumps(
                {"labels": [self._label_for_prompt(prompt)], "reason": "Synthetic response."}
            )
        return LLMClientResponse(
            content=content,
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            latency_seconds=0.01,
        )

    def _label_for_prompt(self, prompt: LLMChatPrompt) -> str:
        content = "\n".join(message.content.lower() for message in prompt.messages)
        if "language translation text linguistic" in content:
            return "cs.CL"
        if "image vision object segmentation" in content:
            return "cs.CV"
        if "learning neural optimization generalization" in content:
            return "cs.LG"
        if "robot navigation manipulation control" in content:
            return "cs.RO"
        if "software testing debugging maintenance" in content:
            return "cs.SE"
        return "cs.AI"


def _concurrent_fake_runtime() -> LLMRuntimeConfig:
    return LLMRuntimeConfig(
        base_url="https://example.test/v1/",
        api_key="secret",
        model="unused-with-fake-client",
        max_concurrency=4,
    )


def test_llm_config_requires_live_credentials() -> None:
    with pytest.raises(LLMConfigError, match="ARXIV_LLM_BASE_URL"):
        parse_llm_runtime_config({}, require_credentials=True)
    config = parse_llm_runtime_config(
        {
            "ARXIV_LLM_BASE_URL": "https://example.test/v1",
            "ARXIV_LLM_API_KEY": "secret",
            "ARXIV_LLM_MAX_CONCURRENCY": "2",
            "ARXIV_LLM_MAX_RETRIES": "1",
        },
        require_credentials=True,
    )
    assert config.model == "gpt-5.6-sol"
    assert config.base_url == "https://example.test/v1/"
    assert config.max_concurrency == 2


def test_taxonomy_prompt_uses_abstract_and_label_definitions_only() -> None:
    prompt = taxonomy_json_v1_prompt("Abstract body about language models.")
    joined = "\n".join(message.content for message in prompt.messages)
    payload = json.loads(prompt.messages[1].content)
    assert "Abstract body about language models." in joined
    assert all(label in joined for label in CATEGORY_DESCRIPTIONS)
    definitions = {item["label"]: item["definition"] for item in payload["labels"]}
    assert definitions == CATEGORY_DESCRIPTIONS
    assert "SECRET TITLE" not in joined
    assert "source_primary_label" not in joined
    assert "ground_truth" not in joined


def test_strict_taxonomy_parser_accepts_only_known_json_labels() -> None:
    valid = parse_taxonomy_output('{"labels":["cs.CL"],"reason":"language"}')
    assert valid.valid_output is True
    assert valid.labels == ("cs.CL",)
    assert parse_taxonomy_output('{"labels":["cs.UNKNOWN"]}').valid_output is False
    assert parse_taxonomy_output('{"label":"cs.CL"}').valid_output is False
    assert parse_taxonomy_output("not json").valid_output is False


def test_openai_client_retries_retryable_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[int] = [429, 200]
    seen_user_agents: list[str | None] = []
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("arxiv_topic_classifier.llm.time.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        seen_user_agents.append(request.headers.get("User-Agent"))
        status = responses.pop(0)
        if status == 429:
            return httpx.Response(
                status,
                headers={"Retry-After": "0.25"},
                json={"error": "rate limit"},
            )
        return httpx.Response(
            status,
            json={
                "choices": [{"message": {"content": '{"labels":["cs.AI"]}'}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )

    config = LLMRuntimeConfig(
        base_url="https://example.test/v1/",
        api_key="secret",
        model="fake-live-model",
        max_retries=1,
        user_agent="unit-test-agent",
    )
    client = OpenAICompatibleChatClient(
        config,
        transport=httpx.MockTransport(handler),
        retry_base_delay_seconds=0.0,
    )
    try:
        response = client.complete(taxonomy_json_v1_prompt("Long enough abstract text."))
    finally:
        client.close()
    assert response.content == '{"labels":["cs.AI"]}'
    assert seen_user_agents == ["unit-test-agent", "unit-test-agent"]
    assert sleeps == [0.25]


def test_llm_evaluation_cache_resume_and_invalid_outputs(
    trained_project: ProjectPaths,
) -> None:
    client = FakeLLMClient(invalid_first=True)
    first = evaluate_llm_split(trained_project, "validation", resume=False, client=client)
    assert first["row_count"] == 155
    assert first["invalid_output_count"] == 1
    assert first["exact_match_ratio"] < 1.0
    assert client.calls == 155

    second = evaluate_llm_split(trained_project, "validation", resume=True, client=client)
    assert second["cache_hit_count"] == 155
    assert client.calls == 155
    assert (trained_project.artifacts / "llm_validation_metrics.json").is_file()
    assert (trained_project.artifacts / "llm_prediction_trace_validation.jsonl").is_file()
    assert (trained_project.artifacts / "llm_validation_label_errors.csv").is_file()


def test_llm_evaluation_caches_successes_and_resume_retries_only_failed_row(
    trained_project: ProjectPaths,
) -> None:
    client = FakeLLMClient(fail_first_request=True, success_delay_seconds=0.01)

    with pytest.raises(LLMRunError, match="LLM evaluation") as error:
        evaluate_llm_split(
            trained_project,
            "validation",
            resume=False,
            client=client,
            runtime_config=_concurrent_fake_runtime(),
        )

    assert "SECRET_API_KEY" not in str(error.value)
    assert "raw response body" not in str(error.value)
    cached_lines = trained_project.llm_cache("validation").read_text(encoding="utf-8").splitlines()
    assert len(cached_lines) == 154
    assert client.calls == 155

    resumed = evaluate_llm_split(
        trained_project,
        "validation",
        resume=True,
        client=client,
        runtime_config=_concurrent_fake_runtime(),
    )
    assert resumed["cache_hit_count"] == 154
    assert client.calls == 156


def test_llm_audit_excludes_test_and_does_not_mutate_labels(trained_project: ProjectPaths) -> None:
    client = FakeLLMClient()
    summary = run_llm_audit(trained_project, sample_size=12, resume=False, client=client)
    assert summary["sample"]["eligible_splits"] == ["train", "validation"]
    rows = [
        json.loads(line)
        for line in (trained_project.artifacts / "llm_label_audit.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 12
    assert {row["split"] for row in rows} <= {"train", "validation"}
    assert all(row["label_mutated"] is False for row in rows)
    assert all(row["source_primary_label"] for row in rows)


def test_llm_audit_caches_successes_and_resume_retries_only_failed_row(
    trained_project: ProjectPaths,
) -> None:
    client = FakeLLMClient(fail_first_request=True, success_delay_seconds=0.01)

    with pytest.raises(LLMRunError, match="LLM label audit"):
        run_llm_audit(
            trained_project,
            sample_size=12,
            resume=False,
            client=client,
            runtime_config=_concurrent_fake_runtime(),
        )

    cached_lines = trained_project.llm_audit_cache.read_text(encoding="utf-8").splitlines()
    assert len(cached_lines) == 11
    assert client.calls == 12

    resumed = run_llm_audit(
        trained_project,
        sample_size=12,
        resume=True,
        client=client,
        runtime_config=_concurrent_fake_runtime(),
    )
    assert resumed["sample"]["cache_hit_count"] == 11
    assert client.calls == 13


def test_cli_parser_exposes_llm_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["llm-evaluate", "--split", "validation"]).command == "llm-evaluate"
    assert parser.parse_args(["llm-audit", "--sample-size", "12"]).command == "llm-audit"
    assert parser.parse_args(["llm-predict", "--text", "x" * 100]).command == "llm-predict"
    report_args = parser.parse_args(["llm-report", "--expected-test-rows", "18"])
    assert report_args.command == "llm-report"
    assert report_args.expected_test_rows == 18
    with pytest.raises(SystemExit):
        parser.parse_args(["audit"])
