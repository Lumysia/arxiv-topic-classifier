from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
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
    LLMProviderError,
    LLMRunError,
    LLMRuntimeConfig,
    OpenAICompatibleChatClient,
    parse_llm_runtime_config,
    parse_taxonomy_output,
    taxonomy_json_v1_prompt,
)
from arxiv_topic_classifier.llm_evaluation import evaluate_llm_split
from arxiv_topic_classifier.opencode_client import OpenCodeCLIClient, OpenCodeProcessResult
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
            content = '{"label":"cs.UNKNOWN"}'
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
                {"label": self._label_for_prompt(prompt), "reason": "Synthetic response."}
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


def _fake_opencode_executable(tmp_path: Path) -> str:
    executable = tmp_path / "opencode"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return str(executable)


def _jsonl_stdout(events: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


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
    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://example.test/v1/"
    assert config.max_concurrency == 2


def test_taxonomy_prompt_uses_abstract_and_label_definitions_only() -> None:
    prompt = taxonomy_json_v1_prompt("Abstract body about language models.")
    joined = "\n".join(message.content for message in prompt.messages)
    assert "Abstract body about language models." in joined
    assert all(label in joined for label in CATEGORY_DESCRIPTIONS)
    assert all(description in joined for description in CATEGORY_DESCRIPTIONS.values())
    assert "SECRET TITLE" not in joined
    assert "source_primary_label" not in joined
    assert "ground_truth" not in joined


def test_strict_taxonomy_parser_accepts_only_known_json_labels() -> None:
    valid = parse_taxonomy_output('{"label":"cs.CL","reason":"language"}')
    assert valid.valid_output is True
    assert valid.label == "cs.CL"
    assert parse_taxonomy_output('{"label":"cs.UNKNOWN"}').valid_output is False
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
                "id": "response-1",
                "choices": [{"message": {"content": '{"label":"cs.AI"}'}}],
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
    assert response.content == '{"label":"cs.AI"}'
    assert seen_user_agents == ["unit-test-agent", "unit-test-agent"]
    assert sleeps == [0.25]


def test_opencode_client_parses_jsonl_usage_cost_latency_and_session_id(
    tmp_path: Path,
) -> None:
    seen_commands: list[Sequence[str]] = []

    def runner(command: Sequence[str], cwd: Path, timeout_seconds: float) -> OpenCodeProcessResult:
        seen_commands.append(command)
        assert cwd != tmp_path
        assert timeout_seconds > 0
        return OpenCodeProcessResult(
            returncode=0,
            stdout=_jsonl_stdout(
                [
                    {
                        "type": "text",
                        "sessionID": "session-1",
                        "part": {"text": '{"label":"cs.AI","reason":"ok"}'},
                    },
                    {
                        "type": "step_finish",
                        "sessionID": "session-1",
                        "part": {
                            "model": "openai/gpt-5.6-sol",
                            "tokens": {
                                "input": 12222,
                                "output": 15,
                                "total": 12237,
                                "reasoning": 3,
                                "cache": {"read": 7, "write": 2},
                            },
                            "cost": 0,
                        },
                    },
                ]
            ),
            stderr="",
        )

    client = OpenCodeCLIClient(
        executable=_fake_opencode_executable(tmp_path),
        work_dir=tmp_path / "isolated",
        runner=runner,
    )
    response = client.complete(taxonomy_json_v1_prompt("Long enough abstract text."))
    command = seen_commands[0]
    assert command[:8] == [
        str((tmp_path / "opencode").resolve()),
        "run",
        "--model",
        "openai/gpt-5.6-sol",
        "--variant",
        "high",
        "--format",
        "json",
    ]
    assert response.content == '{"label":"cs.AI","reason":"ok"}'
    assert response.response_id == "session-1"
    assert response.latency_seconds >= 0
    assert response.usage.input_tokens == 12222
    assert response.usage.output_tokens == 15
    assert response.usage.total_tokens == 12237
    assert response.usage.reasoning_tokens == 3
    assert response.usage.cache_read_tokens == 7
    assert response.usage.cache_write_tokens == 2
    assert response.usage.reported_cost_usd == 0.0


def test_opencode_client_rejects_tool_event(tmp_path: Path) -> None:
    def runner(command: Sequence[str], cwd: Path, timeout_seconds: float) -> OpenCodeProcessResult:
        return OpenCodeProcessResult(
            returncode=0,
            stdout=_jsonl_stdout(
                [
                    {"type": "tool_call", "sessionID": "session-1", "part": {"name": "read"}},
                    {
                        "type": "step_finish",
                        "sessionID": "session-1",
                        "part": {"tokens": {"input": 1, "output": 1, "total": 2}},
                    },
                ]
            ),
            stderr="",
        )

    client = OpenCodeCLIClient(executable=_fake_opencode_executable(tmp_path), runner=runner)
    with pytest.raises(LLMProviderError, match="used tools"):
        client.complete(taxonomy_json_v1_prompt("Long enough abstract text."))


@pytest.mark.parametrize(
    "events",
    [
        [
            {
                "type": "text",
                "sessionID": "session-1",
                "part": {"text": '{"label":"cs.AI"}'},
            }
        ],
        [
            {
                "type": "step_finish",
                "sessionID": "session-1",
                "part": {"tokens": {"input": 1, "output": 1, "total": 2}},
            }
        ],
        [
            {"type": "text", "sessionID": "session-1", "part": {"text": "not json"}},
            {
                "type": "step_finish",
                "sessionID": "session-1",
                "part": {"tokens": {"input": 1, "output": 1, "total": 2}},
            },
        ],
    ],
)
def test_opencode_client_rejects_malformed_or_missing_finish_text(
    tmp_path: Path, events: list[dict[str, object]]
) -> None:
    def runner(command: Sequence[str], cwd: Path, timeout_seconds: float) -> OpenCodeProcessResult:
        return OpenCodeProcessResult(returncode=0, stdout=_jsonl_stdout(events), stderr="")

    client = OpenCodeCLIClient(executable=_fake_opencode_executable(tmp_path), runner=runner)
    with pytest.raises(LLMProviderError):
        client.complete(taxonomy_json_v1_prompt("Long enough abstract text."))


def test_opencode_client_subprocess_failure_is_sanitized(tmp_path: Path) -> None:
    def runner(command: Sequence[str], cwd: Path, timeout_seconds: float) -> OpenCodeProcessResult:
        return OpenCodeProcessResult(
            returncode=2,
            stdout="raw abstract text",
            stderr="SECRET_API_KEY and provider response body",
        )

    client = OpenCodeCLIClient(executable=_fake_opencode_executable(tmp_path), runner=runner)
    with pytest.raises(LLMProviderError) as error:
        client.complete(taxonomy_json_v1_prompt("raw abstract text"))
    message = str(error.value)
    assert "raw abstract text" not in message
    assert "SECRET_API_KEY" not in message
    assert "provider response body" not in message


def test_llm_evaluation_cache_resume_and_invalid_outputs(
    trained_project: ProjectPaths,
) -> None:
    client = FakeLLMClient(invalid_first=True)
    first = evaluate_llm_split(trained_project, "validation", resume=False, client=client)
    assert first["row_count"] == 18
    assert first["invalid_output_count"] == 1
    assert first["accuracy"] < 1.0
    assert client.calls == 18

    second = evaluate_llm_split(trained_project, "validation", resume=True, client=client)
    assert second["cache_hit_count"] == 18
    assert client.calls == 18
    assert (trained_project.artifacts / "llm_validation_metrics.json").is_file()
    assert (trained_project.artifacts / "llm_prediction_trace_validation.jsonl").is_file()
    assert (trained_project.artifacts / "llm_validation_confusion_matrix.csv").is_file()


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
    assert len(cached_lines) == 17
    assert client.calls == 18

    resumed = evaluate_llm_split(
        trained_project,
        "validation",
        resume=True,
        client=client,
        runtime_config=_concurrent_fake_runtime(),
    )
    assert resumed["cache_hit_count"] == 17
    assert client.calls == 19


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
    evaluate_args = parser.parse_args(
        [
            "llm-evaluate",
            "--split",
            "validation",
            "--transport",
            "opencode",
            "--max-concurrency",
            "2",
        ]
    )
    audit_args = parser.parse_args(
        ["llm-audit", "--transport", "opencode", "--max-concurrency", "3"]
    )
    predict_args = parser.parse_args(
        ["llm-predict", "--text", "x" * 100, "--transport", "opencode"]
    )
    assert evaluate_args.transport == "opencode"
    assert evaluate_args.max_concurrency == 2
    assert audit_args.transport == "opencode"
    assert audit_args.max_concurrency == 3
    assert predict_args.transport == "opencode"
    with pytest.raises(SystemExit):
        parser.parse_args(["audit"])
