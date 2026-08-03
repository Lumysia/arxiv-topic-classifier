from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from arxiv_topic_classifier.config import DEFAULT_LLM_TIMEOUT_SECONDS, DEFAULT_OPENCODE_MODEL
from arxiv_topic_classifier.llm import (
    LLMChatPrompt,
    LLMClientResponse,
    LLMConfigError,
    LLMProviderError,
)
from arxiv_topic_classifier.schema import LLMUsage

_TRANSPORT_INSTRUCTION = (
    "Transport instruction: do not use tools, read files, write files, inspect the working "
    "directory, or access the network. Return exactly one JSON object and no surrounding prose."
)
_TOOL_EVENT_TYPES = {
    "tool",
    "tool-call",
    "tool-result",
    "tool_call",
    "tool_result",
    "tool_start",
    "tool_finish",
}
_ALLOWED_ENV_KEYS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "TMPDIR",
    "USER",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}


@dataclass(frozen=True)
class OpenCodeProcessResult:
    returncode: int
    stdout: str
    stderr: str


class OpenCodeSubprocessRunner(Protocol):
    def __call__(
        self, command: Sequence[str], cwd: Path, timeout_seconds: float
    ) -> OpenCodeProcessResult:
        pass


class OpenCodeCLIClient:
    transport = "opencode-cli"

    def __init__(
        self,
        model: str = DEFAULT_OPENCODE_MODEL,
        executable: str = "opencode",
        variant: str = "high",
        work_dir: Path | None = None,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        runner: OpenCodeSubprocessRunner | None = None,
    ) -> None:
        if not model.strip():
            raise LLMConfigError("OpenCode model must not be empty.")
        if not executable.strip():
            raise LLMConfigError("OpenCode executable must not be empty.")
        if timeout_seconds <= 0:
            raise LLMConfigError("OpenCode timeout must be greater than zero.")
        self.model = model
        self._executable = executable
        self._variant = variant
        self._work_dir = work_dir or Path(tempfile.gettempdir()) / "arxiv-topic-classifier-opencode"
        self._timeout_seconds = timeout_seconds
        self._runner = runner or _run_opencode_subprocess

    def complete(self, prompt: LLMChatPrompt) -> LLMClientResponse:
        executable = _resolve_executable(self._executable)
        base_work_dir = self._work_dir.resolve()
        base_work_dir.mkdir(parents=True, exist_ok=True)
        request_work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=base_work_dir))
        command = [
            executable,
            "run",
            "--model",
            self.model,
            "--variant",
            self._variant,
            "--format",
            "json",
            _render_prompt(prompt),
        ]
        started = time.perf_counter()
        try:
            result = self._runner(command, request_work_dir, self._timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise LLMProviderError("OpenCode CLI request timed out.") from error
        except OSError as error:
            raise LLMProviderError("OpenCode CLI request failed to start.") from error
        finally:
            shutil.rmtree(request_work_dir, ignore_errors=True)
        latency_seconds = time.perf_counter() - started
        if result.returncode != 0:
            raise LLMProviderError(f"OpenCode CLI failed with exit code {result.returncode}.")
        return _parse_jsonl_events(result.stdout, self.model, latency_seconds)


def _resolve_executable(executable: str) -> str:
    path = Path(executable)
    if path.parent != Path("."):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise LLMConfigError("OpenCode executable was not found.")
    resolved = shutil.which(executable)
    if resolved is None:
        raise LLMConfigError("OpenCode executable was not found on PATH.")
    return resolved


def _run_opencode_subprocess(
    command: Sequence[str], cwd: Path, timeout_seconds: float
) -> OpenCodeProcessResult:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=_clean_environment(os.environ),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return OpenCodeProcessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _clean_environment(source: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in source.items() if key in _ALLOWED_ENV_KEYS}


def _render_prompt(prompt: LLMChatPrompt) -> str:
    messages = "\n\n".join(
        f"{message.role.upper()}:\n{message.content}" for message in prompt.messages
    )
    return f"{_TRANSPORT_INSTRUCTION}\n\n{messages}"


def _parse_jsonl_events(
    stdout: str, expected_model: str, latency_seconds: float
) -> LLMClientResponse:
    text_parts: list[str] = []
    finish_part: dict[str, Any] | None = None
    session_id: str | None = None
    reported_model: str | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise LLMProviderError("OpenCode CLI emitted malformed JSONL.") from error
        if not isinstance(event, dict):
            raise LLMProviderError("OpenCode CLI emitted a malformed event.")
        event_type = event.get("type")
        if isinstance(event_type, str) and event_type in _TOOL_EVENT_TYPES:
            raise LLMProviderError("OpenCode CLI used tools during the run.")
        event_session_id = _event_session_id(event)
        if event_session_id is not None:
            if session_id is not None and session_id != event_session_id:
                raise LLMProviderError("OpenCode CLI emitted conflicting session IDs.")
            session_id = event_session_id
        event_model = _event_model(event)
        if event_model is not None:
            reported_model = event_model
        if event_type == "text":
            text_parts.append(_text_part(event))
        elif event_type == "step_finish":
            if finish_part is not None:
                raise LLMProviderError("OpenCode CLI emitted multiple step finish events.")
            finish_part = _finish_part(event)
    if session_id is None:
        raise LLMProviderError("OpenCode CLI response is missing a session ID.")
    if finish_part is None:
        raise LLMProviderError("OpenCode CLI response is missing a step finish event.")
    finish_model = _event_model(finish_part)
    reported_model = finish_model or reported_model
    if reported_model is not None and reported_model != expected_model:
        raise LLMProviderError("OpenCode CLI selected an unexpected model.")
    content = _single_json_object_text("".join(text_parts))
    return LLMClientResponse(
        content=content,
        usage=_usage_from_finish(finish_part),
        latency_seconds=latency_seconds,
        response_id=session_id,
    )


def _event_session_id(event: dict[str, Any]) -> str | None:
    raw_value = event.get("sessionID", event.get("sessionId"))
    return raw_value if isinstance(raw_value, str) and raw_value else None


def _event_model(event: dict[str, Any]) -> str | None:
    for key in ("model", "selectedModel", "selected_model"):
        raw_value = event.get(key)
        if isinstance(raw_value, str) and raw_value:
            return raw_value
    part = event.get("part")
    if isinstance(part, dict):
        for key in ("model", "selectedModel", "selected_model"):
            raw_value = part.get(key)
            if isinstance(raw_value, str) and raw_value:
                return raw_value
    return None


def _text_part(event: dict[str, Any]) -> str:
    part = event.get("part")
    if not isinstance(part, dict) or not isinstance(part.get("text"), str):
        raise LLMProviderError("OpenCode CLI text event is malformed.")
    return cast(str, part["text"])


def _finish_part(event: dict[str, Any]) -> dict[str, Any]:
    part = event.get("part")
    if not isinstance(part, dict):
        raise LLMProviderError("OpenCode CLI step finish event is malformed.")
    status = part.get("status", part.get("state"))
    if isinstance(status, str) and status.lower() in {"error", "failed", "cancelled"}:
        raise LLMProviderError("OpenCode CLI step did not finish successfully.")
    return cast(dict[str, Any], part)


def _single_json_object_text(text: str) -> str:
    content = text.strip()
    if not content:
        raise LLMProviderError("OpenCode CLI response is missing assistant text.")
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(content)
    except json.JSONDecodeError as error:
        raise LLMProviderError("OpenCode CLI assistant text is not one JSON object.") from error
    if not isinstance(value, dict) or content[end:].strip():
        raise LLMProviderError("OpenCode CLI assistant text is not one JSON object.")
    return content


def _usage_from_finish(finish_part: dict[str, Any]) -> LLMUsage:
    tokens = finish_part.get("tokens")
    if not isinstance(tokens, dict):
        return LLMUsage(reported_cost_usd=_optional_float(finish_part.get("cost")))
    input_tokens = _first_int(tokens, ("input", "input_tokens", "prompt_tokens"))
    output_tokens = _first_int(tokens, ("output", "output_tokens", "completion_tokens"))
    total_tokens = _first_int(tokens, ("total", "total_tokens"))
    if input_tokens is None or output_tokens is None or total_tokens is None:
        raise LLMProviderError("OpenCode CLI token usage is incomplete.")
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=_first_int(tokens, ("reasoning", "reasoning_tokens")),
        cache_read_tokens=_cache_token(tokens, "read"),
        cache_write_tokens=_cache_token(tokens, "write"),
        reported_cost_usd=_optional_float(finish_part.get("cost")),
    )


def _first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        raw_value = payload.get(key)
        if isinstance(raw_value, int):
            return raw_value
    return None


def _cache_token(tokens: Mapping[str, Any], name: str) -> int | None:
    direct = _first_int(tokens, (f"cache_{name}", f"cache_{name}_tokens"))
    if direct is not None:
        return direct
    cache = tokens.get("cache")
    if isinstance(cache, dict):
        raw_value = cache.get(name)
        if isinstance(raw_value, int):
            return raw_value
        raw_tokens = cache.get(f"{name}_tokens")
        if isinstance(raw_tokens, int):
            return raw_tokens
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
