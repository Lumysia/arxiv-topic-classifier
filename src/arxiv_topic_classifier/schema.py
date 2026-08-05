from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VersionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    version: str
    created: str


class RawPaper(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    title: str
    abstract: str
    categories: str
    versions: list[VersionRecord] = Field(default_factory=list)
    update_date: str | None = None


Split = Literal["train", "validation", "test"]
EvaluationSplit = Literal["validation", "test"]


class DatasetRow(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    text: str
    labels: tuple[str, ...]
    categories: str
    year: int | None
    split: Split
    content_hash: str


class LabelProbability(BaseModel):
    label: str
    category_name: str
    probability: float
    selected: bool


class PredictionResult(BaseModel):
    labels: list[str]
    top_label: str
    top_category_name: str
    top_probability: float
    threshold: float
    probabilities: dict[str, float]
    ranked: list[LabelProbability]


class LLMUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reported_cost_usd: float | None = None


class LLMParsedPrediction(BaseModel):
    labels: tuple[str, ...]
    valid_output: bool
    reason: str | None = None
    error: str | None = None


class LLMEvaluationRecord(BaseModel):
    cache_key: str
    arxiv_id: str
    split: EvaluationSplit
    abstract_hash: str
    model: str
    prompt_version: str
    prompt_hash: str
    actual_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    valid_output: bool
    exact_match: bool
    reason: str | None = None
    parse_error: str | None = None
    latency_seconds: float
    usage: LLMUsage


class LLMAuditRecord(BaseModel):
    cache_key: str
    arxiv_id: str
    split: Literal["train", "validation"]
    abstract_hash: str
    model: str
    prompt_version: str
    prompt_hash: str
    source_primary_label: str
    source_all_categories: list[str]
    suggested_label: str | None
    alternative_labels: list[str]
    ambiguity_flag: bool | None
    short_reason: str | None
    valid_output: bool
    parse_error: str | None = None
    outcome: str
    label_mutated: Literal[False] = False
    latency_seconds: float
    usage: LLMUsage
