from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED = 5775
TEXT_MIN_LENGTH = 80
MAX_INPUT_LENGTH = 12_000
DEFAULT_PER_CLASS = 1_000
DEFAULT_AUDIT_SIZE = 120
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_OPENCODE_MODEL = "openai/gpt-5.6-sol"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_CONCURRENCY = 4
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_USER_AGENT = "arxiv-topic-classifier/0.1"

LLM_BASE_URL_ENV = "ARXIV_LLM_BASE_URL"
LLM_API_KEY_ENV = "ARXIV_LLM_API_KEY"
LLM_MODEL_ENV = "ARXIV_LLM_MODEL"
LLM_TIMEOUT_SECONDS_ENV = "ARXIV_LLM_TIMEOUT_SECONDS"
LLM_MAX_CONCURRENCY_ENV = "ARXIV_LLM_MAX_CONCURRENCY"
LLM_MAX_RETRIES_ENV = "ARXIV_LLM_MAX_RETRIES"
LLM_USER_AGENT_ENV = "ARXIV_LLM_USER_AGENT"

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "cs.AI": (
        "Artificial Intelligence: intelligent agents, reasoning, planning, knowledge "
        "representation, search, decision making, multi-agent systems and general AI methods."
    ),
    "cs.CL": (
        "Computation and Language: natural language processing, language models, machine "
        "translation, text generation, linguistic analysis, speech and dialogue."
    ),
    "cs.CV": (
        "Computer Vision and Pattern Recognition: image and video understanding, object "
        "detection, segmentation, visual recognition, image generation and 3D vision."
    ),
    "cs.LG": (
        "Machine Learning: supervised, unsupervised and reinforcement learning, neural network "
        "optimization, generalization, probabilistic learning and learning theory."
    ),
    "cs.RO": (
        "Robotics: robot perception, localization, mapping, motion planning, manipulation, "
        "navigation, control and autonomous physical systems."
    ),
    "cs.SE": (
        "Software Engineering: software development, testing, debugging, maintenance, program "
        "analysis, requirements, software architecture and developer tools."
    ),
}

CATEGORY_NAMES: dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.RO": "Robotics",
    "cs.SE": "Software Engineering",
}
TARGET_CATEGORIES = tuple(CATEGORY_DESCRIPTIONS)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @property
    def cache(self) -> Path:
        return self.artifacts / "cache"

    @property
    def model(self) -> Path:
        return self.root / "models" / "arxiv_topic_pipeline.joblib"

    @property
    def dataset(self) -> Path:
        return self.processed / "arxiv_topics.csv"

    @property
    def profile(self) -> Path:
        return self.artifacts / "data_profile.json"

    def llm_metrics(self, split: str) -> Path:
        return self.artifacts / f"llm_{split}_metrics.json"

    def llm_prediction_trace(self, split: str) -> Path:
        return self.artifacts / f"llm_prediction_trace_{split}.jsonl"

    def llm_cache(self, split: str) -> Path:
        return self.cache / f"llm_{split}_responses.jsonl"

    def llm_class_metrics(self, split: str) -> Path:
        return self.artifacts / f"llm_{split}_class_metrics.csv"

    def llm_confusion_matrix(self, split: str) -> Path:
        return self.artifacts / f"llm_{split}_confusion_matrix.csv"

    def llm_error_patterns(self, split: str) -> Path:
        return self.artifacts / f"llm_{split}_error_patterns.json"

    def llm_prediction_examples(self, split: str) -> Path:
        return self.artifacts / f"llm_{split}_prediction_examples.json"

    def llm_confusion_figure(self, split: str) -> Path:
        return self.figures / f"llm_{split}_confusion_matrix.png"

    @property
    def llm_audit_summary(self) -> Path:
        return self.artifacts / "llm_audit_summary.json"

    @property
    def llm_label_audit(self) -> Path:
        return self.artifacts / "llm_label_audit.jsonl"

    @property
    def llm_audit_cache(self) -> Path:
        return self.cache / "llm_audit_responses.jsonl"

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        return cls(root=root.resolve())

    def create_output_dirs(self) -> None:
        for path in (self.processed, self.artifacts, self.figures, self.cache, self.model.parent):
            path.mkdir(parents=True, exist_ok=True)
