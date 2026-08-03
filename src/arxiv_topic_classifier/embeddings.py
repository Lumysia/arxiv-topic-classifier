from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

from arxiv_topic_classifier.config import DEFAULT_EMBEDDING_MODEL, ProjectPaths


class EmbeddingHelper(Protocol):
    model_name: str
    cache_dir: Path

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray: ...


@dataclass
class FastEmbedEmbeddingHelper:
    model_name: str
    cache_dir: Path
    threads: int = 2
    _model: TextEmbedding | None = field(default=None, init=False, repr=False)

    def _load(self) -> TextEmbedding:
        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                threads=self.threads,
            )
        return self._model

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)
        values = list(self._load().embed(texts, batch_size=batch_size))
        return np.asarray(values, dtype=float)


def default_embedding_helper(
    paths: ProjectPaths,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FastEmbedEmbeddingHelper:
    return FastEmbedEmbeddingHelper(
        model_name=model_name,
        cache_dir=paths.root / ".cache" / "fastembed",
    )
