from __future__ import annotations

import pytest

from arxiv_topic_classifier.config import CATEGORY_DESCRIPTIONS, ProjectPaths
from arxiv_topic_classifier.data import write_dataset
from arxiv_topic_classifier.model import train_and_evaluate
from arxiv_topic_classifier.schema import DatasetRow
from arxiv_topic_classifier.text import content_fingerprint


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    root = tmp_path_factory.mktemp("trained-project")
    paths = ProjectPaths.from_root(root)
    paths.create_output_dirs()
    keywords = {
        "cs.AI": "agents reasoning planning knowledge",
        "cs.CL": "language translation text linguistic",
        "cs.CV": "image vision object segmentation",
        "cs.LG": "learning neural optimization generalization",
        "cs.RO": "robot navigation manipulation control",
        "cs.SE": "software testing debugging maintenance",
    }
    rows: list[DatasetRow] = []
    for label in CATEGORY_DESCRIPTIONS:
        for index in range(20):
            title = f"Study {index} of {keywords[label]}"
            abstract = (
                f"This paper investigates {keywords[label]} with a reproducible experimental "
                f"method and detailed benchmark evaluation number {index}. Results demonstrate "
                "the strengths and limitations of the proposed approach for this research area."
            )
            split = "train" if index < 14 else "validation" if index < 17 else "test"
            rows.append(
                DatasetRow(
                    arxiv_id=f"{label}-{index}",
                    title=title,
                    abstract=abstract,
                    text=abstract,
                    label=label,
                    categories=label,
                    year=2025,
                    split=split,
                    content_hash=content_fingerprint(abstract),
                )
            )
    write_dataset(rows, paths.dataset)
    train_and_evaluate(paths)
    return paths
