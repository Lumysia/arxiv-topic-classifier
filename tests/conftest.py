from __future__ import annotations

import pytest

from arxiv_topic_classifier.config import CATEGORY_NAMES, ProjectPaths
from arxiv_topic_classifier.data import write_dataset
from arxiv_topic_classifier.files import atomic_write_json, sha256_file
from arxiv_topic_classifier.model import train_and_evaluate
from arxiv_topic_classifier.schema import DatasetRow
from arxiv_topic_classifier.taxonomy import EXPECTED_TAXONOMY_SIZE, taxonomy_fingerprint
from arxiv_topic_classifier.text import content_fingerprint


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory: pytest.TempPathFactory) -> ProjectPaths:
    root = tmp_path_factory.mktemp("trained-project")
    paths = ProjectPaths.from_root(root)
    paths.create_output_dirs()
    special_keywords = {
        "cs.AI": "agents reasoning planning knowledge",
        "cs.CL": "language translation text linguistic",
        "cs.CV": "image vision object segmentation",
        "cs.LG": "learning neural optimization generalization",
        "cs.RO": "robot navigation manipulation control",
        "cs.SE": "software testing debugging maintenance",
    }
    rows: list[DatasetRow] = []
    for label, name in CATEGORY_NAMES.items():
        terms = special_keywords.get(label, f"{name.lower()} {label.lower()} research")
        for index in range(5):
            title = f"Study {index} of {terms}"
            abstract = (
                f"This paper investigates {terms} with a reproducible experimental "
                f"method and detailed benchmark evaluation number {index}. Results demonstrate "
                "the strengths and limitations of the proposed approach for this research area."
            )
            split = "train" if index < 3 else "validation" if index == 3 else "test"
            rows.append(
                DatasetRow(
                    arxiv_id=f"{label}-{index}",
                    title=title,
                    abstract=abstract,
                    text=abstract,
                    labels=(label,),
                    categories=label,
                    year=2025,
                    split=split,
                    content_hash=content_fingerprint(abstract),
                )
            )
    write_dataset(rows, paths.dataset)
    processed_manifest = {
        "task": "official_arxiv_taxonomy_multilabel",
        "source_sha256": "synthetic",
        "source_bytes": 0,
        "taxonomy_sha256": taxonomy_fingerprint(),
        "labels": list(CATEGORY_NAMES),
        "label_count": EXPECTED_TAXONOMY_SIZE,
        "per_label_cap": 5,
        "seed": 5775,
        "dataset_path": "data/processed/arxiv_topics.csv",
        "dataset_sha256": sha256_file(paths.dataset),
        "profile_path": "artifacts/data_profile.json",
        "profile_sha256": "synthetic",
        "row_count": len(rows),
    }
    atomic_write_json(paths.artifacts / "processed_manifest.json", processed_manifest)
    train_and_evaluate(paths, processed_manifest=processed_manifest)
    return paths
