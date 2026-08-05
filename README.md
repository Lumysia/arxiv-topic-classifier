# arXiv Abstract Topic Classification

This repository contains code for abstract-only multilabel classification over the current official 155-label arXiv taxonomy. It uses the Cornell University arXiv Dataset on Kaggle (`Cornell-University/arxiv`, CC0: Public Domain) and excludes titles from model input.

## Scope

The reported run used a bounded, deterministic sample of 149,685 records:

- 104,780 training rows
- 22,453 validation rows
- 22,452 held-out test rows

For each record, the target label set is the intersection of the paper's complete source category list with the official 155-label taxonomy. Abstract whitespace is normalized, version-aware arXiv IDs are grouped before splitting, and normalized-content hashes are used to prevent duplicate leakage across splits. The reported profile found zero duplicate IDs and zero duplicate abstracts across splits.

Train-only unigram/bigram TF-IDF features feed one-vs-rest multilabel classifiers. Validation macro F1 selected logistic regression over multinomial naive Bayes; the test split was then used once for final classical evaluation. The reported logistic regression test metrics are macro F1 0.4504, micro F1 0.4470, hamming loss 0.02852, and exact-match ratio 0.01880 across all 22,452 held-out test rows.

The repository also includes code for prompt-based LLM comparison using the same abstract-only input and official taxonomy. Generated LLM traces, caches, and reports are local outputs and are not committed.

## Repository Contents

Committed files are limited to source code, tests, packaging metadata, the public README, and the AGPL-3.0-only license. The repository does not include raw data, processed data, trained models, metrics artifacts, caches, slides, screenshots, local launchers, archives, or generated reports.

Generated outputs are written under ignored local directories such as `data/`, `models/`, `artifacts/`, and `archives/`. These paths are reproducibility outputs, not packaged repository contents.

## Setup

Install [`uv`](https://docs.astral.sh/uv/) and sync the locked environment:

```bash
uv sync --locked
```

The command-line entry point is available through `uv run`:

```bash
uv run arxiv-classifier --help
```

## Reproduce The Classical Run

The full source file is not committed. To rebuild from Kaggle, configure Kaggle access for `kagglehub`; the pinned source is Kaggle dataset version 297, file `arxiv-metadata-oai-snapshot.json`, expected size `5,437,318,298` bytes, and SHA-256 `fabd00ed721608c57d0e3b9b9cf798062328771c6e6cfcc08cb63d9eb434d43f`.

```bash
uv run arxiv-classifier download-source
uv run arxiv-classifier prepare --source data/raw/arxiv-metadata-oai-snapshot.json
uv run arxiv-classifier train
```

The rebuild writes local data profiles, metrics, figures, model manifests, and the trained model under ignored output directories. After a model has been generated, local inference and the web app can be run with:

```bash
uv run arxiv-classifier predict --text "$(printf 'This paper presents a reproducible neural method for multilingual document classification using transformer language models and benchmark evaluation across several datasets.')"
uv run arxiv-classifier serve
```

## Developer Checks

```bash
git diff --check
uv lock --check
uv run ruff check .
uv run mypy src
uv run pytest -q
uv build
uv run arxiv-classifier --help
```
