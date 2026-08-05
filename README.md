# arXiv Abstract Topic Classification

This is a directly runnable local demo for abstract-only multilabel classification over the current official 155-label arXiv taxonomy. It packages the trained classical model at `models/arxiv_topic_pipeline.joblib` through Git LFS, plus a narrow set of public demo JSON artifacts used by the web page.

The model was trained from the Cornell University arXiv metadata dataset on Kaggle (`Cornell-University/arxiv`, CC0: Public Domain). Titles are not model input. Each target label set is the intersection of a paper's complete source categories with the official taxonomy.

## What Is Packaged

The public demo includes:

- source code, tests, packaging metadata, README, and license
- the Git LFS tracked trained model and its manifest
- `artifacts/data_profile.json`, `artifacts/metrics.json`, `artifacts/demo_selection.json`, and `artifacts/demo_llm_cache.json`
- `run_demo.command` as a macOS launcher

Raw arXiv data, processed training datasets, slides, private materials, screenshots, full LLM traces, broad generated outputs, reports, archives, credentials, and local QA/browser tooling remain excluded by `.gitignore`.

## Install

Prerequisites:

- Git LFS
- `uv`

Install Git LFS once on your machine:

```bash
git lfs install
```

Clone normally. Git LFS downloads the packaged model during clone when LFS is installed:

```bash
git clone <repository-url>
cd arxiv-topic-classifier
git lfs pull
uv sync --locked
```

`git lfs pull` is safe to run after clone or after pulling updates. It ensures `models/arxiv_topic_pipeline.joblib` is the real model file, not an LFS pointer.

## Run The Demo

On macOS, double-click `run_demo.command`. The launcher syncs the locked environment, runs the offline smoke check, starts the local server, and opens `http://127.0.0.1:5000`.

For the cross-platform command line path:

```bash
uv run arxiv-classifier smoke
uv run arxiv-classifier serve --llm-mode offline
```

Then open `http://127.0.0.1:5000`.

You can also run a single local prediction without the server:

```bash
uv run arxiv-classifier predict --text "This paper presents a reproducible neural method for multilingual document classification using transformer language models and benchmark evaluation across several datasets."
```

## Demo Behavior

The classical classifier runs locally for any valid abstract you enter. It uses the packaged 155-label TF-IDF one-vs-rest logistic regression pipeline and returns a fresh prediction at request time.

The three LLM examples are exact documented held-out abstract excerpts. In offline mode, an LLM response is shown only when the normalized abstract exactly matches one of those cached examples. Arbitrary abstracts still receive a fresh local classical prediction, but the LLM panel explicitly reports that no exact cached response is available. The app does not fabricate an LLM result.

## Reported Classical Run

The packaged metrics describe a bounded, deterministic sample of 149,685 records:

- 104,780 training rows
- 22,453 validation rows
- 22,452 held-out test rows

Abstract whitespace is normalized, version-aware arXiv IDs are grouped before splitting, and normalized-content hashes are used to prevent duplicate leakage across splits. The packaged profile reports zero duplicate IDs and zero duplicate abstracts across splits.

Train-only unigram/bigram TF-IDF features feed one-vs-rest multilabel classifiers. Validation macro F1 selected logistic regression over multinomial naive Bayes; the test split was then used once for final classical evaluation. The packaged logistic regression test metrics are macro F1 0.4504, micro F1 0.4470, hamming loss 0.02852, and exact-match ratio 0.01880 across all 22,452 held-out test rows.

## Rebuild Locally

The full source dataset is intentionally not committed. To rebuild from Kaggle, configure Kaggle access for `kagglehub`; the pinned source is Kaggle dataset version 297, file `arxiv-metadata-oai-snapshot.json`, expected size `5,437,318,298` bytes, and SHA-256 `fabd00ed721608c57d0e3b9b9cf798062328771c6e6cfcc08cb63d9eb434d43f`.

```bash
uv run arxiv-classifier download-source
uv run arxiv-classifier prepare --source data/raw/arxiv-metadata-oai-snapshot.json
uv run arxiv-classifier train
```

Rebuild outputs are local generated files under ignored paths such as `data/`, `models/`, `artifacts/`, and `archives/`. The public package intentionally unignores only the exact demo model, manifest, demo JSON files, and launcher listed above.
