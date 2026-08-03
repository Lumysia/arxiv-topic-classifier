# arXiv Abstract Topic Classification

This repository contains public code for an ENGR 5775G course project. The task is abstract-only six-class classification over a balanced sample from the Cornell University arXiv Dataset on Kaggle.

Final held-out test comparison: TF-IDF unigram/bigram plus multinomial logistic regression reached accuracy 80.3% and macro-F1 0.804; fixed zero-shot `taxonomy_json_v1` prompting with OpenAI GPT-5.6 Sol via OpenCode reached accuracy 85.6% and macro-F1 0.849 on the same 900 test abstracts.

Raw data, processed data, trained models, generated artifacts, LLM response caches, slides, speaker notes, Q&A, submission ZIPs, and course submission materials are intentionally not included in this public code repository.

## Reproduce

After obtaining the Cornell University arXiv Dataset metadata file separately, run the pipeline with the existing CLI:

```bash
uv sync --all-groups
uv run arxiv-classifier prepare --source path/to/arxiv-metadata-oai-snapshot.json
uv run arxiv-classifier train
uv run arxiv-classifier llm-evaluate --split validation --transport opencode --max-concurrency 6
uv run arxiv-classifier llm-evaluate --split test --transport opencode --max-concurrency 6
uv run arxiv-classifier llm-audit --transport opencode --sample-size 120 --max-concurrency 2
uv run arxiv-classifier llm-report
uv run arxiv-classifier serve --llm-mode offline --llm-transport opencode
uv run ruff check .
uv run mypy src
uv run pytest
```

Use `--resume` on `llm-evaluate` or `llm-audit` to continue an interrupted OpenCode run from existing response caches. OpenCode is the actual runtime used for the prompt-LLM experiment. Live demo mode is optional; the classroom path is offline exact-cache mode with no credentials packaged in this repository.
