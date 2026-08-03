from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from arxiv_topic_classifier.audit import run_llm_audit
from arxiv_topic_classifier.config import (
    DEFAULT_AUDIT_SIZE,
    DEFAULT_LLM_MAX_CONCURRENCY,
    DEFAULT_PER_CLASS,
    RANDOM_SEED,
    ProjectPaths,
)
from arxiv_topic_classifier.data import prepare_dataset
from arxiv_topic_classifier.final_reporting import run_final_report
from arxiv_topic_classifier.llm import (
    LLMClient,
    OpenAICompatibleChatClient,
    parse_llm_runtime_config,
)
from arxiv_topic_classifier.llm_evaluation import evaluate_llm_split, predict_llm_text
from arxiv_topic_classifier.model import load_artifact, predict_abstract, train_and_evaluate
from arxiv_topic_classifier.opencode_client import OpenCodeCLIClient
from arxiv_topic_classifier.representation import run_representation_benchmark
from arxiv_topic_classifier.web import LLMMode, LLMTransport, create_app


def _paths(value: str) -> ProjectPaths:
    return ProjectPaths.from_root(Path(value))


def _print_summary(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2))


def _add_llm_transport_options(parser: argparse.ArgumentParser, include_concurrency: bool) -> None:
    parser.add_argument(
        "--transport",
        choices=("openai", "opencode"),
        default="openai",
        help=(
            "LLM transport: openai uses the environment-configured OpenAI-compatible API; "
            "opencode uses the local OpenCode CLI."
        ),
    )
    if include_concurrency:
        parser.add_argument(
            "--max-concurrency",
            type=int,
            default=DEFAULT_LLM_MAX_CONCURRENCY,
            help="Maximum worker count for OpenCode CLI transport runs.",
        )


def _opencode_max_concurrency(value: int) -> int:
    if value < 1:
        raise ValueError("--max-concurrency must be at least 1.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible arXiv topic classification pipeline")
    parser.add_argument("--root", default=".", help="Project root directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Profile source and create leakage-safe data")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    prepare.add_argument("--allow-growing-source", action="store_true")

    subparsers.add_parser("train", help="Train, select, evaluate, and serialize models")
    subparsers.add_parser(
        "semantic-benchmark",
        help="Run BGE embedding representation benchmark without changing deployment selection",
    )
    all_command = subparsers.add_parser(
        "all", help="Run offline data preparation and classical model evaluation"
    )
    all_command.add_argument("--source", required=True, type=Path)
    all_command.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    all_command.add_argument("--allow-growing-source", action="store_true")

    predict = subparsers.add_parser("predict", help="Predict one abstract")
    predict.add_argument("--text", required=True)

    llm_evaluate = subparsers.add_parser(
        "llm-evaluate", help="Evaluate prompt-based generative LLM classifier"
    )
    llm_evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    llm_evaluate.add_argument("--resume", action="store_true")
    _add_llm_transport_options(llm_evaluate, include_concurrency=True)

    llm_audit = subparsers.add_parser(
        "llm-audit", help="Run development-only prompt-based label audit"
    )
    llm_audit.add_argument("--sample-size", type=int, default=DEFAULT_AUDIT_SIZE)
    llm_audit.add_argument("--resume", action="store_true")
    _add_llm_transport_options(llm_audit, include_concurrency=True)

    llm_predict = subparsers.add_parser("llm-predict", help="Predict one abstract with LLM")
    llm_predict.add_argument("--text", required=True)
    _add_llm_transport_options(llm_predict, include_concurrency=False)

    llm_report = subparsers.add_parser(
        "llm-report", help="Build final paired classical-vs-LLM test reporting artifacts"
    )
    llm_report.add_argument(
        "--expected-test-rows",
        type=int,
        default=900,
        help="Expected held-out test row count. Defaults to the final 900-row test split.",
    )

    serve = subparsers.add_parser("serve", help="Run local web demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument(
        "--llm-mode",
        choices=("offline", "live", "live-with-cache-fallback"),
        default="offline",
        help="LLM runtime mode for the comparison demo. Defaults to offline for packaged use.",
    )
    serve.add_argument(
        "--llm-transport",
        choices=("openai", "opencode"),
        default="openai",
        help="Live LLM transport used when --llm-mode performs live calls.",
    )

    subparsers.add_parser("smoke", help="Run demo health and prediction smoke checks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = _paths(args.root)
    if args.command == "prepare":
        _print_summary(
            prepare_dataset(
                args.source,
                paths,
                args.per_class,
                RANDOM_SEED,
                args.allow_growing_source,
            )
        )
    elif args.command == "train":
        _print_summary(train_and_evaluate(paths))
    elif args.command == "semantic-benchmark":
        _print_summary(run_representation_benchmark(paths))
    elif args.command == "all":
        profile = prepare_dataset(
            args.source,
            paths,
            args.per_class,
            RANDOM_SEED,
            args.allow_growing_source,
        )
        metrics = train_and_evaluate(paths)
        _print_summary(
            {
                "profile_rows": profile["sample"]["total_rows"],
                "selected_model": metrics["selected_model"],
            }
        )
    elif args.command == "predict":
        _print_summary(predict_abstract(load_artifact(paths.model), args.text).model_dump())
    elif args.command == "llm-evaluate":
        if args.transport == "opencode":
            _print_summary(
                evaluate_llm_split(
                    paths,
                    args.split,
                    resume=args.resume,
                    client=OpenCodeCLIClient(),
                    max_concurrency=_opencode_max_concurrency(args.max_concurrency),
                )
            )
        else:
            _print_summary(evaluate_llm_split(paths, args.split, resume=args.resume))
    elif args.command == "llm-audit":
        if args.transport == "opencode":
            _print_summary(
                run_llm_audit(
                    paths,
                    args.sample_size,
                    resume=args.resume,
                    client=OpenCodeCLIClient(),
                    max_concurrency=_opencode_max_concurrency(args.max_concurrency),
                )
            )
        else:
            _print_summary(run_llm_audit(paths, args.sample_size, resume=args.resume))
    elif args.command == "llm-predict":
        llm_client = OpenCodeCLIClient() if args.transport == "opencode" else None
        _print_summary(predict_llm_text(args.text, client=llm_client))
    elif args.command == "llm-report":
        _print_summary(run_final_report(paths, expected_test_rows=args.expected_test_rows))
    elif args.command == "serve":
        llm_mode = cast(LLMMode, args.llm_mode)
        llm_transport = cast(LLMTransport, args.llm_transport)
        serve_llm_client: LLMClient | None = None
        if llm_mode in {"live", "live-with-cache-fallback"}:
            serve_llm_client = (
                OpenCodeCLIClient()
                if llm_transport == "opencode"
                else OpenAICompatibleChatClient(parse_llm_runtime_config(require_credentials=True))
            )
        create_app(
            paths.root,
            llm_mode=llm_mode,
            llm_transport=llm_transport,
            llm_client=serve_llm_client,
        ).run(host=args.host, port=args.port, debug=False)
    elif args.command == "smoke":
        app = create_app(paths.root)
        client = app.test_client()
        health = client.get("/health")
        sample_text = (
            "We introduce a neural language model for translating documents across languages. "
            "The model uses an attention architecture and is evaluated on multilingual text "
            "benchmarks, improving translation quality and linguistic representation learning."
        )
        prediction = client.post("/api/predict", json={"abstract": sample_text})
        if health.status_code != 200 or prediction.status_code != 200:
            raise RuntimeError(
                "Smoke test failed: "
                f"health={health.status_code}, prediction={prediction.status_code}"
            )
        _print_summary(
            {"health": health.get_json(), "prediction": prediction.get_json(), "status": "passed"}
        )


if __name__ == "__main__":
    main()
