from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from arxiv_topic_classifier.artifact_package import create_artifact_package, verify_artifact_zip
from arxiv_topic_classifier.audit import run_llm_audit
from arxiv_topic_classifier.config import (
    DEFAULT_AUDIT_SIZE,
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
from arxiv_topic_classifier.representation import run_representation_benchmark
from arxiv_topic_classifier.source import SourceSpec, acquire_source, use_local_source
from arxiv_topic_classifier.verification import verify_model
from arxiv_topic_classifier.web import LLMMode, create_app
from arxiv_topic_classifier.workflow import run_full_train_package, write_processed_manifest


def _paths(value: str) -> ProjectPaths:
    return ProjectPaths.from_root(Path(value))


def _print_summary(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2))


def _source_spec_from_args(
    source: Path | None, source_sha256: str | None, source_bytes: int | None
) -> SourceSpec:
    if source is None:
        if source_sha256 is not None or source_bytes is not None:
            raise ValueError("--source-sha256 and --source-bytes are only valid with --source.")
        return SourceSpec()
    if source_sha256 is None or source_bytes is None:
        raise ValueError("--source requires explicit --source-sha256 and --source-bytes.")
    return SourceSpec(expected_bytes=source_bytes, expected_sha256=source_sha256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible arXiv topic classification pipeline")
    parser.add_argument("--root", default=".", help="Project root directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Profile source and create leakage-safe data")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    prepare.add_argument("--allow-growing-source", action="store_true")

    subparsers.add_parser("train", help="Train, select, evaluate, and serialize models")
    download = subparsers.add_parser(
        "download-source",
        help="Acquire the pinned Kaggle arXiv JSONL source",
    )
    download.add_argument("--force-download", action="store_true")

    full = subparsers.add_parser(
        "full-train",
        help="Acquire source, prepare data, train, verify, and create a local artifact ZIP",
    )
    full.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    full.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Explicit local JSONL source for tests/offline reruns",
    )
    full.add_argument("--source-sha256", default=None, help="Required when --source is used")
    full.add_argument(
        "--source-bytes",
        type=int,
        default=None,
        help="Required when --source is used",
    )
    full.add_argument("--force-download", action="store_true")
    full.add_argument("--force-prepare", action="store_true")
    full.add_argument("--force-train", action="store_true")
    full.add_argument("--include-processed-csv", action="store_true")

    verify = subparsers.add_parser("verify-model", help="Run strict model verification")
    verify.add_argument("--source", type=Path, default=None)
    verify.add_argument("--source-sha256", default=None)
    verify.add_argument("--source-bytes", type=int, default=None)
    verify.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)

    package = subparsers.add_parser(
        "package-artifacts",
        help="Create and verify the compact local artifact ZIP",
    )
    package.add_argument("--include-processed-csv", action="store_true")
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

    llm_audit = subparsers.add_parser(
        "llm-audit", help="Run development-only prompt-based label audit"
    )
    llm_audit.add_argument("--sample-size", type=int, default=DEFAULT_AUDIT_SIZE)
    llm_audit.add_argument("--resume", action="store_true")

    llm_predict = subparsers.add_parser("llm-predict", help="Predict one abstract with LLM")
    llm_predict.add_argument("--text", required=True)

    llm_report = subparsers.add_parser(
        "llm-report", help="Build final paired classical-vs-LLM test reporting artifacts"
    )
    llm_report.add_argument(
        "--expected-test-rows",
        type=int,
        default=None,
        help="Expected held-out test row count for regenerated multilabel LLM reporting.",
    )

    serve = subparsers.add_parser("serve", help="Run local web demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument(
        "--llm-mode",
        choices=("offline", "live", "live-with-cache-fallback"),
        default="offline",
        help="LLM runtime mode for the comparison demo. Defaults to offline.",
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
    elif args.command == "download-source":
        status = acquire_source(paths, force_download=args.force_download)
        _print_summary(status.metadata(SourceSpec()))
    elif args.command == "full-train":
        source_spec = _source_spec_from_args(args.source, args.source_sha256, args.source_bytes)
        result = run_full_train_package(
            paths,
            per_class=args.per_class,
            source=args.source,
            source_spec=source_spec,
            force_download=args.force_download,
            force_prepare=args.force_prepare,
            force_train=args.force_train,
            include_processed_csv=args.include_processed_csv,
        )
        _print_summary(result.summary())
    elif args.command == "verify-model":
        source_spec = _source_spec_from_args(args.source, args.source_sha256, args.source_bytes)
        source_status = (
            use_local_source(paths, args.source, source_spec)
            if args.source is not None
            else acquire_source(paths, source_spec)
        )
        processed_manifest = write_processed_manifest(
            paths,
            source_status,
            args.per_class,
            RANDOM_SEED,
        )
        _print_summary(verify_model(paths, processed_manifest))
    elif args.command == "package-artifacts":
        processed_manifest = json.loads(
            (paths.artifacts / "processed_manifest.json").read_text(encoding="utf-8")
        )
        verification_report = verify_model(paths, processed_manifest)
        zip_path = create_artifact_package(
            paths,
            processed_manifest=processed_manifest,
            verification_report=verification_report,
            include_processed_csv=args.include_processed_csv,
        )
        _print_summary(verify_artifact_zip(zip_path))
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
        _print_summary(evaluate_llm_split(paths, args.split, resume=args.resume))
    elif args.command == "llm-audit":
        _print_summary(run_llm_audit(paths, args.sample_size, resume=args.resume))
    elif args.command == "llm-predict":
        _print_summary(predict_llm_text(args.text))
    elif args.command == "llm-report":
        _print_summary(run_final_report(paths, expected_test_rows=args.expected_test_rows))
    elif args.command == "serve":
        llm_mode = cast(LLMMode, args.llm_mode)
        serve_llm_client: LLMClient | None = None
        if llm_mode in {"live", "live-with-cache-fallback"}:
            serve_llm_client = OpenAICompatibleChatClient(
                parse_llm_runtime_config(require_credentials=True)
            )
        create_app(
            paths.root,
            llm_mode=llm_mode,
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
