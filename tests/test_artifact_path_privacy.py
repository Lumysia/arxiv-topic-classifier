import subprocess
from pathlib import Path

from arxiv_topic_classifier.config import ProjectPaths

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {
    "",
    ".command",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
}
EXCLUDED_PATH_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "dist",
    "node_modules",
}
PRIVATE_MARKERS = (
    "ao" + "cro-o",
    "open" + "code",
    "deep" + "seek",
    "/opt/" + "data/",
    "/" + "Users/",
    "response" + "_id",
    "session" + "_id",
)


def test_artifact_path_is_relative_inside_project(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path / "project")
    source = paths.raw / "snapshot.json"

    assert paths.artifact_path(source) == "data/raw/snapshot.json"


def test_artifact_path_hides_external_parent_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path / "project")
    source = tmp_path / "private-user-directory" / "snapshot.json"

    assert paths.artifact_path(source) == "snapshot.json"


def test_public_text_files_exclude_private_runtime_markers() -> None:
    violations: list[str] = []
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        public_files = [
            REPOSITORY_ROOT / relative
            for relative in result.stdout.decode("utf-8").split("\0")
            if relative
        ]
    else:
        public_files = list(REPOSITORY_ROOT.rglob("*"))

    for path in public_files:
        relative = path.relative_to(REPOSITORY_ROOT)
        if not path.is_file() or any(part in EXCLUDED_PATH_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker.lower() in text:
                violations.append(f"{relative.as_posix()}: {marker}")
    assert violations == []
