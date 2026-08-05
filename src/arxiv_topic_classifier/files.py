from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def atomic_write_text(destination: Path, payload: str) -> None:
    atomic_write_bytes(destination, payload.encode("utf-8"))


def atomic_write_json(destination: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_write_text(destination, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_joblib_dump(payload: object, destination: Path, compress: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "wb",
        delete=False,
        dir=destination.parent,
        suffix=".joblib.tmp",
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(payload, temporary, compress=compress)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
