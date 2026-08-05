from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from arxiv_topic_classifier.files import atomic_write_json

DEPENDENCY_NAMES = ("numpy", "scipy", "scikit-learn", "joblib")


def dependency_versions() -> dict[str, str]:
    return {name: version(name) for name in DEPENDENCY_NAMES}


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1]) * 1024
                    except ValueError:
                        return None
    return None


def capture_environment() -> dict[str, Any]:
    return {
        "format_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
            "memory_bytes": _memory_bytes(),
        },
        "dependencies": dependency_versions(),
    }


def write_environment(path: Path) -> dict[str, Any]:
    payload = capture_environment()
    atomic_write_json(path, payload)
    return payload
