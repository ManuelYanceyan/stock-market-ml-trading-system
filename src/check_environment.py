"""Offline pre-flight checks for a reproducible local run.

Run it before the pipeline or in a fresh clone:

    python -m src.check_environment

The checker only inspects the interpreter, installed packages and committed
artefacts; it never downloads data and never contacts a broker.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import config

MINIMUM_PYTHON = (3, 12)
REQUIRED_MODULES = (
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "yfinance",
    "requests",
    "pyarrow",
    "xgboost",
    "pytest",
    "streamlit",
    "nbformat",
)
REQUIRED_FILES = (
    "config.py",
    "requirements.txt",
    "README.md",
    "reports/metrics.json",
    "reports/pytest.xml",
)


def check_python(version: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Compare the running interpreter with the documented minimum."""
    info = tuple(version or sys.version_info[:3])
    ok = tuple(info[:2]) >= MINIMUM_PYTHON
    required = ".".join(str(part) for part in MINIMUM_PYTHON)
    return {
        "name": "python",
        "ok": ok,
        "detail": f"{'.'.join(str(part) for part in info)} (requires >= {required})",
    }


def check_modules(
    modules: tuple[str, ...] = REQUIRED_MODULES,
    importer: Callable[[str], Any] = importlib.import_module,
) -> list[dict[str, Any]]:
    """Import each dependency and report the exact failure when one is missing."""
    results: list[dict[str, Any]] = []
    for name in modules:
        try:
            importer(name)
        except Exception as exc:  # noqa: BLE001 - report the real import error
            results.append({"name": name, "ok": False, "detail": str(exc)})
        else:
            results.append({"name": name, "ok": True, "detail": "imported"})
    return results


def check_files(
    paths: tuple[str, ...] = REQUIRED_FILES,
    root: Path = config.PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Verify that required project and report files exist and are non-empty."""
    results: list[dict[str, Any]] = []
    for relative in paths:
        path = Path(root) / relative
        if not path.is_file():
            results.append({"name": relative, "ok": False, "detail": "missing"})
        elif path.stat().st_size == 0:
            results.append({"name": relative, "ok": False, "detail": "empty file"})
        else:
            results.append(
                {"name": relative, "ok": True, "detail": f"{path.stat().st_size:,} bytes"}
            )
    return results


def run_checks() -> dict[str, Any]:
    """Run every pre-flight check and return a machine-readable summary."""
    python = check_python()
    modules = check_modules()
    files = check_files()
    ok = python["ok"] and all(item["ok"] for item in modules + files)
    return {"ok": ok, "python": python, "modules": modules, "files": files}


def main() -> int:
    report = run_checks()
    for item in [report["python"], *report["modules"], *report["files"]]:
        status = "PASS" if item["ok"] else "FAIL"
        print(f"[{status}] {item['name']}: {item['detail']}")
    print("Environment OK" if report["ok"] else "Environment check failed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
