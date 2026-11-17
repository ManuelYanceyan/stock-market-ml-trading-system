"""Offline consistency checks for the committed report artefacts.

Run after regenerating reports (and in CI) to catch a metrics file whose
strategy sets, session windows or referenced figures no longer agree:

    python -m src.validate_reports

The checker never retrains a model and never contacts the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import config
from src.report import select_strategy
from src.utils import read_json

REQUIRED_TOP_LEVEL = (
    "meta",
    "dataset",
    "best_model",
    "classification",
    "backtests_test",
    "backtests_validation",
    "benchmarks",
    "latest_signal",
    "artifacts",
)
REQUIRED_BENCHMARKS = (
    "spy_open_to_close_net",
    "spy_open_to_close_gross",
    "spy_buy_and_hold_close_to_close",
)
REQUIRED_PERFORMANCE_FIELDS = ("sessions", "total_return", "cagr", "sharpe_ratio", "max_drawdown")


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Return a list of consistency errors (empty when the payload is sound)."""
    if not isinstance(payload, dict):
        return ["metrics payload must be a JSON object"]
    missing = [key for key in REQUIRED_TOP_LEVEL if key not in payload]
    if missing:
        return [f"missing top-level keys: {', '.join(missing)}"]

    errors: list[str] = []
    test_results = payload["backtests_test"]
    validation_results = payload["backtests_validation"]
    try:
        select_strategy(validation_results, test_results)
    except ValueError as exc:
        errors.append(str(exc))

    for window_name, results in (
        ("backtests_test", test_results),
        ("backtests_validation", validation_results),
    ):
        for strategy, block in results.items():
            absent = [field for field in REQUIRED_PERFORMANCE_FIELDS if field not in block]
            if absent:
                errors.append(f"{window_name}[{strategy}] missing fields: {', '.join(absent)}")

    test_sessions = {
        block.get("sessions") for block in test_results.values() if block.get("sessions") is not None
    }
    if len(test_sessions) > 1:
        errors.append(f"test strategies disagree on session count: {sorted(test_sessions)}")

    benchmarks = payload["benchmarks"]
    missing_benchmarks = [name for name in REQUIRED_BENCHMARKS if name not in benchmarks]
    if missing_benchmarks:
        errors.append(f"missing benchmarks: {', '.join(missing_benchmarks)}")
    elif len(test_sessions) == 1:
        expected = next(iter(test_sessions))
        for name in ("spy_open_to_close_net", "spy_open_to_close_gross"):
            if benchmarks[name].get("sessions") != expected:
                errors.append(
                    f"{name} covers {benchmarks[name].get('sessions')} sessions "
                    f"but the strategies cover {expected}"
                )

    best_model = payload["best_model"].get("name")
    classification = payload["classification"]
    if best_model and best_model not in classification:
        errors.append(f"best model {best_model!r} missing from classification block")
    elif best_model:
        test_block = classification[best_model].get("test", {}).get("at_selected_threshold")
        if not test_block:
            errors.append(f"classification for {best_model!r} has no test threshold block")

    expected_rows = payload["dataset"].get("class_balance", {}).get("test", {}).get("sessions")
    if expected_rows is not None and len(test_sessions) == 1:
        actual = next(iter(test_sessions))
        if actual != expected_rows:
            errors.append(
                f"dataset reports {expected_rows} test sessions but metrics use {actual}"
            )
    return errors


def validate_artifacts(payload: dict[str, Any], root: Path = config.PROJECT_ROOT) -> list[str]:
    """Check that every figure and table referenced by metrics.json exists."""
    errors: list[str] = []
    artifacts = payload.get("artifacts", {})
    for group in ("figure_files", "table_files"):
        for name, relative in artifacts.get(group, {}).items():
            path = Path(root) / relative
            if not path.is_file():
                errors.append(f"{group}[{name}] missing: {relative}")
    return errors


def run_checks(
    root: Path = config.PROJECT_ROOT,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Validate metrics.json and every artefact it references."""
    metrics_path = Path(metrics_path or config.REPORTS_DIR / "metrics.json")
    if not metrics_path.is_file():
        return {"ok": False, "errors": [f"metrics file missing: {metrics_path}"]}
    try:
        payload = read_json(metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"could not read {metrics_path}: {exc}"]}
    errors = validate_payload(payload) + validate_artifacts(payload, root)
    return {"ok": not errors, "errors": errors}


def main() -> int:
    report = run_checks()
    if report["ok"]:
        print("Report artefacts are consistent")
        return 0
    for error in report["errors"]:
        print(f"[FAIL] {error}")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
