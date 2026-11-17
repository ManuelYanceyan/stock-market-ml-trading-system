"""Report consistency tests: metrics blocks, benchmarks and artefacts."""

from __future__ import annotations

import json

from src.validate_reports import run_checks, validate_artifacts, validate_payload


def _performance(sessions: int = 2) -> dict:
    return {
        "sessions": sessions,
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
        "total_return": 0.10,
        "cagr": 0.05,
        "sharpe_ratio": 0.50,
        "max_drawdown": -0.02,
    }


def _payload() -> dict:
    return {
        "meta": {"run_mode": "cached", "generated_at_utc": "2024-01-04T00:00:00Z"},
        "dataset": {
            "last_trade_date": "2024-01-03",
            "class_balance": {"test": {"sessions": 2}},
        },
        "best_model": {"name": "model_a", "threshold": 0.5},
        "classification": {
            "model_a": {
                "test": {"at_selected_threshold": {"accuracy": 0.5, "roc_auc": 0.5}},
            }
        },
        "backtests_test": {"S1": _performance()},
        "backtests_validation": {"S1": _performance()},
        "benchmarks": {
            "spy_open_to_close_net": _performance(),
            "spy_open_to_close_gross": _performance(),
            "spy_buy_and_hold_close_to_close": _performance(),
        },
        "latest_signal": {"signal_session": "2024-01-03"},
        "artifacts": {"figure_files": {}, "table_files": {}},
    }


def test_consistent_payload_has_no_errors():
    assert validate_payload(_payload()) == []


def test_missing_top_level_block_is_reported():
    payload = _payload()
    del payload["benchmarks"]

    errors = validate_payload(payload)

    assert any("missing top-level keys" in error and "benchmarks" in error for error in errors)


def test_benchmark_session_mismatch_is_reported():
    payload = _payload()
    payload["benchmarks"]["spy_open_to_close_net"]["sessions"] = 1

    errors = validate_payload(payload)

    assert any("spy_open_to_close_net" in error for error in errors)


def test_missing_artifact_is_reported(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "equity.png").write_text("png", encoding="utf-8")
    payload = _payload()
    payload["artifacts"] = {
        "figure_files": {
            "equity": "reports/equity.png",
            "drawdown": "reports/drawdown.png",
        },
        "table_files": {},
    }

    errors = validate_artifacts(payload, root=tmp_path)

    assert len(errors) == 1
    assert "drawdown.png" in errors[0]


def test_run_checks_reads_metrics_and_returns_a_status(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_payload()), encoding="utf-8")

    report = run_checks(root=tmp_path, metrics_path=metrics_path)

    assert report == {"ok": True, "errors": []}
