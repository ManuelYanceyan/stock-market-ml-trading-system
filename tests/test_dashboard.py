"""Dashboard helper tests: validation-based selection and artefact filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard import performance_table, selected_strategy, trades_for


def _results(returns: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        name: {"total_return": value, "cagr": value / 4, "sharpe_ratio": 0.5}
        for name, value in returns.items()
    }


def _payload() -> dict:
    return {
        "backtests_validation": _results({"S1_top_probability": 0.05, "S4_risk_managed": 0.12}),
        "backtests_test": _results({"S1_top_probability": 0.20, "S4_risk_managed": 0.10}),
    }


def test_strategy_is_selected_on_validation_not_on_test():
    selected, leader = selected_strategy(_payload())

    assert selected == "S4_risk_managed"
    assert leader == "S1_top_probability"


def test_inconsistent_strategy_sets_raise_a_clear_error():
    payload = _payload()
    payload["backtests_test"].pop("S4_risk_managed")

    with pytest.raises(ValueError, match="strategy sets differ"):
        selected_strategy(payload)


def test_performance_table_is_indexed_by_strategy():
    table = performance_table(_payload())

    assert list(table.index) == ["S1_top_probability", "S4_risk_managed"]
    assert table.loc["S1_top_probability", "total_return"] == 0.20


def test_trades_for_filters_the_exported_strategy(tmp_path):
    pd.DataFrame(
        {
            "strategy": ["S1_top_probability", "S4_risk_managed", "S4_risk_managed"],
            "ticker": ["SPY", "QQQ", "XLK"],
        }
    ).to_csv(tmp_path / "trades_test.csv", index=False)

    trades = trades_for({}, "S4_risk_managed", tables_dir=tmp_path)

    assert list(trades["ticker"]) == ["QQQ", "XLK"]


def test_trades_for_missing_table_returns_empty_frame(tmp_path):
    assert trades_for({}, "S1_top_probability", tables_dir=tmp_path).empty
