"""Metric definition tests (so the reported numbers can be recomputed by hand)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_equity_curve_compounds():
    returns = pd.Series([0.10, -0.05, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
    equity = metrics.equity_curve(returns, initial_capital=1_000)
    expected = 1_000 * 1.10 * 0.95 * 1.02
    assert equity.iloc[-1] == pytest.approx(expected)


def test_cagr_uses_initial_capital_and_full_session_count():
    returns = pd.Series([0.001] * 504, index=pd.bdate_range("2020-01-01", periods=504))
    initial_capital = 10_000.0
    equity = metrics.equity_curve(returns, initial_capital=initial_capital)
    expected = (equity.iloc[-1] / initial_capital) ** (252 / 504) - 1
    assert metrics.cagr(equity, initial_capital=initial_capital) == pytest.approx(expected)


def test_cagr_does_not_drop_the_first_session():
    """``equity_curve`` already compounds the first return into ``equity[0]``."""
    initial_capital = 1_000.0
    returns = pd.Series([0.10, 0.0], index=pd.bdate_range("2024-01-01", periods=2))
    equity = metrics.equity_curve(returns, initial_capital=initial_capital)
    expected = (1_100.0 / initial_capital) ** (252 / 2) - 1
    assert metrics.cagr(equity, initial_capital=initial_capital) == pytest.approx(expected)
    # Ignoring the initial capital would annualise only the second session.
    assert metrics.cagr(equity) == pytest.approx((1_100.0 / 1_100.0) ** 126 - 1)
    assert metrics.cagr(equity, initial_capital=initial_capital) != pytest.approx(metrics.cagr(equity))


def test_summary_cagr_uses_the_same_initial_capital():
    returns = pd.Series([0.10, -0.20, 0.0], index=pd.bdate_range("2024-01-01", periods=3))
    result = metrics.summarize_performance(returns, initial_capital=1_000.0)
    expected = (880.0 / 1_000.0) ** (252 / 3) - 1
    assert result["cagr"] == pytest.approx(expected)
    assert result["calmar_ratio"] == pytest.approx(expected / abs(result["max_drawdown"]))


def test_max_drawdown_is_negative_peak_to_trough():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert metrics.max_drawdown(equity) == pytest.approx(-0.25)
    assert metrics.drawdown_curve(equity).min() == pytest.approx(-0.25)


def test_sharpe_ratio_definition():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns) == pytest.approx(expected)
    assert np.isnan(metrics.sharpe_ratio(pd.Series([0.01] * 10)))


def test_annualized_volatility_definition():
    returns = pd.Series([0.01, -0.01] * 50)
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(returns) == pytest.approx(expected)


def test_classification_metrics_are_consistent():
    y_true = pd.Series([1, 0, 1, 0])
    probabilities = pd.Series([0.9, 0.2, 0.4, 0.6])

    result = metrics.classification_metrics(y_true, probabilities, threshold=0.5)

    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["base_rate_up"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)


def test_brier_score():
    assert metrics.brier_score(pd.Series([1, 0]), pd.Series([1.0, 0.0])) == pytest.approx(0.0)
    assert metrics.brier_score(pd.Series([1, 0]), pd.Series([0.0, 1.0])) == pytest.approx(1.0)


def test_trade_statistics_with_no_trades():
    stats = metrics.trade_statistics(pd.DataFrame(), n_sessions=10, total_return=0.0)
    assert stats["total_trades"] == 0
    assert stats["win_rate"] is None
    assert stats["trades_per_session"] == 0.0


def test_summary_exposes_the_required_metrics(prediction_frame):
    from src.backtest import STRATEGIES, build_positions, predictions_to_trades, simulate_exact

    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S1_top_probability"])
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))
    result = simulate_exact(
        positions, predictions_to_trades(prediction_frame), index, cost_bps_side=3.0
    )

    required = {
        "total_return",
        "cagr",
        "annualized_return_arithmetic",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "trades_per_session",
        "win_rate",
        "final_equity",
    }
    assert required.issubset(set(result["metrics"].keys()))
