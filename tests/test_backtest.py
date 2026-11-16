"""Backtest tests: timing, fees, compounding, empty sessions, alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import backtest
from src.backtest import (
    STRATEGIES,
    benchmark_buy_and_hold,
    benchmark_intraday,
    build_positions,
    predictions_to_trades,
    simulate_exact,
    simulate_vectorized,
)

COST = 3.0  # bps per side -> 6 bps round trip
SIDE = COST / 10_000.0


def _trades_from(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions_to_trades(predictions)


def test_signals_execute_on_the_next_session(prediction_frame):
    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S1_top_probability"])

    assert len(positions) == 2
    # Signal on d1 -> position on d2, which is exactly the trade_date column.
    assert positions.iloc[0]["signal_date"] == pd.Timestamp("2024-01-02")
    assert positions.iloc[0]["trade_date"] == pd.Timestamp("2024-01-03")
    assert positions.iloc[0]["ticker"] == "SPY"

    trades = _trades_from(prediction_frame)
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    result = simulate_exact(positions, trades, index, cost_bps_side=COST, initial_capital=10_000)

    day_one_gross = 0.10
    day_one_net = day_one_gross - 2 * SIDE
    assert result["trades"].iloc[0]["gross_ret"] == pytest.approx(day_one_gross)
    assert result["returns"].iloc[0] == pytest.approx(day_one_net)
    # The signal date is NOT the return date: no return is earned on d1.
    assert result["returns"].index[0] == pd.Timestamp("2024-01-03")


def test_fees_are_charged_on_entry_and_exit(prediction_frame):
    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S1_top_probability"])
    trades = _trades_from(prediction_frame)
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))

    result = simulate_exact(positions, trades, index, cost_bps_side=COST)
    first = result["trades"].iloc[0]

    assert first["cost"] == pytest.approx(2 * SIDE * first["weight"])
    assert first["net_ret"] == pytest.approx(first["gross_ret"] - first["cost"])
    # With zero costs the gross return is recovered exactly.
    free = simulate_exact(positions, trades, index, cost_bps_side=0.0)
    assert free["trades"].iloc[0]["net_ret"] == pytest.approx(0.10)
    assert free["returns"].iloc[0] - result["returns"].iloc[0] == pytest.approx(2 * SIDE)


def test_capital_compounds(prediction_frame):
    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S1_top_probability"])
    trades = _trades_from(prediction_frame)
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))

    result = simulate_exact(positions, trades, index, cost_bps_side=COST, initial_capital=5_000)
    r1, r2 = result["returns"].tolist()
    expected = 5_000 * (1 + r1) * (1 + r2)

    assert result["equity"].iloc[0] == pytest.approx(5_000 * (1 + r1))
    assert result["equity"].iloc[-1] == pytest.approx(expected)
    assert result["metrics"]["final_equity"] == pytest.approx(expected)
    assert result["metrics"]["total_return"] == pytest.approx(expected / 5_000 - 1)


def test_empty_sessions_stay_in_cash(prediction_frame):
    index = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"])
    empty_positions = pd.DataFrame(columns=["signal_date", "trade_date", "ticker", "prob", "weight"])
    trades = _trades_from(prediction_frame)

    result = simulate_exact(empty_positions, trades, index, cost_bps_side=COST, initial_capital=1_000)

    assert (result["returns"] == 0).all()
    assert (result["equity"] == 1_000).all()
    assert result["trades"].empty
    assert result["metrics"]["total_trades"] == 0
    assert result["metrics"]["win_rate"] is None
    assert result["metrics"]["exposure_share_of_sessions"] == 0.0


def test_vectorized_and_exact_simulations_agree(prediction_frame):
    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S1_top_probability"])
    trades = _trades_from(prediction_frame)
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))

    exact = simulate_exact(positions, trades, index, cost_bps_side=COST)
    vector = simulate_vectorized(positions, trades, index, cost_bps_side=COST)

    np.testing.assert_allclose(vector["returns"].values, exact["returns"].values, atol=1e-12)
    np.testing.assert_allclose(vector["equity"].values, exact["equity"].values, atol=1e-9)


def test_at_most_one_position_per_session():
    dates = pd.bdate_range("2024-02-01", periods=6)
    rows = []
    for i in range(5):
        for ticker, probability in [("SPY", 0.55), ("QQQ", 0.90), ("XLK", 0.70)]:
            rows.append(
                {
                    "signal_date": dates[i],
                    "trade_date": dates[i + 1],
                    "ticker": ticker,
                    "prob": probability,
                    "target_ret_next_oc": 0.01,
                    "open_next": 100.0,
                    "high_next": 101.0,
                    "low_next": 99.0,
                    "vix_lag1": 15.0,
                    "mkt_above_ma50": 1,
                }
            )
    predictions = pd.DataFrame(rows)
    positions = build_positions(predictions, 0.5, STRATEGIES["S1_top_probability"])

    assert len(positions) == 5
    assert positions["trade_date"].is_unique
    assert set(positions["ticker"]) == {"QQQ"}  # always the highest probability


def test_threshold_and_filters_are_respected(prediction_frame):
    positions = build_positions(prediction_frame, 0.95, STRATEGIES["S1_top_probability"])
    assert positions.empty

    high_vix = prediction_frame.copy()
    high_vix["vix_lag1"] = 40.0
    assert build_positions(high_vix, 0.6, STRATEGIES["S2_conservative"]).empty
    assert len(build_positions(high_vix, 0.6, STRATEGIES["S1_top_probability"])) == 2

    below_trend = prediction_frame.copy()
    below_trend["mkt_above_ma50"] = 0
    assert build_positions(below_trend, 0.6, STRATEGIES["S3_trend_filter"]).empty


def test_probability_scaled_sizing_and_stop_loss(prediction_frame):
    positions = build_positions(prediction_frame, 0.6, STRATEGIES["S4_risk_managed"])
    assert (positions["weight"] >= config.STRATEGY_S4_MIN_WEIGHT).all()
    assert (positions["weight"] <= 1.0).all()

    trades = _trades_from(prediction_frame)
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))
    result = simulate_exact(
        positions,
        trades,
        index,
        cost_bps_side=COST,
        stop_loss=config.STRATEGY_S4_STOP_LOSS,
    )
    second = result["trades"].iloc[1]
    # QQQ falls to 94 from 100 (-6%), below the 1% stop -> exit at -1%.
    assert bool(second["stopped_out"]) is True
    assert second["gross_ret"] == pytest.approx(-config.STRATEGY_S4_STOP_LOSS * second["weight"])


def test_benchmark_uses_the_same_dates_and_costs(prediction_frame):
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))
    trades = _trades_from(prediction_frame)

    benchmark = benchmark_intraday(trades, index, ticker="SPY", cost_bps_side=COST)
    assert list(benchmark["returns"].index) == list(index)
    # Only SPY rows are used, and 6 bps of round-trip cost is deducted.
    assert benchmark["returns"].iloc[0] == pytest.approx(0.10 - 2 * SIDE)
    assert not np.isnan(benchmark["returns"].iloc[1])
    # Sessions without an SPY row are treated as cash (0% gross, still costed).
    assert benchmark["returns"].iloc[1] == pytest.approx(0.0 - 2 * SIDE)


def test_buy_and_hold_is_a_separate_holding_period(prediction_frame):
    closes = pd.Series(
        [100.0, 101.0, 99.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    result = benchmark_buy_and_hold(closes, cost_bps_side=COST)

    expected_final = config.INITIAL_CAPITAL * (1 - 2 * SIDE) * (101 / 100) * (99 / 101)
    assert result["equity"].iloc[-1] == pytest.approx(expected_final)
    # One round trip only, not one per session.
    assert result["returns"].iloc[1] == pytest.approx(101 / 100 - 1)


def test_threshold_selection_only_uses_the_window_it_is_given(prediction_frame):
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))
    selection = backtest.select_threshold(prediction_frame, index, cost_bps_side=COST, min_trades=1)

    assert selection["selected_threshold"] in config.THRESHOLD_GRID
    assert selection["selected"]["trades"] >= 1
    assert len(selection["candidates"]) == len(config.THRESHOLD_GRID)
    # With these returns the only profitable threshold must be above max prob of the losing row.
    assert selection["selected"]["net_total_return"] > 0


def test_run_suite_covers_every_documented_strategy(prediction_frame):
    index = pd.DatetimeIndex(sorted(prediction_frame["trade_date"].unique()))
    results = backtest.run_suite(prediction_frame, index, threshold=0.6, cost_bps_side=COST)

    assert set(results.keys()) == set(STRATEGIES.keys())
    for name, result in results.items():
        assert not result["returns"].empty
        assert "total_return" in result["metrics"]
        if STRATEGIES[name].stop_loss is None:
            assert result["vector_matches_exact"] is True
