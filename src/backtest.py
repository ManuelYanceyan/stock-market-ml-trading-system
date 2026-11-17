"""Trading simulation: strategy rules, exact compounding, benchmarks.

Rules implemented (all long-only, no leverage, at most one ETF per session):

``S1_top_probability``
    Each session, buy the ETF with the highest predicted probability of a
    positive next open-to-close return, but only if that probability is at
    least the threshold selected on the validation window.  Exit at the close.

``S2_conservative``
    Same rule with a higher threshold (+0.05) AND a volatility-regime filter
    (previous-session VIX must be <= 25).  Far fewer trades, only calm days.

``S3_trend_filter``
    Same rule with a market-condition filter: trade only when SPY closed above
    its 50-day moving average (a documented regime rule).

``S4_risk_managed``
    Probability-scaled position size (more capital when the edge is larger)
    plus a 1% intraday stop loss.  This is the risk-management variant.

Timing: a signal produced at the close of ``signal_date`` is executed at the
open of ``trade_date`` (= next session) and closed at that session's close.
Costs (commission + slippage) are charged on BOTH entry and exit in every
simulation, including the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
from src import metrics as metrics_module
from src.utils import get_logger

logger = get_logger("backtest")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    threshold_bump: float = 0.0
    max_vix: float | None = None
    trend_filter: bool = False
    sizing: str = "full"
    stop_loss: float | None = None


STRATEGIES: dict[str, StrategySpec] = {
    "S1_top_probability": StrategySpec(
        name="S1_top_probability",
        description=(
            "Buy the highest-probability ETF each session when its probability is above "
            "the validation-selected threshold; exit at the close."
        ),
        threshold_bump=config.STRATEGY_S1_THRESHOLD_BUMP,
    ),
    "S2_conservative": StrategySpec(
        name="S2_conservative",
        description=(
            "Conservative variant: threshold +0.05 and only when the previous VIX is "
            "<= 25 (calm-regime filter)."
        ),
        threshold_bump=config.STRATEGY_S2_THRESHOLD_BUMP,
        max_vix=config.STRATEGY_S2_MAX_VIX,
    ),
    "S3_trend_filter": StrategySpec(
        name="S3_trend_filter",
        description=(
            "Market-condition strategy: trade only when SPY closed above its 50-day "
            "moving average, with a +0.02 threshold buffer."
        ),
        threshold_bump=config.STRATEGY_S3_THRESHOLD_BUMP,
        trend_filter=config.STRATEGY_S3_TREND_FILTER,
    ),
    "S4_risk_managed": StrategySpec(
        name="S4_risk_managed",
        description=(
            "Risk-managed variant: position size scaled by the probability edge "
            "(25%-100% of capital) and a 1% intraday stop loss."
        ),
        sizing="prob_scaled",
        stop_loss=config.STRATEGY_S4_STOP_LOSS,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cost_fraction(cost_bps_side: float) -> float:
    """Round-trip cost fraction for a fully invested position."""
    return 2.0 * cost_bps_side / 10_000.0


def _validate_cost(cost_bps_side: float) -> None:
    if not np.isfinite(cost_bps_side) or cost_bps_side < 0:
        raise ValueError("cost_bps_side must be a finite non-negative number")


def _validate_capital(initial_capital: float) -> None:
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be a finite positive number")


def _validate_session_index(session_index: pd.DatetimeIndex) -> None:
    if len(session_index) == 0:
        raise ValueError("session_index must contain at least one session")
    if session_index.has_duplicates:
        raise ValueError("session_index must not contain duplicate sessions")
    if not session_index.is_monotonic_increasing:
        raise ValueError("session_index must be sorted in ascending order")


def _validate_positions(positions: pd.DataFrame) -> None:
    required = {"signal_date", "trade_date", "ticker", "prob", "weight"}
    if positions.empty:
        return
    missing = sorted(required - set(positions.columns))
    if missing:
        raise ValueError(f"positions missing columns: {missing}")
    weights = pd.to_numeric(positions["weight"], errors="coerce")
    if weights.isna().any() or (weights < 0).any() or (weights > 1.0 + 1e-9).any():
        raise ValueError("position weights must be finite and between 0 and 1")


def predictions_to_trades(predictions: pd.DataFrame) -> pd.DataFrame:
    """Extract the tradeable fields from a prediction frame."""
    required = ["trade_date", "ticker", config.TARGET_RETURN_COLUMN, "open_next", "low_next"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions missing columns for backtest: {missing}")
    trades = predictions[required].copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    return trades.drop_duplicates(subset=["trade_date", "ticker"], keep="last").reset_index(drop=True)


def build_positions(
    predictions: pd.DataFrame,
    threshold: float,
    spec: StrategySpec,
    prob_full: float = config.STRATEGY_S4_PROB_FULL,
    min_weight: float = config.STRATEGY_S4_MIN_WEIGHT,
) -> pd.DataFrame:
    """Turn per-ETF probabilities into at most one position per session."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not np.isfinite(prob_full) or prob_full <= 0:
        raise ValueError("prob_full must be a finite positive number")
    if not np.isfinite(min_weight) or not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be between 0 and 1")

    frame = predictions.dropna(subset=["prob"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["signal_date", "trade_date", "ticker", "prob", "weight"])

    best_idx = frame.groupby("signal_date")["prob"].idxmax()
    best = frame.loc[best_idx].copy()

    effective_threshold = float(threshold) + float(spec.threshold_bump)
    best = best[best["prob"] >= effective_threshold]

    if spec.max_vix is not None:
        best = best[best["vix_lag1"].fillna(np.inf) <= spec.max_vix]
    if spec.trend_filter:
        best = best[best["mkt_above_ma50"].fillna(0) >= 1]

    if spec.sizing == "prob_scaled":
        edge = (best["prob"] - 0.5) / max(prob_full, 1e-9)
        best["weight"] = edge.clip(lower=min_weight, upper=1.0)
    else:
        best["weight"] = 1.0

    best = best.sort_values("trade_date")
    best.attrs["effective_threshold"] = effective_threshold
    return best[
        ["signal_date", "trade_date", "ticker", "prob", "weight"]
    ].reset_index(drop=True)


def _trade_lookup(trades_data: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], dict[str, float]]:
    lookup: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for row in trades_data.itertuples(index=False):
        lookup[(pd.Timestamp(row.trade_date), row.ticker)] = {
            "ret": float(getattr(row, config.TARGET_RETURN_COLUMN)),
            "open": float(row.open_next),
            "low": float(row.low_next),
        }
    return lookup


def simulate_exact(
    positions: pd.DataFrame,
    trades_data: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
    initial_capital: float = config.INITIAL_CAPITAL,
    stop_loss: float | None = None,
) -> dict[str, Any]:
    """Iterative (row-by-row) simulation with full capital compounding.

    Capital is compounded session by session: ``equity *= (1 + day_return)``.
    A cap of 100% of equity per session (no leverage) is enforced by the
    position builder.  Costs are charged on entry and on exit.
    """
    session_index = pd.DatetimeIndex(session_index)
    _validate_cost(cost_bps_side)
    _validate_capital(initial_capital)
    _validate_session_index(session_index)
    _validate_positions(positions)
    lookup = _trade_lookup(trades_data)

    by_session: dict[pd.Timestamp, list[pd.Series]] = {}
    for row in positions.itertuples(index=False):
        by_session.setdefault(pd.Timestamp(row.trade_date), []).append(
            pd.Series({"ticker": row.ticker, "weight": float(row.weight), "prob": float(row.prob), "signal_date": row.signal_date})
        )

    side_cost = cost_bps_side / 10_000.0
    equity = float(initial_capital)
    daily_returns: list[float] = []
    equity_values: list[float] = []
    trade_records: list[dict[str, Any]] = []

    for session in session_index:
        day_return = 0.0
        for position in by_session.get(session, []):
            key = (session, position["ticker"])
            if key not in lookup:
                raise KeyError(f"No realised return for {position['ticker']} on {session.date()}")
            price = lookup[key]
            weight = float(position["weight"])
            gross_ret = price["ret"]

            stopped = False
            if stop_loss is not None and price["open"] > 0:
                intraday_low_return = price["low"] / price["open"] - 1.0
                if intraday_low_return <= -stop_loss:
                    gross_ret = -abs(stop_loss)
                    stopped = True

            position_cost = 2.0 * side_cost * weight
            net_ret = weight * gross_ret - position_cost
            day_return += net_ret

            trade_records.append(
                {
                    "signal_date": pd.Timestamp(position["signal_date"]),
                    "trade_date": session,
                    "ticker": position["ticker"],
                    "prob": float(position["prob"]),
                    "weight": weight,
                    "gross_ret": float(weight * gross_ret),
                    "cost": float(position_cost),
                    "net_ret": float(net_ret),
                    "stopped_out": bool(stopped),
                }
            )

        equity *= 1.0 + day_return
        daily_returns.append(day_return)
        equity_values.append(equity)

    returns = pd.Series(daily_returns, index=session_index, name="strategy_return")
    equity_series = pd.Series(equity_values, index=session_index, name="equity")
    trades = pd.DataFrame(trade_records)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "signal_date",
                "trade_date",
                "ticker",
                "prob",
                "weight",
                "gross_ret",
                "cost",
                "net_ret",
                "stopped_out",
            ]
        )

    summary = metrics_module.summarize_performance(returns, trades, initial_capital=initial_capital)
    return {"returns": returns, "equity": equity_series, "trades": trades, "metrics": summary}


def simulate_vectorized(
    positions: pd.DataFrame,
    trades_data: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    tickers: list[str] | None = None,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> dict[str, Any]:
    """Vectorised version of the same rules (no stop loss support).

    Used as a cross-check of the exact simulation: both implementations must
    produce the same equity curve (see ``tests/test_backtest.py``).
    """
    session_index = pd.DatetimeIndex(session_index)
    _validate_cost(cost_bps_side)
    _validate_capital(initial_capital)
    _validate_session_index(session_index)
    _validate_positions(positions)
    tickers = tickers or sorted(trades_data["ticker"].unique().tolist())

    weights = pd.DataFrame(0.0, index=session_index, columns=tickers)
    for row in positions.itertuples(index=False):
        weights.loc[pd.Timestamp(row.trade_date), row.ticker] = float(row.weight)

    realised = (
        trades_data.pivot(index="trade_date", columns="ticker", values=config.TARGET_RETURN_COLUMN)
        .reindex(index=session_index, columns=tickers)
        .fillna(0.0)
    )
    side_cost = cost_bps_side / 10_000.0
    gross = (weights * realised).sum(axis=1)
    costs = 2.0 * side_cost * weights.sum(axis=1)
    net = gross - costs
    net.name = "strategy_return"

    equity = initial_capital * (1.0 + net).cumprod()
    equity.name = "equity"
    trades = positions.copy()
    return {
        "returns": net,
        "equity": equity,
        "gross_returns": gross,
        "trades": trades,
        "metrics": metrics_module.summarize_performance(net, None, initial_capital=initial_capital),
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def benchmark_intraday(
    trades_data: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    ticker: str = config.BENCHMARK_TICKER,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Like-for-like benchmark: always long the benchmark ETF open-to-close.

    Same dates, same timing and the same per-side costs as the strategies.
    """
    session_index = pd.DatetimeIndex(session_index)
    _validate_cost(cost_bps_side)
    _validate_session_index(session_index)
    realised = (
        trades_data[trades_data["ticker"] == ticker]
        .set_index("trade_date")[config.TARGET_RETURN_COLUMN]
        .reindex(session_index)
    )
    missing = int(realised.isna().sum())
    if missing:
        logger.warning("Benchmark %s missing %s sessions; filled with 0", ticker, missing)
    net = realised.fillna(0.0) - 2.0 * cost_bps_side / 10_000.0
    net.name = "benchmark_return"
    equity = config.INITIAL_CAPITAL * (1.0 + net).cumprod()
    equity.name = "equity"
    return {
        "returns": net,
        "equity": equity,
        "metrics": metrics_module.summarize_performance(net, None),
    }


def benchmark_buy_and_hold(
    close_prices: pd.Series,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
) -> dict[str, Any]:
    """SPY close-to-close buy & hold - a DIFFERENT holding period.

    Reported only as context: the strategy and the intraday benchmark hold a
    position for one session, while buy & hold is permanently invested.
    """
    closes = close_prices.dropna().sort_index()
    _validate_cost(cost_bps_side)
    returns = closes.pct_change().fillna(0.0)
    if len(returns):
        returns.iloc[0] -= 2.0 * cost_bps_side / 10_000.0
    returns.name = "buy_and_hold_return"
    equity = config.INITIAL_CAPITAL * (1.0 + returns).cumprod()
    equity.name = "equity"
    return {
        "returns": returns,
        "equity": equity,
        "metrics": metrics_module.summarize_performance(returns, None),
    }


# ---------------------------------------------------------------------------
# Threshold selection (validation window only)
# ---------------------------------------------------------------------------
def select_threshold(
    predictions: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
    grid: list[float] | None = None,
    min_trades: int = config.MIN_VALIDATION_TRADES,
) -> dict[str, Any]:
    """Pick the S1 probability threshold on the validation window.

    Score = net cumulative return of the S1 rule after costs, subject to a
    minimum number of trades so that a lucky handful of trades cannot win.
    The test window is never used here.
    """
    grid = grid or config.THRESHOLD_GRID
    trades_data = predictions_to_trades(predictions)
    candidates: list[dict[str, Any]] = []

    for threshold in grid:
        positions = build_positions(predictions, threshold, STRATEGIES["S1_top_probability"])
        simulation = simulate_exact(positions, trades_data, session_index, cost_bps_side)
        candidates.append(
            {
                "threshold": float(threshold),
                "trades": int(len(simulation["trades"])),
                "net_total_return": float(simulation["metrics"].get("total_return", np.nan)),
                "sharpe": float(simulation["metrics"].get("sharpe_ratio", np.nan)),
            }
        )

    eligible = [c for c in candidates if c["trades"] >= min_trades]
    if not eligible:
        logger.warning(
            "No threshold reached the minimum of %s validation trades; using the one with most trades",
            min_trades,
        )
        eligible = sorted(candidates, key=lambda c: c["trades"], reverse=True)[:1]

    best = max(eligible, key=lambda c: (c["net_total_return"], c["threshold"]))
    return {
        "selected_threshold": best["threshold"],
        "selection_rule": (
            "maximise net cumulative return of S1 on the validation window "
            f"subject to >= {min_trades} trades; thresholds {grid[0]:.2f}-{grid[-1]:.2f}"
        ),
        "selected": best,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Full strategy suite
# ---------------------------------------------------------------------------
def run_suite(
    predictions: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    threshold: float,
    cost_bps_side: float = config.COMMISSION_BPS + config.SLIPPAGE_BPS,
    strategies: dict[str, StrategySpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every strategy for one model and one evaluation window."""
    strategies = strategies or STRATEGIES
    trades_data = predictions_to_trades(predictions)
    results: dict[str, dict[str, Any]] = {}
    for name, spec in strategies.items():
        positions = build_positions(predictions, threshold, spec)
        exact = simulate_exact(
            positions,
            trades_data,
            session_index,
            cost_bps_side=cost_bps_side,
            stop_loss=spec.stop_loss,
        )
        vector = simulate_vectorized(positions, trades_data, session_index, cost_bps_side=cost_bps_side)
        exact["positions"] = positions
        exact["vector_returns"] = vector["returns"]
        exact["spec"] = spec
        exact["threshold_used"] = float(threshold) + spec.threshold_bump
        exact["vector_matches_exact"] = bool(
            np.allclose(vector["returns"].values, exact["returns"].values, atol=1e-12)
        )
        results[name] = exact
    return results


def performance_table(results: dict[str, dict[str, Any]], extra: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    """Flatten metric dictionaries into a comparison table."""
    rows = []
    for name, result in results.items():
        row = {"strategy": name}
        row.update(result["metrics"])
        row["threshold_used"] = result.get("threshold_used")
        rows.append(row)
    for name, result in (extra or {}).items():
        row = {"strategy": name}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
