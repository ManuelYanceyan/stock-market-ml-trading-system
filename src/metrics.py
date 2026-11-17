"""Performance and classification metrics.

Conventions (used consistently in every report):

* one year = ``config.TRADING_DAYS_PER_YEAR`` = 252 sessions;
* returns are simple daily returns, the risk-free rate is set to 0;
* Sharpe = mean(daily) / std(daily, ddof=1) * sqrt(252);
* annualised return = mean(daily) * 252 (arithmetic);
* CAGR = (final_equity / initial_capital) ** (252 / number_of_sessions) - 1,
  where ``number_of_sessions`` is the full length of the return series;
* max drawdown = min(equity / running_max(equity) - 1)  (negative number);
* a "trade" is one intraday position (entry at the open, exit at the close).

These definitions are deliberately simple so that a junior engineer can
reproduce the numbers by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config


def equity_curve(returns: pd.Series, initial_capital: float = config.INITIAL_CAPITAL) -> pd.Series:
    """Compound a return series into an equity curve."""
    returns = returns.fillna(0.0).astype(float)
    return initial_capital * (1.0 + returns).cumprod()


def drawdown_curve(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def annualized_return(returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.fillna(0.0).mean() * periods_per_year)


def cagr(
    equity: pd.Series,
    initial_capital: float | None = None,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate over the full session count.

    ``equity`` is normally the curve produced by :func:`equity_curve`, whose
    first value already includes the first session's return.  Pass the actual
    ``initial_capital`` so that the growth rate spans the whole session count;
    falling back to ``equity.iloc[0]`` silently drops that first session and is
    only intended for callers that provide a raw equity series starting at the
    initial capital.
    """
    sessions = len(equity)
    if sessions < 2:
        return float("nan")
    start = float(equity.iloc[0] if initial_capital is None else initial_capital)
    if start <= 0:
        return float("nan")
    growth = float(equity.iloc[-1]) / start
    years = sessions / periods_per_year
    if growth <= 0 or years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.fillna(0.0).std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    values = returns.fillna(0.0)
    std = values.std(ddof=1)
    if std is None or np.isnan(std) or std < 1e-12:
        return float("nan")
    return float(values.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    values = returns.fillna(0.0)
    downside = values.clip(upper=0.0)
    downside_std = np.sqrt((downside**2).mean())
    if downside_std < 1e-12:
        return float("nan")
    return float(values.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def rolling_cumulative(returns: pd.Series, window: int) -> pd.Series:
    """Cumulative return over a trailing window of ``window`` sessions."""
    return (1.0 + returns.fillna(0.0)).rolling(window).apply(np.prod, raw=True) - 1.0


def trade_statistics(trades: pd.DataFrame, n_sessions: int, total_return: float) -> dict[str, Any]:
    """Trade-count, hit-rate and dispersion statistics.

    ``trades`` is expected to have one row per executed intraday position with
    the columns ``net_ret`` (after costs) and ``trade_date``.
    """
    if trades is None or trades.empty:
        return {
            "total_trades": 0,
            "trades_per_session": 0.0,
            "trade_frequency_pct": 0.0,
            "win_rate": None,
            "wins": 0,
            "losses": 0,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "avg_trade_net_ret": None,
            "sessions": int(n_sessions),
        }

    net = trades["net_ret"].astype(float)
    wins = net[net > 0]
    losses = net[net < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "total_trades": int(len(trades)),
        "trades_per_session": float(len(trades) / n_sessions) if n_sessions else None,
        "trade_frequency_pct": float(100.0 * len(trades) / n_sessions) if n_sessions else None,
        "win_rate": float((net > 0).mean()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "avg_trade_net_ret": float(net.mean()),
        "sessions": int(n_sessions),
        "total_return_of_simulated_capital": float(total_return),
    }


def summarize_performance(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    initial_capital: float = config.INITIAL_CAPITAL,
    periods_per_year: int = config.TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    """Full metric block for one return series."""
    returns = returns.fillna(0.0).astype(float)
    if returns.empty:
        return {}
    equity = equity_curve(returns, initial_capital)
    drawdown = drawdown_curve(equity)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    n_sessions = int(len(returns))
    cagr_value = cagr(equity, initial_capital=initial_capital, periods_per_year=periods_per_year)
    max_dd = max_drawdown(equity)

    summary: dict[str, Any] = {
        "sessions": n_sessions,
        "start_date": str(returns.index.min().date()) if isinstance(returns.index, pd.DatetimeIndex) else str(returns.index.min()),
        "end_date": str(returns.index.max().date()) if isinstance(returns.index, pd.DatetimeIndex) else str(returns.index.max()),
        "initial_capital": float(initial_capital),
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr_value,
        "annualized_return_arithmetic": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_dd,
        "calmar_ratio": (float(cagr_value / abs(max_dd)) if max_dd < 0 else None),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "positive_day_share": float((returns > 0).mean()),
        "exposure_share_of_sessions": float((returns != 0).mean()),
        "final_drawdown": float(drawdown.iloc[-1]),
    }
    if trades is not None:
        summary.update(trade_statistics(trades, n_sessions, total_return))
    return summary


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, Any]:
    """Accuracy-style metrics for a probability threshold."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    probabilities = pd.Series(probabilities).astype(float).reset_index(drop=True)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "n_observations": int(len(y_true)),
        "threshold": float(threshold),
        "base_rate_up": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def brier_score(y_true: pd.Series, probabilities: pd.Series) -> float:
    y_true = pd.Series(y_true).astype(float).reset_index(drop=True)
    probabilities = pd.Series(probabilities).astype(float).reset_index(drop=True)
    return float(((probabilities - y_true) ** 2).mean())
