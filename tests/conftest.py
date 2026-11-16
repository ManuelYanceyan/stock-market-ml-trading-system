"""Shared pytest fixtures.

All fixtures are deterministic (fixed seeds, synthetic prices built in memory)
so the test-suite never touches the network and never depends on the real
market data snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_prices(dates: pd.DatetimeIndex, start_price: float, seed: int, volatility: float = 0.011) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, volatility, len(dates))
    close = start_price * np.exp(np.cumsum(returns))
    open_ = close * np.exp(rng.normal(0.0, 0.0025, len(dates)))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = rng.integers(2_000_000, 8_000_000, len(dates)).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """Three ETFs of deterministic daily bars covering train/val/test windows."""
    dates = pd.bdate_range("2013-01-02", "2024-06-28")
    frames = []
    for i, ticker in enumerate(["SPY", "QQQ", "XLK"]):
        frame = _make_prices(dates, 100.0 + 40 * i, seed=100 + i)
        frame["ticker"] = ticker
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return data[["date", "ticker", "open", "high", "low", "close", "volume"]]


@pytest.fixture(scope="session")
def synthetic_macro() -> pd.DataFrame:
    dates = pd.bdate_range("2013-01-02", "2024-06-28")
    rng = np.random.default_rng(7)
    vix = 15 + 4 * np.sin(np.arange(len(dates)) / 40.0) + rng.normal(0, 0.5, len(dates))
    dgs10 = 2.0 + 0.5 * np.cos(np.arange(len(dates)) / 120.0)
    dgs3mo = dgs10 - 0.3
    frame = pd.DataFrame({"date": dates, "vix": vix, "dgs10": dgs10, "dgs3mo": dgs3mo})
    # Simulate FRED style missing observations (holidays / not released).
    frame.loc[frame.index % 37 == 5, ["vix", "dgs10", "dgs3mo"]] = np.nan
    return frame


@pytest.fixture(scope="session")
def synthetic_factors() -> pd.DataFrame:
    dates = pd.bdate_range("2013-01-02", "2024-06-28")
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"date": dates})
    for name in ["ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom"]:
        frame[name] = rng.normal(0.0002, 0.004, len(dates))
    return frame


@pytest.fixture(scope="session")
def feature_panel(synthetic_ohlcv, synthetic_macro, synthetic_factors):
    from src.build_features import build_feature_panel

    return build_feature_panel(synthetic_ohlcv, synthetic_macro, synthetic_factors)


@pytest.fixture()
def sample_frame(feature_panel):
    from src.build_features import assign_splits, build_modeling_sample

    frame, metadata = feature_panel
    sample = build_modeling_sample(frame, metadata)
    sample, info = assign_splits(sample)
    return sample, metadata, info


@pytest.fixture()
def prediction_frame() -> pd.DataFrame:
    """Two sessions, one ETF each, with hand-checkable returns."""
    d1 = pd.Timestamp("2024-01-02")
    d2 = pd.Timestamp("2024-01-03")
    d3 = pd.Timestamp("2024-01-04")
    return pd.DataFrame(
        {
            "signal_date": [d1, d2],
            "trade_date": [d2, d3],
            "ticker": ["SPY", "QQQ"],
            "prob": [0.90, 0.80],
            "target_up": [1, 0],
            "target_ret_next_oc": [0.10, -0.05],
            "open_next": [100.0, 100.0],
            "high_next": [111.0, 101.0],
            "low_next": [99.5, 94.0],
            "close_next": [110.0, 95.0],
            "vix_lag1": [15.0, 15.0],
            "mkt_above_ma50": [1, 1],
        }
    )
