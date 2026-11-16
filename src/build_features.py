"""Feature engineering, target construction and chronological splits.

Timing contract (the most important rule of the project)
--------------------------------------------------------
At the close of session *t* we only know prices up to *t*.  The label is the
NEXT session's open-to-close return:

    target_ret_next_oc = close[t+1] / open[t+1] - 1

so a signal produced at the close of *t* is executed at the open of *t+1* and
closed at the close of *t+1*.  Every feature below is computed from data with
timestamp <= *t*:

* price/volume features use rolling windows that END at *t*;
* FRED series are re-indexed on the session calendar, forward-filled from the
  past and then shifted by ``MACRO_LAG_SESSIONS`` sessions;
* French factors are forward-filled and lagged by ``FACTOR_LAG_SESSIONS``
  sessions (the daily files are published with a delay, see docs);
* no column is ever shifted backwards (except the target itself).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from src.download_data import FACTOR_COLUMNS, MACRO_COLUMNS, load_raw_cache
from src.utils import get_logger, read_json, save_parquet, write_json

logger = get_logger("build_features")

MACRO_LAG_SESSIONS = 1
FACTOR_LAG_SESSIONS = 5  # conservative: French daily files are updated monthly

TO_PREDICT = ["target_up", "target_ret_next_oc"]
CALENDAR_FEATURES = ["dow", "month"]

PROCESSED_PATH = config.PROCESSED_DIR / "features.parquet"
FULL_PATH = config.PROCESSED_DIR / "features_full.parquet"
METADATA_PATH = config.PROCESSED_DIR / "features_metadata.json"


# ---------------------------------------------------------------------------
# Per-ticker technical features
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(
            loss == 0,
            np.where(gain > 0, 100.0, 50.0),
            100.0 - 100.0 / (1.0 + gain / loss),
        )
    return pd.Series(rsi, index=close.index, dtype="float64")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    """All technical features for one ETF, using information up to each date."""
    d = group.sort_values("date")
    close, high, low, open_, volume = (
        d["close"],
        d["high"],
        d["low"],
        d["open"],
        d["volume"],
    )
    ret1 = close.pct_change()
    out = pd.DataFrame(index=d.index)

    # --- lagged momentum ---------------------------------------------------
    for window in (1, 2, 3, 5, 10, 20, 60, 120):
        out[f"ret_{window}d"] = close.pct_change(window)

    # --- realised volatility ----------------------------------------------
    for window in (5, 10, 20, 60):
        out[f"vol_{window}d"] = (
            ret1.rolling(window, min_periods=max(3, window // 2)).std(ddof=1)
            * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        )
    out["downside_vol_20d"] = (
        ret1.clip(upper=0).rolling(20, min_periods=10).std(ddof=1)
        * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    )

    # --- trend / moving averages ------------------------------------------
    ma = {
        window: close.rolling(window, min_periods=max(2, window // 2)).mean()
        for window in (5, 10, 20, 50, 60, 200)
    }
    out["ma_ratio_5_20"] = ma[5] / ma[20] - 1.0
    out["ma_ratio_20_60"] = ma[20] / ma[60] - 1.0
    out["price_vs_ma20"] = close / ma[20] - 1.0
    out["price_vs_ma50"] = close / ma[50] - 1.0
    out["price_vs_ma200"] = close / ma[200] - 1.0

    # --- oscillators -------------------------------------------------------
    out["rsi_14"] = _rsi(close, 14)
    atr = _atr(high, low, close, 14)
    out["atr_14_pct"] = atr / close

    rolling_std_20 = close.rolling(20, min_periods=10).std(ddof=1)
    mid = ma[20]
    upper = mid + 2 * rolling_std_20
    lower = mid - 2 * rolling_std_20
    width = (upper - lower).replace(0, np.nan)
    out["bb_pctb_20"] = (close - lower) / width

    # --- volume ------------------------------------------------------------
    vol_mean_20 = volume.rolling(20, min_periods=10).mean()
    vol_std_20 = volume.rolling(20, min_periods=10).std(ddof=1)
    out["volume_z_20"] = (volume - vol_mean_20) / vol_std_20.replace(0, np.nan)
    vol_mean_5 = volume.rolling(5, min_periods=3).mean()
    out["volume_ratio_5_20"] = vol_mean_5 / vol_mean_20

    # --- intraday structure ------------------------------------------------
    out["intraday_range_pct"] = (high - low) / close
    out["close_pos_in_range"] = (close - low) / (high - low).replace(0, np.nan)
    gap = open_ / close.shift(1) - 1.0
    out["gap_open_pct"] = gap
    out["gap_open_mean_5d"] = gap.rolling(5, min_periods=3).mean()

    # --- position versus recent extremes ----------------------------------
    out["dist_52w_high"] = close / close.rolling(252, min_periods=200).max() - 1.0
    out["dist_20d_high"] = close / close.rolling(20, min_periods=10).max() - 1.0
    out["dist_20d_low"] = close / close.rolling(20, min_periods=10).min() - 1.0

    # --- distribution shape ------------------------------------------------
    out["ret_skew_20"] = ret1.rolling(20, min_periods=10).skew()
    out["ret_kurt_20"] = ret1.rolling(20, min_periods=10).kurt()

    return out


def _market_features(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Benchmark-derived context features (SPY), known at the close of t."""
    spy = panel[panel["ticker"] == config.BENCHMARK_TICKER].set_index("date").sort_index()
    spy = spy.reindex(calendar)
    ret1 = spy["close"].pct_change()
    market = pd.DataFrame(index=calendar)
    market["mkt_ret_1d"] = ret1
    market["mkt_ret_5d"] = spy["close"].pct_change(5)
    market["mkt_ret_20d"] = spy["close"].pct_change(20)
    market["mkt_vol_20d"] = (
        ret1.rolling(20, min_periods=10).std(ddof=1) * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    )
    ma50 = spy["close"].rolling(50, min_periods=25).mean()
    market["mkt_above_ma50"] = (spy["close"] > ma50).astype(float)
    market["mkt_dist_52w_high"] = spy["close"] / spy["close"].rolling(252, min_periods=200).max() - 1.0
    market.index.name = "date"
    return market


# ---------------------------------------------------------------------------
# Macro / factor alignment (past information only)
# ---------------------------------------------------------------------------
def align_daily_source(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lag_sessions: int,
    columns: list[str],
) -> pd.DataFrame:
    """Re-index on the trading calendar, forward-fill and lag.

    ``ffill`` never uses a future observation, and ``shift(lag_sessions)``
    guarantees that the value used on session *t* was observable strictly
    before *t*.
    """
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date").sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.reindex(calendar).ffill()
    data = data.shift(lag_sessions)
    available = [c for c in columns if c in data.columns]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        logger.warning("Source is missing expected columns: %s", missing)
    return data[available]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(panel: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Attach next-session open-to-close return and its sign."""
    panel = panel.sort_values(["ticker", "date"]).copy()
    grouped = panel.groupby("ticker", sort=False)
    panel["trade_date"] = grouped["date"].shift(-1)
    panel["open_next"] = grouped["open"].shift(-1)
    panel["high_next"] = grouped["high"].shift(-1)
    panel["low_next"] = grouped["low"].shift(-1)
    panel["close_next"] = grouped["close"].shift(-1)

    panel["target_ret_next_oc"] = panel["close_next"] / panel["open_next"] - 1.0

    gap_days = (panel["trade_date"] - panel["date"]).dt.days
    invalid = panel["trade_date"].isna() | (gap_days > max_gap_days)
    panel.loc[invalid, ["trade_date", "target_ret_next_oc", "open_next", "high_next", "low_next", "close_next"]] = np.nan

    panel["target_up"] = (panel["target_ret_next_oc"] > 0).astype(float)
    panel.loc[panel["target_ret_next_oc"].isna(), "target_up"] = np.nan
    return panel


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------
def build_feature_panel(
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine every source into one (date, ticker) feature frame."""
    panel = ohlcv.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    calendar = pd.DatetimeIndex(
        sorted(panel.loc[panel["ticker"] == config.BENCHMARK_TICKER, "date"].unique())
    )
    if len(calendar) < 300:
        raise ValueError("Benchmark calendar too short - download problem?")

    panel = panel[panel["date"].isin(calendar)].sort_values(["ticker", "date"])

    technical_parts = [
        _ticker_features(group) for _ticker, group in panel.groupby("ticker", sort=False)
    ]
    technicals = pd.concat(technical_parts).sort_index()
    feature_frame = pd.concat([panel, technicals], axis=1)

    market = _market_features(panel, calendar).reset_index()
    feature_frame = feature_frame.merge(market, on="date", how="left")

    # Cross-sectional / relative features.
    feature_frame["rel_strength_5d"] = feature_frame["ret_5d"] - feature_frame["mkt_ret_5d"]
    feature_frame["rel_strength_20d"] = feature_frame["ret_20d"] - feature_frame["mkt_ret_20d"]

    beta_parts = []
    for _ticker, group in feature_frame.groupby("ticker", sort=False):
        cov = group["ret_1d"].rolling(60, min_periods=40).cov(group["mkt_ret_1d"])
        var = group["mkt_ret_1d"].rolling(60, min_periods=40).var()
        beta_parts.append(cov / var.replace(0, np.nan))
    feature_frame["beta_60d"] = pd.concat(beta_parts).sort_index()

    # Macro block (lagged).
    macro_block = align_daily_source(macro, calendar, MACRO_LAG_SESSIONS, MACRO_COLUMNS)
    macro_block.columns = [f"{c}_lag{MACRO_LAG_SESSIONS}" for c in macro_block.columns]
    macro_block["vix_change_5d_lag1"] = macro_block["vix_lag1"].pct_change(5)
    macro_block["vix_z_60_lag1"] = (
        macro_block["vix_lag1"] - macro_block["vix_lag1"].rolling(60, min_periods=30).mean()
    ) / macro_block["vix_lag1"].rolling(60, min_periods=30).std(ddof=1)
    macro_block["dgs10_change_5d_lag1"] = macro_block["dgs10_lag1"].diff(5)
    macro_block["dgs3mo_change_5d_lag1"] = macro_block["dgs3mo_lag1"].diff(5)
    macro_block["term_spread_lag1"] = macro_block["dgs10_lag1"] - macro_block["dgs3mo_lag1"]
    macro_block["term_spread_change_5d_lag1"] = macro_block["term_spread_lag1"].diff(5)
    macro_block = macro_block.reset_index()
    feature_frame = feature_frame.merge(macro_block, on="date", how="left")

    # French factors block (lagged and forward-filled from the past).
    factor_block = align_daily_source(
        factors, calendar, FACTOR_LAG_SESSIONS, FACTOR_COLUMNS
    )
    factor_block = factor_block.add_suffix(f"_lag{FACTOR_LAG_SESSIONS}")
    factor_block["ff_mom_20d_lag5"] = (
        factor_block["ff_mom_lag5"].rolling(20, min_periods=10).sum()
    )
    factor_block = factor_block.reset_index()
    feature_frame = feature_frame.merge(factor_block, on="date", how="left")

    # Calendar fields and dummies.
    feature_frame["dow"] = feature_frame["date"].dt.dayofweek
    feature_frame["month"] = feature_frame["date"].dt.month
    dummy_columns = []
    for day in range(5):
        column = f"dow_{day}"
        feature_frame[column] = (feature_frame["dow"] == day).astype(int)
        dummy_columns.append(column)
    for month in range(1, 13):
        column = f"month_{month:02d}"
        feature_frame[column] = (feature_frame["month"] == month).astype(int)
        dummy_columns.append(column)
    for ticker in config.TICKERS:
        column = f"ticker_{ticker}"
        feature_frame[column] = (feature_frame["ticker"] == ticker).astype(int)
        dummy_columns.append(column)

    feature_frame = add_targets(feature_frame)
    feature_frame = feature_frame.sort_values(["date", "ticker"]).reset_index(drop=True)

    numeric_columns = [
        c
        for c in feature_frame.columns
        if c
        not in set(
            ["date", "ticker", "trade_date", "split", "open", "high", "low", "close", "volume"]
            + TO_PREDICT
            + ["open_next", "high_next", "low_next", "close_next"]
            + dummy_columns
        )
        and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
    numeric_columns = [c for c in numeric_columns if c not in CALENDAR_FEATURES]

    metadata = {
        "rows_total": int(len(feature_frame)),
        "sessions": int(feature_frame["date"].nunique()),
        "tickers": config.TICKERS,
        "first_date": feature_frame["date"].min().strftime("%Y-%m-%d"),
        "last_date": feature_frame["date"].max().strftime("%Y-%m-%d"),
        "last_trade_date": feature_frame["trade_date"].max(),
        "numeric_features": numeric_columns,
        "dummy_features": dummy_columns,
        "to_predict": TO_PREDICT,
        "macro_lag_sessions": MACRO_LAG_SESSIONS,
        "factor_lag_sessions": FACTOR_LAG_SESSIONS,
        "target_definition": "target_ret_next_oc = close[t+1]/open[t+1]-1; target_up = 1 if >0",
        "n_numeric": len(numeric_columns),
        "n_dummies": len(dummy_columns),
    }
    return feature_frame, metadata


# ---------------------------------------------------------------------------
# Modelling sample and splits
# ---------------------------------------------------------------------------
def get_feature_columns(frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    numeric = [c for c in metadata["numeric_features"] if c in frame.columns]
    dummies = [c for c in metadata["dummy_features"] if c in frame.columns]
    return numeric, dummies


def build_modeling_sample(
    frame: pd.DataFrame, metadata: dict[str, Any], max_missing_share: float = config.MAX_FEATURE_MISSING_SHARE
) -> pd.DataFrame:
    """Rows usable for modelling: known labels, study window, enough features."""
    numeric, dummies = get_feature_columns(frame, metadata)
    features = numeric + dummies

    sample = frame[frame["date"] >= pd.Timestamp(config.STUDY_START)].copy()
    sample = sample.dropna(subset=["target_up", "target_ret_next_oc", "trade_date"])

    missing_share = sample[features].replace([np.inf, -np.inf], np.nan).isna().mean(axis=1)
    sample = sample[missing_share <= max_missing_share].copy()

    sample[features] = sample[features].replace([np.inf, -np.inf], np.nan)
    return sample.reset_index(drop=True)


def assign_splits(
    sample: pd.DataFrame, embargo_days: int = config.SPLIT_EMBARGO_DAYS
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chronological train / validation / test split with a label embargo.

    A row belongs to a split only if its SIGNAL date is inside the split AND
    its label window (the following session) is also inside the split.  The
    last ``embargo_days`` signal rows of each split are therefore dropped,
    which guarantees that no split uses prices from the next split.
    """
    boundaries = {
        "train": (pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END)),
        "validation": (pd.Timestamp(config.VALID_START), pd.Timestamp(config.VALID_END)),
        "test": (
            pd.Timestamp(config.TEST_START),
            pd.Timestamp(config.TEST_END) if config.TEST_END else pd.Timestamp("2262-01-01"),
        ),
    }

    sample = sample.copy()
    sample["split"] = None
    for name, (start, end) in boundaries.items():
        mask = (sample["date"] >= start) & (sample["date"] <= end)
        sample.loc[mask, "split"] = name

    info: dict[str, Any] = {"embargo_days": embargo_days, "boundaries": {}}
    keep_mask = pd.Series(False, index=sample.index)

    for name, (start, end) in boundaries.items():
        subset = sample[sample["split"] == name].sort_values("date")
        if subset.empty:
            info["boundaries"][name] = {"rows": 0}
            continue
        # Drop the tail of the split whose label window crosses the end date.
        if name != "test" or (config.TEST_END is not None):
            tail_cut = subset["trade_date"] > end
            subset = subset[~tail_cut]
            dropped = int(tail_cut.sum())
        else:
            dropped = 0
        keep_mask.loc[subset.index] = True
        info["boundaries"][name] = {
            "signal_start": subset["date"].min().strftime("%Y-%m-%d"),
            "signal_end": subset["date"].max().strftime("%Y-%m-%d"),
            "trade_start": subset["trade_date"].min().strftime("%Y-%m-%d"),
            "trade_end": subset["trade_date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(subset)),
            "sessions": int(subset["date"].nunique()),
            "dropped_at_boundary": dropped,
            "class_balance_up": float(subset["target_up"].mean()),
        }

    sample = sample[keep_mask].sort_values(["date", "ticker"]).reset_index(drop=True)
    return sample, info


def split_frames(sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: sample[sample["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_features(save: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw cache, build features, apply splits and persist artefacts."""
    ohlcv, macro, factors, manifest = load_raw_cache()
    frame, metadata = build_feature_panel(ohlcv, macro, factors)
    sample = build_modeling_sample(frame, metadata)
    sample, split_info = assign_splits(sample)

    metadata["split_info"] = split_info
    metadata["modeling_rows"] = int(len(sample))
    metadata["raw_manifest_cutoff"] = manifest.get("data_cutoff")
    metadata["missing_share_threshold"] = config.MAX_FEATURE_MISSING_SHARE

    metadata["class_balance"] = {
        name: float(part["target_up"].mean()) if len(part) else None
        for name, part in split_frames(sample).items()
    }

    if save:
        save_parquet(sample, PROCESSED_PATH)
        # The full frame keeps the last session, whose label is unknown yet;
        # it is the only frame that can produce a genuine "next session" signal.
        save_parquet(frame, FULL_PATH)
        write_json(METADATA_PATH, metadata)
        logger.info(
            "Features saved: %s modelling rows, %s numeric + %s dummy features",
            len(sample),
            metadata["n_numeric"],
            metadata["n_dummies"],
        )
    return sample, metadata


def load_full_features() -> pd.DataFrame:
    """All feature rows, including the latest session without a label."""
    if FULL_PATH.exists():
        return pd.read_parquet(FULL_PATH)
    logger.warning(
        "%s missing - falling back to the labelled sample for the latest signal",
        FULL_PATH.name,
    )
    frame, _ = load_features()
    return frame


def load_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the processed feature table (cached mode)."""
    if not PROCESSED_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Processed features are missing. Run "
            "'python -m src.run_pipeline --mode cached' after one live run, "
            "or '--mode live' to rebuild everything."
        )
    frame = pd.read_parquet(PROCESSED_PATH)
    metadata = read_json(METADATA_PATH, default={})
    return frame, metadata


if __name__ == "__main__":  # pragma: no cover - manual helper
    build_features()
