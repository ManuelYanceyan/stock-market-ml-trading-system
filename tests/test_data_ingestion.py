"""Data ingestion tests: schema, cleaning rules and chronological ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build_features import align_daily_source
from src.download_data import OHLCV_COLUMNS, validate_ohlcv


def test_ohlcv_schema_and_ordering(synthetic_ohlcv):
    clean, report = validate_ohlcv(synthetic_ohlcv)

    assert list(clean.columns) == OHLCV_COLUMNS
    assert set(clean["ticker"].unique()) == {"SPY", "QQQ", "XLK"}
    assert report["duplicate_dates_remaining"] == 0
    assert report["chronological_order_ok"] is True
    assert clean["date"].is_monotonic_increasing
    assert (clean["high"] >= clean["low"]).all()
    assert (clean[["open", "high", "low", "close"]] > 0).all().all()
    # (date, ticker) is a unique key
    assert not clean.duplicated(subset=["date", "ticker"]).any()
    assert report["per_ticker"]["SPY"]["rows"] > 2000


def test_duplicates_and_bad_prices_are_removed(synthetic_ohlcv):
    dirty = synthetic_ohlcv.copy()
    duplicate = dirty.iloc[[10]].copy()
    for column in ["open", "high", "low", "close"]:
        duplicate[column] = duplicate[column] * 2  # "newer" vendor row must win
    dirty = pd.concat([dirty, duplicate], ignore_index=True)

    bad_low = dirty.iloc[[20]].copy()
    bad_low["high"] = bad_low["low"] * 0.5  # high < low -> impossible bar
    negative = dirty.iloc[[30]].copy()
    negative["close"] = -1.0
    dirty = pd.concat([dirty, bad_low, negative], ignore_index=True)

    clean, report = validate_ohlcv(dirty)

    assert report["dropped_duplicates"] >= 1
    assert report["dropped_inconsistent_bars"] >= 1
    assert report["dropped_non_positive_or_non_finite"] >= 1
    assert (clean["close"] > 0).all()
    assert (clean["high"] >= clean["low"]).all()
    assert not clean.duplicated(subset=["date", "ticker"]).any()
    # The duplicate keeps the LAST row (vendor corrections win).
    key_date = duplicate["date"].iloc[0]
    kept = clean[(clean["date"] == key_date) & (clean["ticker"] == duplicate["ticker"].iloc[0])]
    assert len(kept) == 1
    assert np.isclose(kept["close"].iloc[0], duplicate["close"].iloc[0])


def test_negative_or_missing_volume_rows_are_removed(synthetic_ohlcv):
    dirty = synthetic_ohlcv.copy()
    negative = dirty.iloc[[40]].copy()
    negative["volume"] = -1
    missing = dirty.iloc[[41]].copy()
    missing["volume"] = np.nan
    dirty = pd.concat([dirty, negative, missing], ignore_index=True)

    clean, report = validate_ohlcv(dirty)

    assert report["dropped_invalid_volume"] >= 2
    assert np.isfinite(clean["volume"]).all()
    assert (clean["volume"] >= 0).all()


def test_missing_macro_observations_are_forward_filled_only(synthetic_macro):
    calendar = pd.bdate_range("2020-01-01", periods=10)
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-07"]),
            "vix": [10.0, np.nan, 30.0],
        }
    )
    aligned = align_daily_source(macro, calendar, lag_sessions=1, columns=["vix"])

    # 2020-01-01 observation is only usable from 2020-01-02 (one session lag).
    assert np.isnan(aligned.loc[pd.Timestamp("2020-01-01"), "vix"])
    assert aligned.loc[pd.Timestamp("2020-01-02"), "vix"] == pytest.approx(10.0)
    # A missing observation is forward-filled from the PAST (10.0), never back-filled.
    assert aligned.loc[pd.Timestamp("2020-01-03"), "vix"] == pytest.approx(10.0)
    assert aligned.loc[pd.Timestamp("2020-01-07"), "vix"] == pytest.approx(10.0)
    # The 2020-01-07 observation becomes usable on 2020-01-08.
    assert aligned.loc[pd.Timestamp("2020-01-08"), "vix"] == pytest.approx(30.0)
    # Future values never leak backwards.
    assert aligned.loc[pd.Timestamp("2020-01-02"), "vix"] != pytest.approx(30.0)


def test_macro_alignment_uses_previous_session_only():
    calendar = pd.bdate_range("2021-03-01", periods=5)
    macro = pd.DataFrame({"date": calendar, "vix": [11.0, 12.0, 13.0, 14.0, 15.0]})
    aligned = align_daily_source(macro, calendar, lag_sessions=1, columns=["vix"])

    assert np.isnan(aligned["vix"].iloc[0])
    assert aligned["vix"].iloc[1:].tolist() == [11.0, 12.0, 13.0, 14.0]


def test_empty_download_raises():
    with pytest.raises(ValueError):
        validate_ohlcv(pd.DataFrame(columns=OHLCV_COLUMNS))


def test_schema_error_is_reported():
    with pytest.raises(ValueError, match="missing columns"):
        validate_ohlcv(pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "ticker": ["SPY"]}))
