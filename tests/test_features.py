"""Feature engineering tests: per-ticker lags, targets, leakage and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.build_features import (
    MACRO_LAG_SESSIONS,
    add_targets,
    assign_splits,
    build_feature_panel,
    build_modeling_sample,
    get_feature_columns,
    split_frames,
)


def test_target_is_next_session_open_to_close(synthetic_ohlcv):
    panel = add_targets(
        synthetic_ohlcv[synthetic_ohlcv["ticker"] == "SPY"].sort_values("date").head(120)
    )

    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        expected_trade_date = group["date"].shift(-1)
        expected_return = group["close"].shift(-1) / group["open"].shift(-1) - 1.0

        valid = group["trade_date"].notna()
        assert (group.loc[valid, "trade_date"] == expected_trade_date[valid]).all()
        np.testing.assert_allclose(
            group.loc[valid, "target_ret_next_oc"], expected_return[valid], rtol=1e-12
        )
        # The label sign matches the return sign.
        assert (group.loc[valid, "target_up"] == (expected_return[valid] > 0).astype(float)).all()


def test_per_ticker_lags_do_not_mix_tickers(feature_panel):
    frame, metadata = feature_panel

    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        expected = group["close"].pct_change()
        # First row of every ticker is NaN (no cross-ticker contamination).
        assert np.isnan(group["ret_1d"].iloc[0])
        np.testing.assert_allclose(group["ret_1d"].iloc[1:], expected.iloc[1:], rtol=1e-10)


def test_feature_sets_are_defined(feature_panel):
    frame, metadata = feature_panel
    numeric, dummies = get_feature_columns(frame, metadata)

    assert len(numeric) >= 40
    assert len(dummies) >= 20
    assert "target_up" not in numeric + dummies
    assert "target_ret_next_oc" not in numeric + dummies
    assert "close_next" not in numeric + dummies

    # Dummies are binary and the ticker block is one-hot.
    for column in dummies:
        assert set(frame[column].dropna().unique()).issubset({0, 1}), column
    ticker_columns = [c for c in dummies if c.startswith("ticker_")]
    assert frame[ticker_columns].sum(axis=1).eq(1).all()


def test_no_lookahead_when_history_is_truncated(synthetic_ohlcv, synthetic_macro, synthetic_factors, feature_panel):
    """Recompute features with data truncated before D; values must be identical."""
    frame_full, metadata = feature_panel
    cutoff = pd.Timestamp("2021-06-15")

    ohlcv_truncated = synthetic_ohlcv[synthetic_ohlcv["date"] < cutoff]
    macro_truncated = synthetic_macro[synthetic_macro["date"] < cutoff]
    factors_truncated = synthetic_factors[synthetic_factors["date"] < cutoff]

    frame_truncated, _ = build_feature_panel(ohlcv_truncated, macro_truncated, factors_truncated)

    numeric, dummies = get_feature_columns(frame_full, metadata)
    columns = numeric + dummies + ["date", "ticker"]
    left = frame_full[frame_full["date"] < cutoff][columns].reset_index(drop=True)
    right = frame_truncated[frame_truncated["date"] < cutoff][columns].reset_index(drop=True)

    assert len(left) == len(right)
    for column in numeric + dummies:
        np.testing.assert_allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Feature {column} changes when future data is removed (look-ahead)",
        )


def test_macro_features_are_lagged(synthetic_macro, feature_panel):
    frame, _ = feature_panel
    merged = frame.dropna(subset=["vix_lag1"]).copy()
    dates = pd.DatetimeIndex(sorted(merged["date"].unique()))
    vix_by_date = (
        synthetic_macro.dropna(subset=["vix"]).set_index("date")["vix"].sort_index()
    )

    for date in dates[1:50]:
        previous = dates[dates < date][-1]
        expected = vix_by_date.loc[:previous].iloc[-1]
        actual = merged.loc[merged["date"] == date, "vix_lag1"].iloc[0]
        assert actual == pytest.approx(expected)


def test_french_factors_use_only_values_available_five_sessions_earlier(
    synthetic_factors, feature_panel
):
    from src.build_features import FACTOR_LAG_SESSIONS

    frame, _ = feature_panel
    calendar = pd.DatetimeIndex(sorted(synthetic_factors["date"].unique()))
    target = calendar[300]
    expected = (
        synthetic_factors.set_index("date")["ff_mkt_rf"]
        .sort_index()
        .loc[calendar[300 - FACTOR_LAG_SESSIONS]]
    )

    row = frame[(frame["date"] == target) & (frame["ticker"] == "SPY")].iloc[0]
    assert row["ff_mkt_rf_lag5"] == pytest.approx(expected)


def test_rolling_features_only_use_past(feature_panel):
    frame, _ = feature_panel
    group = frame[frame["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

    index = 600
    window_close = group["close"].iloc[index - 19 : index + 1]
    expected = window_close.mean()
    actual = group["close"].iloc[index] / expected - 1.0
    assert group["price_vs_ma20"].iloc[index] == pytest.approx(actual, rel=1e-9)


def test_splits_are_chronological_and_do_not_overlap(sample_frame):
    sample, metadata, info = sample_frame
    splits = split_frames(sample)

    # Ordered, non-overlapping date ranges.
    assert splits["train"]["date"].max() < splits["validation"]["date"].min()
    assert splits["validation"]["date"].max() < splits["test"]["date"].min()
    for part in splits.values():
        assert part["date"].is_monotonic_increasing

    # No split reuses a signal date from another split.
    signatures = {
        name: set(zip(part["date"], part["ticker"])) for name, part in splits.items()
    }
    assert not (signatures["train"] & signatures["validation"])
    assert not (signatures["validation"] & signatures["test"])
    assert not (signatures["train"] & signatures["test"])


def test_label_windows_stay_inside_each_split(sample_frame):
    sample, metadata, info = sample_frame
    boundaries = {
        "train": (pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END)),
        "validation": (pd.Timestamp(config.VALID_START), pd.Timestamp(config.VALID_END)),
        "test": (pd.Timestamp(config.TEST_START), pd.Timestamp("2262-01-01")),
    }
    for name, (start, end) in boundaries.items():
        part = sample[sample["split"] == name]
        if part.empty:
            continue
        assert part["date"].min() >= start
        assert part["date"].max() <= end
        # The realised return of every row belongs to the same split.
        assert pd.to_datetime(part["trade_date"]).max() <= end
        info_block = info["boundaries"][name]
        if name != "test":
            # The last signal of a closed split is removed when its label would
            # fall into the next split; the test split has no upper boundary.
            assert info_block["dropped_at_boundary"] >= 1
        assert pd.to_datetime(info_block["trade_end"]) <= end


def test_no_missing_targets_in_modelling_sample(sample_frame):
    sample, _, _ = sample_frame
    assert sample["target_up"].notna().all()
    assert sample["target_ret_next_oc"].notna().all()
    assert sample["trade_date"].notna().all()
    assert set(sample["target_up"].unique()).issubset({0.0, 1.0})


def test_modelling_sample_drops_rows_with_too_many_missing_features(feature_panel):
    frame, metadata = feature_panel
    sample = build_modeling_sample(frame, metadata)
    numeric, dummies = get_feature_columns(frame, metadata)
    missing_share = sample[numeric + dummies].isna().mean(axis=1)
    assert missing_share.max() <= config.MAX_FEATURE_MISSING_SHARE + 1e-12
