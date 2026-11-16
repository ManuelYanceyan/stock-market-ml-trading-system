"""Model pipeline tests: chronological CV, preprocessing isolation, predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import train_models
from src.build_features import get_feature_columns
from src.train_models import DatePurgedTimeSeriesSplit, build_estimator, for_backtest, train_all


def test_date_grouped_splitter_is_chronological_and_grouped():
    dates = pd.bdate_range("2020-01-01", periods=40)
    groups = pd.Series(list(dates) * 3)
    splitter = DatePurgedTimeSeriesSplit(n_splits=3, embargo=1)

    folds = list(splitter.split(np.zeros(len(groups)), groups=groups))
    assert len(folds) == 3

    for train_rows, test_rows in folds:
        train_dates = set(groups.iloc[train_rows])
        test_dates = set(groups.iloc[test_rows])
        assert max(train_dates) < min(test_dates), "training data must precede test data"
        assert not (train_dates & test_dates), "a session cannot be in both folds"
        # Rows of a single date stay together (no ETF-level leakage across folds).
        assert len(train_rows) == 3 * len(train_dates)
        assert len(test_rows) == 3 * len(test_dates)
        # Embargo: at least one session gap between the folds.
        gap = pd.DatetimeIndex(sorted(test_dates)).min() - pd.DatetimeIndex(sorted(train_dates)).max()
        assert gap >= pd.Timedelta(days=2)


def test_preprocessing_is_fitted_on_training_rows_only():
    estimator, _, _ = build_estimator("decision_tree")
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 100.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1])

    estimator.fit(frame, y)
    imputer = estimator.named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_[0], 3.0)
    np.testing.assert_allclose(imputer.statistics_[1], 35.0)
    # The scaler is not part of the tree pipeline (trees do not need it).
    assert "scaler" not in estimator.named_steps


def test_for_backtest_selects_the_model_probability():
    predictions = pd.DataFrame({"decision_tree": [0.2, 0.8], "random_forest": [0.6, 0.4]})
    selected = for_backtest(predictions, "random_forest")
    assert selected["prob"].tolist() == [0.6, 0.4]
    with pytest.raises(KeyError):
        for_backtest(predictions, "not_a_model")


def test_models_can_be_built_and_tuned(sample_frame):
    sample, metadata, _ = sample_frame
    numeric, dummies = get_feature_columns(sample, metadata)
    X = sample[sample["split"] == "train"][numeric + dummies]
    y = sample[sample["split"] == "train"]["target_up"].astype(int)

    estimator, grid, needs_tuning = build_estimator("decision_tree")
    assert needs_tuning and grid
    estimator.fit(X, y)
    probabilities = estimator.predict_proba(X)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_train_all_returns_predictions_for_validation_and_test(sample_frame):
    sample, metadata, _ = sample_frame
    artifacts = train_all(sample, metadata, tune=False, models=["decision_tree"])

    predictions = artifacts["predictions"]
    assert set(predictions["split"].unique()) == {"validation", "test"}
    expected_rows = len(sample[sample["split"].isin(["validation", "test"])])
    assert len(predictions) == expected_rows
    assert predictions["decision_tree"].between(0, 1).all()
    assert (predictions["signal_date"] < predictions["trade_date"]).all()
    assert artifacts["feature_columns"] == get_feature_columns(sample, metadata)[0] + get_feature_columns(sample, metadata)[1]
