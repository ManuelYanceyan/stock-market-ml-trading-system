"""Model training, hyper-parameter tuning and prediction.

Design rules
------------
* Chronological, date-based splits: every fold of the internal cross
  validation uses only past dates (``DatePurgedTimeSeriesSplit``).
* All preprocessing (median imputation, scaling) lives inside the sklearn
  ``Pipeline`` and is therefore fitted on training rows only.
* Hyper-parameters are tuned with ``GridSearchCV`` on the TRAIN window only.
* The probability threshold is selected on the VALIDATION window only
  (see ``src.backtest.select_threshold``).
* The TEST window is used once, for the final honest evaluation.

Models
------
1. ``baseline_majority``  - always predicts the majority class (simple baseline)
2. ``baseline_momentum``  - naive rule: up if the trailing 5-day return is > 0
3. ``decision_tree``      - DecisionTreeClassifier (course model)
4. ``random_forest``      - RandomForestClassifier (course model)
5. ``logistic_regression``- regularised linear model
6. ``xgboost``            - gradient boosting (new model)
7. ``mlp``                - small neural network (new model)
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.build_features import get_feature_columns
from src.metrics import brier_score, classification_metrics
from src.utils import get_logger, utc_now_iso

logger = get_logger("train_models")

try:  # xgboost is optional but installed in the reference environment
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    XGBOOST_AVAILABLE = False

MODEL_NAMES = [
    "baseline_majority",
    "baseline_momentum",
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "xgboost",
    "mlp",
]
if not XGBOOST_AVAILABLE:  # pragma: no cover
    MODEL_NAMES.remove("xgboost")

CONTEXT_COLUMNS = [
    "date",
    "trade_date",
    "ticker",
    "split",
    "target_up",
    "target_ret_next_oc",
    "open_next",
    "high_next",
    "low_next",
    "close_next",
    "close",
    "vix_lag1",
    "mkt_above_ma50",
    "mkt_vol_20d",
    "vol_20d",
    "ret_5d",
    "ret_20d",
]


class DatePurgedTimeSeriesSplit:
    """Chronological CV splitter that groups rows by signal date.

    A plain ``TimeSeriesSplit`` on row indices can place two ETFs of the same
    session in different folds.  This splitter splits on unique dates and
    applies an embargo of ``embargo`` sessions so that no validation fold
    starts immediately after a training fold.
    """

    def __init__(self, n_splits: int = 3, embargo: int = 1) -> None:
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("DatePurgedTimeSeriesSplit requires groups=signal dates")
        groups_series = pd.Series(pd.to_datetime(pd.Series(groups).values))
        unique_dates = pd.DatetimeIndex(sorted(groups_series.unique()))
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in base.split(unique_dates.values):
            train_dates = unique_dates[train_idx]
            if self.embargo > 0 and len(train_dates) > self.embargo:
                train_dates = train_dates[: -self.embargo]
            test_dates = unique_dates[test_idx]
            train_rows = np.where(groups_series.isin(train_dates).values)[0]
            test_rows = np.where(groups_series.isin(test_dates).values)[0]
            if len(train_rows) and len(test_rows):
                yield train_rows, test_rows


def _preprocessor(scale: bool) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_estimator(name: str) -> tuple[Any, dict[str, list[Any]], bool]:
    """Return (estimator, parameter grid, needs_tuning) for a model name."""
    seed = config.RANDOM_SEED

    if name == "baseline_majority":
        return DummyClassifier(strategy="most_frequent"), {}, False

    if name == "decision_tree":
        model = Pipeline(
            _preprocessor(scale=False)
            + [
                (
                    "model",
                    DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
                )
            ]
        )
        grid = {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [50, 200],
            "model__criterion": ["gini"],
        }
        return model, grid, True

    if name == "random_forest":
        model = Pipeline(
            _preprocessor(scale=False)
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=seed,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                )
            ]
        )
        grid = {
            "model__max_depth": [6, 10],
            "model__min_samples_leaf": [20, 100],
            "model__max_features": ["sqrt", 0.3],
        }
        return model, grid, True

    if name == "logistic_regression":
        model = Pipeline(
            _preprocessor(scale=True)
            + [
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                )
            ]
        )
        grid = {"model__C": [0.01, 0.1, 1.0]}
        return model, grid, True

    if name == "xgboost":
        if not XGBOOST_AVAILABLE:  # pragma: no cover
            raise ImportError("xgboost is not installed")
        model = Pipeline(
            _preprocessor(scale=False)
            + [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        )
        grid = {
            "model__max_depth": [3, 5],
            "model__n_estimators": [200, 400],
        }
        return model, grid, True

    if name == "mlp":
        model = Pipeline(
            _preprocessor(scale=True)
            + [
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(48,),
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=150,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=seed,
                    ),
                )
            ]
        )
        grid = {
            "model__hidden_layer_sizes": [(32,), (64, 32)],
            "model__alpha": [1e-3, 1e-2],
        }
        return model, grid, True

    raise ValueError(f"Unknown model: {name}")


def _momentum_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """Naive baseline: probability 1 when the trailing 5-day return is positive."""
    return (frame["ret_5d"].fillna(0.0).to_numpy() > 0).astype(float)


def train_all(
    sample: pd.DataFrame,
    metadata: dict[str, Any],
    tune: bool = True,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Fit every model on the training window and predict validation + test."""
    numeric, dummies = get_feature_columns(sample, metadata)
    features = numeric + dummies
    models = models or MODEL_NAMES

    train = sample[sample["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise ValueError("Training split is empty - check the split configuration")

    X_train = train[features]
    y_train = train["target_up"].astype(int)
    groups = train["date"]

    prediction_rows = sample[sample["split"].isin(["validation", "test"])].copy()
    keep_columns = features + [c for c in CONTEXT_COLUMNS if c in prediction_rows.columns]
    prediction_rows = prediction_rows[list(dict.fromkeys(keep_columns))].reset_index(drop=True)

    artifacts: dict[str, Any] = {
        "feature_columns": features,
        "numeric_features": numeric,
        "dummy_features": dummies,
        "models": {},
        "cv_results": {},
        "best_params": {},
        "timings": {},
        "train_rows": int(len(train)),
        "train_sessions": int(train["date"].nunique()),
        "tested_at_utc": utc_now_iso(),
        "random_seed": config.RANDOM_SEED,
    }
    probabilities = pd.DataFrame(index=prediction_rows.index)

    for name in models:
        start = time.perf_counter()
        logger.info("Training %s ...", name)

        if name == "baseline_majority":
            model, grid, needs_tuning = build_estimator(name)
            # DummyClassifier ignores the features; a single column keeps the
            # API identical to the other models.
            model.fit(np.zeros((len(y_train), 1)), y_train)
            probabilities[name] = model.predict_proba(np.zeros((len(prediction_rows), 1)))[:, 1]
            estimator = model
        elif name == "baseline_momentum":
            probabilities[name] = _momentum_probabilities(prediction_rows)
            estimator = None
        else:
            model, grid, needs_tuning = build_estimator(name)
            if tune and needs_tuning and len(grid):
                search = GridSearchCV(
                    model,
                    param_grid=grid,
                    scoring="roc_auc",
                    cv=DatePurgedTimeSeriesSplit(n_splits=3, embargo=1),
                    n_jobs=-1,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train, groups=groups)
                estimator = search.best_estimator_
                artifacts["best_params"][name] = search.best_params_
                artifacts["cv_results"][name] = {
                    "best_cv_roc_auc": float(search.best_score_),
                    "n_candidates": len(search.cv_results_["params"]),
                    "cv_folds": 3,
                    "cv_scheme": "date-grouped TimeSeriesSplit, embargo=1",
                }
            else:
                estimator = model.fit(X_train, y_train)
                artifacts["best_params"][name] = {}
                artifacts["cv_results"][name] = {"best_cv_roc_auc": None}
            probabilities[name] = estimator.predict_proba(prediction_rows[features])[:, 1]

        artifacts["models"][name] = estimator
        artifacts["timings"][name] = round(time.perf_counter() - start, 2)
        logger.info(
            "%s done in %.1fs", name, artifacts["timings"][name]
        )

    predictions = pd.concat([prediction_rows, probabilities], axis=1)
    predictions = predictions.rename(columns={"date": "signal_date"})
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    artifacts["predictions"] = predictions
    artifacts["feature_importance"] = extract_feature_importance(artifacts["models"], features)
    return artifacts


def extract_feature_importance(models: dict[str, Any], features: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Top-15 feature importances (or absolute coefficients) per model."""
    result: dict[str, list[dict[str, Any]]] = {}
    for name, estimator in models.items():
        if estimator is None:
            continue
        model = estimator.named_steps.get("model") if hasattr(estimator, "named_steps") else None
        if model is None:
            continue
        values = None
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_).ravel())
        if values is None or len(values) != len(features):
            continue
        order = np.argsort(values)[::-1][:15]
        result[name] = [
            {"feature": features[i], "importance": float(values[i])} for i in order
        ]
    return result


def for_backtest(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return a copy whose ``prob`` column is the selected model's probability.

    The backtest module works on a single probability column so that the same
    rules can be applied to any model without special casing.
    """
    frame = predictions.copy()
    if model not in frame.columns:
        raise KeyError(f"Model {model} not present in predictions")
    frame["prob"] = frame[model].astype(float)
    return frame


def evaluate_predictions(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Classification metrics per model and split (validation / test)."""
    report: dict[str, dict[str, Any]] = {}
    for model in [c for c in predictions.columns if c in MODEL_NAMES]:
        report[model] = {}
        for split in ("validation", "test"):
            subset = predictions[predictions["split"] == split]
            if subset.empty:
                continue
            threshold = thresholds.get(model, 0.5)
            fixed = classification_metrics(subset["target_up"], subset[model], 0.5)
            tuned = classification_metrics(subset["target_up"], subset[model], threshold)
            tuned["brier_score"] = brier_score(subset["target_up"], subset[model])
            tuned["mean_probability"] = float(subset[model].mean())
            report[model][split] = {"threshold_0_5": fixed, "at_selected_threshold": tuned}
    return report


def refit_on_all(
    sample: pd.DataFrame,
    metadata: dict[str, Any],
    model_name: str,
    best_params: dict[str, Any] | None = None,
) -> Any:
    """Refit the chosen model on ALL labelled data for the live signal.

    This refit happens AFTER the test evaluation and is used only to produce
    the latest simulated signal - it never contributes to the reported test
    metrics.
    """
    if model_name == "baseline_momentum":
        return None
    numeric, dummies = get_feature_columns(sample, metadata)
    features = numeric + dummies
    estimator, _, _ = build_estimator(model_name)
    if best_params:
        estimator.set_params(**best_params)
    estimator.fit(sample[features], sample["target_up"].astype(int))
    return estimator


def predict_latest(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    model_name: str,
    estimator: Any,
) -> pd.DataFrame:
    """Probability for the NEXT session using the latest completed session."""
    numeric, dummies = get_feature_columns(frame, metadata)
    features = numeric + dummies

    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date].copy()
    if latest.empty:
        raise ValueError("No rows available for the latest session")

    if model_name == "baseline_momentum":
        probabilities = _momentum_probabilities(latest)
    else:
        probabilities = estimator.predict_proba(latest[features])[:, 1]

    output = latest[
        ["date", "ticker", "close", "vix_lag1", "mkt_above_ma50", "vol_20d", "ret_5d"]
    ].copy()
    output = output.rename(columns={"date": "signal_date"})
    output["prob_up_next_session"] = probabilities
    output = output.sort_values("prob_up_next_session", ascending=False).reset_index(drop=True)
    output["signal_session"] = str(latest_date.date())
    return output


if __name__ == "__main__":  # pragma: no cover - manual helper
    from src.build_features import load_features

    frame, meta = load_features()
    result = train_all(frame, meta)
    print(result["timings"])
