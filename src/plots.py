"""All figures are generated from the real downloaded data.

Nothing here is synthetic: every plot reads the processed feature table or the
predictions/backtests produced by the pipeline.  Figures are written to
``reports/figures`` and are part of the public evidence of the project.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import metrics as metrics_module
from src.utils import get_logger

logger = get_logger("plots")

try:  # pragma: no cover - style name depends on the matplotlib version
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:  # noqa: BLE001
    pass

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def _save(figure: plt.Figure, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved %s", path.name)
    return f"reports/figures/{name}.png"


# ---------------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------------
def plot_normalized_prices(frame: pd.DataFrame) -> str:
    window = frame[frame["date"] >= pd.Timestamp(config.STUDY_START)]
    pivot = window.pivot(index="date", columns="ticker", values="close")
    normalized = pivot / pivot.iloc[0] * 100.0

    figure, axis = plt.subplots(figsize=(11, 5))
    for i, ticker in enumerate(normalized.columns):
        axis.plot(normalized.index, normalized[ticker], label=ticker, color=COLORS[i % len(COLORS)], linewidth=1.4)
    axis.set_title("Adjusted close prices, rebased to 100 (real data)")
    axis.set_ylabel("Index (100 = first session)")
    axis.set_xlabel("Session")
    axis.legend(ncol=6, fontsize=9)
    return _save(figure, "01_prices_normalized")


def plot_target_balance(sample: pd.DataFrame) -> str:
    by_ticker = sample.groupby("ticker")["target_up"].agg(["mean", "count"])
    by_year = sample.assign(year=sample["date"].dt.year).groupby("year")["target_up"].mean()

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(by_ticker.index, by_ticker["mean"], color=COLORS[0])
    axes[0].axhline(sample["target_up"].mean(), color="black", linestyle="--", linewidth=1,
                    label=f"overall {sample['target_up'].mean():.3f}")
    axes[0].set_ylim(0.4, 0.6)
    axes[0].set_title("Share of positive next open-to-close returns by ETF")
    axes[0].set_ylabel("P(target_up = 1)")
    axes[0].legend(fontsize=9)
    for i, (ticker, row) in enumerate(by_ticker.iterrows()):
        axes[0].text(i, row["mean"] + 0.005, f"{row['mean']:.3f}\nn={int(row['count'])}",
                     ha="center", fontsize=8)

    axes[1].plot(by_year.index, by_year.values, marker="o", color=COLORS[3])
    axes[1].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[1].set_title("Share of positive next open-to-close returns by year")
    axes[1].set_ylabel("P(target_up = 1)")
    axes[1].set_xlabel("Year")
    return _save(figure, "02_target_balance")


def plot_feature_correlations(sample: pd.DataFrame, metadata: dict[str, Any], top_n: int = 20) -> tuple[str, pd.DataFrame]:
    """Correlation between numeric features and the target (TRAIN window only)."""
    train = sample[sample["split"] == "train"]
    numeric = [c for c in metadata["numeric_features"] if c in train.columns]
    correlations = (
        train[numeric + ["target_up"]]
        .corr(numeric_only=True)["target_up"]
        .drop("target_up")
        .dropna()
        .sort_values(key=np.abs, ascending=False)
    )
    table = correlations.head(top_n)

    figure, axis = plt.subplots(figsize=(9, 7))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in table.values]
    axis.barh(table.index[::-1], table.values[::-1], color=colors[::-1])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Correlation with target_up - training window only")
    axis.set_xlabel("Pearson correlation")
    return _save(figure, "03_feature_correlation"), correlations.reset_index().rename(
        columns={"index": "feature", "target_up": "correlation_with_target_up"}
    )


def plot_vix_relationship(sample: pd.DataFrame) -> str:
    data = sample.dropna(subset=["vix_lag1", "target_ret_next_oc"]).copy()
    data["vix_bucket"] = pd.cut(
        data["vix_lag1"],
        bins=[0, 12, 15, 18, 22, 28, 100],
        labels=["<12", "12-15", "15-18", "18-22", "22-28", ">28"],
    )
    bucket_returns = data.groupby("vix_bucket", observed=True)["target_ret_next_oc"].mean() * 100
    bucket_counts = data.groupby("vix_bucket", observed=True).size()

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(data["date"], data["vix_lag1"], linewidth=0.8, color=COLORS[4])
    axes[0].set_title("Lagged VIX used as a feature (one session lag)")
    axes[0].set_ylabel("VIX level")

    axes[1].bar(bucket_returns.index.astype(str), bucket_returns.values, color=COLORS[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean next open-to-close return by lagged VIX bucket")
    axes[1].set_ylabel("Mean return (%)")
    for i, label in enumerate(bucket_returns.index.astype(str)):
        axes[1].text(i, bucket_returns.values[i], f"n={bucket_counts.iloc[i]}", ha="center",
                     va="bottom" if bucket_returns.values[i] >= 0 else "top", fontsize=8)
    return _save(figure, "04_vix_relationship")


def plot_feature_distributions(sample: pd.DataFrame) -> str:
    columns = ["ret_1d", "vol_20d", "rsi_14", "volume_z_20", "dist_52w_high", "beta_60d"]
    available = [c for c in columns if c in sample.columns]
    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, column in zip(axes.ravel(), available):
        values = sample[column].replace([np.inf, -np.inf], np.nan).dropna()
        axis.hist(values, bins=60, color=COLORS[0], alpha=0.85)
        axis.set_title(column)
        axis.set_ylabel("rows")
    for axis in axes.ravel()[len(available):]:
        axis.axis("off")
    figure.suptitle("Distributions of key features (modelling sample)", y=1.01)
    return _save(figure, "05_feature_distributions")


# ---------------------------------------------------------------------------
# Model diagnostics
# ---------------------------------------------------------------------------
def plot_roc_curves(predictions: pd.DataFrame, models: list[str]) -> str:
    from sklearn.metrics import roc_curve

    test = predictions[predictions["split"] == "test"]
    start = pd.to_datetime(test["trade_date"]).min().strftime("%Y-%m-%d")
    end = pd.to_datetime(test["trade_date"]).max().strftime("%Y-%m-%d")
    figure, axis = plt.subplots(figsize=(7, 6))
    for i, model in enumerate(models):
        if model not in test.columns:
            continue
        y_true = test["target_up"].astype(int)
        fpr, tpr, _ = roc_curve(y_true, test[model])
        auc = metrics_module.classification_metrics(y_true, test[model], 0.5)["roc_auc"]
        axis.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})", color=COLORS[i % len(COLORS)], linewidth=1.4)
    axis.plot([0, 1], [0, 1], linestyle=":", color="grey", label="random")
    axis.set_title(f"ROC curves on the test window ({start} to {end})")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(fontsize=8)
    return _save(figure, "06_roc_curves")


def plot_calibration(predictions: pd.DataFrame, model: str, threshold: float) -> str:
    test = predictions[(predictions["split"] == "test")].dropna(subset=[model]).copy()
    test["bucket"] = pd.qcut(test[model], q=10, duplicates="drop")
    grouped = test.groupby("bucket", observed=True).agg(
        mean_prob=(model, "mean"), up_rate=("target_up", "mean"), n=("target_up", "size")
    )

    figure, axis = plt.subplots(figsize=(7, 5.4))
    axis.plot(grouped["mean_prob"], grouped["up_rate"], marker="o", color=COLORS[0], label="observed")
    axis.plot([0.4, 0.75], [0.4, 0.75], linestyle=":", color="grey", label="perfect calibration")
    axis.axvline(threshold, color=COLORS[3], linestyle="--", linewidth=1,
                 label=f"selected threshold {threshold:.2f}")
    axis.set_title(f"Calibration of {model} on the test window (deciles)")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed share of positive returns")
    axis.legend(fontsize=8)
    return _save(figure, "07_calibration")


def plot_confusion_matrix(predictions: pd.DataFrame, model: str, threshold: float) -> str:
    test = predictions[predictions["split"] == "test"]
    stats = metrics_module.classification_metrics(test["target_up"], test[model], threshold)
    matrix = np.array(
        [[stats["true_negatives"], stats["false_positives"]],
         [stats["false_negatives"], stats["true_positives"]]]
    )

    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:,}", ha="center", va="center",
                  color="white" if value > matrix.max() / 2 else "black", fontsize=12)
    axis.set_xticks([0, 1], ["Predicted down", "Predicted up"])
    axis.set_yticks([0, 1], ["Actual down", "Actual up"])
    axis.set_title(f"{model}: confusion matrix (test, threshold={threshold:.2f})")
    figure.colorbar(image, ax=axis, fraction=0.046)
    return _save(figure, "08_confusion_matrix")


def plot_feature_importance(importance: dict[str, list[dict[str, Any]]], model: str = "random_forest") -> str | None:
    rows = importance.get(model) or next(iter(importance.values()), None)
    if not rows:
        return None
    labels = [r["feature"] for r in rows][::-1]
    values = [r["importance"] for r in rows][::-1]

    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.barh(labels, values, color=COLORS[2])
    axis.set_title(f"Top-15 feature importances ({model})")
    axis.set_xlabel("Importance")
    return _save(figure, "09_feature_importance")


def plot_accuracy_by_group(predictions: pd.DataFrame, model: str, threshold: float) -> str:
    test = predictions[predictions["split"] == "test"].copy()
    test["prediction"] = (test[model] >= threshold).astype(int)
    test["correct"] = (test["prediction"] == test["target_up"].astype(int)).astype(float)
    test["month"] = test["trade_date"].dt.month

    by_ticker = test.groupby("ticker")["correct"].mean()
    by_month = test.groupby("month")["correct"].mean()
    trades_by_ticker = test[test["prediction"] == 1].groupby("ticker").size()

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(by_ticker.index, by_ticker.values, color=COLORS[0])
    axes[0].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[0].set_ylim(0.4, 0.65)
    axes[0].set_title(f"Accuracy by ETF (test, {model}, threshold={threshold:.2f})")
    axes[0].set_ylabel("Accuracy")
    for i, ticker in enumerate(by_ticker.index):
        axes[0].text(i, by_ticker.values[i] + 0.004, f"trades={int(trades_by_ticker.get(ticker, 0))}",
                     ha="center", fontsize=8)

    axes[1].plot(by_month.index, by_month.values, marker="o", color=COLORS[3])
    axes[1].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Accuracy by calendar month (test window)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Accuracy")
    return _save(figure, "10_accuracy_by_group")


# ---------------------------------------------------------------------------
# Strategy results
# ---------------------------------------------------------------------------
def plot_equity_curves(
    strategies: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    title: str = "Net equity curves on the test window (after costs)",
) -> str:
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    for i, (name, result) in enumerate(strategies.items()):
        axis.plot(result["equity"].index, result["equity"].values, label=name,
                  color=COLORS[i % len(COLORS)], linewidth=1.6)
    for j, (name, result) in enumerate(benchmarks.items()):
        axis.plot(result["equity"].index, result["equity"].values, label=name,
                  color="black" if j == 0 else "grey", linewidth=1.2,
                  linestyle="--" if j == 0 else ":")
    axis.set_title(title)
    axis.set_ylabel(f"Equity (start = {config.INITIAL_CAPITAL:,.0f} USD, simulated)")
    axis.legend(fontsize=8, ncol=2)
    return _save(figure, "11_equity_curves")


def plot_drawdowns(
    strategies: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
) -> str:
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    for i, (name, result) in enumerate(list(strategies.items())[:2]):
        drawdown = metrics_module.drawdown_curve(result["equity"].dropna())
        axis.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.35,
                          color=COLORS[i], label=name)
    for name, result in benchmarks.items():
        drawdown = metrics_module.drawdown_curve(result["equity"].dropna())
        axis.plot(drawdown.index, drawdown.values * 100, color="black", linewidth=1.0,
                  linestyle="--", label=name)
    axis.set_title("Drawdown (%) - test window")
    axis.set_ylabel("Drawdown (%)")
    axis.legend(fontsize=8)
    return _save(figure, "12_drawdowns")


def plot_rolling_returns(
    strategies: dict[str, dict[str, Any]],
    benchmarks: dict[str, dict[str, Any]],
    window: int = 63,
) -> str:
    figure, axis = plt.subplots(figsize=(11.5, 4.6))
    for i, (name, result) in enumerate(list(strategies.items())[:2]):
        rolling = metrics_module.rolling_cumulative(result["returns"], window) * 100
        axis.plot(rolling.index, rolling.values, label=name, color=COLORS[i], linewidth=1.3)
    for name, result in benchmarks.items():
        rolling = metrics_module.rolling_cumulative(result["returns"], window) * 100
        axis.plot(rolling.index, rolling.values, label=name, color="black", linewidth=1.0,
                  linestyle="--" if name == "SPY open-to-close" else ":")
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.set_title(f"Rolling {window}-session cumulative net return (%)")
    axis.set_ylabel("Return (%)")
    axis.legend(fontsize=8)
    return _save(figure, "13_rolling_returns")


def plot_trade_distribution(trades: pd.DataFrame) -> str | None:
    if trades.empty:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    axes[0].hist(trades["net_ret"] * 100, bins=60, color=COLORS[0], alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Distribution of net trade returns (%)")
    axes[0].set_xlabel("Net return per trade (%)")

    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["trade_date"]).dt.to_period("M")
    monthly = trades.groupby("month").size()
    axes[1].bar(monthly.index.astype(str), monthly.values, color=COLORS[1])
    axes[1].set_title("Number of trades per month (best model, S1)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].set_ylabel("Trades")
    return _save(figure, "14_trade_distribution")


def make_eda_figures(sample: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "prices": plot_normalized_prices(sample),
        "target_balance": plot_target_balance(sample),
        "feature_correlation": plot_feature_correlations(sample, metadata)[0],
        "vix": plot_vix_relationship(sample),
        "distributions": plot_feature_distributions(sample),
    }
