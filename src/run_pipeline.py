"""End-to-end pipeline.

Usage
-----
    python -m src.run_pipeline --mode live     # download + validate + features
                                               # + train + backtest + reports
    python -m src.run_pipeline --mode cached   # same, but no network access
    python -m src.run_pipeline --mode report   # rebuild reports only

The ``cached`` mode requires a successful ``live`` run (raw parquet files and
processed features on disk).  The pipeline prints the latest SIMULATED signal
and never contacts a broker.
"""

from __future__ import annotations

import argparse
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

import config
from src import backtest as backtest_module
from src import plots as plots_module
from src import train_models
from src.build_features import (
    assign_splits,
    build_modeling_sample,
    get_feature_columns,
    load_features,
    load_full_features,
    split_frames,
)
from src.download_data import RAW_MACRO_PATH, RAW_FACTORS_PATH, load_raw_cache, update_cache
from src.metrics import classification_metrics
from src.utils import (
    ensure_directories,
    get_logger,
    read_json,
    save_parquet,
    utc_now_iso,
    write_json,
)

logger = get_logger("run_pipeline")

PREDICTIONS_PATH = config.PROCESSED_DIR / "predictions.parquet"
METRICS_PATH = config.REPORTS_DIR / "metrics.json"
COST_BPS_SIDE = config.COMMISSION_BPS + config.SLIPPAGE_BPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _versions() -> dict[str, str]:
    import matplotlib
    import sklearn
    import yfinance

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "yfinance": yfinance.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:  # noqa: BLE001
        versions["xgboost"] = "not installed"
    return versions


def _load_or_rebuild_features(mode: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if mode == "cached":
        try:
            sample, metadata = load_features()
            manifest = read_json(config.RAW_DIR / "manifest.json", default={})
            logger.info("Cached features loaded: %s rows", len(sample))
            return sample, metadata, manifest
        except FileNotFoundError:
            logger.warning("Processed features missing; rebuilding them from the raw cache")
            if not (RAW_MACRO_PATH.exists() and RAW_FACTORS_PATH.exists()):
                raise RuntimeError(
                    "No cached raw data available. Run 'python -m src.run_pipeline --mode live' first."
                )
            from src.build_features import build_features

            sample, metadata = build_features()
            manifest = read_json(config.RAW_DIR / "manifest.json", default={})
            return sample, metadata, manifest
    raise ValueError(f"Unsupported mode for feature loading: {mode}")


def _gross_metrics(
    predictions: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Gross (no cost) results per strategy, for the cost-drag discussion."""
    trades_data = backtest_module.predictions_to_trades(predictions)
    gross: dict[str, dict[str, float]] = {}
    for name, spec in backtest_module.STRATEGIES.items():
        positions = backtest_module.build_positions(predictions, threshold, spec)
        simulation = backtest_module.simulate_exact(
            positions, trades_data, session_index, cost_bps_side=0.0, stop_loss=spec.stop_loss
        )
        gross[name] = {
            "gross_total_return": simulation["metrics"]["total_return"],
            "gross_cagr": simulation["metrics"]["cagr"],
            "gross_sharpe": simulation["metrics"]["sharpe_ratio"],
        }
    return gross


def _error_analysis(predictions: pd.DataFrame, model: str, threshold: float) -> dict[str, Any]:
    """Where the predictions fail: by probability bucket, ETF and month."""
    test = predictions[predictions["split"] == "test"].dropna(subset=[model]).copy()
    if test.empty:
        return {}
    test["prediction"] = (test[model] >= threshold).astype(int)
    test["correct"] = (test["prediction"] == test["target_up"].astype(int))
    test["month"] = pd.to_datetime(test["trade_date"]).dt.month

    buckets = pd.qcut(test[model], q=10, duplicates="drop")
    bucket_table = (
        test.assign(bucket=buckets)
        .groupby("bucket", observed=True)
        .agg(
            observations=("target_up", "size"),
            mean_predicted_probability=(model, "mean"),
            accuracy=("correct", "mean"),
            mean_next_return=("target_ret_next_oc", "mean"),
            observed_up_rate=("target_up", "mean"),
        )
        .reset_index()
    )
    bucket_table["bucket"] = bucket_table["bucket"].astype(str)

    by_ticker = (
        test.groupby("ticker")
        .apply(
            lambda d: pd.Series(
                {
                    "observations": len(d),
                    "accuracy": float(d["correct"].mean()),
                    "precision_when_predicted_up": float(
                        d.loc[d["prediction"] == 1, "target_up"].mean()
                    ) if (d["prediction"] == 1).any() else None,
                    "predicted_up_share": float((d["prediction"] == 1).mean()),
                    "mean_next_return_when_pred_up": float(
                        d.loc[d["prediction"] == 1, "target_ret_next_oc"].mean()
                    ) if (d["prediction"] == 1).any() else None,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    by_month = (
        test.groupby("month")
        .agg(
            observations=("target_up", "size"),
            accuracy=("correct", "mean"),
            mean_next_return=("target_ret_next_oc", "mean"),
        )
        .reset_index()
    )

    correct = test[test["correct"]]
    wrong = test[~test["correct"]]
    return {
        "by_probability_bucket": [
            {
                "bucket": row["bucket"],
                "observations": int(row["observations"]),
                "mean_predicted_probability": round(float(row["mean_predicted_probability"]), 4),
                "accuracy": round(float(row["accuracy"]), 4),
                "mean_next_return": round(float(row["mean_next_return"]), 5),
                "observed_up_rate": round(float(row["observed_up_rate"]), 4),
            }
            for _, row in bucket_table.iterrows()
        ],
        "by_ticker": [
            {
                "ticker": row["ticker"],
                "observations": int(row["observations"]),
                "accuracy": round(float(row["accuracy"]), 4),
                "precision_when_predicted_up": (
                    round(float(row["precision_when_predicted_up"]), 4)
                    if row["precision_when_predicted_up"] is not None and not pd.isna(row["precision_when_predicted_up"])
                    else None
                ),
                "predicted_up_share": round(float(row["predicted_up_share"]), 4),
                "mean_next_return_when_pred_up": (
                    round(float(row["mean_next_return_when_pred_up"]), 5)
                    if row["mean_next_return_when_pred_up"] is not None and not pd.isna(row["mean_next_return_when_pred_up"])
                    else None
                ),
            }
            for _, row in by_ticker.iterrows()
        ],
        "by_month": [
            {
                "month": int(row["month"]),
                "observations": int(row["observations"]),
                "accuracy": round(float(row["accuracy"]), 4),
                "mean_next_return": round(float(row["mean_next_return"]), 5),
            }
            for _, row in by_month.iterrows()
        ],
        "correct_vs_wrong": {
            "mean_abs_return_when_correct": float(correct["target_ret_next_oc"].abs().mean()),
            "mean_abs_return_when_wrong": float(wrong["target_ret_next_oc"].abs().mean()),
            "mean_prob_correct": float(correct[model].mean()),
            "mean_prob_wrong": float(wrong[model].mean()),
            "share_correct": float(test["correct"].mean()),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def run(mode: str = "live", tune: bool = True, refresh: bool = False, make_figures: bool = True) -> dict[str, Any]:
    ensure_directories()
    started = utc_now_iso()
    logger.info("=== Pipeline start (mode=%s) ===", mode)

    # 1-2. Download + validate -------------------------------------------------
    if mode == "live":
        manifest = update_cache(force_refresh=refresh)
        from src.build_features import build_features

        sample, metadata = build_features()
        logger.info("Features rebuilt from the refreshed raw cache")
    else:
        sample, metadata, manifest = _load_or_rebuild_features(mode)

    splits = split_frames(sample)
    if splits["train"].empty or splits["validation"].empty or splits["test"].empty:
        raise RuntimeError("One of the splits is empty - check config.py date boundaries")

    dataset_summary = {
        "raw_manifest": manifest,
        "rows_modelling_sample": int(len(sample)),
        "sessions": int(sample["date"].nunique()),
        "first_signal_date": sample["date"].min().strftime("%Y-%m-%d"),
        "last_signal_date": sample["date"].max().strftime("%Y-%m-%d"),
        "last_trade_date": pd.to_datetime(sample["trade_date"]).max().strftime("%Y-%m-%d"),
        "numeric_features": metadata["numeric_features"],
        "dummy_features": metadata["dummy_features"],
        "n_numeric": metadata["n_numeric"],
        "n_dummies": metadata["n_dummies"],
        "split_info": metadata["split_info"],
        "class_balance": {
            name: {
                "rows": int(len(part)),
                "sessions": int(part["date"].nunique()),
                "positive_share": float(part["target_up"].mean()),
                "mean_next_oc_return": float(part["target_ret_next_oc"].mean()),
            }
            for name, part in splits.items()
        },
    }
    logger.info("Splits: %s", {k: len(v) for k, v in splits.items()})

    # 3. Train models ----------------------------------------------------------
    artifacts = train_models.train_all(sample, metadata, tune=tune)
    predictions = artifacts["predictions"]
    save_parquet(predictions, PREDICTIONS_PATH)

    validation_predictions = predictions[predictions["split"] == "validation"].copy()
    test_predictions = predictions[predictions["split"] == "test"].copy()
    validation_sessions = pd.DatetimeIndex(sorted(pd.to_datetime(validation_predictions["trade_date"]).unique()))
    test_sessions = pd.DatetimeIndex(sorted(pd.to_datetime(test_predictions["trade_date"]).unique()))

    # 4. Threshold selection on VALIDATION only --------------------------------
    thresholds: dict[str, Any] = {}
    for model in train_models.MODEL_NAMES:
        if model not in validation_predictions.columns:
            continue
        thresholds[model] = backtest_module.select_threshold(
            train_models.for_backtest(validation_predictions, model),
            validation_sessions,
            cost_bps_side=COST_BPS_SIDE,
        )
        logger.info(
            "%s: validation threshold %.2f (validation trades=%s, net return=%.4f)",
            model,
            thresholds[model]["selected_threshold"],
            thresholds[model]["selected"]["trades"],
            thresholds[model]["selected"]["net_total_return"],
        )

    classification = train_models.evaluate_predictions(
        predictions, {m: t["selected_threshold"] for m, t in thresholds.items()}
    )

    # 5. Best model = highest validation net return of the S1 rule -------------
    ranked = sorted(
        (
            (model, info["selected"]["net_total_return"], info["selected"]["trades"])
            for model, info in thresholds.items()
        ),
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    best_model = ranked[0][0]
    best_threshold = thresholds[best_model]["selected_threshold"]
    logger.info("Best model on validation: %s (threshold %.2f)", best_model, best_threshold)

    # 6. Backtests -------------------------------------------------------------
    validation_suite = backtest_module.run_suite(
        train_models.for_backtest(validation_predictions, best_model),
        validation_sessions,
        best_threshold,
        COST_BPS_SIDE,
    )
    test_predictions_best = train_models.for_backtest(test_predictions, best_model)
    test_suite = backtest_module.run_suite(
        test_predictions_best, test_sessions, best_threshold, COST_BPS_SIDE
    )

    trades_test = backtest_module.predictions_to_trades(test_predictions)
    benchmark_oc = backtest_module.benchmark_intraday(trades_test, test_sessions, cost_bps_side=COST_BPS_SIDE)
    benchmark_oc_gross = backtest_module.benchmark_intraday(trades_test, test_sessions, cost_bps_side=0.0)

    spy_closes = (
        sample[sample["ticker"] == config.BENCHMARK_TICKER]
        .set_index("trade_date")["close_next"]
        .reindex(test_sessions)
    )
    buy_and_hold = backtest_module.benchmark_buy_and_hold(spy_closes.dropna(), cost_bps_side=COST_BPS_SIDE)

    gross = _gross_metrics(test_predictions_best, test_sessions, best_threshold)
    for name, values in gross.items():
        test_suite[name]["metrics"].update(values)
        net_return = test_suite[name]["metrics"]["total_return"]
        test_suite[name]["metrics"]["cost_drag_total_return"] = float(net_return - values["gross_total_return"])

    # S1 performance of every model on the test window (model comparison).
    model_comparison = {}
    for model in train_models.MODEL_NAMES:
        if model not in test_predictions.columns:
            continue
        threshold = thresholds[model]["selected_threshold"]
        positions = backtest_module.build_positions(
            train_models.for_backtest(test_predictions, model),
            threshold,
            backtest_module.STRATEGIES["S1_top_probability"],
        )
        simulation = backtest_module.simulate_exact(
            positions, trades_test, test_sessions, cost_bps_side=COST_BPS_SIDE
        )
        model_comparison[model] = {
            "threshold_selected_on_validation": threshold,
            "net_total_return": simulation["metrics"]["total_return"],
            "cagr": simulation["metrics"]["cagr"],
            "sharpe_ratio": simulation["metrics"]["sharpe_ratio"],
            "max_drawdown": simulation["metrics"]["max_drawdown"],
            "total_trades": simulation["metrics"]["total_trades"],
            "win_rate": simulation["metrics"]["win_rate"],
            "accuracy_at_threshold": classification[model]["test"]["at_selected_threshold"]["accuracy"],
            "roc_auc": classification[model]["test"]["at_selected_threshold"]["roc_auc"],
        }

    # 7. Latest simulated signal ----------------------------------------------
    final_estimator = train_models.refit_on_all(
        sample, metadata, best_model, artifacts["best_params"].get(best_model)
    )
    latest_frame = load_full_features()
    latest_signal = train_models.predict_latest(latest_frame, metadata, best_model, final_estimator)
    top = latest_signal.iloc[0]
    would_trade = bool(top["prob_up_next_session"] >= best_threshold)

    latest_payload = {
        "generated_at_utc": utc_now_iso(),
        "signal_session": str(top["signal_session"]),
        "execution": "next session OPEN -> next session CLOSE (simulated only)",
        "model": best_model,
        "threshold": float(best_threshold),
        "selected_ticker": str(top["ticker"]),
        "selected_probability": float(top["prob_up_next_session"]),
        "would_trade": would_trade,
        "ranking": latest_signal.to_dict(orient="records"),
        "note": (
            "SIMULATION ONLY. The model was refit on all labelled data after the "
            "test evaluation; no order is sent anywhere."
        ),
    }

    # 8. Persist metrics and tables -------------------------------------------
    metrics_payload: dict[str, Any] = {
        "meta": {
            "project": config.PROJECT_NAME,
            "title": config.PROJECT_TITLE,
            "generated_at_utc": utc_now_iso(),
            "pipeline_started_utc": started,
            "run_mode": mode,
            "data_cutoff_reported_by_manifest": manifest.get("data_cutoff"),
            "retrieval_timestamp_utc": manifest.get("retrieval_timestamp_utc"),
            "costs": {
                "commission_bps_per_side": config.COMMISSION_BPS,
                "slippage_bps_per_side": config.SLIPPAGE_BPS,
                "total_bps_per_side": COST_BPS_SIDE,
                "charged_on": "entry AND exit of every position, benchmark included",
            },
            "conventions": {
                "trading_days_per_year": config.TRADING_DAYS_PER_YEAR,
                "risk_free_rate": 0.0,
                "initial_capital": config.INITIAL_CAPITAL,
                "position_limit": "one ETF per session, long only, max 100% of equity",
            },
            "random_seed": config.RANDOM_SEED,
            "versions": _versions(),
            "data_sources": {
                "ohlcv": "Yahoo Finance via yfinance",
                "macro": "FRED key-free CSV (VIXCLS, DGS10, DGS3MO) with CBOE/Treasury fallbacks",
                "factors": "Kenneth R. French Data Library (5 factors + momentum)",
            },
        },
        "dataset": dataset_summary,
        "training": {
            "train_rows": artifacts["train_rows"],
            "train_sessions": artifacts["train_sessions"],
            "timings_seconds": artifacts["timings"],
            "best_params": artifacts["best_params"],
            "cv_results": artifacts["cv_results"],
            "feature_importance": artifacts["feature_importance"],
            "tested_at_utc": artifacts["tested_at_utc"],
        },
        "thresholds": {
            model: {k: v for k, v in info.items() if k != "candidates"} for model, info in thresholds.items()
        },
        "threshold_candidates_best_model": thresholds[best_model]["candidates"],
        "best_model": {
            "name": best_model,
            "threshold": float(best_threshold),
            "selection_metric": "validation net cumulative return of the S1 rule (after costs)",
            "ranking": [
                {"model": model, "validation_net_return": float(ret), "validation_trades": int(trades)}
                for model, ret, trades in ranked
            ],
        },
        "classification": classification,
        "model_comparison_test_s1": model_comparison,
        "error_analysis": _error_analysis(test_predictions, best_model, best_threshold),
        "backtests_test": {name: result["metrics"] for name, result in test_suite.items()},
        "strategy_descriptions": {name: spec.description for name, spec in backtest_module.STRATEGIES.items()},
        "backtests_validation": {name: result["metrics"] for name, result in validation_suite.items()},
        "benchmarks": {
            "spy_open_to_close_net": benchmark_oc["metrics"],
            "spy_open_to_close_gross": benchmark_oc_gross["metrics"],
            "spy_buy_and_hold_close_to_close": buy_and_hold["metrics"],
            "benchmark_notes": (
                "SPY open-to-close uses the exact same sessions and per-side costs as the "
                "strategies (like-for-like). Buy & hold is close-to-close with a single "
                "round-trip cost and a DIFFERENT holding period; it is context only."
            ),
        },
        "latest_signal": latest_payload,
        "artifacts": {
            "predictions": "data/processed/predictions.parquet",
            "features": "data/processed/features.parquet",
            "figures": "reports/figures",
            "tables": "reports/tables",
        },
    }
    write_json(METRICS_PATH, metrics_payload)

    from src import report as report_module

    tables = report_module.export_tables(metrics_payload, test_suite, predictions, best_model, best_threshold)

    # 9. Figures ---------------------------------------------------------------
    figures: dict[str, str | None] = {}
    if make_figures:
        figures.update(plots_module.make_eda_figures(sample, metadata))
        figures["roc"] = plots_module.plot_roc_curves(predictions, list(train_models.MODEL_NAMES))
        figures["calibration"] = plots_module.plot_calibration(test_predictions, best_model, best_threshold)
        figures["confusion"] = plots_module.plot_confusion_matrix(test_predictions, best_model, best_threshold)
        figures["importance"] = plots_module.plot_feature_importance(artifacts["feature_importance"])
        figures["accuracy_by_group"] = plots_module.plot_accuracy_by_group(test_predictions, best_model, best_threshold)
        figures["equity"] = plots_module.plot_equity_curves(
            test_suite, {"SPY open-to-close": benchmark_oc, "SPY buy & hold (different period)": buy_and_hold}
        )
        figures["drawdown"] = plots_module.plot_drawdowns(
            test_suite, {"SPY open-to-close": benchmark_oc}
        )
        figures["rolling"] = plots_module.plot_rolling_returns(
            test_suite, {"SPY open-to-close": benchmark_oc}
        )
        figures["trades"] = plots_module.plot_trade_distribution(
            test_suite["S1_top_probability"]["trades"]
        )
    metrics_payload["artifacts"]["figure_files"] = {k: v for k, v in figures.items() if v}
    metrics_payload["artifacts"]["table_files"] = tables
    write_json(METRICS_PATH, metrics_payload)

    # 10. Markdown reports ----------------------------------------------------
    report_module.write_summary(metrics_payload)
    report_module.write_rubric_checklist(metrics_payload)

    # 11. Console hand-off ----------------------------------------------------
    print()
    print("=" * 78)
    print("PIPELINE COMPLETE - simulated research results (not investment advice)")
    print("=" * 78)
    print(f"Data cutoff              : {metrics_payload['dataset']['last_trade_date']}")
    print(f"Modelling rows           : {metrics_payload['dataset']['rows_modelling_sample']:,}")
    print(f"Best model (validation)  : {best_model} (threshold {best_threshold:.2f})")
    for name, row in metrics_payload["backtests_test"].items():
        print(
            f"{name:<20} net={row['total_return']:+.2%}  CAGR={row['cagr']:+.2%}  "
            f"Sharpe={row['sharpe_ratio']:+.2f}  maxDD={row['max_drawdown']:.2%}  trades={row['total_trades']}"
        )
    for name in ("spy_open_to_close_net", "spy_buy_and_hold_close_to_close"):
        row = metrics_payload["benchmarks"][name]
        print(
            f"{name:<20} net={row['total_return']:+.2%}  CAGR={row['cagr']:+.2%}  "
            f"Sharpe={row['sharpe_ratio']:+.2f}  maxDD={row['max_drawdown']:.2%}"
        )
    print("-" * 78)
    print(
        f"LATEST SIMULATED SIGNAL ({latest_payload['signal_session']} close -> next session): "
        f"{latest_payload['selected_ticker']} p={latest_payload['selected_probability']:.3f} "
        f"would_trade={latest_payload['would_trade']}"
    )
    print(f"Reports: {METRICS_PATH} | reports/summary.md | docs/rubric_checklist.md")
    print("=" * 78)
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETF ML trading research pipeline")
    parser.add_argument("--mode", choices=["live", "cached", "report"], default="live")
    parser.add_argument("--no-tune", action="store_true", help="skip GridSearchCV (fast smoke run)")
    parser.add_argument("--refresh", action="store_true", help="force a full re-download in live mode")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args(argv)

    try:
        if args.mode == "report":
            from src import report as report_module

            payload = read_json(METRICS_PATH)
            if not payload:
                raise FileNotFoundError("reports/metrics.json missing - run the pipeline first")
            report_module.write_summary(payload)
            report_module.write_rubric_checklist(payload)
            print("Markdown reports regenerated from reports/metrics.json")
            return 0
        run(mode=args.mode, tune=not args.no_tune, refresh=args.refresh, make_figures=not args.no_figures)
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level guard, logged with traceback
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
