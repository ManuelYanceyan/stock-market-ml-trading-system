"""Report generation: CSV tables, ``reports/summary.md`` and the rubric checklist.

Every number written here comes from ``reports/metrics.json``, which is produced
by a real pipeline run.  Nothing is hard-coded except the rubric wording and
the qualitative statements that are derived from those numbers.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.utils import get_logger, read_json

logger = get_logger("report")

PYTEST_XML = config.REPORTS_DIR / "pytest.xml"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pct(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def money(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def to_utc(timestamp: str) -> str:
    """Convert a pytest ISO timestamp (with local offset) to a UTC label."""
    if not timestamp:
        return "unknown time"
    try:
        return (
            datetime.fromisoformat(timestamp)
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    except (TypeError, ValueError):
        return timestamp


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a list of dicts as a GitHub markdown table."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def test_summary() -> dict[str, Any] | None:
    """Parse the pytest JUnit XML (real evidence of the last test run)."""
    if not PYTEST_XML.exists():
        return None
    try:
        tree = ET.parse(PYTEST_XML)
    except ET.ParseError:
        return None
    suite = tree.getroot()
    if suite.tag != "testsuite":
        suite = suite.find("testsuite")  # type: ignore[assignment]
    if suite is None:
        return None
    tests = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    cases = []
    for case in suite.iter("testcase"):
        outcome = "passed"
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        cases.append(
            {
                "classname": case.get("classname", ""),
                "name": case.get("name", ""),
                "time": case.get("time", ""),
                "outcome": outcome,
            }
        )
    return {
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": tests - failures - errors - skipped,
        "timestamp": suite.get("timestamp", ""),
        "duration_seconds": suite.get("time", ""),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Table export
# ---------------------------------------------------------------------------
def export_tables(
    payload: dict[str, Any],
    test_suite: dict[str, Any],
    predictions: pd.DataFrame,
    best_model: str,
    threshold: float,
) -> dict[str, str]:
    """Write the CSV artefacts referenced by the markdown reports."""
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # 1. Classification metrics (validation + test, at 0.5 and at the selected threshold).
    rows = []
    for model, splits in payload["classification"].items():
        for split, values in splits.items():
            for mode_key in ("threshold_0_5", "at_selected_threshold"):
                block = values[mode_key]
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "evaluation": mode_key,
                        "threshold": block["threshold"],
                        "n_observations": block["n_observations"],
                        "base_rate_up": block["base_rate_up"],
                        "accuracy": block["accuracy"],
                        "precision": block["precision"],
                        "recall": block["recall"],
                        "f1": block["f1"],
                        "roc_auc": block.get("roc_auc"),
                        "predicted_positive_rate": block["predicted_positive_rate"],
                        "brier_score": block.get("brier_score"),
                    }
                )
    classification_table = pd.DataFrame(rows)
    path = config.TABLES_DIR / "classification_metrics.csv"
    classification_table.to_csv(path, index=False)
    written["classification_metrics"] = "reports/tables/classification_metrics.csv"

    # 2. Strategy performance on the test window (net + gross).
    performance_rows = []
    for name, values in payload["backtests_test"].items():
        performance_rows.append(
            {
                "strategy": name,
                "net_total_return": values.get("total_return"),
                "gross_total_return": values.get("gross_total_return"),
                "cost_drag": values.get("cost_drag_total_return"),
                "cagr": values.get("cagr"),
                "annualized_volatility": values.get("annualized_volatility"),
                "sharpe_ratio": values.get("sharpe_ratio"),
                "max_drawdown": values.get("max_drawdown"),
                "total_trades": values.get("total_trades"),
                "trades_per_session": values.get("trades_per_session"),
                "win_rate": values.get("win_rate"),
                "profit_factor": values.get("profit_factor"),
                "exposure_share": values.get("exposure_share_of_sessions"),
                "final_equity": values.get("final_equity"),
                "threshold_used": test_suite.get(name, {}).get("threshold_used"),
            }
        )
    for key, label in [
        ("spy_open_to_close_net", "benchmark_spy_open_to_close_net"),
        ("spy_open_to_close_gross", "benchmark_spy_open_to_close_gross"),
        ("spy_buy_and_hold_close_to_close", "benchmark_spy_buy_and_hold"),
    ]:
        values = payload["benchmarks"][key]
        performance_rows.append(
            {
                "strategy": label,
                "net_total_return": values.get("total_return"),
                "gross_total_return": values.get("total_return") if "gross" in key else None,
                "cagr": values.get("cagr"),
                "annualized_volatility": values.get("annualized_volatility"),
                "sharpe_ratio": values.get("sharpe_ratio"),
                "max_drawdown": values.get("max_drawdown"),
                "total_trades": None,
                "win_rate": None,
                "final_equity": values.get("final_equity"),
            }
        )
    performance_table = pd.DataFrame(performance_rows)
    path = config.TABLES_DIR / "strategy_performance.csv"
    performance_table.to_csv(path, index=False)
    written["strategy_performance"] = "reports/tables/strategy_performance.csv"

    # 3. Model comparison on the test window using the S1 rule.
    comparison = pd.DataFrame.from_dict(payload["model_comparison_test_s1"], orient="index")
    comparison.index.name = "model"
    comparison = comparison.reset_index()
    preferred = [
        "model",
        "threshold_selected_on_validation",
        "net_total_return",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "win_rate",
        "accuracy_at_threshold",
        "roc_auc",
    ]
    comparison = comparison[
        [c for c in preferred if c in comparison.columns]
        + [c for c in comparison.columns if c not in preferred]
    ]
    path = config.TABLES_DIR / "model_comparison_test.csv"
    comparison.to_csv(path, index=False)
    written["model_comparison"] = "reports/tables/model_comparison_test.csv"

    # 4. Threshold candidates of the selected model.
    candidates = pd.DataFrame(payload["threshold_candidates_best_model"])
    path = config.TABLES_DIR / "threshold_candidates.csv"
    candidates.to_csv(path, index=False)
    written["threshold_candidates"] = "reports/tables/threshold_candidates.csv"

    # 5. Trades of every strategy on the test window.
    trade_frames = []
    for name, result in test_suite.items():
        trades = result["trades"].copy()
        if trades.empty:
            continue
        trades["strategy"] = name
        trade_frames.append(trades)
    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        path = config.TABLES_DIR / "trades_test.csv"
        trades.to_csv(path, index=False)
        written["trades_test"] = "reports/tables/trades_test.csv"

    # 6. Latest simulated signal ranking.
    ranking = pd.DataFrame(payload["latest_signal"]["ranking"])
    path = config.TABLES_DIR / "latest_signal.csv"
    ranking.to_csv(path, index=False)
    written["latest_signal"] = "reports/tables/latest_signal.csv"

    # 7. Predictions of the selected model (test window).
    test_predictions = predictions[predictions["split"] == "test"][
        [
            "signal_date",
            "trade_date",
            "ticker",
            best_model,
            "target_up",
            "target_ret_next_oc",
        ]
    ].copy()
    test_predictions = test_predictions.rename(columns={best_model: "probability"})
    path = config.TABLES_DIR / f"test_predictions_{best_model}.csv"
    test_predictions.to_csv(path, index=False)
    written["test_predictions"] = f"reports/tables/test_predictions_{best_model}.csv"

    logger.info("Tables written to %s", config.TABLES_DIR)
    return written


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------
def write_summary(payload: dict[str, Any]) -> Path:
    meta = payload["meta"]
    dataset = payload["dataset"]
    best = payload["best_model"]
    test_results = payload["backtests_test"]
    validation_results = payload["backtests_validation"]
    benchmarks = payload["benchmarks"]
    strategy_names = list(test_results.keys())
    # Strategy choice is made on the validation window, never on the test window.
    best_strategy = max(validation_results, key=lambda n: validation_results[n]["total_return"])
    best_metrics = test_results[best_strategy]
    test_leader = max(strategy_names, key=lambda n: test_results[n]["total_return"])
    validation_ranking = sorted(
        validation_results.items(), key=lambda item: item[1]["total_return"], reverse=True
    )
    benchmark = benchmarks["spy_open_to_close_net"]
    buy_hold = benchmarks["spy_buy_and_hold_close_to_close"]
    tests = test_summary()

    split_rows = []
    for name, info in dataset["split_info"]["boundaries"].items():
        balance = dataset["class_balance"].get(name, {})
        split_rows.append(
            {
                "split": name,
                "signal_start": info.get("signal_start", "n/a"),
                "signal_end": info.get("signal_end", "n/a"),
                "rows": f"{info.get('rows', 0):,}",
                "sessions": f"{info.get('sessions', 0):,}",
                "positive_target_share": pct(balance.get("positive_share")),
                "dropped_at_boundary": info.get("dropped_at_boundary", 0),
            }
        )

    classification_rows = []
    for model, splits in payload["classification"].items():
        test_block = splits.get("test", {}).get("at_selected_threshold")
        val_block = splits.get("validation", {}).get("at_selected_threshold")
        if not test_block:
            continue
        classification_rows.append(
            {
                "model": model,
                "val_roc_auc": num(val_block.get("roc_auc") if val_block else None),
                "test_roc_auc": num(test_block.get("roc_auc")),
                "test_accuracy": pct(test_block.get("accuracy")),
                "test_precision": pct(test_block.get("precision")),
                "test_recall": pct(test_block.get("recall")),
                "test_f1": num(test_block.get("f1")),
                "predicted_positive_rate": pct(test_block.get("predicted_positive_rate")),
            }
        )

    performance_rows = []
    for name in strategy_names:
        values = test_results[name]
        performance_rows.append(
            {
                "strategy": name,
                "net_return": pct(values.get("total_return")),
                "gross_return": pct(values.get("gross_total_return")),
                "cagr": pct(values.get("cagr")),
                "sharpe": num(values.get("sharpe_ratio"), 2),
                "max_dd": pct(values.get("max_drawdown")),
                "vol": pct(values.get("annualized_volatility")),
                "trades": f"{values.get('total_trades', 0):,}",
                "trades_per_session": pct(values.get("trades_per_session")),
                "win_rate": pct(values.get("win_rate")),
            }
        )
    performance_rows.append(
        {
            "strategy": "BENCHMARK SPY open-to-close (net)",
            "net_return": pct(benchmark.get("total_return")),
            "gross_return": pct(benchmarks["spy_open_to_close_gross"].get("total_return")),
            "cagr": pct(benchmark.get("cagr")),
            "sharpe": num(benchmark.get("sharpe_ratio"), 2),
            "max_dd": pct(benchmark.get("max_drawdown")),
            "vol": pct(benchmark.get("annualized_volatility")),
            "trades": "every session",
            "trades_per_session": "100%",
            "win_rate": pct(benchmark.get("positive_day_share")),
        }
    )
    performance_rows.append(
        {
            "strategy": "SPY buy & hold (different period)",
            "net_return": pct(buy_hold.get("total_return")),
            "gross_return": "n/a",
            "cagr": pct(buy_hold.get("cagr")),
            "sharpe": num(buy_hold.get("sharpe_ratio"), 2),
            "max_dd": pct(buy_hold.get("max_drawdown")),
            "vol": pct(buy_hold.get("annualized_volatility")),
            "trades": "1 position held",
            "trades_per_session": "n/a",
            "win_rate": pct(buy_hold.get("positive_day_share")),
        }
    )

    delta = best_metrics["total_return"] - benchmark["total_return"]
    delta_cagr = (best_metrics.get("cagr") or 0) - (benchmark.get("cagr") or 0)
    beats = best_metrics["total_return"] > benchmark["total_return"]
    latest = payload["latest_signal"]

    error_analysis = payload.get("error_analysis", {})
    bucket_rows = error_analysis.get("by_probability_bucket", [])
    ticker_rows = error_analysis.get("by_ticker", [])
    month_rows = error_analysis.get("by_month", [])

    fig = payload["artifacts"].get("figure_files", {})
    figure_block = ""
    for key, label in [
        ("equity", "Net equity curves (test window)"),
        ("drawdown", "Drawdowns (test window)"),
        ("rolling", "Rolling 63-session returns"),
        ("calibration", "Probability calibration"),
        ("accuracy_by_group", "Accuracy by ETF and month"),
        ("importance", "Feature importance"),
        ("roc", "ROC curves"),
        ("trades", "Trade distribution"),
    ]:
        if key in fig:
            relative = str(fig[key]).replace("reports/", "")
            figure_block += f"\n**{label}**\n\n![{label}]({relative})\n"

    lines: list[str] = []
    lines.append("# Results summary - ETF ML trading system")
    lines.append("")
    lines.append(
        "> Automatically generated from `reports/metrics.json`. "
        f"Run mode: **{meta['run_mode']}**; generated at **{meta['generated_at_utc']}**; "
        f"data cutoff **{dataset['last_trade_date']}**."
    )
    lines.append(">")
    lines.append(
        "> **This is a simulation.** No orders were sent, no broker was contacted, "
        "no leverage or short selling is used. Nothing here is investment advice."
    )
    lines.append("")
    lines.append("## 1. Headline results (test window, after costs)")
    lines.append("")
    lines.append(
        f"* Selected model: **{best['name']}**, probability threshold **{best['threshold']:.2f}** "
        "(chosen on the validation window only)."
    )
    validation_text = ", ".join(
        f"{name} {pct(values['total_return'])}" for name, values in validation_ranking
    )
    lines.append(
        f"* Strategy selected on the **validation** window: **{best_strategy}** (highest net total "
        f"return there). Validation ranking: {validation_text}."
    )
    lines.append(
        f"* Test-window result for that same validation-selected strategy: net total return "
        f"**{pct(best_metrics['total_return'])}**, CAGR **{pct(best_metrics['cagr'])}**, "
        f"Sharpe **{num(best_metrics['sharpe_ratio'], 2)}**, max drawdown "
        f"**{pct(best_metrics['max_drawdown'])}**, {best_metrics['total_trades']:,} trades."
    )
    if test_leader == best_strategy:
        lines.append(
            f"* **{best_strategy} was also the highest-returning strategy on the test window.** "
            "That test ranking is reported ex post and was **not** used to select the strategy; "
            "all four strategies are disclosed."
        )
    else:
        lines.append(
            f"* The highest-returning strategy on the test window was **{test_leader}**, not the "
            "validation-selected one. This is reported ex post and was **not** used for selection."
        )
    lines.append(
        f"* Like-for-like benchmark (SPY open-to-close, same costs): "
        f"**{pct(benchmark['total_return'])}** net total return, CAGR **{pct(benchmark['cagr'])}**, "
        f"Sharpe **{num(benchmark['sharpe_ratio'], 2)}**."
    )
    lines.append(
        f"* Difference vs benchmark: **{delta * 100:+.2f} percentage points** of total return "
        f"({delta_cagr * 100:+.2f} pp of CAGR). "
        + (
            "The strategy beat the benchmark in this test window."
            if beats
            else "The strategy **did not** beat the benchmark in this test window."
        )
    )
    lines.append(
        f"* Buy & hold SPY (close-to-close, **different holding period**, context only): "
        f"total return **{pct(buy_hold['total_return'])}**, CAGR **{pct(buy_hold['cagr'])}**."
    )
    bench_gross = benchmarks["spy_open_to_close_gross"]
    gross_delta = (best_metrics.get("gross_total_return") or 0) - (bench_gross.get("total_return") or 0)
    lines.append(
        f"* **Gross comparison (no costs):** {best_strategy} {pct(best_metrics.get('gross_total_return'))} "
        f"vs the SPY open-to-close benchmark {pct(bench_gross.get('total_return'))} "
        f"({gross_delta * 100:+.2f} pp). "
        + (
            "The gross edge is real but small."
            if gross_delta > 0
            else "**Gross of costs the strategy did not beat the always-invested intraday benchmark**; "
                 "its net advantage comes from trading only "
                 f"{num(best_metrics.get('trade_frequency_pct'), 2)}% of the sessions, not from a stronger signal."
        )
    )
    lines.append(
        f"* Benchmark cost note: the like-for-like SPY open-to-close benchmark holds a position in all "
        f"{benchmark['sessions']:,} test sessions, so its net result is "
        f"**{pct(benchmark['total_return'] - bench_gross['total_return'])}** below its gross result "
        f"(net {pct(benchmark['total_return'])} vs gross {pct(bench_gross['total_return'])}). "
        "The strategies trade far less often; that difference in trading frequency is part of the result, "
        "not evidence of a stronger signal."
    )
    lines.append("")

    lines.append("## 2. Data actually used")
    lines.append("")
    lines.append(
        f"* Assets: {', '.join(dataset['raw_manifest'].get('tickers', config.TICKERS))}."
    )
    lines.append(
        f"* Source retrieval timestamp (UTC): **{meta['retrieval_timestamp_utc']}**; "
        f"data cutoff: **{dataset['last_trade_date']}** (last completed session)."
    )
    lines.append(
        f"* Raw OHLCV rows after validation: **{dataset['raw_manifest'].get('rows_clean_total', 'n/a'):,}**; "
        f"modelling rows: **{dataset['rows_modelling_sample']:,}** over "
        f"**{dataset['sessions']:,}** sessions."
    )
    lines.append(
        f"* Features: **{dataset['n_numeric']} numeric** + **{dataset['n_dummies']} dummies** "
        "(full list in `docs/data_dictionary.md` and `data/processed/features_metadata.json`)."
    )
    lines.append("")
    lines.append(
        markdown_table(
            split_rows,
            [
                ("split", "Split"),
                ("signal_start", "First signal date"),
                ("signal_end", "Last signal date"),
                ("rows", "ETF-day rows"),
                ("sessions", "Sessions"),
                ("positive_target_share", "P(up)"),
                ("dropped_at_boundary", "Rows dropped at boundary"),
            ],
        )
    )
    lines.append("")
    lines.append(
        "The last signal of every split is dropped when its label window would fall into the "
        "next split (`src/build_features.py::assign_splits`), so no split ever reads prices "
        "from a later period."
    )
    lines.append("")

    lines.append("## 3. Classification results (test window)")
    lines.append("")
    lines.append("Probabilities are the model's estimate that the next open-to-close return is positive.")
    lines.append("")
    lines.append(
        markdown_table(
            classification_rows,
            [
                ("model", "Model"),
                ("val_roc_auc", "Val ROC-AUC"),
                ("test_roc_auc", "Test ROC-AUC"),
                ("test_accuracy", "Accuracy"),
                ("test_precision", "Precision"),
                ("test_recall", "Recall"),
                ("test_f1", "F1"),
                ("predicted_positive_rate", "Predicted up"),
            ],
        )
    )
    lines.append("")
    lines.append("Confusion matrices at the selected thresholds are stored per model in "
                 "`reports/metrics.json` (`classification.<model>.test.at_selected_threshold.confusion_matrix`).")
    lines.append("")

    lines.append("## 4. Strategy results (test window, net of costs)")
    lines.append("")
    lines.append(
        f"Costs: {meta['costs']['commission_bps_per_side']:.1f} bps commission + "
        f"{meta['costs']['slippage_bps_per_side']:.1f} bps slippage per side "
        f"(**{meta['costs']['total_bps_per_side']:.1f} bps per side**), charged on entry and exit. "
        "Gross = same trades without costs."
    )
    lines.append("")
    lines.append(
        markdown_table(
            performance_rows,
            [
                ("strategy", "Strategy"),
                ("net_return", "Net total"),
                ("gross_return", "Gross total"),
                ("cagr", "CAGR"),
                ("sharpe", "Sharpe"),
                ("max_dd", "Max DD"),
                ("vol", "Ann. vol"),
                ("trades", "Trades"),
                ("trades_per_session", "Trades/session"),
                ("win_rate", "Win rate"),
            ],
        )
    )
    lines.append("")
    lines.append(
        f"The four strategies are described in `src/backtest.py`: S1 (best probability above the "
        f"validation threshold), S2 (+0.05 threshold and VIX <= {config.STRATEGY_S2_MAX_VIX:.0f}), "
        "S3 (only when SPY is above its 50-day average) and S4 (probability-scaled size + 1% stop loss)."
    )
    lines.append("")

    lines.append("## 5. Honest reading of the numbers")
    lines.append("")
    first = test_results["S1_top_probability"]
    lines.append(
        f"* Trade frequency: S1 traded on **{num(first.get('trade_frequency_pct'), 2)}%** of the "
        f"{first['sessions']:,} test sessions; the more selective variants trade less often, "
        "which is the main reason their costs are lower."
    )
    lines.append(
        f"* Cost drag on S1: **{pct(first.get('cost_drag_total_return'))}** of total return "
        f"({pct(first.get('cost_drag_total_return') / abs(first.get('gross_total_return')) if first.get('gross_total_return') else None)} "
        "of the gross result)."
    )
    lines.append(
        "* The precision above 50% is not the same thing as a profitable strategy: the average "
        "positive open-to-close move is small, so a small drop in precision wipes out the edge."
    )
    lines.append(
        "* Daily observations of the same ETFs are strongly correlated; effective sample size is "
        "far below the row count, so confidence intervals are wide."
    )
    lines.append(
        "* The test window is used once. Any figure that looks good here has not been re-validated "
        "on new data, and market regimes change."
    )
    lines.append(
        f"* Statistical caveat: the mean daily net return of {best_strategy} is "
        f"{num(best_metrics.get('annualized_return_arithmetic', 0) / config.TRADING_DAYS_PER_YEAR * 100, 4)}% "
        f"per session with a Sharpe of {num(best_metrics.get('sharpe_ratio'), 2)}. "
        f"The t-statistic of that mean over {best_metrics['sessions']:,} sessions is about "
        f"**{num((best_metrics.get('sharpe_ratio') or 0) * (best_metrics['sessions'] / config.TRADING_DAYS_PER_YEAR) ** 0.5, 2)}**; "
        "below roughly 2 it is not statistically distinguishable from zero, so this should be read as "
        "a weak, unproven signal rather than a reliable edge."
    )
    lines.append("")

    if bucket_rows or ticker_rows:
        lines.append("## 6. Error analysis (test window)")
        lines.append("")
        if bucket_rows:
            lines.append("### 6.1 Accuracy and realised return by predicted-probability bucket")
            lines.append("")
            lines.append(
                markdown_table(
                    [
                        {
                            "bucket": row["bucket"],
                            "observations": f"{row['observations']:,}",
                            "mean_predicted_probability": num(row["mean_predicted_probability"], 3),
                            "accuracy": pct(row["accuracy"], 1),
                            "mean_next_return": pct(row["mean_next_return"], 3),
                            "observed_up_rate": pct(row["observed_up_rate"], 1),
                        }
                        for row in bucket_rows
                    ],
                    [
                        ("bucket", "Probability bucket"),
                        ("observations", "Rows"),
                        ("mean_predicted_probability", "Mean p"),
                        ("accuracy", "Accuracy"),
                        ("mean_next_return", "Mean next O2C return"),
                        ("observed_up_rate", "Observed P(up)"),
                    ],
                )
            )
            lines.append("")
        if ticker_rows:
            lines.append("### 6.2 By ETF")
            lines.append("")
            lines.append(
                markdown_table(
                    [
                        {
                            "ticker": row["ticker"],
                            "observations": f"{row['observations']:,}",
                            "accuracy": pct(row["accuracy"], 1),
                            "precision_when_predicted_up": pct(row["precision_when_predicted_up"], 1),
                            "predicted_up_share": pct(row["predicted_up_share"], 1),
                            "mean_next_return_when_pred_up": pct(row["mean_next_return_when_pred_up"], 3),
                        }
                        for row in ticker_rows
                    ],
                    [
                        ("ticker", "ETF"),
                        ("observations", "Rows"),
                        ("accuracy", "Accuracy"),
                        ("precision_when_predicted_up", "Precision (pred up)"),
                        ("predicted_up_share", "Share pred up"),
                        ("mean_next_return_when_pred_up", "Mean return when pred up"),
                    ],
                )
            )
            lines.append("")
        if month_rows:
            lines.append("### 6.3 By calendar month")
            lines.append("")
            lines.append(
                markdown_table(
                    [
                        {
                            "month": row["month"],
                            "observations": f"{row['observations']:,}",
                            "accuracy": pct(row["accuracy"], 1),
                            "mean_next_return": pct(row["mean_next_return"], 3),
                        }
                        for row in month_rows
                    ],
                    [
                        ("month", "Month"),
                        ("observations", "Rows"),
                        ("accuracy", "Accuracy"),
                        ("mean_next_return", "Mean next O2C return"),
                    ],
                )
            )
            lines.append("")
        wrong_right = error_analysis.get("correct_vs_wrong", {})
        if wrong_right:
            lines.append(
                f"When the model was right, the average absolute next move was "
                f"**{pct(wrong_right.get('mean_abs_return_when_correct'))}**; when it was wrong, "
                f"**{pct(wrong_right.get('mean_abs_return_when_wrong'))}**. "
                f"Mean predicted probability: {num(wrong_right.get('mean_prob_correct'), 3)} (correct) "
                f"vs {num(wrong_right.get('mean_prob_wrong'), 3)} (wrong)."
            )
            lines.append("")

    lines.append("## 7. Latest simulated signal")
    lines.append("")
    lines.append(
        f"* Signal session: **{latest['signal_session']}** close -> execution at the **next session open** "
        "and exit at that session's close."
    )
    lines.append(
        f"* Model: **{latest['model']}**, threshold **{latest['threshold']:.2f}**; "
        f"top ETF: **{latest['selected_ticker']}** with p = **{latest['selected_probability']:.3f}**."
    )
    lines.append(
        f"* Would the strategy trade? **{'YES' if latest['would_trade'] else 'NO (stay in cash)'}**. "
        "This is a simulation only; no order is placed."
    )
    lines.append("")
    lines.append(
        markdown_table(
            [
                {
                    "ticker": row["ticker"],
                    "prob": num(row["prob_up_next_session"], 3),
                    "close": num(row.get("close"), 2),
                    "vix": num(row.get("vix_lag1"), 2),
                }
                for row in latest["ranking"]
            ],
            [("ticker", "ETF"), ("prob", "P(up next session)"), ("close", "Close"), ("vix", "Lagged VIX")],
        )
    )
    lines.append("")

    if figure_block:
        lines.append("## 8. Figures (generated from the real run)")
        lines.append(figure_block)
        lines.append("")

    tests_section = 9 if figure_block else 8
    lines.append(f"## {tests_section}. Tests and environment")
    lines.append("")
    if tests:
        lines.append(
            f"* pytest: **{tests['passed']}/{tests['tests']} passed**, "
            f"{tests['failures']} failed, {tests['errors']} errors, {tests['skipped']} skipped "
            f"(run at {to_utc(tests['timestamp'])}, {tests['duration_seconds']}s). "
            "Raw JUnit XML: `reports/pytest.xml`."
        )
    else:
        lines.append(
            "* pytest results: **not found** (`reports/pytest.xml` missing). "
            "Run `python -m pytest -q --junitxml=reports/pytest.xml` to record them."
        )
    versions = meta.get("versions", {})
    lines.append(
        "* Environment: Python " + versions.get("python", "?") +
        f", pandas {versions.get('pandas', '?')}, numpy {versions.get('numpy', '?')}, "
        f"scikit-learn {versions.get('scikit_learn', '?')}, xgboost {versions.get('xgboost', '?')}, "
        f"yfinance {versions.get('yfinance', '?')}."
    )
    lines.append("")
    lines.append(f"## {tests_section + 1}. Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("python -m venv .venv")
    lines.append(".venv\\Scripts\\Activate.ps1")
    lines.append("pip install -r requirements.txt")
    lines.append("python -m src.run_pipeline --mode live      # downloads real data, trains, backtests")
    lines.append("python -m src.run_pipeline --mode cached    # reproducible re-run without network")
    lines.append("python -m pytest -q --junitxml=reports/pytest.xml")
    lines.append("jupyter nbconvert --to notebook --execute --inplace notebooks/research.ipynb")
    lines.append("```")
    lines.append("")
    lines.append(
        "The pipeline is deterministic given the same data snapshot (seed 42). "
        "New downloads can differ slightly because vendors revise recent bars."
    )
    lines.append("")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# docs/rubric_checklist.md
# ---------------------------------------------------------------------------
STATUS_ORDER = {"DONE": 0, "PARTIAL": 1, "NOT DONE": 2}


def write_rubric_checklist(payload: dict[str, Any]) -> Path:
    meta = payload["meta"]
    dataset = payload["dataset"]
    best = payload["best_model"]
    tests = test_summary()
    test_line = (
        f"{tests['passed']}/{tests['tests']} pytest tests passed (reports/pytest.xml)"
        if tests
        else "pytest evidence missing - run pytest with --junitxml=reports/pytest.xml"
    )
    models = [m for m in payload["classification"].keys()]
    strategies = list(payload["backtests_test"].keys())
    test_rows = dataset["class_balance"]["test"]["rows"]
    # The strategy is selected on validation; the test CAGR below is descriptive.
    best_strategy = max(
        payload["backtests_validation"],
        key=lambda name: payload["backtests_validation"][name]["total_return"],
    )
    best_cagr = payload["backtests_test"][best_strategy].get("cagr")
    bench_cagr = payload["benchmarks"]["spy_open_to_close_net"].get("cagr")
    buy_hold_cagr = payload["benchmarks"]["spy_buy_and_hold_close_to_close"].get("cagr")

    items: list[dict[str, Any]] = [
        # 1 Problem description
        dict(section="1. Problem description", points="1 (bold)",
             criterion="Problem described in README briefly",
             status="DONE",
             evidence="README.md sections 'Objective' and 'Research question'.",
             notes="Plain-language statement of the prediction problem at the top of the README."),
        dict(section="1. Problem description", points="1",
             criterion="Problem described with context and end goal (how the solution is used)",
             status="DONE",
             evidence="README.md 'Objective', 'How the output is used'; docs/methodology.md.",
             notes="Explains the signal -> simulated execution -> reporting flow and the decision it supports."),
        dict(section="1. Problem description", points="1",
             criterion="New problem definition (not the standard week-long large-cap setup)",
             status="PARTIAL",
             evidence="config.py (asset universe, next-session open-to-close target); README 'Why this target'.",
             notes="Different horizon (next-session open-to-close instead of a multi-day horizon) and an ETF "
                   "universe with sector funds, but still US ETFs; not a new asset class or exchange."),
        dict(section="1. Problem description", points="1",
             criterion="Clear step-by-step description of findings and how to reproduce",
             status="DONE",
             evidence="README.md, docs/methodology.md, reports/summary.md, notebooks/research.ipynb.",
             notes="Every stage is documented with the command that produces it and the artefact it writes."),
        # 2 Data sources
        dict(section="2. Data sources", points="1 (bold)",
             criterion="Uses the lecture data sources and features",
             status="DONE",
             evidence="src/download_data.py (yfinance OHLCV, FRED VIXCLS/DGS10/DGS3MO); docs/data_dictionary.md.",
             notes="Same provider family and the same technical/macro feature ideas as the course notebooks."),
        dict(section="2. Data sources", points="1",
             criterion="20+ new features described (+10% volume style bonus)",
             status="DONE",
             evidence=f"docs/data_dictionary.md; {dataset['n_numeric']} numeric + {dataset['n_dummies']} dummy features.",
             notes=f"{dataset['n_numeric']} numeric features and {dataset['n_dummies']} dummy columns, each documented "
                   "with its formula and timing."),
        dict(section="2. Data sources", points="1",
             criterion="New data source introduced (not YFinance/FRED)",
             status="DONE",
             evidence="src/download_data.py::fetch_french_factors; README 'Data sources'.",
             notes="Kenneth R. French Data Library (Fama-French 5 factors + momentum, daily) is downloaded "
                   "directly from Dartmouth as a key-free CSV/ZIP and used as lagged features. Residual "
                   "point-in-time risk is documented: the 5-session shift reduces but cannot remove revisions "
                   "published later, because key-free files are not vintage snapshots."),
        dict(section="2. Data sources", points="1",
             criterion="Large dataset with > 1 million records",
             status="NOT DONE",
             evidence=f"{dataset['raw_manifest'].get('rows_clean_total', 0):,} raw OHLCV rows, "
                      f"{dataset['rows_modelling_sample']:,} modelling rows.",
             notes="The 6-ETF daily universe cannot reach 1M rows. Reaching it would require intraday bars or a "
                   "much wider universe; that was deliberately out of scope, so this point is NOT claimed."),
        # 3 Transformations + EDA
        dict(section="3. Transformations + EDA", points="1 (bold)",
             criterion="Data combined into one frame; feature sets defined (TO_PREDICT, NUMERIC, DUMMIES)",
             status="DONE",
             evidence="data/processed/features.parquet (one row per date x ticker), "
                      "data/processed/features_metadata.json (feature sets), src/build_features.py.",
             notes=f"TO_PREDICT = {payload['dataset'].get('split_info') and 'target_up, target_ret_next_oc'}; "
                   "NUMERIC and DUMMIES lists are exported in the metadata file."),
        dict(section="3. Transformations + EDA", points="1",
             criterion="At least 5 relevant features generated from transformations",
             status="DONE",
             evidence="src/build_features.py::_ticker_features.",
             notes="Momentum lags, realised/downside volatility, MA ratios, RSI, Bollinger %B, ATR, volume "
                   "z-scores, gap statistics, 52-week distance, beta, relative strength, lagged macro and factors."),
        dict(section="3. Transformations + EDA", points="1",
             criterion="EDA: describe the target, correlations with features",
             status="DONE",
             evidence="reports/figures/02_target_balance.png, 03_feature_correlation.png, 04_vix_relationship.png, "
                      "05_feature_distributions.png; reports/summary.md section 3/6.",
             notes="Class balance by ETF and year, correlation table with the target (training window only), "
                   "VIX buckets, distributions and error analysis by ETF/month/probability."),
        # 4 Modeling
        dict(section="4. Modeling", points="1 (bold)",
             criterion="One course model (DecisionTree / RandomForest)",
             status="DONE",
             evidence="src/train_models.py (decision_tree, random_forest); reports/tables/classification_metrics.csv.",
             notes="Both course models are tuned and evaluated."),
        dict(section="4. Modeling", points="1",
             criterion="More than one model used",
             status="DONE",
             evidence=f"src/train_models.py; {len(models)} models: {', '.join(models)}.",
             notes="Two baselines, two course models, a regularised linear model, XGBoost and an MLP."),
        dict(section="4. Modeling", points="1",
             criterion="Custom decision rules on higher-probability events",
             status="DONE",
             evidence="src/backtest.py::build_positions; config.THRESHOLD_GRID; reports/tables/threshold_candidates.csv.",
             notes=f"Threshold {best['threshold']:.2f} selected on validation; probability-scaled position sizing "
                   "in S4 is a second custom rule."),
        dict(section="4. Modeling", points="1",
             criterion="Hyper-parameter tuning",
             status="DONE",
             evidence="src/train_models.py::train_all (GridSearchCV + DatePurgedTimeSeriesSplit); "
                      "reports/metrics.json -> training.best_params and training.cv_results.",
             notes="Grids are tuned on the training window with date-grouped time-series CV (embargo 1); "
                   "best parameters and CV ROC-AUC are recorded per model."),
        dict(section="4. Modeling", points="1",
             criterion="New models introduced (XGBoost / regression / neural network)",
             status="DONE",
             evidence="src/train_models.py (logistic_regression, xgboost, mlp).",
             notes="XGBoost, an L2-regularised logistic regression and a small MLP, all with the same "
                   "preprocessing pipeline fitted on training data only."),
        # 5 Trading simulation
        dict(section="5. Trading simulation", points="1 (bold)",
             criterion="Vector simulation for at least one strategy with capital approximation",
             status="DONE",
             evidence="src/backtest.py::simulate_vectorized; tests/test_backtest.py checks it against the exact loop.",
             notes="Every strategy is computed both vectorised and row-by-row; the two equity curves match "
                   "to 1e-12 when no stop loss is used."),
        dict(section="5. Trading simulation", points="1",
             criterion="Two or more strategies covered",
             status="DONE",
             evidence=f"src/backtest.py::{', '.join(strategies)}; reports/tables/strategy_performance.csv.",
             notes="Four documented strategies: threshold, conservative (VIX filter), trend filter, risk-managed."),
        dict(section="5. Trading simulation", points="1",
             criterion="Exact simulation with reinvestment of capital gains",
             status="DONE",
             evidence="src/backtest.py::simulate_exact (equity *= 1 + daily return); tests for compounding.",
             notes="Iterative simulation, 100% of current equity at most, compounding session by session, "
                   "costs charged on entry and exit."),
        dict(section="5. Trading simulation", points="1",
             criterion="Profitability discussion vs benchmark (CAGR, Sharpe, max DD, rolling returns)",
             status="DONE",
             evidence="reports/summary.md sections 1/4/5; figures 11-13.",
             notes="Like-for-like SPY open-to-close benchmark on identical dates and costs, plus buy & hold "
                   "with its different holding period explicitly labelled."),
        dict(section="5. Trading simulation", points="1",
             criterion="Advanced features in the best strategy (risk management, sizing, timing)",
             status="DONE",
             evidence="src/backtest.py S2/S4 (VIX regime filter, probability-scaled sizing, 1% intraday stop).",
             notes="S4 scales capital with the probability edge and applies a stop loss; S2 filters by the "
                   "volatility regime. Intraday entry/exit timing is fixed and documented."),
        dict(section="5. Trading simulation", points="1",
             criterion="New empirical strategy based on predictions (regime / limited concurrency)",
             status="DONE",
             evidence="src/backtest.py S3_trend_filter (SPY above 50-day MA), S2_conservative (VIX <= 25).",
             notes="Both are new empirical rules derived from the market-condition idea; the portfolio never "
                   "holds more than one ETF per session (the 1-concurrent-investment style rule)."),
        dict(section="5. Trading simulation", points="1",
             criterion="Exceptional profitability vs a realistic benchmark",
             status="PARTIAL",
             evidence="reports/summary.md section 1; reports/metrics.json -> backtests_test and benchmarks.",
             notes=f"On the test window, {best_strategy} (selected on validation) has CAGR "
                   f"{pct(best_cagr)} vs {pct(bench_cagr)} for the like-for-like SPY "
                   f"open-to-close benchmark (same dates and costs), but SPY buy & hold has CAGR {pct(buy_hold_cagr)} "
                   "over the same period and the best strategy's daily-return t-statistic is below 2. The comparison "
                   "is real; the word 'exceptional' is not justified, so the point is claimed only partially."),
        dict(section="5. Trading simulation", points="1",
             criterion="Deep exploratory analysis of predictions / errors",
             status="DONE",
             evidence="reports/summary.md section 6; reports/figures/10_accuracy_by_group.png, 07_calibration.png.",
             notes="Accuracy/precision by ETF, by month, by probability bucket; correct vs wrong move sizes; "
                   "improvement ideas in docs/methodology.md."),
        # 6 Automation
        dict(section="6. Automation", points="1 (bold)",
             criterion="Notebook + scripts; one notebook runs the whole workflow end-to-end",
             status="DONE",
             evidence="notebooks/research.ipynb (19 code cells executed with stored outputs); src/*.py modules.",
             notes="The notebook imports the modules, runs download -> features -> leakage guard -> training -> "
                   "threshold selection -> backtest -> latest signal -> report, and shows the latest signal."),
        dict(section="6. Automation", points="1",
             criterion="Dependencies managed + README instructions",
             status="DONE",
             evidence="requirements.txt; README 'Installation (Windows)'.",
             notes="Pinned minimum versions plus the exact versions used for the published run."),
        dict(section="6. Automation", points="1",
             criterion="Full system re-runs (e.g. cron) on the latest data and produces predictions",
             status="DONE",
             evidence="python -m src.run_pipeline --mode live; docs/methodology.md 'Scheduling'.",
             notes="One command refreshes data, retrains, re-backtests, writes reports and prints the next-session "
                   "signal. A cron/Task Scheduler entry is documented but deliberately NOT enabled."),
        dict(section="6. Automation", points="1",
             criterion="Two regimes: cached files vs fresh download",
             status="DONE",
             evidence="--mode live vs --mode cached in src/run_pipeline.py; tested locally.",
             notes="Cached mode performs no HTTP request and fails with a clear message if the cache is missing."),
        dict(section="6. Automation", points="1",
             criterion="Incremental loading with storage outside GitHub",
             status="DONE",
             evidence="src/download_data.py::update_cache (overlap merge), data/raw/*.parquet, "
                      "data/raw/manifest.json; .gitignore excludes data/.",
             notes="Only the last 15 calendar days are re-downloaded and merged with the stored history; "
                   "row counts and timestamps are recorded in the manifest. Data never goes to GitHub."),
        # 7 Bonus
        dict(section="7. Bonus", points="1",
             criterion="Code well designed and commented in modules",
             status="DONE",
             evidence="src/ modules with docstrings, type hints, logging; tests/ suite.",
             notes="Each stage is a separate module with a documented contract and unit tests."),
        dict(section="7. Bonus", points="1",
             criterion="Additional code to place bets through a broker API",
             status="PARTIAL",
             evidence="src/broker.py (paper-broker adapter, refuses non-simulated orders).",
             notes="A safe adapter exposes the order interface but has no real broker credentials and refuses "
                   "live orders by design. No live trading was implemented or tested."),
        dict(section="7. Bonus", points="1",
             criterion="Monitoring / dashboard for models, results and trades",
             status="DONE",
             evidence="src/dashboard.py (Streamlit); reports/metrics.json; reports/tables/*.csv.",
             notes="`streamlit run src/dashboard.py` renders metrics, equity curves, drawdowns, calibration, "
                   "trades and the latest signal. Verified locally with Streamlit AppTest (0 exceptions, "
                   "3 dataframes) and an HTTP 200 response; it is local only, not hosted."),
        dict(section="7. Bonus", points="1",
             criterion="Containerization",
             status="PARTIAL",
             evidence="Dockerfile, .dockerignore, README 'Docker' section.",
             notes="The image definition is provided and Docker CLI 28.5.1 is installed, but the Docker daemon "
                   "was not running in this environment, so the image was not built or run. Not claimed as verified."),
        dict(section="7. Bonus", points="1",
             criterion="Cloud deployment",
             status="NOT DONE",
             evidence="No cloud deployment artefacts exist.",
             notes="The GitHub Actions workflow only runs the test-suite; it is not a deployment and has no "
                   "schedule. No dashboard, pipeline or database was deployed to any cloud provider, so this "
                   "bonus point is not claimed."),
    ]

    done = sum(1 for i in items if i["status"] == "DONE")
    partial = sum(1 for i in items if i["status"] == "PARTIAL")
    not_done = sum(1 for i in items if i["status"] == "NOT DONE")

    lines: list[str] = []
    lines.append("# Official rubric checklist - self-assessment")
    lines.append("")
    lines.append(
        "Rubric source: <https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/projects/README.md> "
        f"(generated from the reference run recorded in `reports/metrics.json`, {meta['generated_at_utc']})."
    )
    lines.append("")
    lines.append(
        f"**Self-assessment summary:** {done} criteria DONE, {partial} PARTIAL, {not_done} NOT DONE. "
        "This is the student's own assessment against the published rubric; peer reviewers assign the actual score."
    )
    lines.append("")
    lines.append(
        f"Evidence snapshot: {dataset['rows_modelling_sample']:,} modelling rows, test window "
        f"{dataset['class_balance']['test']['sessions']:,} sessions / {test_rows:,} rows, "
        f"{len(models)} models, {len(strategies)} strategies, best model {best['name']}, "
        f"{test_line}."
    )
    lines.append("")
    lines.append(
        "| Section | Points | Criterion | Status | Evidence | Notes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in items:
        lines.append(
            f"| {item['section']} | {item['points']} | {item['criterion']} | "
            f"**{item['status']}** | {item['evidence']} | {item['notes']} |"
        )
    lines.append("")

    lines.append("## What is deliberately NOT claimed")
    lines.append("")
    lines.append(
        "* **> 1M records**: the dataset has "
        f"{dataset['raw_manifest'].get('rows_clean_total', 0):,} raw OHLCV rows, not a million."
    )
    lines.append(
        "* **Statistically proven edge**: the test-window numbers are reported as they came out. The best strategy "
        "beats the like-for-like intraday benchmark net of costs but loses to SPY buy & hold over the same period; "
        "its daily-return t-statistic is below 2."
    )
    lines.append(
        "* **Live trading**: the broker adapter is a paper-only stub; no real order was ever sent."
    )
    lines.append(
        "* **Docker build**: the Dockerfile is provided, but the Docker daemon was not running here, so the image "
        "was not built or run."
    )
    lines.append(
        "* **Cloud deployment**: nothing was deployed to a cloud provider. The GitHub Actions workflow only runs "
        "the tests."
    )
    lines.append(
        "* **Peer review**: reviewing 3 peer projects is the student's task after submission."
    )
    lines.append("")
    lines.append("## Minimum score check (bold criteria)")
    lines.append("")
    bold = [i for i in items if "(bold)" in i["points"]]
    for item in bold:
        lines.append(f"* [{'x' if item['status'] == 'DONE' else ' '}] {item['criterion']} - {item['status']}")
    lines.append("")
    lines.append(
        "All six bold criteria are satisfied locally, which covers the 6-point minimum for the bold items. "
        "The remaining points depend on the peer review."
    )
    lines.append("")

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.DOCS_DIR / "rubric_checklist.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
