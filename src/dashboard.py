"""Streamlit dashboard for the ETF ML trading research project.

Run it locally with:

    streamlit run src/dashboard.py

It only reads the artefacts produced by the pipeline (``reports/metrics.json``,
``reports/figures``, ``reports/tables``) - it never downloads data or trains a
model.  Everything shown is a historical simulation, not investment advice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config

METRICS_PATH = config.REPORTS_DIR / "metrics.json"


@st.cache_data(show_spinner=False)
def load_metrics(path: str, mtime: float) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    st.set_page_config(page_title="ETF ML trading - research dashboard", layout="wide")
    st.title("ETF ML trading system - research dashboard")
    st.caption(
        "SIMULATION ONLY - no live orders, no broker, no leverage. "
        "Historical results do not predict future returns."
    )

    if not METRICS_PATH.exists():
        st.error(
            "reports/metrics.json not found. Run `python -m src.run_pipeline --mode live` first."
        )
        return

    payload = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
    meta = payload["meta"]
    dataset = payload["dataset"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data cutoff", dataset["last_trade_date"])
    col2.metric("Modelling rows", f"{dataset['rows_modelling_sample']:,}")
    col3.metric("Features", f"{dataset['n_numeric']} + {dataset['n_dummies']}")
    col4.metric("Best model", payload["best_model"]["name"])

    st.subheader("Strategy performance (test window, net of costs)")
    backtests = pd.DataFrame(payload["backtests_test"]).T
    columns = ["total_return", "cagr", "sharpe_ratio", "max_drawdown", "total_trades", "win_rate"]
    st.dataframe(
        backtests[[c for c in columns if c in backtests.columns]].style.format(
            {
                "total_return": "{:.2%}",
                "cagr": "{:.2%}",
                "sharpe_ratio": "{:.2f}",
                "max_drawdown": "{:.2%}",
                "win_rate": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    benchmark = payload["benchmarks"]["spy_open_to_close_net"]
    buy_hold = payload["benchmarks"]["spy_buy_and_hold_close_to_close"]
    st.markdown(
        f"**Like-for-like benchmark (SPY open-to-close, same costs):** "
        f"net {_pct(benchmark['total_return'])}, CAGR {_pct(benchmark['cagr'])}, "
        f"Sharpe {benchmark['sharpe_ratio']:.2f}.  \n"
        f"**SPY buy & hold (different holding period):** {_pct(buy_hold['total_return'])}, "
        f"CAGR {_pct(buy_hold['cagr'])}."
    )

    figures = payload.get("artifacts", {}).get("figure_files", {})
    if "equity" in figures:
        equity_path = Path(config.PROJECT_ROOT) / figures["equity"]
        if equity_path.exists():
            st.image(str(equity_path), caption="Net equity curves (test window)")

    left, right = st.columns(2)
    if "drawdown" in figures and (Path(config.PROJECT_ROOT) / figures["drawdown"]).exists():
        left.image(str(Path(config.PROJECT_ROOT) / figures["drawdown"]), caption="Drawdowns")
    if "calibration" in figures and (Path(config.PROJECT_ROOT) / figures["calibration"]).exists():
        right.image(str(Path(config.PROJECT_ROOT) / figures["calibration"]), caption="Calibration")

    st.subheader("Latest simulated signal")
    latest = payload["latest_signal"]
    st.write(
        f"Signal session **{latest['signal_session']}** -> next session open-to-close. "
        f"Model **{latest['model']}**, threshold **{latest['threshold']:.2f}**."
    )
    st.dataframe(pd.DataFrame(latest["ranking"]), use_container_width=True)
    if latest["would_trade"]:
        st.success(f"Simulated intent: buy {latest['selected_ticker']} at the next open.")
    else:
        st.info("No trade: the best probability is below the validated threshold (stay in cash).")

    st.subheader("Trades (best model, S1 rule)")
    trades_path = config.TABLES_DIR / "trades_test.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        trades = trades[trades["strategy"] == "S1_top_probability"]
        st.dataframe(trades.tail(100), use_container_width=True)

    with st.expander("Environment and reproducibility"):
        st.json(meta)


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
