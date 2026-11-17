# Methodology

This document explains **how** the results in `reports/summary.md` were
produced, which decisions were taken, and where the design is weak.  The
project is a historical simulation; it never sends an order.

## 1. Research question

> Can lagged technical and macroeconomic variables predict the sign of the
> **next session's open-to-close** return of a small ETF universe, and does a
> probability-threshold trading rule improve net results versus simple
> baselines after realistic costs?

Hypothesis: short-horizon ETF returns have a weak, regime-dependent
predictability. Even if accuracy is only slightly above 50%, a selective rule
(few, high-confidence trades) might survive costs. The project tests this
honestly, including the possibility that it does not.

Universe: `SPY`, `QQQ`, `IWM`, `XLK`, `XLF`, `XLE` (broad market, growth,
small caps, technology, financials, energy). Long only, at most one ETF at a
time, no leverage, no shorting.

## 2. Timing contract (no look-ahead)

At the close of session **t** we only know data up to **t**:

```
  ... ─── session t ────────────────► session t+1 ────────────────►
          features & signal              execution window
          (close of t)              buy at OPEN → sell at CLOSE
```

* Label: `target_ret_next_oc = close[t+1] / open[t+1] - 1`.
* `target_up = 1` when that return is positive.
* Every technical feature is a rolling window that **ends at t**.
* FRED series are re-indexed on the session calendar, forward-filled from the
  past and shifted by **one session**.
* French factors are re-indexed on the session calendar, forward-filled from
  the past and shifted by **five sessions** (publication lag).  This removes
  same-day availability, but it is **not** a full point-in-time reconstruction:
  the key-free daily files are final-vintage snapshots, so later revisions of a
  factor value can still be embedded in an earlier feature row (see §13).
* The label uses `shift(-1)` on the *same ticker*, so an ETF is never compared
  with another ETF's session.
* The last signal of each split is dropped when its label window would cross
  into the next split (see §5).

This contract is enforced by `tests/test_features.py::test_no_lookahead_when_history_is_truncated`,
which rebuilds the whole feature table with all data after a cutoff removed and
requires the earlier feature values to be identical.

## 3. Data sources

| Source | Content | Access | URL |
| --- | --- | --- | --- |
| Yahoo Finance | Daily OHLCV, auto-adjusted for splits/dividends | `yfinance` package | <https://finance.yahoo.com> |
| FRED | `VIXCLS`, `DGS10`, `DGS3MO` | Key-free CSV endpoint | <https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS> |
| CBOE (fallback) | VIX daily history | Key-free CSV | <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv> |
| U.S. Treasury (fallback) | Daily yield curve | Key-free CSV | <https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve> |
| Kenneth R. French Data Library | Daily Fama-French 5 factors + momentum | Key-free ZIP/CSV | <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html> |

Downloads start at **2013-12-02** so that 200/252-session features are warm by
the start of the official study window **2015-01-01**.  Warm-up rows are used
only to compute features, never as training data.

The last completed session available at run time is the **data cutoff**; any
bar dated "today" is dropped because it may be partial (documented in
`src/download_data.py::validate_ohlcv`).  The retrieval timestamp, row counts,
per-ticker coverage and the cutoff are stored in `data/raw/manifest.json`.

### Cleaning decisions

* duplicated `(date, ticker)` rows → keep the last (vendor corrections win);
* non-positive / non-finite prices → drop;
* impossible bars (`high < low`, or open/close outside `[low, high]` beyond a
  1e-6 relative tolerance) → drop;
* zero-volume bars → kept but counted in `data/raw/validation.json`;
* FRED `.` placeholders → missing values, forward-filled from the past only;
* long calendar gaps (> 5 days) → the label is set to missing so a return is
  never computed across a multi-week halt.

Every rule and every dropped row count is recorded in `data/raw/validation.json`.

## 4. Feature engineering

80 columns: **57 numeric features + 23 dummy columns** (ETF identity, weekday,
month).  The full list with formulas is in `docs/data_dictionary.md` and is
exported programmatically to `data/processed/features_metadata.json`.

Feature blocks:

1. **Momentum** (8 horizons) and **realised / downside volatility** (4 + 1).
2. **Trend**: moving-average ratios and distance from MA20/50/200, RSI, ATR,
   Bollinger %B.
3. **Volume / microstructure**: volume z-score, volume trend, intraday range,
   close position in range, opening gap statistics.
4. **Extremes**: distance from 52-week high, 20-day high/low.
5. **Distribution shape**: 20-day skewness and kurtosis.
6. **Market context** from SPY: market returns/volatility, regime flag,
   relative strength of each ETF vs SPY, 60-day beta.
7. **Macro**: lagged VIX, 10-year and 3-month yields, their 5-session changes,
   VIX z-score, term spread and spread change.
8. **French factors**: lagged market/size/value/profitability/investment and
   momentum factors.
9. **Calendar dummies**: weekday, month, ETF identity.

Rows whose numeric feature vector is more than 30% missing are excluded from
modelling (`config.MAX_FEATURE_MISSING_SHARE`); the remaining gaps are imputed
**inside the model pipeline** with the training-set median.

## 5. Splits (chronological, no overlap)

| Split | Signal dates | Purpose |
| --- | --- | --- |
| Train | 2015-01-01 → 2019-12-31 | fit models and tune hyper-parameters |
| Validation | 2020-01-01 → 2022-12-31 | select the model and the probability threshold |
| Test | 2023-01-01 → last completed session | final, single evaluation |

The validation window deliberately includes the 2020 crash and the 2022
drawdown.  The test window is touched **once**.

Boundary hygiene (`src/build_features.py::assign_splits`): a row belongs to a
split only if both its signal date **and** its executed session (`trade_date`)
fall inside that split.  The last signal of each closed split is therefore
removed (typically 6 rows, one per ETF).  `tests/test_features.py` verifies the
ordering, the absence of overlap and that the label window never crosses a
boundary.

## 6. Models and tuning

| Model | Role | Grid |
| --- | --- | --- |
| `baseline_majority` | DummyClassifier (always the majority class) | – |
| `baseline_momentum` | rule: up if the 5-day return is positive | – |
| `decision_tree` | course model | `max_depth` ∈ {3,5,8}, `min_samples_leaf` ∈ {50,200} |
| `random_forest` | course model | `max_depth` ∈ {6,10}, `min_samples_leaf` ∈ {20,100}, `max_features` ∈ {sqrt, 0.3} |
| `logistic_regression` | linear benchmark | `C` ∈ {0.01,0.1,1.0} |
| `xgboost` | gradient boosting | `max_depth` ∈ {3,5}, `n_estimators` ∈ {200,400} |
| `mlp` | small neural network | `hidden_layer_sizes` ∈ {(32,),(64,32)}, `alpha` ∈ {0.001,0.01} |

* **Preprocessing**: `SimpleImputer(median)` (+ `StandardScaler` for the linear
  model and the MLP) is part of the sklearn `Pipeline`, so it is fitted on
  training rows only.
* **Cross-validation**: `DatePurgedTimeSeriesSplit` (3 folds) splits on unique
  signal dates and applies a one-session embargo, so two ETFs of the same
  session cannot land in different folds and no fold starts immediately after
  another.
* **Scoring**: ROC-AUC.  Class weights are balanced for the tree models.
* **Reproducibility**: `RANDOM_SEED = 42` everywhere.

The best parameters and cross-validation scores of the reference run are stored
in `reports/metrics.json` (`training.best_params`, `training.cv_results`).

## 7. Probability threshold (validation only)

For every model, the S1 rule is simulated on the **validation** window for
thresholds 0.50 → 0.70 in 0.01 steps.  The threshold that maximises **net**
cumulative return after costs, subject to at least **100 trades**, is selected.
The model with the best validation net return is then evaluated on the test
window.  The candidate table is stored in
`reports/tables/threshold_candidates.csv`.

The test window never influences the threshold or the model choice.  The same
rule applies to the four trading strategies: they are ranked on the validation
window, and the validation leader (**S4_risk_managed** in the reference run) is
reported on the test window.  S4 was also the highest-returning strategy on the
test window, but that coincidence is reported **ex post**; the test ranking is
descriptive and was not used to choose or promote a strategy.

## 8. Trading strategies

All strategies: long only, at most one ETF per session, 100% of current equity
at most, flat at the close.

| Strategy | Rule |
| --- | --- |
| `S1_top_probability` | Buy the ETF with the highest `p(up)` if `p ≥ threshold`; exit at the close. |
| `S2_conservative` | S1 with `p ≥ threshold + 0.05` **and** lagged VIX ≤ 25. |
| `S3_trend_filter` | S1 with `p ≥ threshold + 0.02` **and** SPY above its 50-day average. |
| `S4_risk_managed` | S4 sizing: weight = clip((p − 0.5)/0.15, 0.25, 1.0); 1% intraday stop loss. |

The stop loss is approximated with daily OHLC data: if the session's low is at
least 1% below the entry open, the trade is closed at exactly −1% (no gap-through
modelling).  This is stated as an assumption, not a claim of precision.

### Costs

`COMMISSION_BPS = 1.0` and `SLIPPAGE_BPS = 2.0` per side, i.e. **3 bps per
side** and **6 bps per round trip**, charged on both entry and exit.  The
benchmark pays exactly the same costs.

### Benchmarks

1. **SPY open-to-close, like-for-like** – same dates, same one-session holding
   period, same per-side costs (a round trip every session).
2. **SPY buy & hold, close-to-close** – a **different holding period**; shown
   only as context and labelled as such.

## 9. Metrics

* One year = 252 sessions.
* Total return = final equity / initial capital − 1 (compounded).
* CAGR = `(final / initial_capital)^(252 / sessions) - 1`, using the full
  session count, so the first session is never dropped.
* Annualised return = mean daily return × 252 (arithmetic).
* Volatility = std(daily, ddof=1) × √252.
* Sharpe = mean/std × √252, risk-free rate = 0.
* Max drawdown = `min(equity / running_max - 1)`.
* Win rate = share of executed trades whose **net** return is positive.
* Trade frequency = trades / sessions.
* Gross results = the same trades with costs set to zero; the difference is the
  cost drag.

Classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion
matrix, Brier score) are computed at the selected threshold and, for reference,
at 0.5.

## 10. Findings of the reference run

The exact numbers are in `reports/summary.md` and `reports/metrics.json`.  The
qualitative findings are:

* The label is only weakly predictable: test ROC-AUC is close to 0.5 for most
  models, and accuracy hovers around the base rate.
* Precision above 50% does not imply profit: the average positive
  open-to-close move is small compared with the dispersion of outcomes.
* Selectivity helps costs: the high-probability rules trade on a small share of
  sessions, which removes most of the cost drag that the every-session
  benchmark suffers.
* The buy-and-hold comparison is humbling: the intraday strategies concentrate
  their exposure in short windows and capture only a fraction of the market's
  total return over the same years.
* Results differ a lot across ETFs and months, which is consistent with regime
  dependence rather than a stable edge.

Improvement ideas based on the error analysis:

1. Use intraday bars (5–15 minutes) to model the open-to-close path and give the
   stop loss a realistic fill model.
2. Add an explicit regime model (e.g. VIX term structure, realised-volatility
   quantiles) and allow different thresholds per regime.
3. Treat the ETF choice as a ranking problem (learning-to-rank) instead of six
   independent binary classifiers.
4. Add a probability-calibration step (isotonic / Platt) because the tree
   models are over-confident.
5. Test a wider universe (sector ETFs, international ETFs) so that the
   cross-section has more dispersion to exploit.
6. Evaluate on a rolling out-of-sample basis (walk-forward) rather than a single
   fixed test window.

## 11. Tests

`python -m pytest -q --junitxml=reports/pytest.xml`

79 deterministic tests cover: source schema, duplicate/bad-price cleaning,
invalid volume, macro forward-fill without back-fill, per-ticker lags,
French-factor lag, no look-ahead under truncation, split purity, embargo and
boundary validation, target definition, execution timing, empty-signal
sessions, entry/exit fees, compounding, vector-vs-exact equivalence,
one-position-per-session, threshold filtering, stop loss, benchmark alignment,
paper-broker cash/position accounting and invalid-price guards, metric input
validation, validation-window strategy selection, environment pre-flight,
report consistency and metric definitions.  The JUnit XML is committed as
evidence (`reports/pytest.xml`).

Two offline checks complement the tests:

```powershell
python -m src.check_environment   # interpreter, dependencies, committed files
python -m src.validate_reports    # metrics blocks and referenced figures/tables
```

## 12. Automation modes

```powershell
python -m src.run_pipeline --mode live     # refresh data, retrain, re-backtest, write reports
python -m src.run_pipeline --mode cached   # no HTTP request; reuse data/raw + data/processed
python -m src.run_pipeline --mode report   # rebuild the markdown reports only
```

`live` performs an **incremental** update: only the last
`CACHE_REFRESH_OVERLAP_DAYS = 15` calendar days are re-downloaded per ticker and
merged with the stored history, so vendor revisions of recent bars are picked
up without re-downloading ten years of data.  The cache lives in `data/`
(git-ignored) and the manifest records what was fetched and when.

### Scheduling (documented, deliberately NOT enabled)

The brief forbids enabling automatic schedules, so no cron entry or scheduled
workflow is active.  To run it manually every weekday after the US close:

```powershell
# Windows Task Scheduler (run at 22:30 local time, Mon-Fri)
$action = New-ScheduledTaskAction -Execute "python" `
  -Argument "-m src.run_pipeline --mode live" `
  -WorkingDirectory "C:\path\to\stock-market-ml-trading-system"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 22:30
Register-ScheduledTask -TaskName "etf-ml-pipeline" -Action $action -Trigger $trigger
```

```cron
# Linux/macOS equivalent (not installed by this repository)
30 22 * * 1-5 cd /path/to/stock-market-ml-trading-system && python -m src.run_pipeline --mode live
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the test-suite on
push; it has no schedule.

## 13. Limitations

* **Vendor and data revisions**: adjusted prices are restated when dividends or
  splits occur; a re-run on a later date can produce slightly different history.
* **Publication delays and data vintages**: FRED series are lagged one session,
  but exact real-time vintages are not available without ALFRED.  French
  factors are lagged five sessions, which is still an approximation:
  * the public daily files are refreshed on a monthly cadence, so the true
    availability of the most recent month is often later than five sessions;
  * the downloaded file contains **final, subsequently revised** values, not a
    vintage snapshot, so a historical factor row can embed revisions published
    after the simulated decision date;
  * the five-session shift therefore removes same-day and next-day leakage, but
    it cannot guarantee a strict point-in-time feature set.  The residual risk
    is most relevant for the newest observations; eliminating it would require
    archived vintage files (not available key-free) or dropping the factor
    block.  Factor importance is reported in `reports/figures/09_feature_importance.png`
    so the reader can judge the exposure.
* **Costs and liquidity**: 3 bps per side is optimistic for a retail account;
  the model assumes fills exactly at the open and the close.
* **Stop loss**: daily bars cannot reproduce intraday path; the −1% exit is an
  assumption.
* **Overfitting**: thresholds and hyper-parameters are selected on finite
  samples (755 validation sessions, 970 test sessions).  A single test window
  does not establish an edge.
* **Regime change**: the test window is one particular market regime; results
  need walk-forward re-validation.
* **Statistical power**: ETF daily returns are highly correlated, so the
  effective sample size is much smaller than the row count.
* **No live execution**: there is no broker connection, no order routing, no
  slippage calibration against real fills.

## 14. Environment of the reference run

Recorded automatically in `reports/metrics.json` under `meta.versions`
(Python, pandas, numpy, scikit-learn, xgboost, yfinance, matplotlib) together
with the retrieval timestamp and the data cutoff of the run.
