# Data dictionary

All features are computed **at the close of session `t`** and are used to
predict the **next session's open-to-close return**.  No feature reads a price
from `t+1`.

Notation: `C` = adjusted close, `O` = adjusted open, `H` = adjusted high,
`L` = adjusted low, `V` = volume, `r_t = C_t / C_{t-1} - 1`.

An automated export of the exact lists is written by the pipeline to
`data/processed/features_metadata.json` (keys `numeric_features`,
`dummy_features`, `to_predict`).

## 1. Target variables (`TO_PREDICT`)

| Column | Definition | Notes |
| --- | --- | --- |
| `target_up` | `1` if `target_ret_next_oc > 0` else `0` | Binary classification label. |
| `target_ret_next_oc` | `C_{t+1} / O_{t+1} - 1` | Next session **open-to-close** return; the simulated trade enters at the next open and exits at that close. |
| `target_up` / `target_ret_next_oc` null | no next session within 5 calendar days | Prevents linking across long halts. |

Rows also carry the realised next-session OHLC (`open_next`, `high_next`,
`low_next`, `close_next`) and `trade_date`, which the backtester uses; these are
**never** model features.

## 2. Per-ETF technical features (all rolling windows end at `t`)

| Feature | Definition | Why it might help |
| --- | --- | --- |
| `ret_1d` … `ret_120d` | `C_t / C_{t-n} - 1` for n = 1, 2, 3, 5, 10, 20, 60, 120 | Momentum / reversal at several horizons. |
| `vol_5d` … `vol_60d` | rolling std of `r_t` over n sessions × √252 | Realised volatility regime. |
| `downside_vol_20d` | std of `min(r_t, 0)` over 20 sessions × √252 | Asymmetric risk. |
| `ma_ratio_5_20`, `ma_ratio_20_60` | `SMA_n / SMA_m - 1` | Short vs long trend alignment. |
| `price_vs_ma20/50/200` | `C_t / SMA_n - 1` | Distance from the trend. |
| `rsi_14` | Wilder RSI with 14-session EWMA (0–100) | Overbought / oversold. |
| `atr_14_pct` | 14-session ATR divided by `C_t` | Normalised intraday range / risk. |
| `bb_pctb_20` | `(C_t - lower) / (upper - lower)`, Bollinger bands = SMA20 ± 2σ20 | Position inside the volatility band. |
| `volume_z_20` | `(V_t - mean20(V)) / std20(V)` | Unusual participation. |
| `volume_ratio_5_20` | `mean5(V) / mean20(V)` | Volume trend. |
| `intraday_range_pct` | `(H_t - L_t) / C_t` | Intraday turbulence. |
| `close_pos_in_range` | `(C_t - L_t) / (H_t - L_t)` | Where the close finished in the day's range. |
| `gap_open_pct` | `O_t / C_{t-1} - 1` | Overnight gap. |
| `gap_open_mean_5d` | 5-session mean of `gap_open_pct` | Persistent opening behaviour. |
| `dist_52w_high` | `C_t / max252(C) - 1` | Proximity to the 52-week high. |
| `dist_20d_high`, `dist_20d_low` | `C_t / max20(C) - 1`, `C_t / min20(C) - 1` | Short-term breakout / breakdown context. |
| `ret_skew_20`, `ret_kurt_20` | rolling skewness / kurtosis of `r_t` (20 sessions) | Tail behaviour. |

## 3. Market-context features (benchmark = SPY, same session `t`)

| Feature | Definition | Why it might help |
| --- | --- | --- |
| `mkt_ret_1d/5d/20d` | SPY returns over 1, 5, 20 sessions | Market direction. |
| `mkt_vol_20d` | SPY realised volatility (20 sessions, annualised) | Market risk regime. |
| `mkt_above_ma50` | `1` if SPY close > its 50-day SMA | Trend filter used by strategy S3. |
| `mkt_dist_52w_high` | SPY distance from its 52-week high | Drawdown context. |
| `rel_strength_5d`, `rel_strength_20d` | ETF return minus SPY return over 5 / 20 sessions | Relative momentum. |
| `beta_60d` | 60-session rolling covariance with SPY / SPY variance | Sensitivity to the market. |

## 4. Macro features (FRED, lagged one full session)

Series: `VIXCLS` (VIX), `DGS10` (10-year yield), `DGS3MO` (3-month yield).

| Feature | Definition |
| --- | --- |
| `vix_lag1` | VIX level observed at or before session `t-1`. |
| `vix_change_5d_lag1` | 5-session percentage change of the lagged VIX. |
| `vix_z_60_lag1` | 60-session z-score of the lagged VIX. |
| `dgs10_lag1`, `dgs3mo_lag1` | Treasury yields, lagged one session. |
| `dgs10_change_5d_lag1`, `dgs3mo_change_5d_lag1` | 5-session yield changes. |
| `term_spread_lag1` | `DGS10 - DGS3MO` (10y-3m spread). |
| `term_spread_change_5d_lag1` | 5-session change of the spread. |

Alignment procedure (`src/build_features.py::align_daily_source`): re-index on
the SPY session calendar → forward-fill **from the past only** → `shift(1)`.
Missing FRED observations therefore carry the previous value forward and never
borrow a future one.

## 5. Kenneth R. French factors (additional data source, lagged five sessions)

Daily Fama-French 5 factors plus momentum, expressed as decimals (the files
report percent).  The public daily files are updated with a publication delay,
so a **5-session lag** is applied and the series is forward-filled from the
past.

**Residual point-in-time risk.** The key-free download is a final-vintage
snapshot: it does not preserve the values that were published on each historical
date.  The five-session shift removes same-day availability, but later revisions
of a factor value can still be embedded in an earlier feature row, and the true
monthly refresh date can be later than five sessions.  A strictly point-in-time
backtest would need archived vintage files or the factor block removed; the
limitation is disclosed in `docs/methodology.md` (§13) and the README.

| Feature | Definition |
| --- | --- |
| `ff_mkt_rf_lag5` | Market excess return (Mkt-RF). |
| `ff_smb_lag5` | Small-minus-big size factor. |
| `ff_hml_lag5` | High-minus-low value factor. |
| `ff_rmw_lag5` | Robust-minus-weak profitability. |
| `ff_cma_lag5` | Conservative-minus-aggressive investment. |
| `ff_mom_lag5` | Momentum factor. |
| `ff_mom_20d_lag5` | 20-session rolling sum of the lagged momentum factor. |

## 6. Calendar dummies (`DUMMIES`)

| Block | Columns | Values |
| --- | --- | --- |
| ETF identity | `ticker_SPY`, `ticker_QQQ`, `ticker_IWM`, `ticker_XLK`, `ticker_XLF`, `ticker_XLE` | one-hot, exactly one is 1 |
| Weekday | `dow_0` … `dow_4` | Monday = 0 … Friday = 4 |
| Month | `month_01` … `month_12` | January = `month_01` |

## 7. Columns that are NOT features

`date`, `ticker`, `trade_date`, `split`, `open`, `high`, `low`, `close`,
`volume`, `open_next`, `high_next`, `low_next`, `close_next`, `target_up`,
`target_ret_next_oc`, `dow`, `month`.

`close` and `volume` are the raw inputs of the same session used to derive
features; passing them directly as features would mostly encode price level,
so they are excluded from the model matrix (they remain in the stored table for
audits and plots).
