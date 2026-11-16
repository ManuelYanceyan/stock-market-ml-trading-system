"""Central configuration for the ETF ML trading research project.

Every path, date, cost assumption and random seed used by the project lives
here so that experiments are reproducible from a single place.

The project is a *research simulation*: no live orders, no broker
credentials, no short selling, no leverage and no real money.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
LOGS_DIR = REPORTS_DIR / "logs"

DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Universe and study window
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE"]
BENCHMARK_TICKER = "SPY"

# Official modelling window requested by the project brief.
STUDY_START = "2015-01-01"

# Earlier start used only to warm up rolling features (52-week high, 200-day
# moving average, 60-day beta).  These rows are never used as training data.
WARMUP_START = "2013-12-02"

# Chronological splits on the *signal* date (the close of day t).
TRAIN_START = "2015-01-01"
TRAIN_END = "2019-12-31"
VALID_START = "2020-01-01"
VALID_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = None  # open ended: up to the last completed trading day

# Number of signal days removed at the end of every split so that the
# one-session-ahead label window never crosses a split boundary.
SPLIT_EMBARGO_DAYS = 1

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "VIXCLS": "CBOE Volatility Index: VIX (daily close, level)",
    "DGS10": "10-Year Treasury constant maturity yield (percent)",
    "DGS3MO": "3-Month Treasury constant maturity yield (percent)",
}

FRENCH_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FRENCH_MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)

# Fallbacks used only when FRED is unreachable (both are key-free CSVs).
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
TREASURY_YIELD_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)

HTTP_TIMEOUT_SECONDS = 60
HTTP_RETRIES = 3

# When the cache is refreshed we re-download a small overlapping window so
# that vendor revisions of recent bars are picked up.
CACHE_REFRESH_OVERLAP_DAYS = 15

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET_COLUMN = "target_up"
TARGET_RETURN_COLUMN = "target_ret_next_oc"

# A row is kept for modelling only if at most this share of numeric features
# is missing (the remaining gaps are imputed inside the sklearn pipeline).
MAX_FEATURE_MISSING_SHARE = 0.30

# ---------------------------------------------------------------------------
# Costs (applied on BOTH entry and exit)
# ---------------------------------------------------------------------------
COMMISSION_BPS = 1.0   # broker commission per side, basis points
SLIPPAGE_BPS = 2.0     # assumed slippage per side, basis points

# ---------------------------------------------------------------------------
# Backtest conventions
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 10_000.0
MIN_VALIDATION_TRADES = 100  # minimum trades before a threshold is accepted
THRESHOLD_GRID = [round(0.50 + 0.01 * i, 2) for i in range(21)]  # 0.50 .. 0.70

# Strategy parameters (all selected / justified on the validation window).
STRATEGY_S1_THRESHOLD_BUMP = 0.00   # "best probability ETF" rule
STRATEGY_S2_THRESHOLD_BUMP = 0.05   # conservative threshold
STRATEGY_S2_MAX_VIX = 25.0          # volatility regime filter
STRATEGY_S3_TREND_FILTER = True     # SPY close above its 50-day average
STRATEGY_S3_THRESHOLD_BUMP = 0.02
STRATEGY_S4_PROB_FULL = 0.15        # probability edge that maps to full capital
STRATEGY_S4_MIN_WEIGHT = 0.25
STRATEGY_S4_STOP_LOSS = 0.010       # 1% intraday stop loss (approximate)

# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------
PROJECT_NAME = "stock-market-ml-trading-system"
PROJECT_TITLE = "Stock Market ML Trading System: ETF Signals and Backtesting"
RANDOM_STATE_DOC = "All models use RANDOM_SEED = 42."
