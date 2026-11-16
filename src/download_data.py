"""Download, validate and cache the raw market / macro data.

Sources
-------
1. Yahoo Finance via ``yfinance``  -> per-ticker daily OHLCV (auto adjusted).
2. FRED key-free CSV endpoint       -> VIXCLS, DGS10, DGS3MO.
3. Kenneth R. French Data Library   -> daily Fama-French 5 factors + momentum
   (an additional free source that is not Yahoo Finance and not FRED).

All downloads are cached under ``data/raw`` (git-ignored).  A ``live`` run
refreshes the cache incrementally (only the last few days are re-downloaded),
a ``cached`` run reads the existing files and never touches the network.

Nothing in this module fabricates market data: if a source fails the run
stops with a clear error and the failure is recorded in the manifest.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import config
from src.utils import (
    ensure_directories,
    get_logger,
    latest_completed_session,
    load_parquet,
    read_json,
    save_parquet,
    utc_now_iso,
    write_json,
)

logger = get_logger("download_data")

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
MACRO_COLUMNS = ["vix", "dgs10", "dgs3mo"]
FACTOR_COLUMNS = ["ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_mom"]

RAW_OHLCV_PATH = config.RAW_DIR / "ohlcv.parquet"
RAW_MACRO_PATH = config.RAW_DIR / "macro.parquet"
RAW_FACTORS_PATH = config.RAW_DIR / "french_factors.parquet"
VALIDATION_PATH = config.RAW_DIR / "validation.json"
MANIFEST_PATH = config.RAW_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = config.HTTP_TIMEOUT_SECONDS) -> requests.Response:
    """GET with retries; raises a RuntimeError with the real reason on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-project/1.0)"}
    last_error: Exception | None = None
    for attempt in range(1, config.HTTP_RETRIES + 1):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200 and response.content:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
        except Exception as exc:  # noqa: BLE001 - we re-raise below
            last_error = exc
        logger.warning("Request failed (attempt %s/%s): %s", attempt, config.HTTP_RETRIES, last_error)
    raise RuntimeError(f"Could not download {url}: {last_error}")


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------
def _flatten_yfinance(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise the different yfinance return shapes into a long frame.

    yfinance returns a flat frame for a single ticker and a (field, ticker)
    or (ticker, field) MultiIndex for several tickers depending on ``group_by``
    and on the library version.  This function accepts all of them.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = set(frame.columns.get_level_values(0))
        # group_by="ticker" -> level 0 is the ticker; otherwise field first.
        if tickers[0] in level0 or any(t in level0 for t in tickers):
            frame.columns = frame.columns.swaplevel(0, 1)
        frame = frame.stack(level=-1, future_stack=True)
        frame.index.names = ["date", "ticker"]
        frame = frame.reset_index()
    else:
        if len(tickers) != 1:
            raise ValueError("Flat yfinance frame received for multiple tickers")
        frame = frame.reset_index()
        frame.columns = [str(c).lower() for c in frame.columns]
        frame["ticker"] = tickers[0]

    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    if "date" not in frame.columns:
        date_col = next((c for c in frame.columns if c in {"index", "datetime"}), None)
        if date_col is None:
            raise ValueError(f"No date column in yfinance output: {list(frame.columns)}")
        frame = frame.rename(columns={date_col: "date"})

    if "adj_close" in frame.columns and "close" not in frame.columns:
        frame = frame.rename(columns={"adj_close": "close"})

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"yfinance output missing columns: {missing}")

    frame = frame[OHLCV_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker"])
    return frame.drop_duplicates(subset=["date", "ticker"], keep="last")


def fetch_ohlcv(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV for ``tickers`` (auto-adjusted prices)."""
    import yfinance as yf

    logger.info("Downloading OHLCV for %s from %s", ",".join(tickers), start)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    frame = _flatten_yfinance(raw, tickers)
    frame = frame[frame["ticker"].isin(tickers)]
    logger.info("Downloaded %s raw OHLCV rows", len(frame))
    return frame


# ---------------------------------------------------------------------------
# FRED (key-free CSV) and fallbacks
# ---------------------------------------------------------------------------
def _fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id: str) -> tuple[pd.Series, str]:
    """Return (series, source_label) for one FRED series."""
    response = _http_get(_fred_url(series_id))
    frame = pd.read_csv(io.StringIO(response.text))
    date_col = frame.columns[0]
    value_col = frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    series = (
        frame.dropna(subset=[date_col])
        .set_index(date_col)[value_col]
        .sort_index()
    )
    series = series[~series.index.duplicated(keep="last")]
    return series, f"FRED csv ({series_id})"


def _fetch_cboe_vix() -> pd.Series:
    response = _http_get(config.CBOE_VIX_URL)
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = [c.strip().upper() for c in frame.columns]
    frame["DATE"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y", errors="coerce")
    series = frame.dropna(subset=["DATE"]).set_index("DATE")["CLOSE"].astype(float).sort_index()
    return series[~series.index.duplicated(keep="last")]


def _fetch_treasury_yields(column: str) -> pd.Series:
    """Treasury.gov daily yield curve (key-free CSV), one request per year."""
    current_year = pd.Timestamp.today().year
    frames = []
    for year in range(int(config.WARMUP_START[:4]), current_year + 1):
        url = config.TREASURY_YIELD_URL.format(year=year)
        try:
            response = _http_get(url)
        except RuntimeError as exc:  # pragma: no cover - network dependent
            logger.warning("Treasury %s year %s unavailable: %s", column, year, exc)
            continue
        frame = pd.read_csv(io.StringIO(response.text))
        if column not in frame.columns:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="coerce")
        frames.append(frame.dropna(subset=["Date"]).set_index("Date")[[column]])
    if not frames:
        raise RuntimeError(f"Could not download Treasury column {column}")
    series = pd.concat(frames).sort_index()[column].astype(float)
    return series[~series.index.duplicated(keep="last")]


def fetch_macro(start: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download VIX, 10Y and 3M yields.  Falls back to key-free CSV mirrors."""
    start_ts = pd.Timestamp(start)
    sources: dict[str, str] = {}
    columns: dict[str, pd.Series] = {}

    for series_id, name in [("VIXCLS", "vix"), ("DGS10", "dgs10"), ("DGS3MO", "dgs3mo")]:
        try:
            series, label = fetch_fred_series(series_id)
            sources[name] = label
        except Exception as exc:  # noqa: BLE001 - documented fallback
            logger.warning("FRED %s failed (%s); trying documented fallback", series_id, exc)
            if name == "vix":
                series = _fetch_cboe_vix()
                sources[name] = "CBOE VIX_History.csv (FRED fallback)"
            else:
                column = "10 Yr" if name == "dgs10" else "3 Mo"
                series = _fetch_treasury_yields(column)
                sources[name] = f"Treasury.gov daily yield curve ({column}, FRED fallback)"
        columns[name] = series[series.index >= start_ts]

    macro = pd.DataFrame(columns).sort_index()
    macro.index.name = "date"
    macro = macro[~macro.index.duplicated(keep="last")]
    logger.info("Macro series: %s rows, last date %s", len(macro), macro.index.max().date())
    return macro.reset_index(), sources


# ---------------------------------------------------------------------------
# Backwards-compatible re-export
# ---------------------------------------------------------------------------
def align_daily_source(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lag_sessions: int,
    columns: list[str],
) -> pd.DataFrame:
    """Re-export :func:`src.build_features.align_daily_source`.

    The implementation lives in ``build_features`` because it is part of the
    feature-engineering contract.  The local import avoids a circular import
    at module load time while keeping the public ingestion API stable for
    tests written against earlier versions of this module.
    """
    from src.build_features import align_daily_source as _align

    return _align(frame, calendar, lag_sessions, columns)


# ---------------------------------------------------------------------------
# Kenneth French data library
# ---------------------------------------------------------------------------
def _parse_french_csv(text: str, header_token: str) -> pd.DataFrame:
    """Parse the daily section of a French library CSV (values are percent)."""
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(",") and header_token in line),
        None,
    )
    if header_idx is None:
        raise ValueError(f"French CSV: could not find header with {header_token!r}")

    names = [c.strip() for c in lines[header_idx].split(",")]
    records: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        records.append(parts)

    frame = pd.DataFrame(records, columns=names[: len(records[0])])
    frame = frame.set_index(names[0])
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    numeric = frame.apply(pd.to_numeric, errors="coerce") / 100.0
    return numeric.sort_index()


def _download_zip_csv(url: str) -> str:
    response = _http_get(url, timeout=120)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        return archive.read(name).decode("latin-1", errors="replace")


def fetch_french_factors() -> pd.DataFrame:
    """Daily Fama-French 5 factors plus momentum (percent -> decimal)."""
    five = _parse_french_csv(_download_zip_csv(config.FRENCH_FACTORS_URL), "Mkt-RF")
    five.columns = [c.strip() for c in five.columns]
    five = five.rename(
        columns={
            "Mkt-RF": "ff_mkt_rf",
            "SMB": "ff_smb",
            "HML": "ff_hml",
            "RMW": "ff_rmw",
            "CMA": "ff_cma",
        }
    )
    momentum = _parse_french_csv(_download_zip_csv(config.FRENCH_MOMENTUM_URL), "Mom")
    momentum.columns = ["ff_mom"] + [
        f"ff_mom_extra_{i}" for i in range(momentum.shape[1] - 1)
    ]

    factors = five[[c for c in ["ff_mkt_rf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma"] if c in five]]
    factors = factors.join(momentum[["ff_mom"]], how="outer")
    factors = factors[factors.index >= pd.Timestamp(config.WARMUP_START)]
    factors = factors.sort_index()
    factors = factors[~factors.index.duplicated(keep="last")]
    logger.info("French factors: %s rows, last date %s", len(factors), factors.index.max().date())
    return factors.reset_index()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_ohlcv(frame: pd.DataFrame, retrieval_ts: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check schema / integrity and return a cleaned long frame plus a report.

    Cleaning rules (all counted in the report):
      * drop duplicated (date, ticker) rows keeping the last observation;
      * drop rows with non-positive or non-finite prices;
      * drop rows where high < low or where open/close fall outside [low, high]
        beyond a small relative tolerance;
      * drop bars from the current (possibly incomplete) UTC date.
    """
    report: dict[str, Any] = {
        "retrieval_timestamp_utc": retrieval_ts or utc_now_iso(),
        "rules": [
            "duplicate (date, ticker) rows -> keep last",
            "non-positive or non-finite prices -> drop row",
            "high < low or open/close outside [low, high] (1e-6 rel. tol) -> drop row",
            "bars dated today or later in UTC -> drop (session may be incomplete)",
            "zero-volume bars -> kept but counted",
        ],
    }

    if frame.empty:
        raise ValueError("OHLCV frame is empty - download failed")

    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV schema error, missing columns: {missing}")

    data = frame.copy()
    report["rows_raw"] = int(len(data))

    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    report["missing_values_raw"] = {
        c: int(data[c].isna().sum()) for c in OHLCV_COLUMNS
    }

    before = len(data)
    data = data.drop_duplicates(subset=["date", "ticker"], keep="last")
    report["dropped_duplicates"] = int(before - len(data))

    before = len(data)
    price_cols = ["open", "high", "low", "close"]
    finite = np.isfinite(data[price_cols]).all(axis=1)
    positive = (data[price_cols] > 0).all(axis=1)
    data = data[finite & positive]
    report["dropped_non_positive_or_non_finite"] = int(before - len(data))

    before = len(data)
    tolerance = 1e-6
    ordered = data["high"] >= data["low"] * (1 - tolerance)
    open_inside = (data["open"] >= data["low"] * (1 - tolerance)) & (
        data["open"] <= data["high"] * (1 + tolerance)
    )
    close_inside = (data["close"] >= data["low"] * (1 - tolerance)) & (
        data["close"] <= data["high"] * (1 + tolerance)
    )
    data = data[ordered & open_inside & close_inside]
    report["dropped_inconsistent_bars"] = int(before - len(data))

    before = len(data)
    cutoff = latest_completed_session()
    data = data[data["date"] <= cutoff]
    report["dropped_incomplete_current_date"] = int(before - len(data))
    report["requested_cutoff"] = cutoff.strftime("%Y-%m-%d")

    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    report["rows_clean"] = int(len(data))
    report["duplicate_dates_remaining"] = int(
        data.duplicated(subset=["date", "ticker"]).sum()
    )
    report["chronological_order_ok"] = bool(
        data.sort_values(["date", "ticker"]).reset_index(drop=True).equals(data)
    )
    report["zero_volume_rows"] = int((data["volume"] == 0).sum())

    per_ticker = {}
    for ticker, group in data.groupby("ticker"):
        ret = group["close"].pct_change()
        per_ticker[ticker] = {
            "rows": int(len(group)),
            "first_date": group["date"].min().strftime("%Y-%m-%d"),
            "last_date": group["date"].max().strftime("%Y-%m-%d"),
            "missing_close": int(group["close"].isna().sum()),
            "max_abs_daily_close_jump": float(np.nanmax(np.abs(ret))) if len(group) > 1 else None,
        }
    report["per_ticker"] = per_ticker

    gaps = {}
    for ticker, group in data.groupby("ticker"):
        diffs = group["date"].diff().dt.days
        large = int((diffs > 5).sum())
        gaps[ticker] = large
    report["calendar_gaps_over_5_days"] = gaps

    report["data_cutoff"] = data["date"].max().strftime("%Y-%m-%d")
    report["tickers"] = sorted(data["ticker"].unique().tolist())
    report["total_sessions"] = int(data["date"].nunique())
    return data, report


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------
def load_raw_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the cached raw data; raises a helpful error when it is missing."""
    missing = [
        path.name
        for path in (RAW_OHLCV_PATH, RAW_MACRO_PATH, RAW_FACTORS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cached raw data is missing: "
            + ", ".join(missing)
            + ". Run 'python -m src.run_pipeline --mode live' once first."
        )
    manifest = read_json(MANIFEST_PATH, default={})
    return (
        load_parquet(RAW_OHLCV_PATH),
        load_parquet(RAW_MACRO_PATH),
        load_parquet(RAW_FACTORS_PATH),
        manifest,
    )


def update_cache(force_refresh: bool = False) -> dict[str, Any]:
    """Refresh the raw cache from the providers (network required).

    Incremental behaviour: if a cache exists, only the last
    ``CACHE_REFRESH_OVERLAP_DAYS`` calendar days are requested again and merged
    with the stored history.  ``force_refresh=True`` re-downloads everything
    from ``WARMUP_START``.
    """
    ensure_directories()
    retrieval_ts = utc_now_iso()

    existing = None
    if RAW_OHLCV_PATH.exists() and not force_refresh:
        try:
            existing = load_parquet(RAW_OHLCV_PATH)
        except Exception as exc:  # noqa: BLE001 - corrupt cache is re-downloaded
            logger.warning("Cache unreadable (%s); performing a full download", exc)

    if existing is not None and not existing.empty:
        last_cached = pd.to_datetime(existing["date"]).max()
        start = (last_cached - pd.Timedelta(days=config.CACHE_REFRESH_OVERLAP_DAYS)).strftime("%Y-%m-%d")
        start = max(start, config.WARMUP_START)
        logger.info("Incremental update from %s (overlap %s days)", start, config.CACHE_REFRESH_OVERLAP_DAYS)
    else:
        start = config.WARMUP_START

    downloaded = fetch_ohlcv(config.TICKERS, start=start)
    if existing is not None and not existing.empty:
        downloaded = pd.concat([existing, downloaded], ignore_index=True)
    downloaded = downloaded.drop_duplicates(subset=["date", "ticker"], keep="last")

    clean, report = validate_ohlcv(downloaded, retrieval_ts=retrieval_ts)
    save_parquet(clean, RAW_OHLCV_PATH)
    write_json(VALIDATION_PATH, report)

    macro_raw, macro_sources = fetch_macro(config.WARMUP_START)
    macro_raw.to_parquet(RAW_MACRO_PATH, index=False)

    factors = fetch_french_factors()
    factors.to_parquet(RAW_FACTORS_PATH, index=False)

    manifest = {
        "project": config.PROJECT_NAME,
        "run_mode": "live",
        "retrieval_timestamp_utc": retrieval_ts,
        "requested_start": config.WARMUP_START,
        "study_start": config.STUDY_START,
        "data_cutoff": report["data_cutoff"],
        "rows_raw_this_request": int(len(downloaded)),
        "rows_clean_total": int(len(clean)),
        "sessions_total": report["total_sessions"],
        "tickers": config.TICKERS,
        "sources": {
            "ohlcv": "Yahoo Finance via yfinance (auto_adjust=True)",
            "macro": macro_sources,
            "factors": "Kenneth R. French Data Library (5 factors + momentum, daily)",
            "macro_urls": [
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO",
            ],
            "factor_urls": [config.FRENCH_FACTORS_URL, config.FRENCH_MOMENTUM_URL],
        },
        "macro_last_dates": {
            c: pd.to_datetime(macro_raw["date"]).max().strftime("%Y-%m-%d")
            for c in macro_raw.columns
            if c != "date"
        },
        "factors_last_date": pd.to_datetime(factors["date"]).max().strftime("%Y-%m-%d"),
        "validation_report": "data/raw/validation.json",
        "notes": [
            "Prices are auto-adjusted dividends/splits (yfinance auto_adjust).",
            "Raw datasets are git-ignored and reproduced by re-downloading.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    logger.info(
        "Cache updated: %s clean rows, cutoff %s, %s sessions",
        manifest["rows_clean_total"],
        manifest["data_cutoff"],
        manifest["sessions_total"],
    )
    return manifest


if __name__ == "__main__":  # pragma: no cover - manual helper
    update_cache()
