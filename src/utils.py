"""Small shared helpers: logging, JSON/parquet IO and time utilities."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config


def utc_now_iso() -> str:
    """Current UTC timestamp with second resolution (ISO-8601)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_directories() -> None:
    for path in (
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Console + file logger.  Calling it twice does not duplicate handlers."""
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        config.LOGS_DIR / f"{name}_{stamp}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON atomically (tmp file + replace) with UTF-8 encoding.

    If serialisation fails, the temporary file is removed and any existing
    target is left untouched, so a failed run never truncates the artefact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Atomic parquet write (the cache never ends up half-written)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def latest_completed_session(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last calendar day whose US session can be considered complete.

    We never keep the current UTC date: if the pipeline runs while the market
    is open (or shortly after the close, before vendors finalise the bar) the
    current bar may be partial.  Dropping it guarantees a completed session.
    """
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    candidate = today - pd.Timedelta(days=1)
    # Skip weekends; the caller still validates against real trading dates.
    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate


def format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def format_num(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"
