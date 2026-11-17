"""Validate the central configuration before a pipeline run.

The configuration is intentionally a plain module, which makes it easy to edit
but also easy to break.  These checks catch inconsistent dates, duplicate
tickers, impossible costs and invalid strategy parameters before any data is
downloaded or a model is trained.
"""

from __future__ import annotations

import sys

import pandas as pd

import config


def validate_config() -> list[str]:
    """Return a list of configuration errors (empty when the config is valid)."""
    errors: list[str] = []

    tickers = list(config.TICKERS)
    if not tickers:
        errors.append("TICKERS must not be empty")
    if len(set(tickers)) != len(tickers):
        errors.append("TICKERS must not contain duplicates")
    if config.BENCHMARK_TICKER not in tickers:
        errors.append("BENCHMARK_TICKER must be part of TICKERS")

    if config.INITIAL_CAPITAL <= 0:
        errors.append("INITIAL_CAPITAL must be positive")
    if config.COMMISSION_BPS < 0:
        errors.append("COMMISSION_BPS must be non-negative")
    if config.SLIPPAGE_BPS < 0:
        errors.append("SLIPPAGE_BPS must be non-negative")
    if config.SPLIT_EMBARGO_DAYS < 1:
        errors.append("SPLIT_EMBARGO_DAYS must be at least 1")
    if config.RANDOM_SEED is None:
        errors.append("RANDOM_SEED must be set for reproducibility")

    windows = [
        ("train", config.TRAIN_START, config.TRAIN_END),
        ("validation", config.VALID_START, config.VALID_END),
        ("test", config.TEST_START, config.TEST_END),
    ]
    parsed: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = []
    for name, start, end in windows:
        try:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) if end is not None else None
        except (TypeError, ValueError):
            errors.append(f"{name} split has an invalid date")
            continue
        if end_ts is not None and start_ts > end_ts:
            errors.append(f"{name} split starts after it ends")
        parsed.append((name, start_ts, end_ts))
    for (previous_name, _, previous_end), (name, start, _) in zip(parsed, parsed[1:]):
        if previous_end is not None and start <= previous_end:
            errors.append(f"{name} split must start after the {previous_name} split ends")

    if not config.THRESHOLD_GRID:
        errors.append("THRESHOLD_GRID must not be empty")
    elif any(value < 0 or value > 1 for value in config.THRESHOLD_GRID):
        errors.append("THRESHOLD_GRID values must be between 0 and 1")
    if not 0 <= config.MAX_FEATURE_MISSING_SHARE <= 1:
        errors.append("MAX_FEATURE_MISSING_SHARE must be between 0 and 1")
    if config.STRATEGY_S4_PROB_FULL <= 0:
        errors.append("STRATEGY_S4_PROB_FULL must be positive")
    if not 0 <= config.STRATEGY_S4_MIN_WEIGHT <= 1:
        errors.append("STRATEGY_S4_MIN_WEIGHT must be between 0 and 1")
    return errors


def main() -> int:
    errors = validate_config()
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print("Configuration OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
