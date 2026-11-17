"""Configuration validation tests: dates, costs, universe and strategy ranges."""

from __future__ import annotations

import config
import pytest
from src.validate_config import validate_config


def test_pipeline_refuses_to_start_with_invalid_config(monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", -1.0)
    from src.run_pipeline import run

    with pytest.raises(RuntimeError, match="Invalid configuration"):
        run(mode="cached")


def test_default_configuration_is_valid():
    assert validate_config() == []


def test_duplicate_or_empty_universe_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "TICKERS", ["SPY", "SPY", "QQQ"])
    assert any("duplicates" in error for error in validate_config())

    monkeypatch.setattr(config, "TICKERS", [])
    assert any("must not be empty" in error for error in validate_config())


def test_benchmark_must_belong_to_the_universe(monkeypatch):
    monkeypatch.setattr(config, "BENCHMARK_TICKER", "DIA")
    errors = validate_config()

    assert any("BENCHMARK_TICKER" in error for error in errors)


def test_impossible_costs_and_capital_are_rejected(monkeypatch):
    monkeypatch.setattr(config, "INITIAL_CAPITAL", 0.0)
    monkeypatch.setattr(config, "COMMISSION_BPS", -1.0)
    monkeypatch.setattr(config, "SLIPPAGE_BPS", -2.0)
    errors = validate_config()

    assert any("INITIAL_CAPITAL" in error for error in errors)
    assert any("COMMISSION_BPS" in error for error in errors)
    assert any("SLIPPAGE_BPS" in error for error in errors)


def test_zero_embargo_and_missing_seed_are_rejected(monkeypatch):
    monkeypatch.setattr(config, "SPLIT_EMBARGO_DAYS", 0)
    monkeypatch.setattr(config, "RANDOM_SEED", None)
    errors = validate_config()

    assert any("SPLIT_EMBARGO_DAYS" in error for error in errors)
    assert any("RANDOM_SEED" in error for error in errors)


def test_reversed_and_overlapping_split_dates_are_rejected(monkeypatch):
    monkeypatch.setattr(config, "TRAIN_START", "2020-01-01")
    monkeypatch.setattr(config, "TRAIN_END", "2019-12-31")
    errors = validate_config()

    assert any("train split starts after it ends" in error for error in errors)

    monkeypatch.setattr(config, "TRAIN_START", "2015-01-01")
    monkeypatch.setattr(config, "TRAIN_END", "2019-12-31")
    monkeypatch.setattr(config, "VALID_START", "2019-12-31")
    errors = validate_config()

    assert any("validation split must start after" in error for error in errors)


def test_invalid_dates_are_reported_without_raising(monkeypatch):
    monkeypatch.setattr(config, "TEST_START", "not-a-date")
    errors = validate_config()

    assert any("test split has an invalid date" in error for error in errors)


def test_thresholds_and_strategy_ranges_are_checked(monkeypatch):
    monkeypatch.setattr(config, "THRESHOLD_GRID", [0.5, 1.5])
    monkeypatch.setattr(config, "MAX_FEATURE_MISSING_SHARE", 1.5)
    monkeypatch.setattr(config, "STRATEGY_S4_PROB_FULL", 0.0)
    monkeypatch.setattr(config, "STRATEGY_S4_MIN_WEIGHT", -0.1)
    errors = validate_config()

    assert any("THRESHOLD_GRID" in error for error in errors)
    assert any("MAX_FEATURE_MISSING_SHARE" in error for error in errors)
    assert any("STRATEGY_S4_PROB_FULL" in error for error in errors)
    assert any("STRATEGY_S4_MIN_WEIGHT" in error for error in errors)
