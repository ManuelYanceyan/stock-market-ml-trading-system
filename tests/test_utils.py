"""Shared utility tests: atomic IO, logging, session dates and formatting."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    format_num,
    format_pct,
    get_logger,
    latest_completed_session,
    load_parquet,
    read_json,
    save_parquet,
    utc_now_iso,
    write_json,
)


def test_write_and_read_json_round_trip_unicode(tmp_path):
    path = tmp_path / "nested" / "payload.json"
    payload = {"name": "señal", "value": 1.5, "items": [1, 2, 3]}

    written = write_json(path, payload)

    assert written == path
    assert read_json(path) == payload
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "señal"


def test_read_json_returns_default_for_missing_file(tmp_path):
    assert read_json(tmp_path / "absent.json", default={"a": 1}) == {"a": 1}
    assert read_json(tmp_path / "absent.json") is None


def test_failed_json_write_keeps_the_previous_artefact_and_cleans_up(tmp_path):
    path = tmp_path / "metrics.json"
    write_json(path, {"version": 1})

    circular: dict = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="Circular"):
        write_json(path, circular)

    assert read_json(path) == {"version": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_parquet_round_trip_preserves_values(tmp_path):
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [1.0, 2.5]})
    path = tmp_path / "processed" / "features.parquet"

    save_parquet(frame, path)

    loaded = load_parquet(path)
    pd.testing.assert_frame_equal(loaded, frame)


def test_failed_parquet_write_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "features.parquet"

    with pytest.raises(AttributeError):
        save_parquet("not a dataframe", path)

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        ("2024-01-08", "2024-01-05"),  # Monday -> Friday
        ("2024-01-06", "2024-01-05"),  # Saturday -> Friday
        ("2024-01-07", "2024-01-05"),  # Sunday -> Friday
        ("2024-01-10", "2024-01-09"),  # Wednesday -> Tuesday
    ],
)
def test_latest_completed_session_skips_weekends(today, expected):
    assert latest_completed_session(pd.Timestamp(today)) == pd.Timestamp(expected)


def test_logger_does_not_duplicate_handlers(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    name = "test_utils_logger"

    first = get_logger(name)
    second = get_logger(name)

    assert first is second
    assert sum(isinstance(handler, logging.FileHandler) for handler in first.handlers) == 1
    for handler in list(first.handlers):
        handler.close()
    first.handlers.clear()


def test_utc_timestamp_is_parseable():
    stamp = utc_now_iso()
    parsed = pd.Timestamp(stamp)

    assert stamp.endswith("Z")
    assert parsed.tz is not None


def test_formatters_handle_missing_values():
    assert format_pct(0.1234) == "12.34%"
    assert format_pct(None) == "n/a"
    assert format_pct(float("nan")) == "n/a"
    assert format_num(1.23456, digits=2) == "1.23"
    assert format_num(np.nan) == "n/a"
