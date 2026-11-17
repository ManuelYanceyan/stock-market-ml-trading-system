"""Report-selection tests: validation drives the choice, test is descriptive."""

from __future__ import annotations

import pytest

from src.report import select_strategy


def test_strategy_is_selected_on_validation_not_test():
    validation = {
        "S1": {"total_return": 0.10},
        "S2": {"total_return": 0.30},
    }
    test = {
        "S1": {"total_return": 0.50},
        "S2": {"total_return": 0.20},
    }

    selected, test_leader, ranking = select_strategy(validation, test)

    assert selected == "S2"
    assert test_leader == "S1"
    assert [name for name, _ in ranking] == ["S2", "S1"]


def test_strategy_sets_must_match_between_windows():
    with pytest.raises(ValueError, match="strategy sets differ"):
        select_strategy({"S1": {"total_return": 0.1}}, {"S2": {"total_return": 0.2}})


def test_empty_validation_results_are_rejected():
    with pytest.raises(ValueError, match="validation results are required"):
        select_strategy({}, {"S1": {"total_return": 0.1}})
