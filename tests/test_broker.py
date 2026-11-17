"""Paper-broker accounting tests: cash, positions and rejected orders."""

from __future__ import annotations

import pytest

from src.broker import Order, PaperBroker


@pytest.fixture()
def broker(tmp_path) -> PaperBroker:
    return PaperBroker(
        starting_cash=10_000.0,
        commission_bps=1.0,
        slippage_bps=2.0,
        log_path=tmp_path / "orders_simulated.csv",
    )


def test_sell_without_holdings_is_rejected(broker):
    sell = broker.place_order(
        Order(ticker="SPY", side="sell", quantity=10.0, reference_price=100.0, session="2024-01-02")
    )

    assert sell.status == "rejected_insufficient_position"
    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}
    assert broker.orders == []


def test_sell_cannot_exceed_the_position(broker):
    buy = broker.place_order(
        Order(ticker="SPY", side="buy", quantity=10.0, reference_price=100.0, session="2024-01-02")
    )
    assert buy.status == "filled_paper"
    cash_after_buy = broker.cash
    position_after_buy = broker.positions["SPY"]

    sell = broker.place_order(
        Order(ticker="SPY", side="sell", quantity=11.0, reference_price=101.0, session="2024-01-02")
    )

    assert sell.status == "rejected_insufficient_position"
    assert broker.cash == pytest.approx(cash_after_buy)
    assert broker.positions["SPY"] == pytest.approx(position_after_buy)
    assert [order.side for order in broker.orders] == ["buy"]


def test_round_trip_updates_cash_and_clears_the_position(broker):
    buy, sell = broker.simulate_day(
        "SPY", capital_fraction=1.0, open_price=100.0, close_price=105.0, session="2024-01-02"
    )

    assert buy.status == "filled_paper"
    assert sell.status == "filled_paper"
    assert broker.positions == {}
    # 5% gross gain minus two sides of commission and slippage.
    assert broker.cash > 10_000.0
    assert broker.equity() == pytest.approx(broker.cash)


def test_rejected_buy_does_not_execute_the_matching_sell(broker):
    buy, sell = broker.simulate_day(
        "SPY", capital_fraction=2.0, open_price=100.0, close_price=105.0, session="2024-01-02"
    )

    assert buy.status == "rejected_insufficient_cash"
    assert sell.status == "rejected_buy_not_filled"
    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}
    assert broker.orders == []


def test_equity_marks_open_positions_to_market(broker):
    buy = broker.place_order(
        Order(ticker="SPY", side="buy", quantity=10.0, reference_price=100.0, session="2024-01-02")
    )
    assert buy.status == "filled_paper"

    cash = broker.cash
    assert broker.equity({"SPY": 110.0}) == pytest.approx(cash + 10.0 * 110.0)
    assert broker.equity({"SPY": 90.0}) == pytest.approx(cash + 10.0 * 90.0)


def test_equity_requires_a_mark_for_every_open_position(broker):
    buy = broker.place_order(
        Order(ticker="SPY", side="buy", quantity=10.0, reference_price=100.0, session="2024-01-02")
    )
    assert buy.status == "filled_paper"

    broker.last_prices.clear()

    with pytest.raises(ValueError, match="no mark available"):
        broker.equity()


def test_order_rejects_non_finite_quantity(broker):
    with pytest.raises(ValueError, match="finite positive"):
        broker.place_order(
            Order(ticker="SPY", side="buy", quantity=float("nan"), reference_price=100.0, session="2024-01-02")
        )

    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}


def test_order_rejects_non_positive_reference_price(broker):
    with pytest.raises(ValueError, match="finite positive"):
        broker.place_order(
            Order(ticker="SPY", side="buy", quantity=10.0, reference_price=0.0, session="2024-01-02")
        )

    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}


def test_simulate_day_rejects_non_positive_close_price(broker):
    with pytest.raises(ValueError, match="close_price"):
        broker.simulate_day(
            "SPY", capital_fraction=1.0, open_price=100.0, close_price=-1.0, session="2024-01-02"
        )

    assert broker.cash == pytest.approx(10_000.0)
    assert broker.positions == {}
