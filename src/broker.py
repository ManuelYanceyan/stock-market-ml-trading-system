"""Paper broker adapter - deliberately INCAPABLE of sending real orders.

The project brief forbids live trading.  This module exists to show the
interface a real broker integration would use, while making the safety
boundary explicit:

* no credentials are read, stored or transmitted;
* ``PaperBroker`` only records simulated fills in ``reports/orders_simulated.csv``;
* constructing a "live" broker raises, so a caller cannot accidentally trade.

A real integration (Alpaca, IBKR, ...) would implement the same ``place_order``
contract in a separate, explicitly enabled adapter.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

import config
from src.utils import ensure_directories, get_logger, utc_now_iso

logger = get_logger("broker")

ORDERS_LOG = config.REPORTS_DIR / "orders_simulated.csv"


class LiveTradingDisabled(RuntimeError):
    """Raised whenever real-money trading is requested."""


@dataclass
class Order:
    ticker: str
    side: str            # "buy" or "sell"
    quantity: float
    reference_price: float
    session: str
    reason: str = ""
    order_id: str = ""
    status: str = "new"
    filled_price: float | None = None
    created_at: str = field(default_factory=utc_now_iso)


class PaperBroker:
    """Simulated broker: fills at the reference price plus slippage, no network."""

    def __init__(
        self,
        starting_cash: float = config.INITIAL_CAPITAL,
        commission_bps: float = config.COMMISSION_BPS,
        slippage_bps: float = config.SLIPPAGE_BPS,
        log_path: Path = ORDERS_LOG,
    ) -> None:
        self.cash = float(starting_cash)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.log_path = Path(log_path)
        self.orders: list[Order] = []
        ensure_directories()

    # -- internal ---------------------------------------------------------
    def _apply_slippage(self, side: str, price: float) -> float:
        drift = self.slippage_bps / 10_000.0
        return price * (1 + drift) if side == "buy" else price * (1 - drift)

    def _commission(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.commission_bps / 10_000.0

    def _append_log(self, order: Order) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(order).keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(asdict(order))

    # -- public API -------------------------------------------------------
    def place_order(self, order: Order) -> Order:
        if order.side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if order.quantity <= 0:
            raise ValueError("quantity must be positive")

        fill = self._apply_slippage(order.side, float(order.reference_price))
        notional = order.quantity * fill
        commission = self._commission(order.quantity, fill)

        if order.side == "buy" and notional + commission > self.cash + 1e-9:
            order.status = "rejected_insufficient_cash"
            self._append_log(order)
            logger.warning("Order rejected (cash): %s", order)
            return order

        self.cash += (-notional - commission) if order.side == "buy" else (notional - commission)
        order.filled_price = round(fill, 6)
        order.status = "filled_paper"
        order.order_id = order.order_id or f"paper-{len(self.orders) + 1:05d}"
        self.orders.append(order)
        self._append_log(order)
        logger.info("Paper fill: %s %s %.4f @ %.4f", order.side, order.ticker, order.quantity, fill)
        return order

    def equity(self, prices: dict[str, float]) -> float:
        """Cash only: the strategy is flat at the end of every session."""
        return self.cash

    def simulate_day(self, ticker: str, capital_fraction: float, open_price: float, close_price: float, session: str) -> tuple[Order, Order]:
        """Buy at the open and sell at the close (paper only)."""
        quantity = max(capital_fraction * self.cash / open_price, 0.0)
        buy = self.place_order(
            Order(ticker=ticker, side="buy", quantity=quantity, reference_price=open_price,
                  session=session, reason="intraday entry at the open")
        )
        sell = self.place_order(
            Order(ticker=ticker, side="sell", quantity=quantity, reference_price=close_price,
                  session=session, reason="intraday exit at the close")
        )
        return buy, sell


def build_broker(paper: bool = True, **kwargs) -> PaperBroker:
    """Factory that refuses anything other than the paper broker."""
    if not paper:
        raise LiveTradingDisabled(
            "Live trading is disabled in this research project. No broker credentials "
            "exist here and no real order can be sent."
        )
    return PaperBroker(**kwargs)


def positions_from_signal(signal: pd.DataFrame, threshold: float) -> list[dict]:
    """Translate the latest model output into an order intent (for display only)."""
    if signal.empty:
        return []
    row = signal.iloc[0]
    if float(row["prob_up_next_session"]) < threshold:
        return []
    return [
        {
            "ticker": row["ticker"],
            "side": "buy",
            "weight": 1.0,
            "probability": float(row["prob_up_next_session"]),
            "session": row["signal_session"],
            "note": "SIMULATED intent only - not sent to any venue",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    ]


if __name__ == "__main__":  # pragma: no cover - manual demo
    broker = build_broker(paper=True)
    broker.simulate_day("SPY", 1.0, open_price=500.0, close_price=505.0, session="demo")
    print(f"Demo paper equity: {broker.cash:,.2f}")
