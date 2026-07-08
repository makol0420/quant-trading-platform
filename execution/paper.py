"""
Paper trading execution client: simulates fills against a live price feed
without risking real money. This is what sits between "backtest" and
"live" -- same code path as live trading (goes through the orchestrator,
same Strategy, same RiskManager), but orders fill against a simulated
book instead of a real exchange/broker.

Running this well (for days/weeks on a real, currently-updating price
feed) before ever switching to a live execution client is the whole point
of having three modes instead of two. A strategy that looks fine in
backtest can still fail in paper trading for reasons a backtest can't
surface -- data feed hiccups, a model that predicts confidently but is
miscalibrated on data structurally different from its training window,
timestamps/timezones not lining up the way you assumed, etc.
"""

from __future__ import annotations
import uuid
from datetime import datetime

from .base import ExecutionClient, OrderResult


class PaperExecutionClient(ExecutionClient):
    def __init__(self, starting_equity: float = 100_000.0, fee_bps: float = 10.0, slippage_bps: float = 5.0):
        self.cash = starting_equity
        self.positions: dict[str, float] = {}
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._last_prices: dict[str, float] = {}
        self.fill_log: list[dict] = []

    def mark_price(self, symbol: str, price: float) -> None:
        """Call this every time a new bar/tick arrives, before evaluating
        signals, so get_equity() reflects current mark-to-market value."""
        self._last_prices[symbol] = price

    def get_equity(self) -> float:
        equity = self.cash
        for symbol, qty in self.positions.items():
            price = self._last_prices.get(symbol)
            if price:
                equity += qty * price
        return equity

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def place_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        price = self._last_prices.get(symbol)
        if price is None:
            return OrderResult(
                symbol=symbol, side=side, requested_qty=qty, filled_qty=0.0,
                fill_price=0.0, fee=0.0, order_id="", status="rejected",
                message=f"no current price for {symbol}; call mark_price() first",
            )

        slip_mult = (1 + self.slippage_bps / 10_000) if side == "buy" else (1 - self.slippage_bps / 10_000)
        fill_price = price * slip_mult
        notional = qty * fill_price
        fee = notional * (self.fee_bps / 10_000)

        signed_qty = qty if side == "buy" else -qty
        self.cash -= signed_qty * fill_price
        self.cash -= fee
        self.positions[symbol] = self.positions.get(symbol, 0.0) + signed_qty

        order_id = str(uuid.uuid4())[:8]
        result = OrderResult(
            symbol=symbol, side=side, requested_qty=qty, filled_qty=qty,
            fill_price=fill_price, fee=fee, order_id=order_id, status="filled",
        )
        self.fill_log.append({"timestamp": datetime.utcnow().isoformat(), **result.__dict__})
        return result
