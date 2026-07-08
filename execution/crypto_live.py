"""
Live crypto order execution via ccxt. Places REAL orders with REAL money
once wired up with real API keys. Same untested-in-this-sandbox caveat as
data/providers/crypto_ccxt.py applies here -- verify against your
exchange's testnet (Binance, Bybit, and others offer one) before this ever
touches a funded account.

Safety-by-default: this client refuses to place any order unless the
environment variable LIVE_TRADING_CONFIRMED is set to exactly "yes". This
is deliberately not a config.yaml flag -- an environment variable that
must be set explicitly in the shell you're launching from is a slightly
higher-friction, harder-to-leave-on-by-accident confirmation than a
setting buried in a config file. It is not a substitute for your own
judgment, only a guard against launching in live mode by mistake (wrong
config file, copy-pasted command, etc).
"""

from __future__ import annotations
import os
import ccxt

from .base import ExecutionClient, OrderResult


class CryptoLiveExecutionClient(ExecutionClient):
    def __init__(self, exchange_id: str, api_key: str, secret: str):
        if os.environ.get("LIVE_TRADING_CONFIRMED") != "yes":
            raise RuntimeError(
                "Live crypto trading blocked: set the environment variable "
                "LIVE_TRADING_CONFIRMED=yes to enable real order placement. "
                "This is a deliberate safety gate, not a bug -- see execution/crypto_live.py."
            )
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
        )

    def get_equity(self) -> float:
        balance = self.exchange.fetch_balance()
        # Sums free+used balances converted at last price would need a
        # pricing pass per asset; for a single quote-currency portfolio
        # (e.g. everything denominated in USDT) this simplifies to:
        total = balance.get("total", {})
        return float(total.get("USDT", 0.0))

    def get_position(self, symbol: str) -> float:
        base = symbol.split("/")[0]
        balance = self.exchange.fetch_balance()
        return float(balance.get("total", {}).get(base, 0.0))

    def place_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        try:
            order = self.exchange.create_order(symbol, type="market", side=side, amount=qty)
            fill_price = float(order.get("average") or order.get("price") or 0.0)
            filled = float(order.get("filled") or qty)
            fee = 0.0
            if order.get("fee"):
                fee = float(order["fee"].get("cost", 0.0))
            return OrderResult(
                symbol=symbol, side=side, requested_qty=qty, filled_qty=filled,
                fill_price=fill_price, fee=fee, order_id=str(order.get("id", "")),
                status="filled" if filled >= qty * 0.99 else "partial",
            )
        except Exception as e:
            return OrderResult(
                symbol=symbol, side=side, requested_qty=qty, filled_qty=0.0,
                fill_price=0.0, fee=0.0, order_id="", status="rejected", message=str(e),
            )
