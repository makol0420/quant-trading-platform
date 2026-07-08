"""
Live forex order execution via OANDA's v20 API. Same safety gate as
execution/crypto_live.py -- requires LIVE_TRADING_CONFIRMED=yes -- and the
same "not exercised against a live server from this sandbox" caveat.

Test against OANDA's practice environment first (practice=True uses
their demo servers with fake money on a real account -- this is the
correct place to validate order placement, not the live environment).
"""

from __future__ import annotations
import os
from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.accounts import AccountSummary
from oandapyV20.endpoints.positions import OpenPositions

from .base import ExecutionClient, OrderResult
from data.providers.forex_oanda import _to_oanda_instrument


class ForexLiveExecutionClient(ExecutionClient):
    def __init__(self, api_token: str, account_id: str, practice: bool = True):
        if os.environ.get("LIVE_TRADING_CONFIRMED") != "yes":
            raise RuntimeError(
                "Live forex trading blocked: set the environment variable "
                "LIVE_TRADING_CONFIRMED=yes to enable real order placement. "
                "This is a deliberate safety gate, not a bug -- see execution/forex_live.py."
            )
        env = "practice" if practice else "live"
        self.client = API(access_token=api_token, environment=env)
        self.account_id = account_id

    def get_equity(self) -> float:
        req = AccountSummary(accountID=self.account_id)
        resp = self.client.request(req)
        return float(resp["account"]["NAV"])

    def get_position(self, symbol: str) -> float:
        instrument = _to_oanda_instrument(symbol)
        req = OpenPositions(accountID=self.account_id)
        resp = self.client.request(req)
        for pos in resp.get("positions", []):
            if pos["instrument"] == instrument:
                long_units = float(pos["long"]["units"])
                short_units = float(pos["short"]["units"])
                return long_units + short_units  # short units are already negative
        return 0.0

    def place_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        instrument = _to_oanda_instrument(symbol)
        units = int(qty) if side == "buy" else -int(qty)
        order_payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        try:
            req = OrderCreate(accountID=self.account_id, data=order_payload)
            resp = self.client.request(req)
            fill = resp.get("orderFillTransaction")
            if fill is None:
                return OrderResult(
                    symbol=symbol, side=side, requested_qty=qty, filled_qty=0.0,
                    fill_price=0.0, fee=0.0, order_id=resp.get("orderCreateTransaction", {}).get("id", ""),
                    status="rejected", message=str(resp.get("orderCancelTransaction", resp)),
                )
            return OrderResult(
                symbol=symbol, side=side, requested_qty=qty,
                filled_qty=abs(float(fill["units"])), fill_price=float(fill["price"]),
                fee=abs(float(fill.get("commission", 0.0))), order_id=fill["id"], status="filled",
            )
        except Exception as e:
            return OrderResult(
                symbol=symbol, side=side, requested_qty=qty, filled_qty=0.0,
                fill_price=0.0, fee=0.0, order_id="", status="rejected", message=str(e),
            )
