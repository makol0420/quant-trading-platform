"""
Live/historical forex data via OANDA's v20 REST API.

OANDA is used as the reference forex broker because it gives retail users
a free practice ("demo") account with full API access to both historical
candles and live pricing/execution under one login -- no separate data
vendor needed. Get a practice account + API token at:
https://www.oanda.com/demo-account/tpa/personal_token

This code is real and complete, but -- same caveat as the crypto provider
-- this sandbox has no route to OANDA's servers, so it has not been
exercised live. Test on your own machine:

    python -c "from data.providers.forex_oanda import ForexOandaProvider; \
               p = ForexOandaProvider(api_token='YOUR_TOKEN', practice=True); \
               print(p.historical('EUR_USD', 'M5', limit=5))"

Note OANDA uses underscore instrument names (EUR_USD, not EUR/USD) and its
own granularity codes (M1, M5, M15, H1, H4, D). The rest of this platform
uses the ccxt-style slash/lowercase convention, so config.yaml maps
between the two -- see `_TIMEFRAME_MAP` below.
"""

from __future__ import annotations
from datetime import datetime
from typing import Iterator, Optional
import time
import pandas as pd

try:
    from oandapyV20 import API
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    from oandapyV20.endpoints.pricing import PricingInfo
except ImportError as e:
    raise ImportError(
        "oandapyV20 is required for forex data: pip install oandapyV20"
    ) from e

from .base import DataProvider, Bar

_TIMEFRAME_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D",
}


def _to_oanda_instrument(symbol: str) -> str:
    return symbol.replace("/", "_")


class ForexOandaProvider(DataProvider):
    name = "forex_oanda"

    def __init__(self, api_token: str, practice: bool = True, account_id: str = ""):
        env = "practice" if practice else "live"
        self.client = API(access_token=api_token, environment=env)
        self.account_id = account_id  # only needed for order placement, not data

    def historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        instrument = _to_oanda_instrument(symbol)
        granularity = _TIMEFRAME_MAP.get(timeframe, timeframe)
        params = {"granularity": granularity, "price": "M"}  # midpoint prices
        if start and end:
            params["from"] = start.isoformat("T") + "Z"
            params["to"] = end.isoformat("T") + "Z"
        else:
            params["count"] = limit or 500

        req = InstrumentsCandles(instrument=instrument, params=params)
        resp = self.client.request(req)

        rows = []
        for candle in resp.get("candles", []):
            if not candle["complete"]:
                continue
            mid = candle["mid"]
            rows.append(
                {
                    "timestamp": pd.Timestamp(candle["time"]).tz_convert("UTC")
                    if pd.Timestamp(candle["time"]).tzinfo
                    else pd.Timestamp(candle["time"], tz="UTC"),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(candle["volume"]),
                }
            )

        if not rows:
            raise RuntimeError(
                f"No candles returned for {instrument}. Check the instrument "
                f"name and that your API token has data access."
            )

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """
        Polling-based live stream. OANDA also exposes a true streaming
        pricing endpoint (`PricingStream`) for tick-level data; polling
        the candle endpoint is simpler and adequate for bar-close strategies.
        """
        last_ts = None
        while True:
            df = self.historical(symbol, timeframe, limit=2)
            if last_ts is None or df.index[-1] > last_ts:
                last_ts = df.index[-1]
                latest = df.iloc[-1]
                yield Bar(
                    symbol=symbol,
                    timestamp=last_ts.to_pydatetime(),
                    open=float(latest["open"]),
                    high=float(latest["high"]),
                    low=float(latest["low"]),
                    close=float(latest["close"]),
                    volume=float(latest["volume"]),
                )
            time.sleep(5)

    def current_price(self, symbol: str) -> dict:
        """Live bid/ask -- used by the execution client for order pricing."""
        instrument = _to_oanda_instrument(symbol)
        req = PricingInfo(
            accountID=self.account_id, params={"instruments": instrument}
        )
        resp = self.client.request(req)
        price = resp["prices"][0]
        return {"bid": float(price["bids"][0]["price"]), "ask": float(price["asks"][0]["price"])}
