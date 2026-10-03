"""
Live/historical crypto data via ccxt (https://github.com/ccxt/ccxt).

ccxt gives one unified interface across ~100 exchanges. Default here is
Binance because it's the most liquid and commonly used, but this class
works unmodified against Coinbase, Kraken, Bybit, etc. -- just change
`exchange_id` in config.yaml.

This code is real and complete. It is NOT reachable from the sandbox this
platform was built in (no route to api.binance.com there) -- it has not
been exercised against a live exchange. Test it on your own machine before
you trust it:

    python -c "from data.providers.crypto_ccxt import CryptoCCXTProvider; \
               p = CryptoCCXTProvider('binance'); \
               print(p.historical('BTC/USDT', '5m', limit=5))"

If that print statement gives you 5 rows of real OHLCV, the connector works.

Notes:
- Binance.com blocks US IPs for spot trading; US users typically need
  Binance.US (exchange_id='binanceus') or a different exchange entirely.
- Public market data (fetch_ohlcv) needs no API key. Only order placement
  (see execution/crypto_live.py) needs keys.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterator, Optional
import time
import pandas as pd
import ccxt

from .base import DataProvider, Bar


class CryptoCCXTProvider(DataProvider):
    name = "crypto_ccxt"

    def __init__(self, exchange_id: str = "binance", api_key: str = "", secret: str = ""):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": api_key or None,
                "secret": secret or None,
                "enableRateLimit": True,
            }
        )
        self.exchange_id = exchange_id

    def historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        limit = limit or 1000
        since = int(start.timestamp() * 1000) if start else None

        if since is None:
            # ccxt returns the *most recent* bars when `since` is omitted,
            # not the oldest. Paging forward from there runs off the end of
            # the series immediately, so asking for 20000 bars used to
            # silently return 1000 and report success. Deriving an explicit
            # start makes the forward paging below behave correctly, and
            # stays exchange-agnostic -- no venue-specific `endTime` param.
            tf_seconds = self.exchange.parse_timeframe(timeframe)
            since = self.exchange.milliseconds() - limit * tf_seconds * 1000

        all_rows = []
        cursor = since
        while len(all_rows) < limit:
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor,
                limit=min(limit - len(all_rows), 1000),
            )
            if not batch:
                break
            all_rows.extend(batch)

            # Advance past the last bar received. If the cursor ever fails
            # to move forward the exchange is echoing the same page, and
            # looping forever would hang the caller -- stop instead.
            next_cursor = batch[-1][0] + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < 1000:
                break  # short page == the exchange has no more history left
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_rows:
            raise RuntimeError(
                f"No OHLCV data returned for {symbol} on {self.exchange_id}. "
                f"Check the symbol is listed and the exchange is reachable."
            )

        df = pd.DataFrame(
            all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        # Pages can overlap by a bar; de-duplicate before reindexing so the
        # index is unique (and trim to the requested size).
        df = df.drop_duplicates(subset="ts", keep="last").tail(limit)
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("timestamp").drop(columns=["ts"]).sort_index()
        if end is not None:
            # pandas 3.0 rejects passing a tz-aware datetime together with
            # tz="UTC" ("Cannot pass a datetime or Timestamp with tzinfo with
            # the tz parameter"). Normalize to a UTC Timestamp first.
            end_ts = pd.Timestamp(end)
            end_ts = (
                end_ts.tz_convert("UTC") if end_ts.tzinfo is not None
                else end_ts.tz_localize("UTC")
            )
            df = df[df.index <= end_ts]
        return df

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """
        Polling-based live stream (simple, works everywhere ccxt does).
        For lower latency, use the exchange's native websocket via ccxt.pro
        instead -- left out here to avoid an extra paid dependency.
        """
        last_ts = None
        while True:
            df = self.historical(symbol, timeframe, limit=2)
            latest = df.iloc[-1]
            if last_ts is None or df.index[-1] > last_ts:
                last_ts = df.index[-1]
                yield Bar(
                    symbol=symbol,
                    timestamp=last_ts.to_pydatetime(),
                    open=float(latest["open"]),
                    high=float(latest["high"]),
                    low=float(latest["low"]),
                    close=float(latest["close"]),
                    volume=float(latest["volume"]),
                )
            time.sleep(max(5, self.exchange.rateLimit / 1000))

    def supported_symbols(self) -> list[str]:
        markets = self.exchange.load_markets()
        return list(markets.keys())
