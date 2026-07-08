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

        all_rows = []
        remaining = limit
        cursor = since
        # ccxt caps rows per call (usually 500-1500 depending on exchange);
        # page backward/forward until we have `limit` rows.
        while remaining > 0:
            batch = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=cursor, limit=min(remaining, 1000)
            )
            if not batch:
                break
            all_rows.extend(batch)
            remaining -= len(batch)
            if len(batch) < min(remaining + len(batch), 1000):
                break
            cursor = batch[-1][0] + 1
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_rows:
            raise RuntimeError(
                f"No OHLCV data returned for {symbol} on {self.exchange_id}. "
                f"Check the symbol is listed and the exchange is reachable."
            )

        df = pd.DataFrame(
            all_rows, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("timestamp").drop(columns=["ts"]).sort_index()
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
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
