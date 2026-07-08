"""
Abstract interface every data source must implement.

Why this exists: the backtester, the paper-trading loop, and the live
orchestrator all consume price data through this exact interface. That means
the SAME strategy and risk code runs unmodified whether it's replaying
history, paper trading, or executing for real. Only the provider swaps.

This is the single most important architectural decision in this codebase.
Platforms that write separate "backtest logic" and "live logic" reliably
discover — expensively, in production — that the two paths behaved
differently. Don't fork the logic; fork the data source.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional
import pandas as pd


@dataclass
class Bar:
    """A single OHLCV candle for one symbol."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class DataProvider(ABC):
    """
    Base class for anything that can supply OHLCV bars.

    Two modes:
      - historical(): bulk fetch for backtesting / model training.
      - stream(): yields bars one at a time as they "arrive", for paper
        trading or live trading. For historical replay providers, this
        just iterates a DataFrame. For live providers, this polls or
        subscribes to a websocket and yields real-time bars.
    """

    name: str = "base"

    @abstractmethod
    def historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame indexed by timestamp (UTC) with columns:
        open, high, low, close, volume. Sorted ascending, no gaps assumed.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """
        Yield Bar objects as they become available. Blocking generator —
        callers should run this in its own loop/thread/async task.
        """
        raise NotImplementedError

    def supported_symbols(self) -> list[str]:
        """Optional: override to expose what this provider can quote."""
        return []
