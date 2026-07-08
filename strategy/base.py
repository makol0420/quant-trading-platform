"""
Abstract strategy interface.

A Strategy's only job is: given price history (+ any model predictions),
decide a target position in {-1, 0, +1} (short, flat, long) for a symbol.
It does NOT decide position size or whether the trade clears risk checks
-- that's RiskManager's job (see risk.py). Keeping these separate means
you can swap the strategy (a different model, a different rule set)
without touching position sizing or kill-switch logic, and vice versa.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class Signal:
    symbol: str
    target_position: float  # -1.0 (full short) .. 0.0 (flat) .. 1.0 (full long)
    confidence: float        # 0..1, used by the risk manager for sizing
    reason: str = ""


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, symbol: str, history: pd.DataFrame) -> Signal:
        """
        history: OHLCV DataFrame up to and including the current bar.
        Must not be given anything after "now" -- callers (backtester,
        paper loop, live loop) are responsible for only ever passing data
        up to the current point in time.
        """
        raise NotImplementedError
