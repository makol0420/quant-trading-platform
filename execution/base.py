"""
Abstract execution interface. Same idea as data/providers/base.py: the
orchestrator (orchestrator/runner.py) places orders through this interface
without knowing or caring whether it's paper, crypto-live, or forex-live
underneath.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrderResult:
    symbol: str
    side: str          # 'buy' or 'sell'
    requested_qty: float
    filled_qty: float
    fill_price: float
    fee: float
    order_id: str
    status: str          # 'filled', 'rejected', 'partial'
    message: str = ""


class ExecutionClient(ABC):
    @abstractmethod
    def get_equity(self) -> float:
        """Total account equity (cash + mark-to-market positions)."""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Current position quantity for symbol (negative = short)."""
        raise NotImplementedError

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Market order for simplicity -- extend with limit/stop types as needed."""
        raise NotImplementedError
