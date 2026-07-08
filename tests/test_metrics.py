"""
Tests for backtest/metrics.py -- particularly _trade_pnls, the FIFO
round-trip matcher. This function was rewritten mid-development after an
earlier version shipped with leftover placeholder arithmetic (see git
history / development notes) -- these tests exist so that class of bug
can't quietly come back.
"""

from backtest.engine import Trade
from backtest.metrics import _trade_pnls
import pandas as pd


def _trade(ts, symbol, side, qty, price, fee=0.0, reason=""):
    return Trade(timestamp=pd.Timestamp(ts), symbol=symbol, side=side, qty=qty, price=price, fee=fee, reason=reason)


def test_simple_long_round_trip_profit():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 1.0, 100.0),
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 1.0, 110.0),
    ]
    pnls = _trade_pnls(trades)
    assert len(pnls) == 1
    assert pnls[0] == 10.0  # bought at 100, sold at 110, no fees


def test_simple_long_round_trip_loss():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 1.0, 100.0),
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 1.0, 95.0),
    ]
    pnls = _trade_pnls(trades)
    assert pnls[0] == -5.0


def test_simple_short_round_trip_profit():
    trades = [
        _trade("2026-01-01T00:00", "EUR/USD", "sell", 10_000.0, 1.10),  # open short
        _trade("2026-01-01T00:05", "EUR/USD", "buy", 10_000.0, 1.08),   # cover
    ]
    pnls = _trade_pnls(trades)
    assert len(pnls) == 1
    # short profit = (entry - exit) * qty = (1.10 - 1.08) * 10000
    assert abs(pnls[0] - 200.0) < 1e-6


def test_fees_reduce_pnl_proportionally():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 2.0, 100.0, fee=1.0),
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 2.0, 110.0, fee=1.0),
    ]
    pnls = _trade_pnls(trades)
    # gross = (110-100)*2 = 20; minus the sell's fee (1.0, fully attributed
    # since the whole sell matches this one lot) = 19. The buy's fee isn't
    # charged against P&L here since only the closing trade's fee is
    # attributed per matched chunk in this simplified model.
    assert abs(pnls[0] - 19.0) < 1e-6


def test_partial_close_produces_two_pnl_entries():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 2.0, 100.0),
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 1.0, 110.0),  # closes half
        _trade("2026-01-01T00:10", "BTC/USDT", "sell", 1.0, 120.0),  # closes the rest
    ]
    pnls = _trade_pnls(trades)
    assert len(pnls) == 2
    assert abs(pnls[0] - 10.0) < 1e-6  # 1 unit: 110-100
    assert abs(pnls[1] - 20.0) < 1e-6  # 1 unit: 120-100


def test_position_flip_in_one_trade_realizes_pnl_and_opens_new_lot():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 10.0, 100.0),   # long 10
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 15.0, 105.0),  # closes 10 long, opens 5 short
        _trade("2026-01-01T00:10", "BTC/USDT", "buy", 5.0, 102.0),    # covers the 5 short
    ]
    pnls = _trade_pnls(trades)
    assert len(pnls) == 2
    assert abs(pnls[0] - 50.0) < 1e-6   # closed the 10 long: (105-100)*10
    assert abs(pnls[1] - 15.0) < 1e-6   # closed the 5 short: (105-102)*5


def test_independent_symbols_do_not_interfere():
    trades = [
        _trade("2026-01-01T00:00", "BTC/USDT", "buy", 1.0, 100.0),
        _trade("2026-01-01T00:00", "ETH/USDT", "buy", 1.0, 50.0),
        _trade("2026-01-01T00:05", "BTC/USDT", "sell", 1.0, 110.0),
        _trade("2026-01-01T00:05", "ETH/USDT", "sell", 1.0, 40.0),
    ]
    pnls = _trade_pnls(trades)
    assert sorted(pnls) == sorted([10.0, -10.0])
