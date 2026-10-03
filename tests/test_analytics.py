"""
Tests for api/analytics.py -- the realized-PnL and win-rate derivation.

These exist because the dashboard's "Win Rate" card was previously a
hardcoded 71% with nothing behind it. Now that it is computed, the
computation needs to be right, and the honest cases matter as much as the
arithmetic: an open position has no realized outcome, and a metric that
cannot be computed must return None rather than 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import analytics


def _trade(symbol, side, qty, price):
    return {"symbol": symbol, "side": side, "qty": qty, "price": price}


def test_no_closed_trades_reports_none():
    stats = analytics.realized_stats([_trade("BTC/USDT", "buy", 1.0, 100.0)])

    assert stats["closed_trades"] == 0
    assert stats["win_rate"] is None
    assert stats["realized_pnl"] is None


def test_single_winning_round_trip():
    stats = analytics.realized_stats([
        _trade("BTC/USDT", "buy", 1.0, 100.0),
        _trade("BTC/USDT", "sell", 1.0, 110.0),
    ])

    assert stats["closed_trades"] == 1
    assert stats["wins"] == 1
    assert stats["realized_pnl"] == 10.0
    assert stats["win_rate"] == 100.0


def test_single_losing_round_trip():
    stats = analytics.realized_stats([
        _trade("BTC/USDT", "buy", 2.0, 100.0),
        _trade("BTC/USDT", "sell", 2.0, 95.0),
    ])

    assert stats["closed_trades"] == 1
    assert stats["losses"] == 1
    assert stats["realized_pnl"] == -10.0
    assert stats["win_rate"] == 0.0


def test_short_round_trip_is_signed_correctly():
    # Sell high, buy back low == profit on a short.
    stats = analytics.realized_stats([
        _trade("ETH/USDT", "sell", 1.0, 200.0),
        _trade("ETH/USDT", "buy", 1.0, 180.0),
    ])

    assert stats["realized_pnl"] == 20.0
    assert stats["win_rate"] == 100.0


def test_partial_close_counts_once_and_keeps_basis():
    stats = analytics.realized_stats([
        _trade("BTC/USDT", "buy", 10.0, 100.0),
        _trade("BTC/USDT", "sell", 4.0, 120.0),   # +80 realized, 6 still open
    ])

    assert stats["closed_trades"] == 1
    assert stats["realized_pnl"] == 80.0


def test_averaged_entry_basis():
    stats = analytics.realized_stats([
        _trade("BTC/USDT", "buy", 1.0, 100.0),
        _trade("BTC/USDT", "buy", 1.0, 200.0),   # avg basis 150
        _trade("BTC/USDT", "sell", 2.0, 160.0),  # +20
    ])

    assert stats["realized_pnl"] == 20.0


def test_mixed_wins_and_losses():
    stats = analytics.realized_stats([
        _trade("BTC/USDT", "buy", 1.0, 100.0),
        _trade("BTC/USDT", "sell", 1.0, 110.0),  # win
        _trade("BTC/USDT", "buy", 1.0, 100.0),
        _trade("BTC/USDT", "sell", 1.0, 90.0),   # loss
    ])

    assert stats["closed_trades"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["win_rate"] == 50.0


def test_malformed_entries_are_skipped_not_crashed():
    stats = analytics.realized_stats([
        {"symbol": "BTC/USDT", "side": "buy", "qty": 0, "price": 100.0},
        {"symbol": None, "side": "buy", "qty": 1.0, "price": 100.0},
        {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 0},
    ])

    assert stats["closed_trades"] == 0
    assert stats["win_rate"] is None


def test_summarize_snapshot_never_invents_values():
    summary = analytics.summarize_snapshot({
        "equity": 100_000.0,
        "positions": {},
        "latest_prices": {},
        "trade_log": [],
        "drawdown_pct": 0.0,
        "trading_halted": False,
    })

    assert summary["portfolio"] == 100_000.0
    assert summary["open_trades"] == 0
    # Nothing has closed, so there is no win rate -- not 0%, not 71%.
    assert summary["win_rate"] is None
    assert summary["daily_pl"] is None


def test_summarize_snapshot_lists_open_positions():
    summary = analytics.summarize_snapshot({
        "equity": 100_000.0,
        "positions": {"BTC/USDT": -0.5, "ETH/USDT": 0.0},
        "latest_prices": {"BTC/USDT": 80_000.0, "ETH/USDT": 3_000.0},
        "trade_log": [],
    })

    assert summary["open_trades"] == 1  # the flat ETH position is not a trade
    pos = summary["positions"][0]
    assert pos["symbol"] == "BTC/USDT"
    assert pos["side"] == "SHORT"
    assert pos["notional"] == 40_000.0
