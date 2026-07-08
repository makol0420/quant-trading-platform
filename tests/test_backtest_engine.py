"""
Integration tests for backtest/engine.py -- checks that the event loop's
bookkeeping (fees, slippage, cash/position accounting) actually enforces
what strategy/risk.py computes, rather than just calling those functions
and ignoring the answer.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, LOOKBACK_BARS
from strategy.base import Strategy, Signal
from strategy.risk import RiskManager, RiskLimits


def _flat_price_df(n, price=100.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 1000.0},
        index=idx,
    )


class AlwaysLongStrategy(Strategy):
    """Deterministic: always wants a full-confidence long position."""
    def generate_signal(self, symbol, history):
        return Signal(symbol=symbol, target_position=1.0, confidence=1.0, reason="test_always_long")


class FlatStrategy(Strategy):
    def generate_signal(self, symbol, history):
        return Signal(symbol=symbol, target_position=0.0, confidence=0.0, reason="test_flat")


def test_engine_never_exceeds_max_position_pct():
    n = LOOKBACK_BARS + 50
    price_data = {"TEST/USD": _flat_price_df(n)}
    # Deliberately absurd risk_per_trade_pct + tiny stop distance so the
    # vol-scaled sizing formula alone would demand a huge position --
    # this isolates whether the hard max_position_pct cap actually binds.
    limits = RiskLimits(max_position_pct=20.0, risk_per_trade_pct=50.0, stop_loss_atr_mult=0.1)
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data=price_data,
        strategies={"TEST/USD": AlwaysLongStrategy()},
        risk_manager=risk,
        starting_equity=100_000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = engine.run()

    for _, row in result.positions_history.iterrows():
        notional = abs(row["TEST/USD"]) * 100.0
        assert notional <= 100_000.0 * 0.20 + 1e-6


def test_engine_equity_equals_cash_plus_mark_to_market():
    n = LOOKBACK_BARS + 80
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    df = pd.DataFrame(
        {"open": prices, "high": prices * 1.002, "low": prices * 0.998, "close": prices, "volume": 1000.0},
        index=idx,
    )
    price_data = {"TEST/USD": df}
    limits = RiskLimits(risk_per_trade_pct=1.0, stop_loss_atr_mult=2.0, max_position_pct=50.0, min_rebalance_fraction=0.0)
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data=price_data, strategies={"TEST/USD": AlwaysLongStrategy()},
        risk_manager=risk, starting_equity=100_000.0, fee_bps=10.0, slippage_bps=5.0,
    )
    result = engine.run()

    joined = result.equity_curve.join(result.positions_history)
    reconstructed = joined["cash"] + joined["TEST/USD"] * df["close"].reindex(joined.index)
    assert np.allclose(reconstructed.values, joined["equity"].values, atol=1e-6)


def test_engine_charges_fees_correctly_on_every_fill():
    n = LOOKBACK_BARS + 30
    df = _flat_price_df(n, price=100.0)
    rng = np.random.default_rng(2)
    df["high"] = df["close"] + np.abs(rng.normal(0, 0.5, n))
    df["low"] = df["close"] - np.abs(rng.normal(0, 0.5, n))

    price_data = {"TEST/USD": df}
    limits = RiskLimits(risk_per_trade_pct=1.0, stop_loss_atr_mult=2.0, max_position_pct=50.0, min_rebalance_fraction=0.0)
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data=price_data, strategies={"TEST/USD": AlwaysLongStrategy()},
        risk_manager=risk, starting_equity=100_000.0, fee_bps=10.0, slippage_bps=5.0,
    )
    result = engine.run()

    assert len(result.trades) > 0, "expected at least an initial entry trade"
    for t in result.trades:
        expected_fee = t.qty * t.price * (10.0 / 10_000)
        assert abs(t.fee - expected_fee) < 1e-9


def test_flat_strategy_never_trades_and_equity_stays_flat():
    n = LOOKBACK_BARS + 30
    price_data = {"TEST/USD": _flat_price_df(n)}
    risk = RiskManager(RiskLimits(), starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data=price_data, strategies={"TEST/USD": FlatStrategy()},
        risk_manager=risk, starting_equity=100_000.0,
    )
    result = engine.run()
    assert len(result.trades) == 0
    assert (result.equity_curve["equity"] == 100_000.0).all()


def test_mismatched_symbol_indices_raise():
    idx1 = pd.date_range("2026-01-01", periods=200, freq="5min", tz="UTC")
    idx2 = pd.date_range("2026-01-02", periods=200, freq="5min", tz="UTC")  # different range entirely
    df1 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx1)
    df2 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx2)
    risk = RiskManager(RiskLimits(), starting_equity=100_000.0)

    with pytest.raises(ValueError):
        BacktestEngine(
            price_data={"A": df1, "B": df2},
            strategies={"A": FlatStrategy(), "B": FlatStrategy()},
            risk_manager=risk,
        )
