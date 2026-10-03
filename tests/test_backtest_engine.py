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


def test_engine_enforces_total_exposure_cap_across_symbols_within_one_bar():
    """
    Regression test for the aggregate-cap bug.

    open_exposure_pct used to be computed once per bar, before the per-symbol
    loop, and never updated as positions were opened. Every symbol was then
    gated against that same pre-loop number, so with four symbols each
    wanting 20% of a 60%-capped book, all four were individually approved
    (0 + 20 <= 60) and the portfolio landed at 80% exposure. The cap is a
    statement about the portfolio, so it has to bind on the portfolio.
    """
    symbols = ["A/USD", "B/USD", "C/USD", "D/USD"]
    n = LOOKBACK_BARS + 50
    price_data = {s: _flat_price_df(n) for s in symbols}

    limits = RiskLimits(
        max_position_pct=20.0,
        max_total_exposure_pct=60.0,
        risk_per_trade_pct=50.0,
        stop_loss_atr_mult=0.1,   # vol-scaled sizing alone would demand far more than 20%
        min_rebalance_fraction=0.2,
    )
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data=price_data,
        strategies={s: AlwaysLongStrategy() for s in symbols},
        risk_manager=risk,
        starting_equity=100_000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = engine.run()

    assert result.trades, "test is vacuous unless the book actually opened positions"

    # Tolerance for fees and slippage: they shrink equity slightly, so a book
    # holding three ~20% legs can read a hair over 60%. The bug this guards
    # against is 80%, twenty points away.
    worst = result.equity_curve["exposure_pct"].max()
    assert worst <= 60.0 + 0.5, (
        f"gross exposure reached {worst:.1f}% against a 60% cap -- four symbols "
        f"at 20% each must not all be approved"
    )

    # ...and the cap should genuinely be the thing holding it back, not the
    # sizing formula happening to want less than 20% per symbol.
    assert worst > 55.0, f"expected the book to approach its 60% cap, got {worst:.1f}%"

    held = (result.positions_history.abs() > 0).sum(axis=1).max()
    assert held == 3, (
        f"expected exactly 3 of 4 symbols to be admitted against a 60% cap at "
        f"20% each, saw {held}"
    )


def test_engine_trims_a_drifted_book_back_inside_the_cap():
    """
    The companion to the test above: the cap has to bound the book while it
    is HELD, not only at the moment of entry.

    A position sized at 20% of the equity that existed when it was opened
    represents more than 20% once that equity grows underneath it, and the
    no-trade band deliberately suppresses the resize that would otherwise
    correct it (the drift here is ~10%, inside the 20% band). So the book
    drifts past its cap and sits there. On the bundled backtest this read
    61.7% against a 60% cap across 112 bars with cash untouched throughout --
    purely equity moving under a fixed notional.

    A 10% ramp over the run is chosen because it stays inside the no-trade
    band: with the trim disabled this reaches 62.0%, so the test cannot pass
    on the strength of ordinary re-sizing.
    """
    symbols = ["A/USD", "B/USD", "C/USD"]
    n = LOOKBACK_BARS + 600
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    prices = 100.0 * np.exp(np.log(1.10) * np.arange(n) / n)  # steady +10%
    df = pd.DataFrame(
        {"open": prices, "high": prices * 1.001, "low": prices * 0.999,
         "close": prices, "volume": 1000.0},
        index=idx,
    )
    limits = RiskLimits(
        max_position_pct=20.0,
        max_total_exposure_pct=60.0,
        risk_per_trade_pct=50.0,
        stop_loss_atr_mult=0.1,
    )
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data={s: df for s in symbols},
        strategies={s: AlwaysLongStrategy() for s in symbols},
        risk_manager=risk,
        starting_equity=100_000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = engine.run()

    worst = result.equity_curve["exposure_pct"].max()
    assert worst <= 60.0 * 1.005 + 0.1, (
        f"held book drifted to {worst:.2f}% against a 60% cap; the trim should "
        f"hold it within the 0.5% tolerance band"
    )

    # Three entries plus at least one trim -- proof the clamp actually fired
    # rather than the book simply never having drifted.
    assert len(result.trades) > len(symbols), (
        f"expected trim fills beyond the {len(symbols)} entries, saw "
        f"{len(result.trades)} trades"
    )


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
