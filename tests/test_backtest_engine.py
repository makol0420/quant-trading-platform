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


class LongAfterStrategy(Strategy):
    """Long at a fixed confidence, but only once the clock passes `after`."""
    def __init__(self, confidence=1.0, after=None):
        self.confidence = confidence
        self.after = after

    def generate_signal(self, symbol, history):
        if self.after is not None and history.index[-1] < self.after:
            return Signal(symbol=symbol, target_position=0.0, confidence=0.0, reason="test_waiting")
        return Signal(symbol=symbol, target_position=1.0, confidence=self.confidence, reason="test_long")


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

    The four symbols are deliberately identical: the cap has to be shared
    between them, and the only defensible way to share it is equally.
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
    # holding its legs at the cap can read a hair over 60%. The bug this
    # guards against is 80%, twenty points away.
    worst = result.equity_curve["exposure_pct"].max()
    assert worst <= 60.0 + 0.5, (
        f"gross exposure reached {worst:.1f}% against a 60% cap -- four symbols "
        f"at 20% each must not all be approved at full size"
    )

    # ...and the cap should genuinely be the thing holding it back, not the
    # sizing formula happening to want less than 20% per symbol.
    assert worst > 55.0, f"expected the book to approach its 60% cap, got {worst:.1f}%"

    # Every symbol is admitted, at a reduced size. Rejecting the last one
    # outright would also hold the cap, and would be the first-come-first-
    # served rule this replaced -- so the share each symbol gets is the
    # assertion that matters, not just the total.
    last = result.positions_history.iloc[-1]
    equity = result.equity_curve["equity"].iloc[-1]
    notionals = {s: abs(last[s]) * 100.0 for s in symbols}

    assert all(qty != 0 for qty in last[symbols]), (
        f"expected all four symbols to be allocated a share, got {dict(last)}"
    )
    assert sum(notionals.values()) / equity * 100 == pytest.approx(60.0, abs=0.6)

    largest, smallest = max(notionals.values()), min(notionals.values())
    assert largest - smallest <= largest * 0.01 + 1e-6, (
        f"identical symbols got unequal shares ({notionals}); allocation must "
        f"not depend on symbol ordering"
    )


def test_engine_allocation_does_not_depend_on_symbol_ordering():
    """
    The allocation rule, stated as a property: permuting the symbol order must
    not change what the portfolio ends up holding.

    Under first-come-first-served this fails by construction -- whoever sorts
    first takes the budget and whoever sorts last is rejected -- which is how
    symbol iteration order silently became a portfolio decision.
    """
    symbols = ["A/USD", "B/USD", "C/USD", "D/USD"]
    n = LOOKBACK_BARS + 120
    price_data = {s: _flat_price_df(n) for s in symbols}

    limits = RiskLimits(
        max_position_pct=20.0,
        max_total_exposure_pct=60.0,
        risk_per_trade_pct=50.0,
        stop_loss_atr_mult=0.1,
    )

    def run(order):
        risk = RiskManager(limits, starting_equity=100_000.0)
        engine = BacktestEngine(
            price_data={s: price_data[s] for s in order},
            strategies={s: AlwaysLongStrategy() for s in order},
            risk_manager=risk,
            starting_equity=100_000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
        )
        return engine.run()

    forward = run(symbols)
    backward = run(list(reversed(symbols)))

    assert forward.final_positions == pytest.approx(backward.final_positions)
    assert (
        forward.equity_curve["equity"].values
        == pytest.approx(backward.equity_curve["equity"].values)
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


def test_engine_cap_holds_when_a_suppressed_leg_meets_a_new_entry():
    """
    Regression test: a book that is compliant can still be pushed over its cap
    by a new entry, because the no-trade band lets a leg sit above its
    allocation.

    A leg is suppressed while it is within `min_rebalance_fraction` of its
    allocated size -- so a leg can hold MORE than its allocation and be left
    alone. With three such legs (20% each, allocation 17.1%, a 14.3% gap, just
    inside the 20% band) the book reads 60% and looks compliant. A fourth leg
    then wants to open at its full allocated 8.6% on top of that, which is
    68.6% of equity.

    The failure mode depends on the order the two corrections are applied in,
    and the test pins down the one that preserves the allocation rule. Without
    the correction, the book does NOT breach -- the entry gate is handed the
    banded book, sees 68.6% against a 60% cap, and refuses the new leg
    outright. The cap holds, but by rejecting the symbol that asked last:
    first-come-first-served, with the book frozen at three legs of 20% and the
    fourth locked out. Measured on the bundled backtest in the other order --
    gate handed the pre-band allocation -- the same shape put the book at
    71.1% against the 60% cap instead.

    So the assertion that carries the teeth is not the exposure ceiling; it is
    that all four legs are held, that they sum to the cap, and that the
    half-confidence leg holds the smaller share.
    """
    symbols = ["A/USD", "B/USD", "C/USD", "D/USD"]
    n = LOOKBACK_BARS + 200
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    # A wide high/low gives an ATR of 0.2 * price, which puts the vol-scaled
    # size BELOW the 20% position cap -- so confidence, not the cap, is what
    # sets each leg's size here, and the fourth leg can ask for a smaller one.
    df = pd.DataFrame(
        {"open": 100.0, "high": 110.0, "low": 90.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )

    limits = RiskLimits(
        risk_per_trade_pct=4.0,     # -> 20% of equity per leg at full confidence
        stop_loss_atr_mult=1.0,
        max_position_pct=20.0,
        max_total_exposure_pct=60.0,
        min_rebalance_fraction=0.2,
    )
    risk = RiskManager(limits, starting_equity=100_000.0)
    engine = BacktestEngine(
        price_data={s: df for s in symbols},
        strategies={
            "A/USD": LongAfterStrategy(confidence=1.0),
            "B/USD": LongAfterStrategy(confidence=1.0),
            "C/USD": LongAfterStrategy(confidence=1.0),
            # Half confidence -> 10% of equity, so the book it joins is
            # 70% of equity and the allocation scales it back to 60%.
            "D/USD": LongAfterStrategy(confidence=0.5, after=idx[LOOKBACK_BARS + 50]),
        },
        risk_manager=risk,
        starting_equity=100_000.0,
        fee_bps=10.0,
        slippage_bps=5.0,
    )
    result = engine.run()

    worst = result.equity_curve["exposure_pct"].max()
    assert worst <= 60.0 * 1.005 + 0.1, (
        f"book reached {worst:.2f}% against a 60% cap: three legs just inside "
        f"the no-trade band plus one new entry must not add up past the cap"
    )
    assert worst > 55.0, f"expected the book to reach its cap, got {worst:.1f}%"

    # The fourth leg is admitted rather than rejected, and the three existing
    # legs are trimmed to their allocated shares rather than left alone.
    last = result.positions_history.iloc[-1]
    notionals = {s: abs(last[s]) * 100.0 for s in symbols}
    assert all(qty != 0 for qty in last[symbols]), f"expected all four legs held, got {dict(last)}"
    assert sum(notionals.values()) == pytest.approx(60_000.0, rel=0.01)
    assert notionals["D/USD"] < notionals["A/USD"], (
        "the half-confidence leg should hold a smaller share: "
        f"{notionals}"
    )


def test_kill_switch_flattens_the_whole_book():
    """
    The drawdown kill-switch is a liquidation, not a pause.

    On halt the engine used to zero only the symbols that bar was about to ADD
    to. A position that is merely being held is neither adding nor trimming,
    so it was never put to the entry gate at all, and the book rode straight
    through the halt still carrying the exposure the switch exists to remove.

    The slide is deliberate: 12% down over 20 bars. Large enough to take a 60%
    book past a 5% drawdown limit, and small enough that the re-size it
    implies (about 5%) stays inside the 20% no-trade band -- so the position
    is suppressed, never reaches the gate, and the old code left it standing.
    A violent enough crash hides the bug instead, because the vol-scaled
    target genuinely grows and the symbol does get asked about.
    """
    symbols = ["A/USD", "B/USD", "C/USD"]
    n = LOOKBACK_BARS + 400
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")

    prices = np.full(n, 100.0)
    prices[LOOKBACK_BARS + 200: LOOKBACK_BARS + 220] = 100.0 * np.linspace(1.0, 0.88, 20)
    prices[LOOKBACK_BARS + 220:] = 88.0
    df = pd.DataFrame(
        {"open": prices, "high": prices * 1.001, "low": prices * 0.999,
         "close": prices, "volume": 1000.0},
        index=idx,
    )

    limits = RiskLimits(max_drawdown_pct=5.0, max_position_pct=20.0, min_rebalance_fraction=0.2)
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

    assert result.equity_curve["halted"].any(), "test is vacuous unless the kill-switch fired"
    halt_bar = result.equity_curve["halted"].idxmax()
    assert (result.positions_history.loc[:halt_bar, symbols] != 0.0).any().any(), (
        "test is vacuous unless the book was actually invested when the halt fired"
    )

    # From the bar the switch fires onward the book is empty -- not merely no
    # longer growing.
    after = result.positions_history.loc[halt_bar:]
    assert (after[symbols] == 0.0).all().all(), (
        f"the book was still held through the halt: {after.iloc[-1].to_dict()}"
    )
    assert all(qty == 0.0 for qty in result.final_positions.values()), (
        f"expected a flat book at the end, got {result.final_positions}"
    )

    # ...and being flat, equity stops moving entirely.
    assert result.equity_curve.loc[halt_bar:, "equity"].nunique() == 1, (
        "equity kept moving after a halt with an empty book"
    )

    # The liquidation fills say what they are. A trade log that attributes a
    # forced exit to the model's signal misreports the one thing it records.
    closing = [t for t in result.trades if t.timestamp == halt_bar]
    assert closing, "expected liquidation fills on the bar the switch fired"
    assert all("max_drawdown_kill_switch" in t.reason for t in closing), (
        f"liquidation fills were labelled as signal trades: {[t.reason for t in closing]}"
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
