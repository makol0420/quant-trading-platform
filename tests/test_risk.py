"""
Tests for strategy/risk.py -- the module most worth trusting before this
platform ever touches real money.
"""

from strategy.risk import (
    RiskManager,
    RiskLimits,
    exposure_trim_scale,
    gross_exposure_pct,
    projected_exposure_pct,
)

import pytest


def make_manager(**overrides) -> RiskManager:
    limits = RiskLimits(**overrides)
    return RiskManager(limits, starting_equity=100_000.0)


def test_position_size_scales_with_confidence():
    rm = make_manager(risk_per_trade_pct=1.0, stop_loss_atr_mult=2.0, max_position_pct=100.0)
    full_conf = rm.position_size(equity=100_000, price=100.0, atr=2.0, confidence=1.0)
    half_conf = rm.position_size(equity=100_000, price=100.0, atr=2.0, confidence=0.5)
    assert half_conf == full_conf / 2, "position size should scale linearly with confidence"


def test_position_size_respects_max_position_cap():
    rm = make_manager(risk_per_trade_pct=50.0, stop_loss_atr_mult=0.01, max_position_pct=10.0)
    # Deliberately huge risk-implied size (tiny stop distance) so the hard
    # cap, not the vol-scaled formula, is what actually binds.
    qty = rm.position_size(equity=100_000, price=100.0, atr=1.0, confidence=1.0)
    max_allowed_notional = 100_000 * 0.10
    assert qty * 100.0 <= max_allowed_notional + 1e-6


def test_position_size_zero_atr_returns_zero():
    rm = make_manager()
    assert rm.position_size(equity=100_000, price=100.0, atr=0.0, confidence=1.0) == 0.0


def test_daily_loss_limit_blocks_new_entries():
    rm = make_manager(max_daily_loss_pct=3.0)
    rm.update_equity(100_000, "2026-01-01T00:00:00")
    # Equity drops 4% intraday -- past the 3% daily loss limit.
    allowed, reason = rm.check_entry_allowed(equity=96_000, projected_exposure_pct=5.0)
    assert not allowed
    assert "daily_loss_limit" in reason


def test_daily_loss_limit_resets_on_new_day():
    rm = make_manager(max_daily_loss_pct=3.0)
    rm.update_equity(100_000, "2026-01-01T00:00:00")
    rm.update_equity(96_000, "2026-01-01T12:00:00")  # -4% intraday
    allowed, _ = rm.check_entry_allowed(equity=96_000, projected_exposure_pct=5.0)
    assert not allowed

    # New day -> day_start_equity resets to the new day's opening equity,
    # so the same absolute equity level is no longer "down 4% today".
    rm.update_equity(96_000, "2026-01-02T00:00:00")
    allowed, _ = rm.check_entry_allowed(equity=96_000, projected_exposure_pct=5.0)
    assert allowed


def test_max_drawdown_triggers_permanent_halt_until_reset():
    rm = make_manager(max_drawdown_pct=15.0)
    rm.update_equity(100_000, "2026-01-01T00:00:00")  # peak
    rm.update_equity(84_000, "2026-01-02T00:00:00")    # -16% drawdown -- breach

    assert rm.state.trading_halted is True
    allowed, reason = rm.check_entry_allowed(equity=84_000, projected_exposure_pct=1.0)
    assert not allowed
    assert "max_drawdown_kill_switch" in reason

    # Even if equity recovers, halt persists without an explicit reset --
    # this is intentional: a drawdown breach should require a human look,
    # not silently clear itself once the number looks better again.
    rm.update_equity(99_000, "2026-01-03T00:00:00")
    allowed, _ = rm.check_entry_allowed(equity=99_000, projected_exposure_pct=1.0)
    assert not allowed

    rm.reset_halt()
    allowed, _ = rm.check_entry_allowed(equity=99_000, projected_exposure_pct=1.0)
    assert allowed


def test_max_total_exposure_cap():
    rm = make_manager(max_total_exposure_pct=60.0)
    # The gate compares the book's PROJECTED TOTAL, not the size of the trade.
    allowed, reason = rm.check_entry_allowed(equity=100_000, projected_exposure_pct=70.0)
    assert not allowed
    assert "max_total_exposure_exceeded" in reason

    allowed, _ = rm.check_entry_allowed(equity=100_000, projected_exposure_pct=58.0)
    assert allowed


def test_max_total_exposure_cap_allows_exactly_the_limit():
    """>` not `>=`: a book sitting precisely on its cap is compliant."""
    rm = make_manager(max_total_exposure_pct=60.0)
    allowed, _ = rm.check_entry_allowed(equity=100_000, projected_exposure_pct=60.0)
    assert allowed


def test_gross_exposure_counts_shorts_as_exposure():
    """Netting a long against a short would report 0% for a book that is
    fully exposed in both directions."""
    positions = {"BTC/USDT": 1.0, "ETH/USDT": -1.0}
    prices = {"BTC/USDT": 10_000.0, "ETH/USDT": 10_000.0}
    assert gross_exposure_pct(positions, prices, 100_000.0) == 20.0


def test_gross_exposure_is_zero_for_an_empty_book():
    assert gross_exposure_pct({}, {}, 100_000.0) == 0.0


def test_gross_exposure_does_not_divide_by_zero():
    assert gross_exposure_pct({"A": 1.0}, {"A": 100.0}, 0.0) == 0.0


def test_projected_exposure_replaces_the_existing_leg_rather_than_adding_to_it():
    """
    Sizing an existing position up must not count its current leg twice.
    The old gate did `open_exposure + proposed_size`, so a symbol already
    holding 10% and asked for 15% while the rest of the book held 45%
    was scored as 70% against a 60% cap and wrongly rejected.
    """
    positions = {"A": 0.10, "B": 0.45}  # notional fractions of a $1 equity
    prices = {"A": 1.0, "B": 1.0}

    # A already holds 10%; going to 15% leaves the book at 45% + 15% = 60%.
    assert projected_exposure_pct(positions, prices, 1.0, "A", 0.15) == pytest.approx(60.0)


def test_projected_exposure_handles_flipping_through_zero():
    positions = {"A": 0.20, "B": 0.20}
    prices = {"A": 1.0, "B": 1.0}

    # Flipping A from +20% to -10% frees 10pp of the book, not adds 10pp.
    assert projected_exposure_pct(positions, prices, 1.0, "A", -0.10) == pytest.approx(30.0)


def test_projected_exposure_of_a_first_entry_is_just_that_position():
    assert projected_exposure_pct({}, {"A": 1.0}, 1.0, "A", 0.20) == pytest.approx(20.0)


def test_trim_scale_is_one_when_the_book_is_inside_its_cap():
    positions = {"A": 0.30, "B": 0.30}   # 60% of a $1 equity, exactly on cap
    prices = {"A": 1.0, "B": 1.0}
    assert exposure_trim_scale(positions, prices, 1.0, 60.0) == 1.0


def test_trim_scale_is_one_inside_the_tolerance_band():
    """Trimming is itself a trade. Reacting to a hair over the cap would pay
    fees to fix a rounding error."""
    positions = {"A": 0.3015, "B": 0.30}   # 60.15% -- within 0.5% of the 60% cap
    prices = {"A": 1.0, "B": 1.0}
    assert exposure_trim_scale(positions, prices, 1.0, 60.0) == 1.0


def test_trim_scale_brings_a_drifted_book_back_to_the_cap():
    positions = {"A": 0.40, "B": 0.30}   # 70% against a 60% cap
    prices = {"A": 1.0, "B": 1.0}

    scale = exposure_trim_scale(positions, prices, 1.0, 60.0)

    assert scale == pytest.approx(60.0 / 70.0)
    after = {s: q * scale for s, q in positions.items()}
    assert gross_exposure_pct(after, prices, 1.0) == pytest.approx(60.0)


def test_trim_scale_is_one_with_no_positions():
    assert exposure_trim_scale({}, {}, 1.0, 60.0) == 1.0


def test_trim_scale_does_not_divide_by_zero():
    assert exposure_trim_scale({"A": 1.0}, {"A": 100.0}, 0.0, 60.0) == 1.0


def test_trim_scale_flattens_a_book_under_a_zero_cap():
    positions = {"A": 0.10}
    prices = {"A": 1.0}
    assert exposure_trim_scale(positions, prices, 1.0, 0.0) == 0.0


def test_significant_change_always_allows_opening_closing_and_flipping():
    rm = make_manager(min_rebalance_fraction=0.5)  # generous band -- would otherwise block small resizes
    assert rm.is_significant_change(current_qty=0.0, desired_qty=1.0)   # opening
    assert rm.is_significant_change(current_qty=1.0, desired_qty=0.0)   # closing
    assert rm.is_significant_change(current_qty=1.0, desired_qty=-1.0)  # flipping


def test_significant_change_blocks_small_resizes_within_band():
    rm = make_manager(min_rebalance_fraction=0.20)
    # 5% change, same direction, well inside a 20% no-trade band.
    assert not rm.is_significant_change(current_qty=1.00, desired_qty=1.05)


def test_significant_change_allows_large_resizes_outside_band():
    rm = make_manager(min_rebalance_fraction=0.20)
    # 50% change, same direction -- clearly outside a 20% band.
    assert rm.is_significant_change(current_qty=1.00, desired_qty=1.50)
