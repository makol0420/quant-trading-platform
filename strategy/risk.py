"""
Risk management: position sizing and portfolio-level kill switches.

If you only read one file in this codebase before going live, make it
this one. The model decides direction; this decides how much money is
ever actually at stake, and when to stop trading altogether. A mediocre
model with strict risk management survives being wrong. A brilliant
model with no risk management is a matter of when, not if, a bad week
wipes out months of gains -- this is the actual, boring, unglamorous
reason most retail trading bots that blow up do so, far more often than
"the model was bad."
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    risk_per_trade_pct: float = 0.5      # % of equity risked per trade (via stop distance)
    stop_loss_atr_mult: float = 2.0       # stop placed this many ATRs from entry
    take_profit_atr_mult: float = 3.5     # optional target, informational for now
    max_position_pct: float = 20.0        # hard cap: no single position > this % of equity
    max_total_exposure_pct: float = 60.0  # hard cap: sum of all open position sizes
    max_daily_loss_pct: float = 3.0       # halt new entries for the day past this loss
    max_drawdown_pct: float = 15.0        # halt ALL trading past this drawdown from peak equity
    min_rebalance_fraction: float = 0.20  # ignore position-size changes smaller than this fraction of the larger of (current, desired) size


@dataclass
class RiskState:
    """Mutable running state the RiskManager tracks across the session."""
    peak_equity: float
    day_start_equity: float
    current_day: str = ""  # 'YYYY-MM-DD', used to detect day rollover
    trading_halted: bool = False
    halt_reason: str = ""
    open_exposure_pct: float = 0.0


class RiskManager:
    def __init__(self, limits: RiskLimits, starting_equity: float):
        self.limits = limits
        self.state = RiskState(peak_equity=starting_equity, day_start_equity=starting_equity)

    def position_size(self, equity: float, price: float, atr: float, confidence: float) -> float:
        """
        Volatility-scaled position sizing: size such that if price moves
        `stop_loss_atr_mult` ATRs against the position, the loss equals
        `risk_per_trade_pct` of equity -- then scale that down further by
        the strategy's own confidence in the signal. Returns a quantity
        (units of the asset), not a dollar amount.
        """
        if atr <= 0 or price <= 0:
            return 0.0

        risk_dollars = equity * (self.limits.risk_per_trade_pct / 100.0) * confidence
        stop_distance = atr * self.limits.stop_loss_atr_mult
        if stop_distance <= 0:
            return 0.0

        qty = risk_dollars / stop_distance

        # Hard cap regardless of the vol-scaled result above.
        max_notional = equity * (self.limits.max_position_pct / 100.0)
        max_qty = max_notional / price
        return min(qty, max_qty)

    def check_entry_allowed(self, equity: float, projected_exposure_pct: float) -> tuple[bool, str]:
        """
        Call before opening/increasing a position. Returns (allowed, reason_if_not).

        `projected_exposure_pct` is the portfolio's TOTAL gross exposure as a
        percentage of equity *if this trade executes* -- not the size of the
        trade itself. That distinction is load-bearing. An earlier version took
        the proposed position's size and added it to `state.open_exposure_pct`,
        which both counted a symbol's existing leg twice when sizing up, and
        stayed stale for the rest of the bar because callers computed total
        exposure once and then judged several symbols against that one
        pre-computed number. Four symbols at 20% each sailed through a 60% cap
        and landed at 80% exposure.

        Exposure is now allocated upstream in one shot by
        `exposure_scale_to_cap`, which is the only way to make the rule
        order-independent; callers therefore pass the exposure of the book
        those allocations add up to, not a per-symbol projection (a per-symbol
        projection would reject whichever symbol happened to be evaluated once
        the budget looked spent -- first-come-first-served by another name).
        The exposure branch here is consequently a backstop that the allocator
        should never let fire, and it shares that allocator's deadband so the
        two cannot disagree about whether a book is compliant. The halt and
        daily-loss branches are not backstops; they are the reason this is
        still called for every entry.
        """
        if self.state.trading_halted:
            return False, self.state.halt_reason

        daily_loss_pct = (self.state.day_start_equity - equity) / self.state.day_start_equity * 100
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            return False, f"daily_loss_limit_hit ({daily_loss_pct:.2f}% >= {self.limits.max_daily_loss_pct}%)"

        drawdown_pct = (self.state.peak_equity - equity) / self.state.peak_equity * 100
        if drawdown_pct >= self.limits.max_drawdown_pct:
            self.state.trading_halted = True
            self.state.halt_reason = f"max_drawdown_kill_switch ({drawdown_pct:.2f}% >= {self.limits.max_drawdown_pct}%)"
            return False, self.state.halt_reason

        if projected_exposure_pct > self.limits.max_total_exposure_pct * (1.0 + EXPOSURE_TOLERANCE):
            return False, (
                f"max_total_exposure_exceeded "
                f"({projected_exposure_pct:.1f}% > {self.limits.max_total_exposure_pct}% cap)"
            )

        return True, ""

    def update_equity(self, equity: float, timestamp) -> None:
        """Call once per bar/tick with current mark-to-market equity."""
        day_str = str(timestamp)[:10]
        if day_str != self.state.current_day:
            self.state.current_day = day_str
            self.state.day_start_equity = equity

        self.state.peak_equity = max(self.state.peak_equity, equity)

        drawdown_pct = (self.state.peak_equity - equity) / self.state.peak_equity * 100
        if drawdown_pct >= self.limits.max_drawdown_pct and not self.state.trading_halted:
            self.state.trading_halted = True
            self.state.halt_reason = f"max_drawdown_kill_switch ({drawdown_pct:.2f}%)"

    def is_significant_change(self, current_qty: float, desired_qty: float) -> bool:
        """
        True if moving from current_qty to desired_qty is worth the
        transaction cost of actually trading. Opening from flat, closing
        to flat, and flipping direction always count as significant.
        Pure resizing (same direction, still nonzero) only counts if the
        change exceeds min_rebalance_fraction of the larger position --
        without this, a strategy that continuously re-sizes based on a
        smoothly varying confidence score will pay fees+slippage on
        nearly every bar for changes too small to matter, which drowns
        out whatever real edge the signal has. This is a standard
        "no-trade band" / turnover control, not a way of making backtest
        numbers look better -- it reflects how any real execution desk
        would actually implement continuous-confidence sizing.
        """
        opening_or_closing = (current_qty == 0) != (desired_qty == 0)
        flipping = current_qty != 0 and desired_qty != 0 and (current_qty > 0) != (desired_qty > 0)
        if opening_or_closing or flipping:
            return True

        denom = max(abs(current_qty), abs(desired_qty), 1e-12)
        return abs(desired_qty - current_qty) / denom > self.limits.min_rebalance_fraction

    def reset_halt(self) -> None:
        """Manual reset after a halt -- deliberately not automatic. A
        drawdown breach should require a human to look at what happened
        before trading resumes."""
        self.state.trading_halted = False
        self.state.halt_reason = ""


def gross_exposure_pct(positions: dict[str, float], prices: dict[str, float], equity: float) -> float:
    """
    Total gross notional exposure as a percentage of equity: the sum of
    |qty * price| across open positions, long and short alike.

    Shorts count against the cap exactly as longs do. The cap exists to
    bound how much a bad bar can move the portfolio, and a short moves it
    as much as the equivalent long -- netting them would report a
    market-neutral-looking 0% for a book that is fully exposed in both
    directions.

    Returns 0.0 for non-positive equity rather than dividing by zero.

    Note that this measures a book as a whole, so it is not a substitute for
    asking what a single trade would do: adding a symbol's proposed size to a
    total that already includes that symbol's existing position counts its
    current leg twice, and rejects trades that would land exactly on the cap.
    """
    if equity <= 0:
        return 0.0
    gross = sum(abs(qty) * prices[symbol] for symbol, qty in positions.items())
    return gross / equity * 100


# Deadband on acting against the exposure cap, shared by the allocator
# (`exposure_scale_to_cap`) and the entry gate (`check_entry_allowed`) so the
# two cannot disagree about whether a book is compliant. Acting is itself a
# trade -- it pays fees and slippage -- so reacting to a 0.01pp breach would
# bleed cost to fix a rounding error.
EXPOSURE_TOLERANCE = 0.005


def is_over_exposure_cap(
    positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
    max_total_exposure_pct: float,
    tolerance: float = EXPOSURE_TOLERANCE,
) -> bool:
    """
    True when a HELD book is past its exposure cap, beyond the deadband.

    Callers use this to decide whether the cap outranks the no-trade band. The
    band exists to avoid paying fees for changes too small to matter, which is
    a judgement about signal noise -- but it will happily decline to shrink a
    book that is genuinely over its risk limit, because a slow drift produces
    exactly the small per-bar deltas the band is designed to ignore. When this
    returns True the cap wins and the correction goes through unscaled.
    """
    if equity <= 0:
        return False
    return gross_exposure_pct(positions, prices, equity) > max_total_exposure_pct * (1.0 + tolerance)


def exposure_scale_to_cap(
    positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
    max_total_exposure_pct: float,
    tolerance: float = EXPOSURE_TOLERANCE,
) -> float:
    """
    Factor to multiply a whole book of desired positions by so that its total
    gross exposure lands inside `max_total_exposure_pct`. Returns 1.0 when the
    book already fits.

    This is the platform's position-allocation rule. Callers hand over the book
    they WOULD hold and get back the fraction of it they are allowed to hold,
    which makes the cap order-independent: when demand exceeds it, every
    position is scaled by the same factor rather than whichever symbols happen
    to be evaluated last being rejected outright while the first ones take the
    entire budget. First-come-first-served is what a naive sequential loop
    does, and it quietly turns symbol ordering into a portfolio decision -- on
    the bundled backtest it rejected the fourth of four equivalent signals
    purely for arriving last, which accounted for about two-thirds of a 5.6pp
    win-rate gap between two runs of the same strategy.

    Scaling the desired book also subsumes clamping the held one. A position
    sized against the equity that existed at entry represents a larger share
    once that equity moves, and the no-trade band deliberately suppresses the
    resize that would correct it, so the book would otherwise drift past its
    cap and sit there -- measured at 61.7% against a 60% cap across 112 bars,
    with cash untouched throughout, i.e. purely equity moving under a fixed
    notional.

    The tolerance is a deadband on acting, not slack in the cap: the resulting
    book may sit up to `tolerance` above the limit rather than churning to
    correct a rounding error.
    """
    if equity <= 0:
        return 1.0
    if not is_over_exposure_cap(positions, prices, equity, max_total_exposure_pct, tolerance):
        return 1.0
    return max_total_exposure_pct / gross_exposure_pct(positions, prices, equity)
