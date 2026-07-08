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

    def check_entry_allowed(self, equity: float, proposed_exposure_pct: float) -> tuple[bool, str]:
        """Call before opening/increasing a position. Returns (allowed, reason_if_not)."""
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

        if self.state.open_exposure_pct + proposed_exposure_pct > self.limits.max_total_exposure_pct:
            return False, (
                f"max_total_exposure_exceeded "
                f"({self.state.open_exposure_pct + proposed_exposure_pct:.1f}% > {self.limits.max_total_exposure_pct}%)"
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
