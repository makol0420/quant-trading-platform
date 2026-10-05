"""
Event-driven backtest engine.

"Event-driven" (loop bar-by-bar, maintain positions/cash explicitly) rather
than "vectorized" (compute returns * position-sign across a whole array
at once) on purpose: it's slower, but it's the same loop shape as the
paper-trading and live-trading orchestrator, which is what lets identical
Strategy and RiskManager code run in all three modes. A vectorized
backtest is easy to write but structurally can't share code with the live
path -- which reintroduces exactly the backtest/live divergence risk this
whole architecture exists to avoid.

Models here, deliberately: trading fees (bps per side), slippage (bps,
applied as an adverse price adjustment on every fill), and position-level
risk limits. NOT modeled: partial fills / order-book depth, funding rates
on crypto perpetuals, margin interest, or latency -- a real venue-specific
execution simulator would need those; this gives you a realistic-enough
first pass and an architecture that's straightforward to extend.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from strategy.base import Strategy
from strategy.risk import (
    RiskManager,
    exposure_scale_to_cap,
    gross_exposure_pct,
    is_over_exposure_cap,
)
from features.indicators import atr as compute_atr

LOOKBACK_BARS = 100  # bounded history window passed to strategies each step


@dataclass
class Trade:
    timestamp: pd.Timestamp
    symbol: str
    side: str          # 'buy' or 'sell'
    qty: float
    price: float        # execution price, AFTER slippage
    fee: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame     # index=timestamp, columns=[equity, cash, drawdown_pct, exposure_pct]
    trades: list[Trade]
    positions_history: pd.DataFrame  # index=timestamp, one column per symbol (qty held)
    final_positions: dict[str, float]


class BacktestEngine:
    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        strategies: dict[str, Strategy],
        risk_manager: RiskManager,
        starting_equity: float = 100_000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.price_data = price_data
        self.strategies = strategies
        self.risk = risk_manager
        self.starting_equity = starting_equity
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

        symbols = list(price_data.keys())
        # This reference implementation assumes all symbols share one
        # timestamp index (true for the bundled synthetic generator, which
        # stamps every symbol on the same grid). Feeding real multi-venue
        # data with gaps/misaligned timestamps would need a resampling or
        # forward-fill step here first.
        self.index = price_data[symbols[0]].index
        for s in symbols[1:]:
            if not price_data[s].index.equals(self.index):
                raise ValueError(
                    f"Symbol '{s}' has a different timestamp index than '{symbols[0]}'. "
                    f"Align/resample all symbols to a common index before backtesting."
                )

        self._atr_series = {
            sym: compute_atr(df["high"], df["low"], df["close"], window=14)
            for sym, df in price_data.items()
        }

    def run(self) -> BacktestResult:
        symbols = list(self.price_data.keys())
        cash = self.starting_equity
        positions: dict[str, float] = {s: 0.0 for s in symbols}

        equity_rows = []
        position_rows = []
        trades: list[Trade] = []

        start_i = LOOKBACK_BARS
        for i in range(start_i, len(self.index)):
            ts = self.index[i]
            prices = {s: float(self.price_data[s]["close"].iloc[i]) for s in symbols}

            # Mark-to-market equity BEFORE this bar's decisions, using this
            # bar's close (i.e. decisions this bar execute "at the close",
            # a common and simple backtest convention).
            equity = cash + sum(positions[s] * prices[s] for s in symbols)

            self.risk.update_equity(equity, ts)
            self.risk.state.open_exposure_pct = gross_exposure_pct(positions, prices, equity)

            # --- Pass 1: what every strategy wants, before the portfolio cap.
            # Gathered for all symbols before anything executes, because the
            # cap constrains the book as a whole: how much of it each symbol
            # may hold is not knowable until you know what all of them asked
            # for. Deciding symbol-by-symbol is what makes a cap
            # first-come-first-served.
            intents: dict[str, float] = {}
            reasons: dict[str, str] = {}
            for s in symbols:
                window = self.price_data[s].iloc[max(0, i - LOOKBACK_BARS + 1): i + 1]
                signal = self.strategies[s].generate_signal(s, window)
                reasons[s] = signal.reason

                atr_val = self._atr_series[s].iloc[i]
                want = 0.0
                if signal.target_position != 0 and atr_val and atr_val > 0 and not np.isnan(atr_val):
                    raw_qty = self.risk.position_size(equity, prices[s], atr_val, signal.confidence)
                    want = raw_qty if signal.target_position > 0 else -raw_qty
                intents[s] = want

            # One allocation factor for the whole book -- uniform, so the result
            # cannot depend on symbol ordering. Scaled from the raw intents:
            # what the strategies want is the demand being rationed, and the
            # band below is a per-symbol cost filter that has no business
            # changing the size of the book being rationed.
            scale = exposure_scale_to_cap(
                intents, prices, equity, self.risk.limits.max_total_exposure_pct
            )
            allocated = {s: qty * scale for s, qty in intents.items()}

            # The no-trade band, applied to the sizes actually being asked for
            # and therefore AFTER the allocation, not before. Allocation moves
            # a position by an amount nothing has vetted: a book pinned at its
            # cap is rescaled on every bar as equity and ATR move, and those
            # rescaled sizes are precisely the small continuous adjustments the
            # band exists to refuse. Banding the pre-allocation intent instead
            # lets them through unvetted -- measured at 3,617 fills against
            # 626, and the band is there to avoid paying fees, so that is a
            # real regression and not just a number moving.
            banded = {
                s: allocated[s]
                if self.risk.is_significant_change(positions[s], allocated[s])
                else positions[s]
                for s in symbols
            }

            # A suppressed leg keeps what it is already holding, which for a
            # leg above its allocated share is MORE than its allocation -- and
            # a leg can sit up to `min_rebalance_fraction` above its
            # allocation and still be suppressed. So a book that is itself
            # compliant can be pushed over the cap by a different leg opening
            # at full allocated size. That is not hypothetical: three legs
            # each 18.4% above their allocation (inside the 20% band, so
            # suppressed, 20% each) plus one new leg opening at its allocated
            # 11% put the book at 71% against a 60% cap on the bundled
            # backtest -- one bar, because the next bar's correction was no
            # longer suppressed, but a cap that is only enforced on the
            # following bar is not a cap.
            #
            # When the banded book breaches, the legs holding above their
            # allocation lose the argument. This stays order-independent and
            # needs no iteration: every unforced leg ends at or below its
            # allocation, so the book cannot exceed the allocated book, which
            # the scale above has already brought inside the cap.
            if is_over_exposure_cap(banded, prices, equity, self.risk.limits.max_total_exposure_pct):
                banded = {
                    s: allocated[s] if abs(positions[s]) > abs(allocated[s]) else qty
                    for s, qty in banded.items()
                }

            # The exposure these allocations add up to, passed to the entry
            # gate rather than a per-symbol projection -- see
            # strategy/risk.py::check_entry_allowed.
            allocated_pct = gross_exposure_pct(banded, prices, equity)

            # --- Pass 2: gate and execute.
            for s in symbols:
                price = prices[s]
                current_qty = positions[s]
                desired_qty = banded[s]

                is_adding_risk = abs(desired_qty) > abs(current_qty) or (
                    current_qty != 0 and desired_qty != 0 and np.sign(current_qty) != np.sign(desired_qty)
                )

                if is_adding_risk:
                    allowed, reason = self.risk.check_entry_allowed(equity, allocated_pct)
                    if not allowed:
                        desired_qty = 0.0 if self.risk.state.trading_halted else current_qty

                delta = desired_qty - current_qty
                if abs(delta) * price < 1e-8:
                    continue  # not worth executing a dust-sized trade

                side = "buy" if delta > 0 else "sell"
                slip_mult = (1 + self.slippage_bps / 10_000) if side == "buy" else (1 - self.slippage_bps / 10_000)
                exec_price = price * slip_mult
                notional = abs(delta) * exec_price
                fee = notional * (self.fee_bps / 10_000)

                cash -= delta * exec_price  # buying spends cash (delta>0), selling adds cash (delta<0)
                cash -= fee
                positions[s] = desired_qty

                trades.append(
                    Trade(
                        timestamp=ts, symbol=s, side=side, qty=abs(delta),
                        price=exec_price, fee=fee, reason=reasons[s],
                    )
                )

            equity_after = cash + sum(positions[s] * prices[s] for s in symbols)
            drawdown_pct = (self.risk.state.peak_equity - equity_after) / self.risk.state.peak_equity * 100
            # Re-read the book after this bar's fills, so the reported
            # exposure is what the portfolio is actually carrying rather
            # than what it carried before the trades.
            self.risk.state.open_exposure_pct = gross_exposure_pct(positions, prices, equity_after)
            equity_rows.append(
                {
                    "timestamp": ts,
                    "equity": equity_after,
                    "cash": cash,
                    "drawdown_pct": max(0.0, drawdown_pct),
                    "exposure_pct": self.risk.state.open_exposure_pct,
                    "halted": self.risk.state.trading_halted,
                }
            )
            position_rows.append({"timestamp": ts, **positions})

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        positions_history = pd.DataFrame(position_rows).set_index("timestamp")

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            positions_history=positions_history,
            final_positions=positions,
        )
