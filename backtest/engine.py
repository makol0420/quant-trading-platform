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
from strategy.risk import RiskManager
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

            # Mark-to-market equity BEFORE this bar's decisions, using this
            # bar's close (i.e. decisions this bar execute "at the close",
            # a common and simple backtest convention).
            equity = cash
            for s in symbols:
                price_now = self.price_data[s]["close"].iloc[i]
                equity += positions[s] * price_now

            self.risk.update_equity(equity, ts)

            total_exposure_pct = sum(
                abs(positions[s] * self.price_data[s]["close"].iloc[i]) for s in symbols
            ) / equity * 100 if equity > 0 else 0.0
            self.risk.state.open_exposure_pct = total_exposure_pct

            for s in symbols:
                df = self.price_data[s]
                price = df["close"].iloc[i]
                window = df.iloc[max(0, i - LOOKBACK_BARS + 1): i + 1]

                signal = self.strategies[s].generate_signal(s, window)
                atr_val = self._atr_series[s].iloc[i]

                current_qty = positions[s]
                desired_qty = 0.0

                if signal.target_position != 0 and atr_val and atr_val > 0 and not np.isnan(atr_val):
                    raw_qty = self.risk.position_size(equity, price, atr_val, signal.confidence)
                    desired_qty = raw_qty if signal.target_position > 0 else -raw_qty

                if not self.risk.is_significant_change(current_qty, desired_qty):
                    desired_qty = current_qty  # change too small to be worth the transaction cost

                is_adding_risk = abs(desired_qty) > abs(current_qty) or (
                    current_qty != 0 and desired_qty != 0 and np.sign(current_qty) != np.sign(desired_qty)
                )

                if is_adding_risk:
                    proposed_notional = abs(desired_qty) * price
                    proposed_exposure_pct = proposed_notional / equity * 100 if equity > 0 else 0
                    allowed, reason = self.risk.check_entry_allowed(equity, proposed_exposure_pct)
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
                        price=exec_price, fee=fee, reason=signal.reason,
                    )
                )

            equity_after = cash + sum(positions[s] * self.price_data[s]["close"].iloc[i] for s in symbols)
            drawdown_pct = (self.risk.state.peak_equity - equity_after) / self.risk.state.peak_equity * 100
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
