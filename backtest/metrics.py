"""
Standard performance metrics computed from a backtest's equity curve and trade log.

Nothing exotic here on purpose -- Sharpe, Sortino, max drawdown, win rate,
and profit factor are the metrics any experienced reviewer will ask for
first, and reporting anything fancier instead of these (rather than in
addition to them) is itself a small red flag.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .engine import BacktestResult, Trade


@dataclass
class PerformanceReport:
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    n_days_observed: int
    bars_per_year_assumed: float

    def as_dict(self) -> dict:
        return asdict(self)


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    median_gap = pd.Series(index).diff().median()
    seconds = median_gap.total_seconds()
    if seconds <= 0:
        return 252.0
    return (365 * 24 * 3600) / seconds


def _trade_pnls(trades: list[Trade]) -> list[float]:
    """
    Reconstruct realized P&L per round trip using FIFO matching of buys
    against sells, per symbol. Used for win rate / profit factor, which
    are defined per closed trade, not per bar.

    open_lots[symbol] is a FIFO queue of [qty_signed, price] where
    qty_signed > 0 is an open long lot and < 0 is an open short lot. Each
    incoming trade first closes out opposite-direction lots (realizing
    P&L per matched chunk), then opens a new lot with whatever quantity
    is left over -- which naturally handles a trade that flips the
    position from long to short (or vice versa) in one fill.
    """
    pnls: list[float] = []
    open_lots: dict[str, list[list[float]]] = {}

    for t in trades:
        lots = open_lots.setdefault(t.symbol, [])
        remaining = t.qty if t.side == "buy" else -t.qty
        fee_per_unit = t.fee / t.qty if t.qty > 0 else 0.0

        while remaining != 0 and lots and np.sign(lots[0][0]) != np.sign(remaining):
            lot_qty, lot_price = lots[0]
            matched = min(abs(remaining), abs(lot_qty))
            close_direction = 1 if lot_qty > 0 else -1  # closing a long vs. covering a short

            pnl = matched * close_direction * (t.price - lot_price) - fee_per_unit * matched
            pnls.append(pnl)

            lot_qty -= close_direction * matched
            remaining -= (-close_direction) * matched

            if abs(lot_qty) < 1e-9:
                lots.pop(0)
            else:
                lots[0][0] = lot_qty

        if abs(remaining) > 1e-9:
            lots.append([remaining, t.price])

    return pnls


def compute_performance(result: BacktestResult, starting_equity: float) -> PerformanceReport:
    eq = result.equity_curve["equity"]
    if len(eq) < 2:
        raise ValueError("Equity curve too short to compute metrics")

    total_return_pct = (eq.iloc[-1] / starting_equity - 1) * 100
    bars_per_year = _bars_per_year(eq.index)

    # Sharpe/Sortino/CAGR are computed on DAILY-resampled returns, not raw
    # intrabar (e.g. 5-minute) returns annualized by sqrt(bars_per_year).
    # The latter is a common mistake for intraday strategies: it assumes
    # returns are i.i.d. bar to bar, and any small persistent per-bar bias
    # -- noise, not necessarily real edge or real failure -- gets amplified
    # by a very large sqrt(N) (105,120 for 5-minute bars vs. 252 for daily),
    # producing Sharpe magnitudes far outside what the strategy's actual
    # day-to-day behavior would suggest. Resampling to daily first is the
    # field-standard fix.
    daily_eq = eq.resample("1D").last().dropna()
    daily_returns = daily_eq.pct_change().dropna()
    n_days = len(daily_eq)
    n_years = n_days / 252.0

    cagr_pct = ((eq.iloc[-1] / starting_equity) ** (1 / max(n_years, 1e-6)) - 1) * 100 if eq.iloc[-1] > 0 else -100.0

    if len(daily_returns) >= 2 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = daily_returns[daily_returns < 0]
    if len(downside) >= 2 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    max_drawdown_pct = result.equity_curve["drawdown_pct"].max()

    pnls = _trade_pnls(result.trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate_pct = (len(wins) / len(pnls) * 100) if pnls else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_trade_pnl = float(np.mean(pnls)) if pnls else 0.0

    return PerformanceReport(
        total_return_pct=round(total_return_pct, 3),
        cagr_pct=round(cagr_pct, 3),
        sharpe=round(float(sharpe), 3),
        sortino=round(float(sortino), 3),
        max_drawdown_pct=round(float(max_drawdown_pct), 3),
        win_rate_pct=round(win_rate_pct, 2),
        profit_factor=round(profit_factor, 3) if np.isfinite(profit_factor) else 999.0,
        num_trades=len(pnls),
        avg_trade_pnl=round(avg_trade_pnl, 4),
        n_days_observed=n_days,
        bars_per_year_assumed=round(bars_per_year, 1),
    )
