"""
Derived metrics for the dashboard, computed from the orchestrator's own
trade log rather than invented.

The original dashboard hardcoded "Win Rate 71%" and "Daily P/L $125" with no
computation behind them. Those numbers are the single most misleading thing
in the UI -- a fabricated win rate on an unauthenticated public trading
dashboard is exactly the genre of dishonesty this codebase's README is built
to argue against. Everything here returns None when it genuinely cannot be
computed, and the UI renders that as "—" rather than as zero or a guess.
"""

from __future__ import annotations


def realized_stats(trade_log: list[dict]) -> dict:
    """
    Match entries against exits (average-cost basis) to get realized PnL and
    a real win rate.

    A trade only counts as a win/loss once it has been closed -- an open
    position has no realized outcome, and counting it would make a strategy
    look good purely for holding a position that happens to be up.
    """
    positions: dict[str, tuple[float, float]] = {}  # symbol -> (qty, avg_price)
    wins = losses = closed = 0
    realized = 0.0

    for t in trade_log:
        symbol = t.get("symbol")
        qty = float(t.get("qty") or 0.0)
        price = float(t.get("price") or 0.0)
        if not symbol or qty == 0 or price == 0:
            continue
        signed = qty if t.get("side") == "buy" else -qty

        held_qty, avg = positions.get(symbol, (0.0, 0.0))

        opening_or_adding = held_qty == 0 or (held_qty > 0) == (signed > 0)
        if opening_or_adding:
            new_qty = held_qty + signed
            new_avg = (
                (avg * held_qty + price * signed) / new_qty if new_qty != 0 else 0.0
            )
            positions[symbol] = (new_qty, new_avg)
            continue

        closing_qty = min(abs(signed), abs(held_qty))
        pnl = closing_qty * (price - avg) * (1 if held_qty > 0 else -1)
        realized += pnl
        closed += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        new_qty = held_qty + signed
        if new_qty == 0:
            positions[symbol] = (0.0, 0.0)
        elif (new_qty > 0) != (held_qty > 0):
            positions[symbol] = (new_qty, price)  # flipped through zero
        else:
            positions[symbol] = (new_qty, avg)

    return {
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "realized_pnl": round(realized, 2) if closed else None,
        "win_rate": round(wins / closed * 100, 1) if closed else None,
    }


def summarize_snapshot(snapshot: dict) -> dict:
    """
    Reshape an orchestrator snapshot (orchestrator/runner.py::_build_snapshot)
    into the flat shape the dashboard cards read, without inventing anything
    the snapshot doesn't contain.
    """
    positions_raw = snapshot.get("positions") or {}
    latest_prices = snapshot.get("latest_prices") or {}
    trade_log = snapshot.get("trade_log") or []

    open_positions = []
    for symbol, qty in positions_raw.items():
        if not qty:
            continue
        price = latest_prices.get(symbol)
        # Entry basis isn't in the snapshot; surface what is known rather
        # than reconstructing a plausible-looking entry that may be wrong.
        open_positions.append(
            {
                "symbol": symbol,
                "qty": qty,
                "side": "LONG" if qty > 0 else "SHORT",
                "current": price,
                "notional": abs(qty * price) if price else None,
            }
        )

    stats = realized_stats(trade_log)
    equity = snapshot.get("equity")

    return {
        "portfolio": equity,
        "peak_equity": snapshot.get("peak_equity"),
        "daily_pl": stats["realized_pnl"],
        "open_trades": len(open_positions),
        "win_rate": stats["win_rate"],
        "closed_trades": stats["closed_trades"],
        "drawdown_pct": snapshot.get("drawdown_pct"),
        "open_exposure_pct": snapshot.get("open_exposure_pct"),
        "trading_halted": snapshot.get("trading_halted", False),
        "halt_reason": snapshot.get("halt_reason"),
        "positions": open_positions,
        "latest_prices": latest_prices,
        "recent_signals": snapshot.get("recent_signals") or {},
        "trade_log": trade_log,
        "timestamp": snapshot.get("timestamp"),
    }
