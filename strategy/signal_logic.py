"""
Single source of truth for turning a model probability into a Signal.

Both the live strategy (strategy/ml_strategy.py, backed by the final
model) and the backtest replay strategy (backtest/oof_strategy.py, backed
by walk-forward out-of-fold predictions) call this exact function. If this
logic were copy-pasted into both places instead of shared, they would
eventually drift -- someone tunes a threshold for backtesting and forgets
to update live, or vice versa -- and the backtest would silently stop
representing what the live bot actually does. Don't duplicate this.
"""

from __future__ import annotations
from .base import Signal


def proba_to_signal(
    symbol: str,
    proba: float | None,
    long_threshold: float,
    short_threshold: float,
) -> Signal:
    if proba is None:
        return Signal(symbol=symbol, target_position=0.0, confidence=0.0, reason="insufficient_history")

    if proba >= long_threshold:
        confidence = (proba - long_threshold) / max(1e-9, 1 - long_threshold)
        return Signal(symbol=symbol, target_position=1.0, confidence=min(confidence, 1.0), reason=f"p_up={proba:.3f}")

    if proba <= short_threshold:
        confidence = (short_threshold - proba) / max(1e-9, short_threshold)
        return Signal(symbol=symbol, target_position=-1.0, confidence=min(confidence, 1.0), reason=f"p_up={proba:.3f}")

    return Signal(symbol=symbol, target_position=0.0, confidence=0.0, reason=f"p_up={proba:.3f}_no_edge")
