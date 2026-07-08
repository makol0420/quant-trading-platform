"""
Backtest-only strategy adapter: replays walk-forward OUT-OF-FOLD
predictions instead of calling the live model.

This distinction matters and is easy to get wrong: models/train.py's
`final_model` is retrained on the entire historical dataset so it's ready
to use going forward -- which means using it to "backtest" over that same
historical period would be scoring the model on data it already saw
during training. That's not a backtest, it's an in-sample fit, and it's a
major reason naive trading-bot demos show implausibly good historical
performance.

The fix: TrainingResult.oof_predictions holds, for every historical bar,
the prediction made by a model that was trained ONLY on data strictly
before that bar (with an embargo gap -- see models/dataset.py). Backtest
against those instead, and the backtest numbers represent what the
strategy would genuinely have done, bar by bar, with no hindsight.

Paper trading and live trading don't use this class at all -- they use
strategy/ml_strategy.py (MLStrategy), because for bars that haven't
happened yet, the final model genuinely hasn't seen them, so there's no
leakage concern.
"""

from __future__ import annotations
import pandas as pd

from strategy.base import Strategy, Signal
from strategy.signal_logic import proba_to_signal


class OOFReplayStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        oof_predictions: pd.Series,
        long_threshold: float = 0.58,
        short_threshold: float = 0.42,
    ):
        self.symbol = symbol
        self.oof_predictions = oof_predictions
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signal(self, symbol: str, history: pd.DataFrame) -> Signal:
        current_ts = history.index[-1]
        proba = self.oof_predictions.get(current_ts, None)
        if proba is not None and pd.isna(proba):
            proba = None
        return proba_to_signal(symbol, proba, self.long_threshold, self.short_threshold)
