"""
Converts the model's P(up-move) into a long/flat/short signal.

Deliberately conservative: the model was trained to predict "will this
move exceed round-trip costs", so a probability near 0.5 means "no edge",
not "uncertain but lean long". We only take a position when probability
clears a confidence band on either side, and flatten (not "always be in
the market") in between -- a huge share of real edge in retail algo
trading comes from correctly staying out, not from picking a side every bar.
"""

from __future__ import annotations
import pandas as pd

from .base import Strategy, Signal
from .signal_logic import proba_to_signal
from models.predictor import Predictor


class MLStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        long_threshold: float = 0.58,
        short_threshold: float = 0.42,
    ):
        """
        long_threshold / short_threshold: probability thresholds to enter
        long / short. Note these ARE NOT symmetric around a naive 0.5 --
        the label itself already only fires on moves that clear costs, so
        0.5 is not automatically "no edge" the way it would be for an
        uncalibrated coin-flip predictor. Thresholds above/below 0.5 add a
        further margin of safety on top of that, and should be tuned via
        backtesting, not assumed.
        """
        self.symbol = symbol
        self.predictor = Predictor(symbol)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signal(self, symbol: str, history: pd.DataFrame) -> Signal:
        proba = self.predictor.latest_signal(history)
        return proba_to_signal(symbol, proba, self.long_threshold, self.short_threshold)
