"""
Thin inference wrapper: loads a saved model and scores new bars.

Kept separate from train.py because the live/paper trading loop should
only ever need this file -- it shouldn't need sklearn training internals,
walk-forward logic, or anything else that isn't relevant once a model is
already trained and saved to the registry.
"""

from __future__ import annotations
import pandas as pd

from features.indicators import build_feature_matrix
from models.train import load_model


class Predictor:
    def __init__(self, symbol: str):
        bundle = load_model(symbol)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.symbol = symbol

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        df: OHLCV history up to and including "now" (needs enough lookback
        for the longest rolling window feature, i.e. >= ~50 bars). Returns
        P(up-move exceeding cost threshold) for every row where features
        are fully formed; earlier rows are NaN.
        """
        X = build_feature_matrix(df)[self.feature_names]
        valid = X.notna().all(axis=1)
        proba = pd.Series(index=df.index, dtype=float)
        if valid.any():
            proba.loc[valid] = self.model.predict_proba(X.loc[valid])[:, 1]
        return proba

    def latest_signal(self, df: pd.DataFrame) -> float | None:
        """Convenience: just the most recent bar's probability, or None if not enough history."""
        proba = self.predict_proba(df)
        latest = proba.iloc[-1]
        return None if pd.isna(latest) else float(latest)
