"""
Turns raw OHLCV + features into an ML-ready, leakage-safe dataset.

Two design choices here are doing the real work of keeping the backtest
honest, and are worth understanding before you touch this file:

1. COST-AWARE LABELING
   The label isn't "did price go up" -- it's "did price move up by more
   than round-trip trading costs". A model that's 55% accurate at calling
   direction is worthless if the average winning move is smaller than fees
   + slippage. Baking the cost threshold into the label itself means the
   model is only ever rewarded for predicting moves worth acting on, and a
   flat 50/50 model naturally predicts "no trade" instead of "trade
   randomly by keeping quiet if unsure only some of the time."

2. WALK-FORWARD SPLITTING WITH AN EMBARGO GAP
   Financial time series can't use a random train/test split (or k-fold
   CV with shuffling) -- that leaks future information backward into
   training, because adjacent bars are correlated and a random split puts
   bar t+1 in train while bar t is in test. This module instead does
   *anchored* walk-forward validation: train on everything up to a point,
   skip an embargo window equal to the label horizon (so no test-set label
   was computed using data the model trained on), then test on the next
   chunk. Repeat, expanding the training window each time. This is slower
   and gives worse-looking numbers than a naive split -- that's the point,
   they're numbers you can trust.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from features.indicators import build_feature_matrix


@dataclass
class Fold:
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    fold_number: int


def make_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    cost_threshold_bps: float = 30.0,
) -> pd.Series:
    """
    Binary label: 1 if the forward return over `horizon` bars exceeds the
    ROUND-TRIP cost threshold, 0 otherwise (includes "moved against you"
    AND "moved but not enough to matter"). `cost_threshold_bps` should
    reflect your real entry-plus-exit fees and typical slippage combined,
    not just one side -- e.g. a crypto exchange at ~10bps taker fee plus a
    few bps of slippage, PER SIDE, is ~30bps round trip; a wider-spread
    forex pair might need more. Getting this wrong in the cheap direction
    (using a one-way cost instead of round-trip) trains the model to treat
    moves as "worth trading" that don't actually clear what it costs to
    get back out of the position.
    """
    forward_return = df["close"].shift(-horizon) / df["close"] - 1
    threshold = cost_threshold_bps / 10_000
    label = (forward_return > threshold).astype(float)
    label[forward_return.isna()] = np.nan  # can't label the last `horizon` bars
    return label


def build_dataset(df: pd.DataFrame, horizon: int = 12, cost_threshold_bps: float = 30.0):
    """
    Returns (X, y) aligned and with NaN rows (warm-up period + un-labelable
    tail) dropped. X is features only -- no OHLCV leakage into the model.
    """
    X = build_feature_matrix(df)
    y = make_labels(df, horizon=horizon, cost_threshold_bps=cost_threshold_bps)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def walk_forward_folds(
    index: pd.DatetimeIndex,
    n_folds: int = 5,
    embargo_bars: int = 12,
    min_train_bars: int = 500,
) -> list[Fold]:
    """
    Anchored (expanding-window) walk-forward split with an embargo gap
    between train and test to prevent label leakage across the boundary.

    embargo_bars should be >= the label horizon: if horizon=12, a label
    computed at bar t uses close[t+12], so bars in [t, t+12) must not
    appear in a test fold that starts right after a training fold ending
    at t, or the model would effectively have seen forward-looking
    information smeared across the boundary.
    """
    n = len(index)
    usable = n - min_train_bars
    if usable <= n_folds:
        raise ValueError(
            f"Not enough bars ({n}) for {n_folds} folds with "
            f"min_train_bars={min_train_bars}. Reduce n_folds or fetch more history."
        )
    fold_size = usable // n_folds

    folds = []
    for i in range(n_folds):
        train_end = min_train_bars + i * fold_size
        test_start = train_end + embargo_bars
        test_end = test_start + fold_size if i < n_folds - 1 else n

        if test_start >= n:
            break

        train_idx = index[:train_end]
        test_idx = index[test_start:test_end]
        if len(test_idx) == 0:
            continue
        folds.append(Fold(train_idx=train_idx, test_idx=test_idx, fold_number=i + 1))

    return folds
