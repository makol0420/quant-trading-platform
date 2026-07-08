"""
Tests for models/dataset.py -- the labeling and walk-forward split logic
that the whole "honest backtest" claim rests on. If these are wrong,
everything downstream (training, backtesting, the report) is wrong too.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from models.dataset import make_labels, walk_forward_folds, build_dataset


def _make_price_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.random(n) * 100,
        },
        index=idx,
    )
    return df


class TestMakeLabels:
    def test_known_up_move_labels_positive(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
        # Price doubles over the horizon -- unambiguously above any cost threshold.
        df = pd.DataFrame(
            {"open": [1, 1, 1, 1, 1], "high": [1, 1, 1, 1, 1], "low": [1, 1, 1, 1, 1],
             "close": [1.0, 1.0, 1.0, 1.0, 2.0], "volume": [1, 1, 1, 1, 1]},
            index=idx,
        )
        labels = make_labels(df, horizon=4, cost_threshold_bps=30)
        assert labels.iloc[0] == 1.0

    def test_flat_price_labels_negative(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {"open": [1]*5, "high": [1]*5, "low": [1]*5, "close": [1.0]*5, "volume": [1]*5},
            index=idx,
        )
        labels = make_labels(df, horizon=4, cost_threshold_bps=30)
        assert labels.iloc[0] == 0.0

    def test_tail_is_unlabelable(self):
        """The last `horizon` bars can't have a forward return computed --
        they must be NaN, not silently dropped or zero-filled, or the
        dataset would quietly train on labels that don't mean what they
        claim to."""
        df = _make_price_series(n=50)
        labels = make_labels(df, horizon=10, cost_threshold_bps=30)
        assert labels.iloc[-10:].isna().all()
        assert labels.iloc[:-10].notna().all()

    def test_build_dataset_drops_unlabelable_and_warmup_rows(self):
        df = _make_price_series(n=500)
        X, y = build_dataset(df, horizon=12, cost_threshold_bps=30)
        assert len(X) == len(y)
        assert X.notna().all().all()
        assert y.notna().all()
        assert len(X) < len(df)  # warmup + tail must have been trimmed


class TestWalkForwardFolds:
    def test_folds_are_sequential_and_non_overlapping(self):
        df = _make_price_series(n=3000)
        folds = walk_forward_folds(df.index, n_folds=5, embargo_bars=12, min_train_bars=500)
        assert len(folds) == 5
        for prev, cur in zip(folds, folds[1:]):
            # each fold's test window must start after the previous one ends
            assert cur.test_idx[0] > prev.test_idx[-1]
            # anchored/expanding: later folds train on strictly more data
            assert len(cur.train_idx) > len(prev.train_idx)

    def test_embargo_gap_enforced(self):
        """No bar within `embargo_bars` after a fold's training window may
        appear in that fold's own test window -- that gap is what
        prevents a label computed near the boundary from leaking
        forward-looking information across it."""
        df = _make_price_series(n=3000)
        embargo = 12
        folds = walk_forward_folds(df.index, n_folds=5, embargo_bars=embargo, min_train_bars=500)
        for fold in folds:
            train_end_pos = df.index.get_loc(fold.train_idx[-1])
            test_start_pos = df.index.get_loc(fold.test_idx[0])
            assert test_start_pos - train_end_pos >= embargo

    def test_raises_when_insufficient_data(self):
        df = _make_price_series(n=50)
        with pytest.raises(ValueError):
            walk_forward_folds(df.index, n_folds=5, embargo_bars=12, min_train_bars=500)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
