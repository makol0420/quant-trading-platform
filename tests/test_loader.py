"""
Tests for config/loader.py -- specifically symbol/provider resolution and
the index alignment that real exchange data requires.

BacktestEngine refuses to run unless every symbol shares a byte-identical
timestamp index (see backtest/engine.py). The bundled synthetic provider
guarantees that by stamping all symbols on one generated grid, so the
requirement was invisible during development. Real venue data has no such
guarantee: a symbol listed later, halted, or returned short during
pagination produces a different index, and the backtest aborts. Alignment
is the step that makes real data usable, so it gets tested.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from config import loader


def _frame(start: str, n: int, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": base, "high": base, "low": base, "close": base, "volume": 1.0},
        index=idx,
    )


def test_align_keeps_only_shared_timestamps():
    a = _frame("2026-01-01", 10)
    b = _frame("2026-01-01", 12)  # two extra bars at the end

    aligned = loader.align_to_common_index({"A": a, "B": b})

    assert len(aligned["A"]) == 10
    assert len(aligned["B"]) == 10
    assert aligned["A"].index.equals(aligned["B"].index)


def test_align_handles_offset_series():
    a = _frame("2026-01-01", 10)
    b = _frame("2026-01-01 00:10", 10)  # starts two bars later

    aligned = loader.align_to_common_index({"A": a, "B": b})

    assert len(aligned["A"]) == 8
    assert aligned["A"].index.equals(aligned["B"].index)


def test_align_does_not_mutate_inputs():
    a = _frame("2026-01-01", 10)
    b = _frame("2026-01-01", 12)

    loader.align_to_common_index({"A": a, "B": b})

    assert len(a) == 10 and len(b) == 12


def test_align_raises_when_nothing_is_shared():
    a = _frame("2026-01-01", 10)
    b = _frame("2026-06-01", 10)  # months apart

    with pytest.raises(ValueError, match="no common timestamps"):
        loader.align_to_common_index({"A": a, "B": b})


def test_resolve_symbols_reads_config():
    cfg = loader.load_config()

    crypto = loader.resolve_symbols("crypto", cfg)
    assert crypto, "config.yaml should define crypto symbols"
    assert all(s.asset_class == "crypto" for s in crypto)

    combined = loader.resolve_symbols("all", cfg)
    assert len(combined) >= len(crypto)


def test_resolve_symbols_rejects_unknown_scope():
    with pytest.raises(ValueError):
        loader.resolve_symbols("equities", loader.load_config())


def test_cache_name_tracks_provider_type():
    spec = loader.SymbolSpec("BTC/USDT", "crypto", "ccxt")
    assert spec.cache_name == "crypto_ccxt"
    assert spec.is_real

    synthetic = loader.SymbolSpec("EUR/USD", "forex", "synthetic")
    assert synthetic.cache_name == "synthetic"
    assert not synthetic.is_real
