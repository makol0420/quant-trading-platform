"""
Regression tests for the data providers.

The pagination test here covers a bug that silently corrupted every
real-data fetch: ccxt's fetch_ohlcv, called with since=None, returns the
*most recent* bars rather than the oldest, so paging forward from there
immediately ran off the end of the series. Asking for 20,000 bars returned
1,000 and reported success -- no exception, no warning, just a fifth of the
requested history and a walk-forward validation quietly starved of data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data.providers.crypto_ccxt import CryptoCCXTProvider


class FakeExchange:
    """Minimal ccxt-shaped stub: a fixed series of bars, pageable forward."""

    rateLimit = 0  # no sleeping in tests

    def __init__(self, total_bars: int = 2500, interval_ms: int = 300_000):
        self.total_bars = total_bars
        self.interval_ms = interval_ms
        self.start_ms = 1_600_000_000_000
        self.calls: list[tuple[int | None, int]] = []

    def _end_ms(self) -> int:
        return self.start_ms + self.total_bars * self.interval_ms

    def milliseconds(self) -> int:
        return self._end_ms()

    def parse_timeframe(self, timeframe: str) -> int:
        return self.interval_ms // 1000

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=1000):
        self.calls.append((since, limit))

        # Real ccxt behaviour: with no `since`, return the MOST RECENT bars.
        if since is None:
            since = self._end_ms() - limit * self.interval_ms
        # The venue has no bars before the symbol's first one.
        since = max(since, self.start_ms)

        end = self._end_ms()
        rows = []
        ts = since
        while ts < end and len(rows) < limit:
            rows.append([ts, 100.0, 101.0, 99.0, 100.5, 10.0])
            ts += self.interval_ms
        return rows


def _provider_with_fake(total_bars: int = 2500) -> tuple[CryptoCCXTProvider, FakeExchange]:
    provider = CryptoCCXTProvider.__new__(CryptoCCXTProvider)  # skip real network setup
    fake = FakeExchange(total_bars=total_bars)
    provider.exchange = fake
    provider.exchange_id = "fake"
    return provider, fake


def test_historical_pages_past_one_exchange_page():
    """Requesting more than one page must actually return more than one page."""
    provider, fake = _provider_with_fake(total_bars=2500)

    df = provider.historical("BTC/USDT", "5m", limit=2500)

    assert len(df) == 2500, (
        f"expected 2500 bars, got {len(df)} -- pagination stopped after the "
        f"first page (calls: {fake.calls})"
    )
    assert len(fake.calls) > 1, "should have paged more than once"


def test_historical_result_is_sorted_and_unique():
    provider, _ = _provider_with_fake(total_bars=2500)

    df = provider.historical("BTC/USDT", "5m", limit=2500)

    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_historical_stops_cleanly_when_history_runs_out():
    """Asking for more bars than exist returns what exists, without hanging."""
    provider, _ = _provider_with_fake(total_bars=800)

    df = provider.historical("BTC/USDT", "5m", limit=5000)

    assert len(df) == 800
    assert len(df) <= 5000


def test_historical_respects_end_bound():
    provider, fake = _provider_with_fake(total_bars=2500)
    cutoff = pd.Timestamp(fake.start_ms + 1000 * fake.interval_ms, unit="ms", tz="UTC")

    df = provider.historical("BTC/USDT", "5m", limit=2500, end=cutoff.to_pydatetime())

    assert df.index.max() <= cutoff


def test_historical_empty_response_raises():
    provider, fake = _provider_with_fake(total_bars=0)

    try:
        provider.historical("NOPE/USDT", "5m", limit=100)
    except RuntimeError as exc:
        assert "No OHLCV data" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for an empty response")
