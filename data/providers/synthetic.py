"""
Synthetic OHLCV generator.

THIS IS NOT REAL MARKET DATA. It exists so the rest of the platform
(feature engineering, model training, backtesting, risk management,
dashboard) can be built, run, and sanity-checked in an environment with no
route to a real exchange or broker.

It generates prices via a regime-switching random walk: the market
alternates between a few "moods" (trending-up, trending-down, choppy,
quiet) via a Markov chain, each with its own drift/volatility, so the
series has realistic-ish volatility clustering instead of being pure
white noise. Volume is loosely correlated with realized volatility, the
way it tends to be in real markets.

Swap this provider for CryptoCCXTProvider or ForexOandaProvider to run
against real data — nothing else in the codebase needs to change.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional
import time
import zlib
import numpy as np
import pandas as pd

from .base import DataProvider, Bar

# Regime definitions: (annualized_drift, annualized_vol, mean_bars_in_regime)
_REGIMES = {
    "trend_up": dict(drift=0.35, vol=0.30, persistence=120),
    "trend_down": dict(drift=-0.30, vol=0.35, persistence=90),
    "choppy": dict(drift=0.0, vol=0.55, persistence=60),
    "quiet": dict(drift=0.02, vol=0.12, persistence=180),
}

_TRANSITION = {
    # from -> {to: probability per bar of switching}
    "trend_up": dict(trend_up=0.985, trend_down=0.003, choppy=0.008, quiet=0.004),
    "trend_down": dict(trend_up=0.004, trend_down=0.985, choppy=0.008, quiet=0.003),
    "choppy": dict(trend_up=0.006, trend_down=0.006, choppy=0.980, quiet=0.008),
    "quiet": dict(trend_up=0.003, trend_down=0.003, choppy=0.010, quiet=0.984),
}

# Rough per-symbol starting price + bar-level vol scaling (crypto trades
# much noisier per-bar than G10 FX). Purely for surface realism.
_SYMBOL_PARAMS = {
    "BTC/USDT": dict(start_price=62000.0, vol_scale=1.6, asset_class="crypto"),
    "ETH/USDT": dict(start_price=3400.0, vol_scale=1.9, asset_class="crypto"),
    "EUR/USD": dict(start_price=1.0850, vol_scale=0.35, asset_class="forex"),
    "GBP/USD": dict(start_price=1.2650, vol_scale=0.40, asset_class="forex"),
}

_TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._demo_clock: dict[str, datetime] = {}  # per-symbol; advances one bar per call when `end` is omitted

    def _bars_per_year(self, timeframe: str) -> float:
        minutes = _TIMEFRAME_MINUTES[timeframe]
        return (365 * 24 * 60) / minutes

    def historical(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if symbol not in _SYMBOL_PARAMS:
            raise ValueError(
                f"Unknown synthetic symbol '{symbol}'. "
                f"Available: {list(_SYMBOL_PARAMS)}"
            )
        if timeframe not in _TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown timeframe '{timeframe}'")

        n = limit or 10_000
        params = _SYMBOL_PARAMS[symbol]
        bars_per_year = self._bars_per_year(timeframe)

        # Seed per-symbol so BTC and ETH don't move in lockstep, but stay
        # reproducible across runs.
        # NOTE: uses zlib.crc32, not Python's builtin hash() -- the latter
        # is randomized per-process for strings (PYTHONHASHSEED, a
        # security feature), which would silently make `seed` NOT actually
        # reproducible across separate runs despite the parameter's name
        # and docstring promising exactly that.
        symbol_offset = zlib.crc32(symbol.encode()) % 10_000
        rng = np.random.default_rng(self.seed + symbol_offset)

        regime = "quiet"
        regimes_seq = []
        for _ in range(n):
            trans = _TRANSITION[regime]
            regimes_seq.append(regime)
            roll = rng.random()
            cum = 0.0
            nxt = regime
            for to_regime, p in trans.items():
                cum += p
                if roll <= cum:
                    nxt = to_regime
                    break
            regime = nxt

        drifts = np.array([_REGIMES[r]["drift"] for r in regimes_seq])
        vols = np.array([_REGIMES[r]["vol"] for r in regimes_seq]) * params["vol_scale"]

        dt = 1.0 / bars_per_year
        mu_per_bar = (drifts - 0.5 * vols**2) * dt
        sigma_per_bar = vols * np.sqrt(dt)
        shocks = rng.standard_normal(n)
        log_returns = mu_per_bar + sigma_per_bar * shocks

        # Fat-tailed jump component (rare, small probability) so returns
        # aren't perfectly Gaussian -- real markets have more extreme moves
        # than a pure GBM implies.
        jump_mask = rng.random(n) < 0.002
        jump_size = rng.standard_normal(n) * sigma_per_bar * 6
        log_returns = log_returns + jump_mask * jump_size

        log_prices = np.log(params["start_price"]) + np.cumsum(log_returns)
        close = np.exp(log_prices)
        open_ = np.empty(n)
        open_[0] = params["start_price"]
        open_[1:] = close[:-1]

        # Intrabar noise for high/low, scaled to the bar's own volatility.
        intrabar_noise = np.abs(rng.standard_normal(n)) * sigma_per_bar * close * 0.6
        high = np.maximum(open_, close) + intrabar_noise
        low = np.minimum(open_, close) - intrabar_noise
        low = np.maximum(low, close * 0.001)  # guard against non-positive prices

        base_volume = 1000.0 if params["asset_class"] == "crypto" else 5_000_000.0
        volume = base_volume * (1.0 + 4.0 * (vols / vols.max())) * (
            0.5 + rng.random(n)
        )

        if end is None:
            # Demo convenience: advance a virtual per-symbol clock one bar
            # per call instead of using wall-clock now() every time.
            # Without this, calling historical() twice for the same symbol
            # within the same minute (as a paper-trading loop naturally
            # does across consecutive cycles) would return identical bars,
            # since the RNG draw is deterministic given (seed, symbol) and
            # `end` is what varies the window. Keyed per-symbol so
            # generating BTC/USDT then ETH/USDT in a loop doesn't bleed
            # one symbol's clock advancement into another's -- each
            # symbol's first call anchors to "now"; only later calls for
            # THAT SAME symbol advance from there. Pass an explicit `end=`
            # if you want fully deterministic, non-advancing output (e.g.
            # in unit tests).
            if symbol not in self._demo_clock:
                self._demo_clock[symbol] = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            else:
                self._demo_clock[symbol] += timedelta(minutes=_TIMEFRAME_MINUTES[timeframe])
            end = self._demo_clock[symbol]
        freq = f"{_TIMEFRAME_MINUTES[timeframe]}min"
        timestamps = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "regime": regimes_seq,  # kept for diagnostics; drop before ML
            },
            index=timestamps,
        )
        df.index.name = "timestamp"
        return df

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """
        Simulate a live feed: yield one new synthetic bar at a time,
        sleeping to approximate real-time pacing (sped up for demo use).
        """
        minutes = _TIMEFRAME_MINUTES[timeframe]
        hist = self.historical(symbol, timeframe, limit=500)
        last_close = hist["close"].iloc[-1]
        rng = np.random.default_rng(self.seed + int(time.time()))
        params = _SYMBOL_PARAMS[symbol]
        t = hist.index[-1]
        while True:
            vol = _REGIMES["choppy"]["vol"] * params["vol_scale"]
            dt = 1.0 / self._bars_per_year(timeframe)
            shock = rng.standard_normal()
            ret = -0.5 * vol**2 * dt + vol * np.sqrt(dt) * shock
            new_close = last_close * np.exp(ret)
            t = t + timedelta(minutes=minutes)
            bar = Bar(
                symbol=symbol,
                timestamp=t,
                open=last_close,
                high=max(last_close, new_close) * 1.0003,
                low=min(last_close, new_close) * 0.9997,
                close=new_close,
                volume=abs(rng.standard_normal()) * 1000,
            )
            last_close = new_close
            yield bar

    def supported_symbols(self) -> list[str]:
        return list(_SYMBOL_PARAMS.keys())
