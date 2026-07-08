"""
Local on-disk cache for OHLCV data, keyed by provider/symbol/timeframe.

Re-fetching months of history every time you tweak a feature is slow and,
for rate-limited exchange APIs, rude. This caches to parquet and only
re-fetches the gap since the last cached bar.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache"


def _cache_path(provider_name: str, symbol: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    return CACHE_DIR / f"{provider_name}_{safe_symbol}_{timeframe}.parquet"


def load_cached(provider_name: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(provider_name, symbol, timeframe)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_cache(provider_name: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    path = _cache_path(provider_name, symbol, timeframe)
    df.to_parquet(path)


def get_or_fetch(provider, symbol: str, timeframe: str, limit: int = 5000, force_refresh: bool = False) -> pd.DataFrame:
    """
    Return cached history if present, else fetch fresh via the provider
    and cache the result. Does not attempt incremental gap-filling in this
    minimal version -- for long-running deployments, extend this to fetch
    only bars newer than the cache's last timestamp.
    """
    if not force_refresh:
        cached = load_cached(provider.name, symbol, timeframe)
        if cached is not None and len(cached) >= limit:
            return cached.tail(limit)

    df = provider.historical(symbol, timeframe, limit=limit)
    save_cache(provider.name, symbol, timeframe, df)
    return df
