"""
Config-driven resolution of symbols -> provider/cache.

config/config.yaml has always been described as "the intended single source
of truth for symbols" (see its header comment), with a note that the bundled
scripts read hardcoded constants instead while you're getting oriented. This
module is the wiring that comment describes: the scripts call in here rather
than each hardcoding their own SYMBOLS/PROVIDER_NAME pair, so pointing the
platform at a different provider or universe is a config edit, not a code
edit across three files.

Two things this deliberately does NOT do:

  - It does not merge providers into one universe. Crypto and forex come
    from different providers with different bar grids; a backtest mixing a
    real feed with a synthetic one produces a blended equity curve that
    describes no real market. Symbols are grouped by provider and each
    group is meant to be run on its own.
  - It does not silently fall back to synthetic when real data is
    requested but missing. A missing cache raises, because "quietly
    substituted random numbers for market data" is the single worst
    failure mode this platform could have.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml

from data import storage

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Provider type (as written in config.yaml) -> the cache key that
# data.storage uses, i.e. the provider's `name` attribute.
PROVIDER_CACHE_NAMES = {
    "synthetic": "synthetic",
    "ccxt": "crypto_ccxt",
    "oanda": "forex_oanda",
}

# config.yaml asset-class section -> the scope string the pipeline scripts
# use to select a universe.
_SECTION_ORDER = ["crypto", "forex"]


@dataclass(frozen=True)
class SymbolSpec:
    """One symbol, and where its bars come from."""

    symbol: str
    asset_class: str
    provider_type: str

    @property
    def cache_name(self) -> str:
        """Provider `name` used as the key in data/_cache/."""
        return PROVIDER_CACHE_NAMES[self.provider_type]

    @property
    def is_real(self) -> bool:
        """True when these bars come from an actual venue, not a generator."""
        return self.provider_type != "synthetic"


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_symbols(scope: str = "crypto", config: dict | None = None) -> list[SymbolSpec]:
    """
    Return the SymbolSpecs for one asset class.

    scope: 'crypto' | 'forex' | 'all'
    """
    cfg = config or load_config()
    sections = _SECTION_ORDER if scope == "all" else [scope]
    if scope not in _SECTION_ORDER and scope != "all":
        raise ValueError(f"scope must be one of {_SECTION_ORDER + ['all']}, got {scope!r}")

    provider_cfg = cfg.get("data_provider", {})
    specs: list[SymbolSpec] = []
    for section in sections:
        provider_type = provider_cfg.get(section, {}).get("type", "synthetic")
        if provider_type not in PROVIDER_CACHE_NAMES:
            raise ValueError(
                f"Unknown data_provider.{section}.type {provider_type!r}; "
                f"expected one of {list(PROVIDER_CACHE_NAMES)}"
            )
        for symbol in cfg.get("symbols", {}).get(section, []) or []:
            specs.append(SymbolSpec(symbol, section, provider_type))
    return specs


def timeframe(config: dict | None = None) -> str:
    return (config or load_config()).get("timeframe", "5m")


def load_symbol_data(
    spec: SymbolSpec,
    timeframe: str,
) -> pd.DataFrame:
    """
    Load one symbol's cached bars, raising a clear error if the cache is
    missing rather than returning None for the caller to mishandle.
    """
    df = storage.load_cached(spec.cache_name, spec.symbol, timeframe)
    if df is None:
        raise FileNotFoundError(
            f"No cached data for {spec.symbol} (provider={spec.cache_name}, "
            f"timeframe={timeframe}). Run the matching fetch script first:\n"
            f"  crypto -> python scripts/fetch_market_data.py\n"
            f"  forex  -> python scripts/generate_sample_data.py (synthetic) "
            f"or a configured OANDA fetch (real)"
        )
    return df


def align_to_common_index(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Restrict every frame to the timestamps they all share.

    BacktestEngine requires a byte-identical index across symbols
    (backtest/engine.py raises otherwise). Real venue data does not
    guarantee that: a symbol that was listed later, halted, or simply
    returned a short page during pagination will have a different index
    than its peers. Intersecting is the honest fix -- it drops only bars
    that genuinely aren't shared, and never invents a price for a bar a
    symbol didn't trade.

    Returns a new dict; inputs are not mutated. Raises if the intersection
    is empty, since an empty intersection means the alignment premise is
    wrong and silently producing a zero-bar backtest would be worse.
    """
    if not frames:
        return {}

    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)

    if common is None or len(common) == 0:
        spans = ", ".join(f"{s}: {len(d.index)} bars" for s, d in frames.items())
        raise ValueError(
            f"Symbols share no common timestamps ({spans}). They are probably "
            f"on different bar grids or from different providers -- check that "
            f"config.yaml's data_provider section points every symbol in a run "
            f"at the same source."
        )

    return {symbol: df.loc[common] for symbol, df in frames.items()}
