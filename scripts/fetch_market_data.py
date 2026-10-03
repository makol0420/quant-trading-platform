"""
Fetches REAL OHLCV history from a crypto exchange via ccxt and caches it to
data/_cache/, replacing the synthetic generator for the crypto universe.

Public market data needs no API key -- only order placement does (see
execution/crypto_live.py). Nothing is traded here.

    python scripts/fetch_market_data.py              # bars from config.yaml
    python scripts/fetch_market_data.py --bars 6000  # smaller window
    python scripts/fetch_market_data.py --exchange binanceus

After this, train_model.py / run_backtest.py / build_report.py run
unchanged -- they consume a DataProvider's cached output, not the provider
itself. That indirection is the whole point of data/providers/base.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import config`, etc.

import argparse
from datetime import datetime, timedelta, timezone

from config import loader
from data import storage
from data.providers.crypto_ccxt import CryptoCCXTProvider

_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def main() -> int:
    cfg = loader.load_config()
    default_bars = cfg.get("history", {}).get("bars", 20_000)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=int, default=default_bars,
                    help=f"bars of history per symbol (default {default_bars})")
    ap.add_argument("--exchange", default=None,
                    help="ccxt exchange id (default: data_provider.crypto.exchange_id from config.yaml)")
    ap.add_argument("--scope", default="crypto", help="config.yaml asset class to fetch (default: crypto)")
    args = ap.parse_args()

    tf = loader.timeframe(cfg)
    specs = loader.resolve_symbols(args.scope, cfg)
    if not specs:
        print(f"No symbols configured under symbols.{args.scope}", file=sys.stderr)
        return 1

    if any(not s.is_real for s in specs):
        print(
            f"WARNING: some symbols in scope '{args.scope}' are configured as "
            f"'synthetic' in config.yaml, but this script only fetches real "
            f"exchange data. Set data_provider.{args.scope}.type to 'ccxt'.",
            file=sys.stderr,
        )
        return 1

    exchange_id = args.exchange or cfg["data_provider"][args.scope].get("exchange_id", "binance")
    minutes = _TIMEFRAME_MINUTES.get(tf)
    if minutes is None:
        print(f"Unsupported timeframe {tf!r}", file=sys.stderr)
        return 1

    provider = CryptoCCXTProvider(exchange_id=exchange_id)

    # Explicit start/end, rather than relying on the provider's default
    # window: the whole run must span one common grid, and asking for the
    # same window per symbol is what makes that true.
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes * args.bars)

    print(f"Fetching {args.bars} x {tf} bars per symbol from {exchange_id}")
    print(f"Window: {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC\n")

    frames = {}
    for spec in specs:
        try:
            df = provider.historical(spec.symbol, tf, start=start, end=end, limit=args.bars)
        except Exception as exc:
            print(f"{spec.symbol:10s}: FETCH FAILED -- {exc}", file=sys.stderr)
            return 1

        if len(df) < args.bars:
            # Not fatal (a newly-listed pair genuinely has less history),
            # but the caller should know the window isn't what was asked
            # for, since it affects how much walk-forward validation can
            # actually be done.
            print(f"{spec.symbol:10s}: WARNING only {len(df)} of {args.bars} bars available")

        storage.save_cache(provider.name, spec.symbol, tf, df)
        frames[spec.symbol] = df
        print(f"{spec.symbol:10s}: {len(df):6d} bars  [{df.index[0]:%Y-%m-%d %H:%M} -> "
              f"{df.index[-1]:%Y-%m-%d %H:%M}]  close {df['close'].min():,.2f}-{df['close'].max():,.2f}")

    # Drop bars not shared by every symbol, so the backtest's
    # identical-index requirement holds for real venue data.
    aligned = loader.align_to_common_index(frames)
    dropped = {s: len(frames[s]) - len(aligned[s]) for s in frames}
    n_common = len(next(iter(aligned.values())))
    if any(dropped.values()):
        detail = ", ".join(f"{s} -{n}" for s, n in dropped.items() if n)
        print(f"\nAligned to {n_common} common bars (dropped: {detail})")
    else:
        print(f"\nAll symbols share the same {n_common} bars -- no alignment needed")

    if n_common < 1000:
        print(
            f"\nWARNING: only {n_common} common bars. Walk-forward validation needs "
            f"enough history for {cfg['model']['n_folds']} folds; consider --bars with a "
            f"larger value or an earlier --start.",
            file=sys.stderr,
        )

    print(f"\nCached to {storage.CACHE_DIR}")
    print("This is REAL exchange data -- prices are actual traded levels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
