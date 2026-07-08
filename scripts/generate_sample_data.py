"""
Generates aligned synthetic OHLCV history for all four demo symbols and
caches it to data/_cache/, so train_model.py and run_backtest.py have
something to run against without needing network access to a real
exchange or broker.

Real deployments skip this script entirely: point config.yaml's
data_provider at 'ccxt' or 'oanda' instead of 'synthetic', and everything
downstream (features, training, backtest, orchestrator) is unchanged --
they all consume a DataProvider, not a specific implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

from data.providers.synthetic import SyntheticProvider
from data import storage

SYMBOLS = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
TIMEFRAME = "5m"
N_BARS = 20_000  # ~69 days of 5-minute bars -- long enough for daily-resampled Sharpe/Sortino
                 # to mean something (a 2-week sample gives ~14 daily observations, too few for
                 # a stable estimate), while still running quickly end to end in this environment.


def main():
    provider = SyntheticProvider(seed=42)
    for symbol in SYMBOLS:
        df = provider.historical(symbol, TIMEFRAME, limit=N_BARS)
        storage.save_cache(provider.name, symbol, TIMEFRAME, df)
        print(f"{symbol:10s}: {len(df)} bars  [{df.index[0]} -> {df.index[-1]}]  "
              f"close range {df['close'].min():.2f}-{df['close'].max():.2f}")

    print(f"\nCached to {storage.CACHE_DIR}")
    print("NOTE: this is synthetic data for pipeline testing, not a real market.")


if __name__ == "__main__":
    main()
