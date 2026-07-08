"""
Runs the historical backtest using walk-forward OUT-OF-FOLD predictions
(never the final model scored on its own training data -- see
backtest/oof_strategy.py for why that distinction matters) and writes
results/backtest_results.json for the dashboard to read.

Must run after train_model.py (needs results/oof_predictions/*.parquet).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

import json
import pandas as pd

from data import storage
from backtest.oof_strategy import OOFReplayStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_performance
from strategy.risk import RiskManager, RiskLimits

SYMBOLS = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
TIMEFRAME = "5m"
STARTING_EQUITY = 100_000.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0

RESULTS_DIR = Path(__file__).parent.parent / "results"
PROVIDER_NAME = "synthetic"


def main():
    price_data = {}
    strategies = {}

    for symbol in SYMBOLS:
        df = storage.load_cached(PROVIDER_NAME, symbol, TIMEFRAME)
        if df is None:
            raise RuntimeError(f"No cached data for {symbol}; run generate_sample_data.py first")
        price_data[symbol] = df

        safe = symbol.replace("/", "-")
        oof_path = RESULTS_DIR / "oof_predictions" / f"{safe}.parquet"
        if not oof_path.exists():
            raise RuntimeError(f"No OOF predictions for {symbol}; run train_model.py first")
        oof = pd.read_parquet(oof_path)["proba"]
        strategies[symbol] = OOFReplayStrategy(symbol, oof)

    risk = RiskManager(RiskLimits(), starting_equity=STARTING_EQUITY)
    engine = BacktestEngine(
        price_data=price_data,
        strategies=strategies,
        risk_manager=risk,
        starting_equity=STARTING_EQUITY,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    result = engine.run()
    perf = compute_performance(result, STARTING_EQUITY)

    print("=== Backtest performance (walk-forward, out-of-fold, fees+slippage applied) ===")
    for k, v in perf.as_dict().items():
        print(f"  {k}: {v}")

    equity_curve = result.equity_curve.reset_index()
    equity_curve["timestamp"] = equity_curve["timestamp"].astype(str)

    trades_out = [
        {
            "timestamp": str(t.timestamp), "symbol": t.symbol, "side": t.side,
            "qty": round(t.qty, 6), "price": round(t.price, 4), "fee": round(t.fee, 4),
            "reason": t.reason,
        }
        for t in result.trades
    ]

    per_symbol = {}
    for symbol in SYMBOLS:
        sym_trades = [t for t in result.trades if t.symbol == symbol]
        per_symbol[symbol] = {"num_trades": len(sym_trades)}

    training_summary_path = RESULTS_DIR / "training_summary.json"
    training_summary = {}
    if training_summary_path.exists():
        with open(training_summary_path) as f:
            training_summary = json.load(f)

    payload = {
        "starting_equity": STARTING_EQUITY,
        "fee_bps": FEE_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "performance": perf.as_dict(),
        "equity_curve": equity_curve.to_dict(orient="records"),
        "trades": trades_out[-500:],
        "per_symbol": per_symbol,
        "symbols": SYMBOLS,
        "training_summary": training_summary,
        "note": "SYNTHETIC DATA -- generated in-sandbox for pipeline testing, not a real market. See README.",
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "backtest_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nWrote results/backtest_results.json ({len(trades_out)} trades total, "
          f"{len(result.trades)} fills)")


if __name__ == "__main__":
    main()
