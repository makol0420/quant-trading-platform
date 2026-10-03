"""
Runs the historical backtest using walk-forward OUT-OF-FOLD predictions
(never the final model scored on its own training data -- see
backtest/oof_strategy.py for why that distinction matters) and writes
results/backtest_results.json for the dashboard to read.

Must run after train_model.py (needs results/oof_predictions/*.parquet).

Universe, timeframe, and costs come from config/config.yaml.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

import argparse
import json
import pandas as pd

from config import loader
from backtest.oof_strategy import OOFReplayStrategy
from backtest.engine import BacktestEngine
from backtest.metrics import compute_performance
from strategy.risk import RiskManager, RiskLimits

STARTING_EQUITY = 100_000.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="crypto",
                    help="config.yaml asset class to backtest (default: crypto)")
    args = ap.parse_args()

    cfg = loader.load_config()
    timeframe = loader.timeframe(cfg)
    specs = loader.resolve_symbols(args.scope, cfg)

    price_data = {}
    strategies = {}
    skipped = []

    for spec in specs:
        try:
            df = loader.load_symbol_data(spec, timeframe)
        except FileNotFoundError as exc:
            skipped.append((spec.symbol, str(exc)))
            continue

        safe = spec.symbol.replace("/", "-")
        oof_path = RESULTS_DIR / "oof_predictions" / f"{safe}.parquet"
        if not oof_path.exists():
            skipped.append((spec.symbol, f"no OOF predictions at {oof_path}; run train_model.py"))
            continue

        price_data[spec.symbol] = df
        oof = pd.read_parquet(oof_path)["proba"]
        strategies[spec.symbol] = OOFReplayStrategy(spec.symbol, oof)

    for symbol, why in skipped:
        print(f"{symbol:10s}: SKIPPED -- {why}", file=sys.stderr)

    if not price_data:
        print("\nNothing to backtest.", file=sys.stderr)
        return 1

    # Real venue data does not guarantee a shared grid across symbols
    # (late listings, halts, short pages). BacktestEngine requires an
    # identical index, so align explicitly and say what was dropped.
    raw_counts = {s: len(d) for s, d in price_data.items()}
    price_data = loader.align_to_common_index(price_data)
    n_common = len(next(iter(price_data.values())))
    dropped = {s: raw_counts[s] - n_common for s in price_data}
    if any(dropped.values()):
        detail = ", ".join(f"{s} -{n}" for s, n in dropped.items() if n)
        print(f"Aligned to {n_common} common bars (dropped: {detail})")

    symbols = list(price_data.keys())

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
    for symbol in symbols:
        sym_trades = [t for t in result.trades if t.symbol == symbol]
        per_symbol[symbol] = {"num_trades": len(sym_trades)}

    training_summary_path = RESULTS_DIR / "training_summary.json"
    training_summary = {}
    if training_summary_path.exists():
        with open(training_summary_path) as f:
            training_summary = json.load(f)

    # Data provenance, surfaced in the payload so the dashboard and report
    # can label the numbers honestly rather than implying every run is real
    # market data.
    is_real = all(
        training_summary.get(s, {}).get("is_real_data", False) for s in symbols
    )
    providers = sorted({training_summary.get(s, {}).get("provider", "unknown") for s in symbols})

    payload = {
        "starting_equity": STARTING_EQUITY,
        "fee_bps": FEE_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "performance": perf.as_dict(),
        "equity_curve": equity_curve.to_dict(orient="records"),
        "trades": trades_out[-500:],
        "per_symbol": per_symbol,
        "symbols": symbols,
        "training_summary": training_summary,
        "data_source": {
            "is_real_data": is_real,
            "providers": providers,
            "timeframe": timeframe,
            "n_bars": n_common,
            "start": str(next(iter(price_data.values())).index[0]),
            "end": str(next(iter(price_data.values())).index[-1]),
        },
        "note": (
            "REAL exchange data (ccxt/Binance). Fees and slippage applied per side; "
            "predictions are walk-forward out-of-fold."
            if is_real else
            "SYNTHETIC DATA -- generated for pipeline testing, not a real market. See README."
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "backtest_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nWrote results/backtest_results.json ({len(trades_out)} trades shown, "
          f"{len(result.trades)} fills, {n_common} bars, real_data={is_real})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
