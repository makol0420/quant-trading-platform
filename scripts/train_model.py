"""
Trains one direction-probability model per symbol using walk-forward
validation, and saves:

  - the production model, retrained on all data: models/registry/<symbol>_model.joblib
  - walk-forward fold diagnostics: results/training_summary.json
  - out-of-fold predictions (for honest backtesting): results/oof_predictions/<symbol>.parquet

Run after generate_sample_data.py (or after pointing config.yaml at a
real provider and letting it cache actual history).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

import json

from data import storage
from models.train import train_symbol, save_model

SYMBOLS = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
TIMEFRAME = "5m"
HORIZON = 12
# Must reflect ROUND-TRIP cost (entry + exit), not one-way. The backtest
# and orchestrator both apply fee_bps=10 + slippage_bps=5 = 15bps of
# friction PER SIDE (see scripts/run_backtest.py, execution/paper.py) --
# so a round trip costs ~30bps total. Labeling moves as "worth trading"
# at a lower threshold than that would train the model to chase moves
# that don't actually clear real costs once you're paying to get out too.
COST_THRESHOLD_BPS = 30
N_FOLDS = 5

RESULTS_DIR = Path(__file__).parent.parent / "results"
PROVIDER_NAME = "synthetic"  # matches whichever provider generated the cached data


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    oof_dir = RESULTS_DIR / "oof_predictions"
    oof_dir.mkdir(exist_ok=True)

    summary = {}

    for symbol in SYMBOLS:
        df = storage.load_cached(PROVIDER_NAME, symbol, TIMEFRAME)
        if df is None:
            raise RuntimeError(f"No cached data for {symbol}; run generate_sample_data.py first")

        result = train_symbol(
            df, symbol, horizon=HORIZON, cost_threshold_bps=COST_THRESHOLD_BPS, n_folds=N_FOLDS
        )
        save_model(result)

        safe = symbol.replace("/", "-")
        result.oof_predictions.to_frame("proba").to_parquet(oof_dir / f"{safe}.parquet")

        summary[symbol] = {
            "mean_auc": result.mean_auc,
            "n_folds_used": len(result.fold_results),
            "n_labeled_bars": int(result.oof_predictions.notna().sum()),
            "folds": [
                {
                    "fold_number": f.fold_number,
                    "train_size": f.train_size,
                    "test_size": f.test_size,
                    "auc": round(f.auc, 4),
                    "brier": round(f.brier, 4),
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                }
                for f in result.fold_results
            ],
        }

        auc_str = f"{result.mean_auc:.3f}" if result.fold_results else "N/A (no valid folds)"
        print(f"{symbol:10s}: mean walk-forward AUC = {auc_str}  "
              f"across {len(result.fold_results)} folds, "
              f"{int(result.oof_predictions.notna().sum())} OOF predictions")

    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nModels saved to models/registry/, summary written to results/training_summary.json")
    print("Reminder: AUC ~0.5 means the model found no exploitable edge on this data -- "
          "that's a legitimate, informative result, not a bug to fix by tuning until it goes away.")


if __name__ == "__main__":
    main()
