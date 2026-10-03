"""
Trains one direction-probability model per symbol using walk-forward
validation, and saves:

  - the production model, retrained on all data: models/registry/<symbol>_model.joblib
  - walk-forward fold diagnostics: results/training_summary.json
  - out-of-fold predictions (for honest backtesting): results/oof_predictions/<symbol>.parquet

Run after fetch_market_data.py (real data) or generate_sample_data.py
(synthetic), or after pointing config.yaml at a real provider and letting it
cache actual history.

The universe and timeframe come from config/config.yaml via config/loader.py,
so this trains on whatever that file currently specifies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

import argparse
import json

from config import loader
from models.train import train_symbol, save_model

# Must reflect ROUND-TRIP cost (entry + exit), not one-way. The backtest
# and orchestrator both apply fee_bps=10 + slippage_bps=5 = 15bps of
# friction PER SIDE (see scripts/run_backtest.py, execution/paper.py) --
# so a round trip costs ~30bps total. Labeling moves as "worth trading"
# at a lower threshold than that would train the model to chase moves
# that don't actually clear real costs once you're paying to get out too.
COST_THRESHOLD_BPS = 30
N_FOLDS = 5

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="crypto",
                    help="config.yaml asset class to train on (default: crypto)")
    args = ap.parse_args()

    cfg = loader.load_config()
    timeframe = loader.timeframe(cfg)
    specs = loader.resolve_symbols(args.scope, cfg)

    RESULTS_DIR.mkdir(exist_ok=True)
    oof_dir = RESULTS_DIR / "oof_predictions"
    oof_dir.mkdir(exist_ok=True)

    summary = {}
    trained = 0

    for spec in specs:
        try:
            df = loader.load_symbol_data(spec, timeframe)
        except FileNotFoundError as exc:
            # Skip rather than abort: a partially-populated cache shouldn't
            # block training on the symbols that do have data. The missing
            # ones are reported so this can't pass unnoticed.
            print(f"{spec.symbol:10s}: SKIPPED -- {exc}", file=sys.stderr)
            continue

        result = train_symbol(
            df, spec.symbol, horizon=cfg["model"]["horizon_bars"],
            cost_threshold_bps=COST_THRESHOLD_BPS, n_folds=N_FOLDS,
        )
        save_model(result)
        trained += 1

        safe = spec.symbol.replace("/", "-")
        result.oof_predictions.to_frame("proba").to_parquet(oof_dir / f"{safe}.parquet")

        summary[spec.symbol] = {
            "mean_auc": result.mean_auc,
            "n_folds_used": len(result.fold_results),
            "n_labeled_bars": int(result.oof_predictions.notna().sum()),
            "provider": spec.cache_name,
            "is_real_data": spec.is_real,
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
        print(f"{spec.symbol:10s}: mean walk-forward AUC = {auc_str}  "
              f"across {len(result.fold_results)} folds, "
              f"{int(result.oof_predictions.notna().sum())} OOF predictions")

    if trained == 0:
        print("\nNo symbols trained. Fetch data first (scripts/fetch_market_data.py).",
              file=sys.stderr)
        return 1

    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{trained} model(s) saved to models/registry/, summary written to results/training_summary.json")
    print("Reminder: AUC ~0.5 means the model found no exploitable edge on this data -- "
          "that's a legitimate, informative result, not a bug to fix by tuning until it goes away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
