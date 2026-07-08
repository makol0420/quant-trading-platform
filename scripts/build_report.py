"""
Consolidates results/backtest_results.json + results/training_summary.json
into a single, downsampled payload for the standalone HTML report, and
writes results/report.html by inlining that payload into the template.

Downsamples the equity curve to hourly resolution -- the underlying
backtest runs on 5-minute bars (tens of thousands of points), which is
more resolution than a chart needs and would bloat a self-contained HTML
file for no visual benefit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"


def _fold_to_fractions(folds: list[dict]) -> list[dict]:
    """Convert each fold's train/test timestamps into 0..1 fractions of
    the full time range spanned by all folds, for proportional rendering
    of the walk-forward timeline diagram."""
    if not folds:
        return []
    t0 = pd.Timestamp(folds[0]["train_start"])
    t1 = pd.Timestamp(folds[-1]["test_end"])
    total = (t1 - t0).total_seconds()
    if total <= 0:
        return []

    out = []
    for f in folds:
        train_start = pd.Timestamp(f["train_start"])
        train_end = pd.Timestamp(f["train_end"])
        test_start = pd.Timestamp(f["test_start"])
        test_end = pd.Timestamp(f["test_end"])
        out.append(
            {
                "fold_number": f["fold_number"],
                "auc": f["auc"],
                "train_frac_start": (train_start - t0).total_seconds() / total,
                "train_frac_width": (train_end - train_start).total_seconds() / total,
                "test_frac_start": (test_start - t0).total_seconds() / total,
                "test_frac_width": (test_end - test_start).total_seconds() / total,
                "train_size": f["train_size"],
                "test_size": f["test_size"],
            }
        )
    return out


def main():
    with open(RESULTS_DIR / "backtest_results.json") as f:
        backtest = json.load(f)

    training_summary_path = RESULTS_DIR / "training_summary.json"
    training_summary = {}
    if training_summary_path.exists():
        with open(training_summary_path) as f:
            training_summary = json.load(f)

    eq = pd.DataFrame(backtest["equity_curve"])
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq = eq.set_index("timestamp")
    hourly = eq.resample("1h").last().dropna(how="all").reset_index()
    hourly["timestamp"] = hourly["timestamp"].astype(str)
    equity_curve_downsampled = hourly[["timestamp", "equity", "drawdown_pct", "halted"]].to_dict(orient="records")

    symbols = backtest["symbols"]
    per_symbol = {}
    for symbol in symbols:
        auc = training_summary.get(symbol, {}).get("mean_auc")
        n_trades = backtest["per_symbol"].get(symbol, {}).get("num_trades", 0)
        per_symbol[symbol] = {"auc": auc, "num_trades": n_trades}

    folds_by_symbol = {
        symbol: _fold_to_fractions(training_summary.get(symbol, {}).get("folds", []))
        for symbol in symbols
    }

    final_row = eq.iloc[-1]
    status = "HALTED" if bool(final_row["halted"]) else "ACTIVE"
    halt_ts = None
    halted_mask = eq["halted"] == True  # noqa: E712
    if halted_mask.any():
        halt_ts = str(eq.index[halted_mask][0])

    report_data = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "note": backtest["note"],
        "starting_equity": backtest["starting_equity"],
        "fee_bps": backtest["fee_bps"],
        "slippage_bps": backtest["slippage_bps"],
        "performance": backtest["performance"],
        "status": status,
        "halt_timestamp": halt_ts,
        "equity_curve": equity_curve_downsampled,
        "symbols": symbols,
        "per_symbol": per_symbol,
        "folds_by_symbol": folds_by_symbol,
        "trades": backtest["trades"][-60:],
    }

    template_path = DASHBOARD_DIR / "report_template.html"
    with open(template_path) as f:
        template = f.read()

    injected = template.replace(
        "/*__REPORT_DATA__*/{}",
        json.dumps(report_data),
    )

    out_path = RESULTS_DIR / "report.html"
    with open(out_path, "w") as f:
        f.write(injected)

    print(f"Wrote {out_path} ({len(injected):,} bytes, {len(equity_curve_downsampled)} equity points)")


if __name__ == "__main__":
    main()
