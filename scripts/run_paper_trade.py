"""
Demo paper-trading run. Uses SyntheticProvider (since no real feed is
reachable from the environment this platform was built in) to simulate a
short live session and prove the orchestrator loop -- data -> signal ->
risk check -> execution -> state file -- works end to end using the
SAME MLStrategy class that would run against real live data.

For an actual deployment:
  1. Replace `provider = SyntheticProvider(...)` below with
     CryptoCCXTProvider(...) or ForexOandaProvider(...).
  2. Set `poll_seconds` in OrchestratorConfig to match your timeframe
     (e.g. 300 for 5-minute bars) instead of the fast demo polling below.
  3. Replace the `for i in range(N_DEMO_CYCLES)` loop with a call to
     `orchestrator.run_forever()`, and run this under a process
     supervisor (systemd, tmux, supervisord, etc) since it's meant to
     run indefinitely.

Requires train_model.py to have been run first (loads saved models).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import data`, `import models`, etc.

from data.providers.synthetic import SyntheticProvider
from execution.paper import PaperExecutionClient
from strategy.ml_strategy import MLStrategy
from strategy.risk import RiskManager, RiskLimits
from orchestrator.runner import TradingOrchestrator, OrchestratorConfig

SYMBOLS = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
TIMEFRAME = "5m"
STARTING_EQUITY = 100_000.0
N_DEMO_CYCLES = 8  # demo only -- real deployments call run_forever() instead


def main():
    # Different seed than training data, standing in for "new data the
    # model has never seen" -- the honest analogue of live market data
    # arriving after the model was trained.
    provider = SyntheticProvider(seed=123)
    execution = PaperExecutionClient(starting_equity=STARTING_EQUITY, fee_bps=10.0, slippage_bps=5.0)
    strategies = {s: MLStrategy(s) for s in SYMBOLS}
    risk = RiskManager(RiskLimits(), starting_equity=STARTING_EQUITY)

    config = OrchestratorConfig(mode="paper", timeframe=TIMEFRAME, poll_seconds=1, lookback_bars=150)
    orchestrator = TradingOrchestrator(config, provider, execution, strategies, risk)

    print(f"Starting {N_DEMO_CYCLES}-cycle paper trading demo ({', '.join(SYMBOLS)})...\n")
    for i in range(N_DEMO_CYCLES):
        snapshot = orchestrator.run_cycle()
        positions_str = ", ".join(f"{s}={q:.4f}" for s, q in snapshot["positions"].items())
        print(
            f"cycle {i + 1}/{N_DEMO_CYCLES}  "
            f"equity=${snapshot['equity']:,.2f}  "
            f"dd={snapshot['drawdown_pct']:.2f}%  "
            f"halted={snapshot['trading_halted']}  "
            f"positions=[{positions_str}]"
        )

    print(f"\nDemo complete. Full state written to runtime_state/paper_state.json")
    print("Serve the dashboard with: uvicorn api.main:app --reload  (reads this same state file)")


if __name__ == "__main__":
    main()
