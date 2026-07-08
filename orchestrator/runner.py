"""
The paper/live trading loop: fetch latest data -> generate signal -> size
via risk manager -> check risk gates -> execute -> persist state.

Backtesting does NOT go through this class -- see backtest/engine.py and
the module docstring in backtest/oof_strategy.py for why (it replays
walk-forward out-of-fold predictions instead of calling the live model,
to avoid lookahead). This orchestrator is what paper trading and live
trading actually run, and it's intentionally the thinnest possible layer
over Strategy + RiskManager + ExecutionClient so behavior here matches
what the backtest's event loop does, structurally, as closely as two
genuinely different code paths (historical replay vs. real-time polling)
can.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import traceback

from data.providers.base import DataProvider
from execution.base import ExecutionClient
from execution.paper import PaperExecutionClient
from strategy.base import Strategy
from strategy.risk import RiskManager
from features.indicators import atr as compute_atr

STATE_DIR = Path(__file__).parent.parent / "runtime_state"


@dataclass
class OrchestratorConfig:
    mode: str                 # 'paper' or 'live'
    timeframe: str
    lookback_bars: int = 150
    poll_seconds: int = 60


class TradingOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        provider: DataProvider,
        execution_client: ExecutionClient,
        strategies: dict[str, Strategy],
        risk_manager: RiskManager,
    ):
        self.config = config
        self.provider = provider
        self.execution = execution_client
        self.strategies = strategies
        self.risk = risk_manager
        self.recent_signals: dict[str, dict] = {}
        self.trade_log: list[dict] = []
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def run_cycle(self) -> dict:
        symbols = list(self.strategies.keys())
        latest_prices = {}
        histories = {}

        for symbol in symbols:
            history = self.provider.historical(symbol, self.config.timeframe, limit=self.config.lookback_bars)
            histories[symbol] = history
            price = float(history["close"].iloc[-1])
            latest_prices[symbol] = price
            if isinstance(self.execution, PaperExecutionClient):
                self.execution.mark_price(symbol, price)

        equity = self.execution.get_equity()
        now = datetime.now(timezone.utc)
        self.risk.update_equity(equity, now)

        total_exposure_pct = 0.0
        for symbol in symbols:
            qty = self.execution.get_position(symbol)
            total_exposure_pct += abs(qty * latest_prices[symbol])
        total_exposure_pct = (total_exposure_pct / equity * 100) if equity > 0 else 0.0
        self.risk.state.open_exposure_pct = total_exposure_pct

        for symbol in symbols:
            history = histories[symbol]
            signal = self.strategies[symbol].generate_signal(symbol, history)
            self.recent_signals[symbol] = {
                "timestamp": now.isoformat(),
                "target_position": signal.target_position,
                "confidence": signal.confidence,
                "reason": signal.reason,
            }

            price = latest_prices[symbol]
            atr_val = float(compute_atr(history["high"], history["low"], history["close"]).iloc[-1])
            current_qty = self.execution.get_position(symbol)

            desired_qty = 0.0
            if signal.target_position != 0 and atr_val > 0:
                raw_qty = self.risk.position_size(equity, price, atr_val, signal.confidence)
                desired_qty = raw_qty if signal.target_position > 0 else -raw_qty

            if not self.risk.is_significant_change(current_qty, desired_qty):
                desired_qty = current_qty  # change too small to be worth the transaction cost

            is_adding_risk = abs(desired_qty) > abs(current_qty) or (
                current_qty != 0 and desired_qty != 0
                and (current_qty > 0) != (desired_qty > 0)
            )
            if is_adding_risk:
                proposed_notional = abs(desired_qty) * price
                proposed_exposure_pct = (proposed_notional / equity * 100) if equity > 0 else 0
                allowed, reason = self.risk.check_entry_allowed(equity, proposed_exposure_pct)
                if not allowed:
                    desired_qty = 0.0 if self.risk.state.trading_halted else current_qty

            delta = desired_qty - current_qty
            if abs(delta) * price >= 1.0:  # skip dust-sized adjustments
                side = "buy" if delta > 0 else "sell"
                result = self.execution.place_order(symbol, side, abs(delta))
                self.trade_log.append(
                    {
                        "timestamp": now.isoformat(), "symbol": symbol, "side": side,
                        "qty": result.filled_qty, "price": result.fill_price,
                        "status": result.status, "reason": signal.reason,
                    }
                )

        snapshot = self._build_snapshot(equity, latest_prices, now)
        self._persist(snapshot)
        return snapshot

    def _build_snapshot(self, equity: float, latest_prices: dict, now: datetime) -> dict:
        return {
            "mode": self.config.mode,
            "timestamp": now.isoformat(),
            "equity": equity,
            "peak_equity": self.risk.state.peak_equity,
            "drawdown_pct": (self.risk.state.peak_equity - equity) / self.risk.state.peak_equity * 100
            if self.risk.state.peak_equity > 0 else 0.0,
            "trading_halted": self.risk.state.trading_halted,
            "halt_reason": self.risk.state.halt_reason,
            "open_exposure_pct": self.risk.state.open_exposure_pct,
            "positions": {s: self.execution.get_position(s) for s in self.strategies},
            "latest_prices": latest_prices,
            "recent_signals": self.recent_signals,
            "trade_log": self.trade_log[-200:],  # cap so the state file doesn't grow unbounded
        }

    def _persist(self, snapshot: dict) -> None:
        path = STATE_DIR / f"{self.config.mode}_state.json"
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    def run_forever(self) -> None:
        print(f"[orchestrator] starting {self.config.mode} loop, "
              f"polling every {self.config.poll_seconds}s for {list(self.strategies)}")
        while True:
            try:
                snapshot = self.run_cycle()
                print(
                    f"[{snapshot['timestamp']}] equity={snapshot['equity']:.2f} "
                    f"dd={snapshot['drawdown_pct']:.2f}% halted={snapshot['trading_halted']}"
                )
            except Exception:
                print("[orchestrator] error during cycle, will retry next interval:")
                traceback.print_exc()
            time.sleep(self.config.poll_seconds)
