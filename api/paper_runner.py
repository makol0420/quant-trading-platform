"""
Lifecycle management for the paper-trading session the dashboard's
Start/Stop buttons drive.

This module deliberately contains NO trading logic. It constructs the
already-existing pieces -- a real data provider, PaperExecutionClient,
MLStrategy per symbol, RiskManager, and TradingOrchestrator -- and then just
starts, supervises, and stops a thread that calls the orchestrator's own
run_cycle() in a loop. Signals, sizing, risk gates, and fills all come from
the same code the backtest and a live deployment use; that shared path is
the platform's central design decision (data/providers/base.py) and nothing
here forks it.

PAPER ONLY, by construction. There is no code path in this module that can
construct a live execution client, and no endpoint may set
LIVE_TRADING_CONFIRMED. execution/crypto_live.py refuses to place an order
without that env var, so even a mistake here cannot reach a real venue. The
dashboard is public and unauthenticated by request; a reachable "start real
trading" button would let anyone who finds the URL trade the account.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import traceback

from config import loader
from data.providers.crypto_ccxt import CryptoCCXTProvider
from data.providers.synthetic import SyntheticProvider
from execution.paper import PaperExecutionClient
from strategy.ml_strategy import MLStrategy
from strategy.risk import RiskManager, RiskLimits
from orchestrator.runner import TradingOrchestrator, OrchestratorConfig

STATE_DIR = Path(__file__).parent.parent / "runtime_state"
EQUITY_HISTORY_PATH = STATE_DIR / "paper_equity_history.json"
MAX_EQUITY_POINTS = 2000

STARTING_EQUITY = 100_000.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0


class PaperSession:
    """One start/stop cycle of the paper-trading loop. Not reusable after stop()."""

    def __init__(self, scope: str = "crypto", poll_seconds: int | None = None):
        cfg = loader.load_config()
        self.cfg = cfg
        self.scope = scope
        self.timeframe = loader.timeframe(cfg)
        self.specs = loader.resolve_symbols(scope, cfg)
        if not self.specs:
            raise RuntimeError(f"No symbols configured for scope '{scope}' in config.yaml")

        self.poll_seconds = poll_seconds or cfg.get("orchestrator", {}).get("poll_seconds", 60)
        self.lookback_bars = cfg.get("orchestrator", {}).get("lookback_bars", 150)

        self.provider = self._build_provider()
        self.execution = PaperExecutionClient(
            starting_equity=STARTING_EQUITY, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS
        )
        self.risk = RiskManager(RiskLimits(), starting_equity=STARTING_EQUITY)

        # MLStrategy loads models/registry/<symbol>_model.joblib in its
        # constructor, so a missing registry surfaces here rather than as a
        # confusing failure mid-cycle.
        self.strategies = {spec.symbol: MLStrategy(spec.symbol) for spec in self.specs}

        self.orchestrator = TradingOrchestrator(
            OrchestratorConfig(
                mode="paper", timeframe=self.timeframe,
                poll_seconds=self.poll_seconds, lookback_bars=self.lookback_bars,
            ),
            self.provider, self.execution, self.strategies, self.risk,
        )

        self.started_at: datetime | None = None
        self.cycles = 0
        self.last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._equity_history: list[dict] = []

    def _build_provider(self):
        """Real exchange data when config says so; synthetic only if configured."""
        crypto_cfg = self.cfg.get("data_provider", {}).get("crypto", {})
        ptype = crypto_cfg.get("type", "ccxt")
        if ptype == "ccxt":
            return CryptoCCXTProvider(exchange_id=crypto_cfg.get("exchange_id", "binance"))
        return SyntheticProvider(seed=123)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.is_alive():
            return
        self.started_at = datetime.now(timezone.utc)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="paper-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._write_equity_history()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self.orchestrator.run_cycle()
                self.cycles += 1
                self.last_error = None
                self._equity_history.append(
                    {
                        "timestamp": snapshot["timestamp"],
                        "equity": snapshot["equity"],
                        "drawdown_pct": snapshot["drawdown_pct"],
                    }
                )
                del self._equity_history[:-MAX_EQUITY_POINTS]
                self._write_equity_history()
            except Exception:
                # A single bad cycle (network blip, exchange hiccup) should
                # not kill the session -- record it and retry next interval,
                # same as TradingOrchestrator.run_forever() does.
                self.last_error = traceback.format_exc(limit=3)
            self._stop.wait(self.poll_seconds)

    def _write_equity_history(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(EQUITY_HISTORY_PATH, "w") as f:
            json.dump(self._equity_history, f)

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self.is_alive(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "cycles": self.cycles,
            "poll_seconds": self.poll_seconds,
            "symbols": [s.symbol for s in self.specs],
            "timeframe": self.timeframe,
            "last_error": self.last_error,
            "data_source": self.provider.name,
        }


# Module-level singleton: the API is the only writer, and there is exactly
# one paper session per process.
_session: PaperSession | None = None
_lock = threading.Lock()


def get_session() -> PaperSession | None:
    return _session


def start_session(scope: str = "crypto", poll_seconds: int | None = None) -> PaperSession:
    global _session
    with _lock:
        if _session is not None and _session.is_alive():
            return _session  # already running; idempotent
        _session = PaperSession(scope=scope, poll_seconds=poll_seconds)
        _session.start()
        return _session


def stop_session() -> bool:
    """Returns True if a running session was stopped, False if none was running."""
    with _lock:
        if _session is None or not _session.is_alive():
            return False
        _session.stop()
        return True


def load_equity_history() -> list[dict]:
    if not EQUITY_HISTORY_PATH.exists():
        return []
    try:
        with open(EQUITY_HISTORY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
