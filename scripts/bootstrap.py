"""
Builds the artifacts the dashboard reads, on a machine that doesn't have them.

Every artifact this platform serves -- cached OHLCV, trained models, backtest
results, the HTML report -- is listed in .gitignore. That is the right call
for a repo (they're large, and they're build output), but it has a
consequence that was easy to miss: a freshly deployed instance has none of
them, so `/api/results/backtest` 404s, `/api/state/paper` has no state to
report, and the dashboard has nothing real to draw.

The previous answer to that was a hardcoded fallback in the API -- a
portfolio of $10,000, a win rate of 71.3% -- which meant every deploy showed
plausible-looking numbers with nothing behind them. This script is the honest
alternative: build the real artifacts at startup, and let the API report
"still building" until they exist.

    python scripts/bootstrap.py                # skip steps whose output exists
    python scripts/bootstrap.py --force        # rebuild everything
    python scripts/bootstrap.py --bars 3000    # smaller window for a slow container
    python scripts/bootstrap.py --check        # report status, build nothing

Designed to be safe to run on every container start: completed steps are
skipped, so a restart costs a few stat() calls rather than a full refetch.

Progress is written to runtime_state/bootstrap_status.json, which /api/health
reports, so the dashboard can say "building artifacts (train)" instead of
showing an empty page or a fabricated number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

from config import loader
from data import storage

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models" / "registry"
STATE_DIR = BASE_DIR / "runtime_state"
STATUS_PATH = STATE_DIR / "bootstrap_status.json"


@dataclass
class Step:
    name: str
    argv: list[str]
    #: Relative paths that must all exist for this step to count as done.
    outputs: list[Path] = field(default_factory=list)

    def is_done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)

    @property
    def display(self) -> str:
        return f"{sys.executable} {' '.join(self.argv)}"


def _write_status(**fields) -> None:
    """Publish bootstrap progress for /api/health. Never fatal if unwritable."""
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        # Progress reporting is a nicety; it must never be the thing that
        # kills a build. The mkdir is inside the try for the same reason --
        # an unwritable state directory raises here, not at the open().
        print(f"[bootstrap] could not write status file: {exc}", file=sys.stderr)


def _rel(path: Path) -> str:
    """Project-relative path for log messages, absolute if it lies outside."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def read_status() -> dict | None:
    """Used by api/main.py. Returns None when bootstrap has never run."""
    if not STATUS_PATH.exists():
        return None
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_steps(cfg: dict, bars: int, scope: str) -> list[Step]:
    tf = loader.timeframe(cfg)
    specs = loader.resolve_symbols(scope, cfg)

    return [
        Step(
            "fetch",
            ["scripts/fetch_market_data.py", "--bars", str(bars), "--scope", scope],
            outputs=[storage.cache_path(s.cache_name, s.symbol, tf) for s in specs],
        ),
        Step(
            "train",
            ["scripts/train_model.py", "--scope", scope],
            outputs=[RESULTS_DIR / "training_summary.json"],
        ),
        Step(
            "backtest",
            ["scripts/run_backtest.py", "--scope", scope],
            outputs=[RESULTS_DIR / "backtest_results.json"],
        ),
        Step(
            "report",
            ["scripts/build_report.py"],
            outputs=[RESULTS_DIR / "report.html"],
        ),
    ]


def run_step(step: Step) -> bool:
    """
    Run one step, streaming its output into this process's stdout so the
    platform's build log is one continuous record rather than four
    interleaved buffers.
    """
    print(f"\n{'=' * 70}\n[bootstrap] {step.name}: {' '.join(step.argv)}\n{'=' * 70}", flush=True)
    started = time.monotonic()

    proc = subprocess.run([sys.executable, *step.argv], cwd=str(BASE_DIR))
    elapsed = time.monotonic() - started

    if proc.returncode != 0:
        print(f"[bootstrap] {step.name} FAILED after {elapsed:.1f}s "
              f"(exit {proc.returncode})", file=sys.stderr, flush=True)
        return False

    # A step that exits 0 without producing its artifact would otherwise be
    # recorded as success and skipped forever on every subsequent restart.
    if not step.is_done():
        missing = ", ".join(_rel(p) for p in step.outputs if not p.exists())
        print(f"[bootstrap] {step.name} exited 0 but produced nothing at: {missing}",
              file=sys.stderr, flush=True)
        return False

    print(f"[bootstrap] {step.name} done in {elapsed:.1f}s", flush=True)
    return True


def main() -> int:
    cfg = loader.load_config()
    default_bars = cfg.get("history", {}).get("bootstrap_bars", 6000)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=int, default=default_bars,
                    help=f"bars of history (default: history.bootstrap_bars = {default_bars})")
    ap.add_argument("--scope", default="crypto", help="config.yaml asset class (default: crypto)")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--check", action="store_true",
                    help="print which steps are done and exit without building")
    args = ap.parse_args()

    steps = build_steps(cfg, args.bars, args.scope)

    if args.check:
        for step in steps:
            print(f"{'ok     ' if step.is_done() else 'missing'}  {step.name}")
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"[bootstrap] scope={args.scope} bars={args.bars} force={args.force}")

    for i, step in enumerate(steps):
        if not args.force and step.is_done():
            print(f"[bootstrap] {step.name}: already built, skipping")
            continue

        _write_status(state="running", step=step.name, bars=args.bars, scope=args.scope,
                      started_at=started_at, completed=[s.name for s in steps[:i]],
                      total_steps=len(steps))

        if not run_step(step):
            _write_status(state="failed", step=step.name, bars=args.bars, scope=args.scope,
                          started_at=started_at, error=f"{step.name} failed; see build log",
                          completed=[s.name for s in steps[:i]], total_steps=len(steps))
            return 1

    _write_status(state="complete", step=None, bars=args.bars, scope=args.scope,
                  started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                  completed=[s.name for s in steps], total_steps=len(steps))

    print(f"\n[bootstrap] all artifacts ready.")
    print(f"[bootstrap]   models  -> {MODELS_DIR}")
    print(f"[bootstrap]   results -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
