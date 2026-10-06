"""
FastAPI backend serving the dashboard.

Serves the dashboard/ static app, the pages/ sub-pages, the shared
components/ fragments, and the generated results/ artifacts (including the
full interactive backtest report at /report).

On reporting honestly: the previous version of this file returned a
hardcoded portfolio ("$10,000", "Daily P/L $125", "Win Rate 71.3%") whenever
no runtime state existed, and always claimed "running": true. Since
runtime_state/ is gitignored, that fallback is what every deployed instance
actually served -- a public trading dashboard displaying invented numbers.
Endpoints here return null plus an explanatory status instead, and the UI
renders those as "—".

No authentication, by request. That is fine for the read-only surface, and
the one write surface (start/stop the paper loop) is deliberately confined to
paper trading -- see api/paper_runner.py for why a live-trading button on an
unauthenticated public dashboard would be unsafe.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import analytics, paper_runner

BASE_DIR = Path(__file__).parent.parent
PAGES_DIR = BASE_DIR / "pages"
STATE_DIR = BASE_DIR / "runtime_state"
RESULTS_DIR = BASE_DIR / "results"
DASHBOARD_DIR = BASE_DIR / "dashboard"
COMPONENTS_DIR = BASE_DIR / "components"

app = FastAPI(title="Quant Bot Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> dict | list | None:
    """Read a JSON artifact, treating a missing or corrupt file as absent."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@app.get("/api/health")
def health():
    """What this instance actually has on disk, so the UI can say so plainly."""
    backtest = RESULTS_DIR / "backtest_results.json"
    summary = RESULTS_DIR / "training_summary.json"
    models = sorted(p.name for p in (BASE_DIR / "models" / "registry").glob("*.joblib"))
    session = paper_runner.get_session()
    ready = backtest.exists() and bool(models)

    # Artifacts are gitignored, so a fresh deploy has none and builds them at
    # startup via scripts/bootstrap.py. Reporting that state lets the UI say
    # "building" rather than showing an empty page -- or, as it used to,
    # inventing numbers to fill the gap.
    bootstrap = _read_json(STATE_DIR / "bootstrap_status.json")

    return {
        "status": "ok",
        "artifacts": {
            "backtest_results": backtest.exists(),
            "training_summary": summary.exists(),
            "report_html": (RESULTS_DIR / "report.html").exists(),
            "trained_models": models,
        },
        "ready": ready,
        "bootstrap": bootstrap or {"state": "unknown"},
        "paper_session": session.status() if session else {"running": False},
    }


def _downsample_equity(points: list[dict], keep: int = 2000) -> list[dict]:
    """
    Thin the equity curve to at most `keep` points (last point per hour).

    The raw curve is one point per 5-minute bar -- 20,000 rows, ~3.3 MB of
    JSON -- which is far more resolution than a chart can show and a lot of
    payload for a free-tier host to serialize on every dashboard load. The
    full-resolution curve stays on disk in results/backtest_results.json and
    is what the generated report uses; this only affects the HTTP response.
    """
    if len(points) <= keep:
        return points
    by_hour: dict[str, dict] = {}
    for p in points:
        ts = str(p.get("timestamp", ""))
        by_hour[ts[:13]] = p  # ISO strings sort lexicographically; last wins
    thinned = [by_hour[k] for k in sorted(by_hour)]
    return thinned if thinned else points


@app.get("/api/results/backtest")
def get_backtest_results(full: bool = False):
    payload = _read_json(RESULTS_DIR / "backtest_results.json")
    if payload is None:
        raise HTTPException(
            404,
            "No backtest results yet. Run: python scripts/run_backtest.py "
            "(see README Quickstart).",
        )
    if not full and isinstance(payload, dict) and payload.get("equity_curve"):
        payload = {**payload, "equity_curve": _downsample_equity(payload["equity_curve"])}
    return payload


@app.get("/api/training/summary")
def get_training_summary():
    payload = _read_json(RESULTS_DIR / "training_summary.json")
    if payload is None:
        raise HTTPException(404, "No training summary yet. Run: python scripts/train_model.py")
    return payload


@app.get("/api/state/{mode}")
def get_state(mode: str):
    """
    Current trading state. Reports only what actually exists -- null metrics
    and running:false when nothing has run, never placeholder values.
    """
    if mode not in ("paper", "live"):
        raise HTTPException(400, "mode must be 'paper' or 'live'")

    if mode == "live":
        # Live execution is gated behind LIVE_TRADING_CONFIRMED and is not
        # reachable from this API by design.
        return {
            "mode": "live",
            "running": False,
            "enabled": False,
            "message": (
                "Live trading is not enabled on this instance. execution/crypto_live.py "
                "requires LIVE_TRADING_CONFIRMED=yes in the environment, and no endpoint "
                "here can set it."
            ),
            "portfolio": None, "daily_pl": None, "open_trades": None,
            "win_rate": None, "drawdown_pct": None, "positions": [],
        }

    session = paper_runner.get_session()
    session_status = session.status() if session else {"running": False, "cycles": 0}

    snapshot = _read_json(STATE_DIR / f"{mode}_state.json")
    if snapshot is None:
        return {
            "mode": mode,
            "running": False,
            "message": (
                "No paper-trading session has run yet. Start one from the dashboard, "
                "or run: python scripts/run_paper_trade.py"
            ),
            "portfolio": None, "daily_pl": None, "open_trades": None,
            "win_rate": None, "drawdown_pct": None, "positions": [],
            **session_status,
        }

    summary = analytics.summarize_snapshot(snapshot)
    return {"mode": mode, **summary, **session_status}


class PaperStartRequest(BaseModel):
    poll_seconds: int | None = None
    scope: str = "crypto"


@app.post("/api/paper/start")
def start_paper(req: PaperStartRequest | None = None):
    """
    Start the paper-trading loop against live market data. Paper only --
    see api/paper_runner.py.
    """
    req = req or PaperStartRequest()
    try:
        session = paper_runner.start_session(scope=req.scope, poll_seconds=req.poll_seconds)
    except FileNotFoundError as exc:
        raise HTTPException(
            400,
            f"Trained models are required before paper trading can start. {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(500, f"Could not start paper session: {exc}") from exc
    return {"started": True, **session.status()}


@app.post("/api/paper/stop")
def stop_paper():
    stopped = paper_runner.stop_session()
    session = paper_runner.get_session()
    return {
        "stopped": stopped,
        "message": None if stopped else "No paper session was running.",
        **(session.status() if session else {"running": False}),
    }


@app.get("/api/paper/status")
def paper_status():
    session = paper_runner.get_session()
    if session is None:
        return {"running": False, "message": "No paper session has been started."}
    return session.status()


@app.get("/api/paper/equity")
def paper_equity():
    """Equity points recorded during the running paper session."""
    return {"points": paper_runner.load_equity_history()}


# --------------------------------------------------------------------------
# Dashboard assets and pages
# --------------------------------------------------------------------------


@app.get("/")
def dashboard_index():
    index_path = DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Dashboard not found.")
    return FileResponse(index_path)


@app.get("/style.css")
def style():
    return FileResponse(DASHBOARD_DIR / "style.css")


@app.get("/app.js")
def javascript():
    return FileResponse(DASHBOARD_DIR / "app.js")


@app.get("/common.js")
def common_js():
    return FileResponse(DASHBOARD_DIR / "common.js")


@app.get("/pages.js")
def pages_js():
    return FileResponse(DASHBOARD_DIR / "pages.js")


@app.get("/layout.js")
def layout_js():
    return FileResponse(DASHBOARD_DIR / "layout.js")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """
    Every browser asks for this on every page. No icon is shipped, and the
    default 404 puts a red error in the console of all 17 pages -- noise that
    looks exactly like a real broken asset while you're debugging one.
    """
    return Response(status_code=204)


@app.get("/report")
def report():
    """The full interactive backtest report, generated by scripts/build_report.py."""
    path = RESULTS_DIR / "report.html"
    if not path.exists():
        raise HTTPException(404, "No report yet. Run: python scripts/build_report.py")
    return FileResponse(path)


@app.get("/deposit")
def deposit():
    return FileResponse(PAGES_DIR / "deposit.html")


@app.get("/withdraw")
def withdraw():
    return FileResponse(PAGES_DIR / "withdraw.html")


@app.get("/transactions")
def transactions():
    return FileResponse(PAGES_DIR / "transactions.html")


@app.get("/portfolio")
def portfolio():
    return FileResponse(PAGES_DIR / "portfolio.html")


# Static mounts. /components is what dashboard/layout.js fetches the shared
# sidebar and topbar from -- it was never mounted before, so every sub-page
# rendered with an empty nav. /results exposes the generated report and raw
# artifacts; /pages serves the sub-pages.
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")
if PAGES_DIR.exists():
    app.mount("/pages", StaticFiles(directory=str(PAGES_DIR)), name="pages")
if COMPONENTS_DIR.exists():
    app.mount("/components", StaticFiles(directory=str(COMPONENTS_DIR)), name="components")
if RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
