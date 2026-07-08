"""
FastAPI backend serving the dashboard.

Run with: uvicorn api.main:app --reload --port 8000

This process only READS state -- it does not run the trading loop
itself. Run `python scripts/run_paper_trade.py` or `run_live_trade.py`
as a separate process to actually trade; this keeps "is my dashboard
running" and "is my bot placing orders" clearly decoupled, so opening
the dashboard can never accidentally start trading.
"""

from __future__ import annotations
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).parent.parent
STATE_DIR = BASE_DIR / "runtime_state"
RESULTS_DIR = BASE_DIR / "results"
DASHBOARD_DIR = BASE_DIR / "dashboard"

app = FastAPI(title="Quant Bot Platform API")

# Local-dev convenience. If you deploy this beyond your own machine,
# restrict allow_origins to the actual dashboard origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/state/{mode}")
def get_state(mode: str):
    if mode not in ("paper", "live"):
        raise HTTPException(400, "mode must be 'paper' or 'live'")
    path = STATE_DIR / f"{mode}_state.json"
    if not path.exists():
        return {"running": False, "mode": mode, "message": f"No {mode} session has run yet."}
    with open(path) as f:
        state = json.load(f)
    state["running"] = True
    return state


@app.get("/api/results/backtest")
def get_backtest_results():
    path = RESULTS_DIR / "backtest_results.json"
    if not path.exists():
        raise HTTPException(404, "No backtest results found. Run scripts/run_backtest.py first.")
    with open(path) as f:
        return json.load(f)


@app.get("/")
def dashboard_index():
    index_path = DASHBOARD_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Dashboard not built yet.")
    return FileResponse(index_path)


# Serve any other dashboard assets (css/js) if the dashboard grows beyond one file.
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")
