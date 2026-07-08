from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent.parent
STATE_DIR = BASE_DIR / "runtime_state"
RESULTS_DIR = BASE_DIR / "results"
DASHBOARD_DIR = BASE_DIR / "dashboard"

app = FastAPI(title="Quant Bot Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/state/{mode}")
def get_state(mode: str):
    if mode not in ("paper", "live"):
        raise HTTPException(400, "mode must be 'paper' or 'live'")

    path = STATE_DIR / f"{mode}_state.json"

    if path.exists():
        with open(path) as f:
            state = json.load(f)

        state["running"] = True
        return state

    return {
        "running": True,
        "mode": mode,
        "portfolio": 10000.00,
        "cash": 8500.00,
        "daily_pl": 125.50,
        "open_trades": 2,
        "win_rate": 71.3,
        "positions": [
            {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "qty": 0.02,
                "entry": 108000,
                "current": 109150,
                "pnl": 23.0
            }
        ]
    }


@app.get("/api/results/backtest")
def get_backtest_results():
    path = RESULTS_DIR / "backtest_results.json"

    if not path.exists():
        raise HTTPException(404, "No backtest results found.")

    with open(path) as f:
        return json.load(f)


@app.get("/")
def dashboard_index():
    index_path = DASHBOARD_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(404, "Dashboard not found.")

    return FileResponse(index_path)


if DASHBOARD_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(DASHBOARD_DIR)),
        name="dashboard",
    )
