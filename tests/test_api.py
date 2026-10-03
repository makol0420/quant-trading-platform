"""
Smoke tests for the FastAPI surface.

These are hermetic: every test that touches results/, runtime_state/, or the
artifact directories points the app at a tmp_path instead of the real ones,
so the suite passes identically whether or not the pipeline has been run and
whether or not a paper session happens to be live.

The behaviour worth pinning down here is the honesty contract. The deployed
version of this API returned a hardcoded portfolio ("$10,000", "Win Rate
71.3%") and always claimed "running": true, because the artifacts it reads are
gitignored and therefore never present on a fresh deploy. Every deployed
instance served invented numbers. These tests assert the opposite: absent
artifacts produce an explanatory message and null metrics, never a plausible-
looking placeholder.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from api import main, paper_runner

# 5-minute bars, so this is ~208 hours of curve -- comfortably past the
# 2000-point threshold below which _downsample_equity returns early.
N_POINTS = 2500
HOURS_SPANNED = 209


def _curve(n: int = N_POINTS) -> list[dict]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "timestamp": (start + timedelta(minutes=5 * i)).isoformat(),
            "equity": float(i),
        }
        for i in range(n)
    ]


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def empty_artifacts(tmp_path, monkeypatch):
    """Point every artifact directory at an empty tree."""
    empty = tmp_path / "results"
    empty.mkdir()
    monkeypatch.setattr(main, "RESULTS_DIR", empty)
    monkeypatch.setattr(main, "STATE_DIR", tmp_path / "runtime_state")
    return empty


# --------------------------------------------------------------------------
# Routes resolve at all (the deployed app 404'd on several of these)
# --------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["artifacts"]) == {
        "backtest_results", "training_summary", "report_html", "trained_models",
    }


def test_dashboard_index_served(client):
    assert client.get("/").status_code == 200


def test_shared_components_are_mounted(client):
    """/components was never mounted, so every sub-page had an empty sidebar."""
    r = client.get("/components/sidebar.html")
    assert r.status_code == 200, "sidebar.html must be reachable for sub-pages to render"


def test_static_assets_served(client):
    for path in ("/style.css", "/app.js", "/common.js", "/pages.js", "/layout.js"):
        assert client.get(path).status_code == 200, f"{path} should be served"


def test_every_sidebar_page_route_exists(client):
    for path in ("/portfolio", "/transactions", "/deposit", "/withdraw"):
        assert client.get(path).status_code == 200, f"{path} should render"


# --------------------------------------------------------------------------
# Honest state reporting
# --------------------------------------------------------------------------


def test_paper_state_without_artifacts_invents_nothing(client, empty_artifacts):
    r = client.get("/api/state/paper")
    assert r.status_code == 200
    body = r.json()

    assert body["running"] is False
    for field in ("portfolio", "daily_pl", "win_rate", "drawdown_pct", "open_trades"):
        assert body[field] is None, f"{field} must be null, not a fabricated value"
    assert body["positions"] == []
    assert "message" in body


def test_live_state_is_disabled_not_faked(client):
    body = client.get("/api/state/live").json()

    assert body["running"] is False
    assert body["enabled"] is False
    assert body["portfolio"] is None


def test_unknown_mode_rejected(client):
    assert client.get("/api/state/equities").status_code == 400


def test_missing_backtest_results_is_404_with_instructions(client, empty_artifacts):
    r = client.get("/api/results/backtest")
    assert r.status_code == 404
    assert "run_backtest" in r.json()["detail"]


def test_missing_training_summary_is_404(client, empty_artifacts):
    assert client.get("/api/training/summary").status_code == 404


def test_corrupt_artifact_reads_as_absent(client, empty_artifacts):
    """A half-written JSON file must not 500 the endpoint."""
    (empty_artifacts / "backtest_results.json").write_text("{not json")

    assert client.get("/api/results/backtest").status_code == 404
    assert main._read_json(empty_artifacts / "backtest_results.json") is None


# --------------------------------------------------------------------------
# Equity-curve thinning
# --------------------------------------------------------------------------


def test_downsamples_equity_to_one_point_per_hour():
    curve = _curve()

    thinned = main._downsample_equity(curve)

    assert len(thinned) == HOURS_SPANNED, "one point per hour, ~209 hours"
    assert len(thinned) < len(curve) / 10, "should be a large reduction, not a no-op"
    # The last bar of the window survives, so the curve still ends where it did.
    assert thinned[-1] == curve[-1]
    assert [p["timestamp"] for p in thinned] == sorted(p["timestamp"] for p in thinned)


def test_downsample_leaves_short_curves_untouched():
    points = [{"timestamp": "2026-01-01T00:00:00+00:00", "equity": 1.0}]

    assert main._downsample_equity(points) is points


def test_backtest_endpoint_thins_by_default_and_serves_full_on_request(
    client, empty_artifacts
):
    curve = _curve()
    (empty_artifacts / "backtest_results.json").write_text(
        json.dumps({"performance": {"total_return_pct": -15.12}, "equity_curve": curve})
    )

    default = client.get("/api/results/backtest").json()
    full = client.get("/api/results/backtest?full=true").json()

    assert len(default["equity_curve"]) == HOURS_SPANNED   # thinned
    assert len(full["equity_curve"]) == N_POINTS           # everything, on request
    # Thinning is a display concern only; it must not disturb the metrics.
    assert default["performance"] == full["performance"] == {"total_return_pct": -15.12}


def test_default_backtest_response_is_small_enough_to_serve(client, empty_artifacts):
    """The raw curve was 3.3 MB -- too much to serialize on every page load."""
    (empty_artifacts / "backtest_results.json").write_text(
        json.dumps({"performance": {}, "equity_curve": _curve()})
    )

    assert len(client.get("/api/results/backtest").content) < 200_000


# --------------------------------------------------------------------------
# Paper session lifecycle
# --------------------------------------------------------------------------


def test_paper_status_before_any_session(client):
    body = client.get("/api/paper/status").json()

    assert body["running"] is False


def test_paper_equity_endpoint_returns_points(client):
    body = client.get("/api/paper/equity").json()

    assert isinstance(body["points"], list)


def test_start_paper_explains_missing_models(client, monkeypatch):
    """No trained models is a 400 with a reason, not a 500."""
    def boom(*args, **kwargs):
        raise FileNotFoundError("no model at models/registry/BTC_USDT.joblib")

    monkeypatch.setattr(paper_runner, "start_session", boom)

    r = client.post("/api/paper/start", json={"scope": "crypto"})

    assert r.status_code == 400
    assert "Trained models" in r.json()["detail"]


def test_stop_paper_when_nothing_running(client):
    body = client.post("/api/paper/stop").json()

    assert body["stopped"] is False
    assert "message" in body
