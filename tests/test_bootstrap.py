"""
Tests for scripts/bootstrap.py, the startup artifact builder.

This script only matters on a machine that does NOT already have the
artifacts -- i.e. every fresh deploy, and no developer's laptop. That
asymmetry is exactly why it needs tests: a bug here is invisible locally and
fatal in production, where the symptom is a dashboard with nothing to draw.

The two failure modes worth pinning down are the ones that hide. A step that
exits non-zero is loud. A step that exits ZERO without producing its artifact
is silent -- and worse, because the output-existence check would then mark it
done and skip it on every subsequent restart, so the instance stays broken
across redeploys while reporting healthy.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

import bootstrap
from config import loader
from data import storage


# --------------------------------------------------------------------------
# Status file
# --------------------------------------------------------------------------


def test_read_status_is_none_before_any_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "STATUS_PATH", tmp_path / "nope.json")

    assert bootstrap.read_status() is None


def test_read_status_survives_a_corrupt_file(tmp_path, monkeypatch):
    """A half-written status file must not take down /api/health."""
    path = tmp_path / "bootstrap_status.json"
    path.write_text('{"state": "run')
    monkeypatch.setattr(bootstrap, "STATUS_PATH", path)

    assert bootstrap.read_status() is None


def test_write_then_read_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "STATUS_PATH", tmp_path / "bootstrap_status.json")

    bootstrap._write_status(state="running", step="train", bars=6000)

    status = bootstrap.read_status()
    assert status["state"] == "running"
    assert status["step"] == "train"
    assert status["bars"] == 6000
    assert "updated_at" in status


def test_write_status_does_not_raise_when_unwritable(tmp_path, monkeypatch):
    """Progress reporting must never be the thing that kills a build."""
    blocker = tmp_path / "afile"
    blocker.write_text("not a directory")
    monkeypatch.setattr(bootstrap, "STATE_DIR", blocker / "sub")
    monkeypatch.setattr(bootstrap, "STATUS_PATH", blocker / "sub" / "s.json")

    bootstrap._write_status(state="running")  # must not raise


# --------------------------------------------------------------------------
# Step detection
# --------------------------------------------------------------------------


def test_step_is_not_done_without_outputs():
    assert not bootstrap.Step("x", ["x.py"]).is_done()


def test_step_done_only_when_every_output_exists(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    step = bootstrap.Step("x", ["x.py"], outputs=[a, b])

    assert not step.is_done(), "one of two outputs missing is not done"

    b.write_text("y")
    assert step.is_done()


def test_fetch_step_paths_match_where_fetch_actually_writes():
    """
    The skip check is only correct if it looks where the fetch script
    actually saves. Deriving the filename twice is how that drifts.
    """
    cfg = loader.load_config()
    steps = bootstrap.build_steps(cfg, bars=6000, scope="crypto")
    fetch = next(s for s in steps if s.name == "fetch")

    tf = loader.timeframe(cfg)
    expected = [
        storage.cache_path(spec.cache_name, spec.symbol, tf)
        for spec in loader.resolve_symbols("crypto", cfg)
    ]

    assert fetch.outputs == expected
    assert all(str(p).endswith(".parquet") for p in fetch.outputs)


def test_steps_run_in_dependency_order():
    steps = bootstrap.build_steps(loader.load_config(), bars=6000, scope="crypto")

    assert [s.name for s in steps] == ["fetch", "train", "backtest", "report"]


# --------------------------------------------------------------------------
# run_step
# --------------------------------------------------------------------------


class _Proc:
    def __init__(self, returncode):
        self.returncode = returncode


def _fake_popen(returncode, lines=(), on_start=None):
    """
    Stand-in for subprocess.Popen: run_step streams from `.stdout` and then
    calls `.wait()`, so the fake needs both.
    """
    class FakePopen:
        def __init__(self, cmd, **kwargs):
            if on_start:
                on_start()
            self.stdout = iter([f"{ln}\n" for ln in lines])
            self._returncode = returncode

        def wait(self):
            return self._returncode

    return FakePopen


def test_run_step_fails_on_nonzero_exit(tmp_path, monkeypatch):
    out = tmp_path / "artifact.json"
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _fake_popen(1))

    ok, error = bootstrap.run_step(bootstrap.Step("train", ["t.py"], outputs=[out]))

    assert not ok
    assert "train" in error


def test_run_step_fails_when_exit_zero_but_nothing_was_produced(tmp_path, monkeypatch):
    """
    The silent failure. Returning ok here would write state=complete and
    mark the step done, so every later restart would skip it and the
    instance would stay artifact-less while reporting healthy.
    """
    out = tmp_path / "never_written.json"
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _fake_popen(0))

    ok, error = bootstrap.run_step(bootstrap.Step("train", ["t.py"], outputs=[out]))

    assert not ok
    assert "no output" in error


def test_run_step_succeeds_when_the_artifact_appears(tmp_path, monkeypatch):
    out = tmp_path / "written.json"
    monkeypatch.setattr(
        bootstrap.subprocess, "Popen", _fake_popen(0, on_start=lambda: out.write_text("{}"))
    )

    ok, error = bootstrap.run_step(bootstrap.Step("train", ["t.py"], outputs=[out]))

    assert ok
    assert error == ""


def test_failure_error_carries_the_actual_message(tmp_path, monkeypatch):
    """
    The reason a step died has to survive into the status file: on a hosted
    platform /api/health is often the only surface you can reach, so a bare
    "fetch failed" would leave the cause buried in a build log behind a
    dashboard login.
    """
    out = tmp_path / "artifact.json"
    monkeypatch.setattr(
        bootstrap.subprocess,
        "Popen",
        _fake_popen(1, lines=[
            "Fetching 6000 x 5m bars",
            "binance: FAILED -- ExchangeNotAvailable: binance is restricted (HTTP 451)",
        ]),
    )

    ok, error = bootstrap.run_step(bootstrap.Step("fetch", ["f.py"], outputs=[out]))

    assert not ok
    assert "451" in error
    assert "restricted" in error


# --------------------------------------------------------------------------
# End-to-end status transitions
# --------------------------------------------------------------------------


def test_main_reports_failure_and_stops_at_the_broken_step(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "STATUS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(bootstrap.subprocess, "Popen", _fake_popen(1, lines=["boom"]))
    # --force so this exercises the build path even on a machine that already
    # has the artifacts checked out.
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--force"])

    rc = bootstrap.main()

    assert rc == 1
    status = bootstrap.read_status()
    assert status["state"] == "failed"
    assert status["step"] == "fetch"           # first step, nothing completed
    assert status["completed"] == []
    assert "boom" in status["error"]


def test_main_marks_complete_when_everything_is_already_built(tmp_path, monkeypatch):
    """The restart path: a redeploy must not refetch anything."""
    outputs = [
        tmp_path / "crypto_ccxt_BTC-USDT_5m.parquet",
        tmp_path / "training_summary.json",
        tmp_path / "backtest_results.json",
        tmp_path / "report.html",
    ]
    for p in outputs:
        p.write_text("{}")

    steps = [
        bootstrap.Step(name, ["x.py"], outputs=[out])
        for name, out in zip(["fetch", "train", "backtest", "report"], outputs)
    ]
    monkeypatch.setattr(bootstrap, "build_steps", lambda *a, **k: steps)
    monkeypatch.setattr(bootstrap, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "STATUS_PATH", tmp_path / "s.json")

    def explode(*args, **kwargs):
        raise AssertionError("nothing should run when all outputs already exist")

    monkeypatch.setattr(bootstrap.subprocess, "Popen", explode)
    monkeypatch.setattr(sys, "argv", ["bootstrap.py"])

    assert bootstrap.main() == 0
    assert bootstrap.read_status()["state"] == "complete"


def test_force_rebuilds_even_when_outputs_exist(tmp_path, monkeypatch):
    out = tmp_path / "backtest_results.json"
    out.write_text("{}")
    step = bootstrap.Step("backtest", ["b.py"], outputs=[out])

    monkeypatch.setattr(bootstrap, "build_steps", lambda *a, **k: [step])
    monkeypatch.setattr(bootstrap, "STATE_DIR", tmp_path)
    monkeypatch.setattr(bootstrap, "STATUS_PATH", tmp_path / "s.json")

    ran = []

    def fake_popen(cmd, **kwargs):
        ran.append(cmd)
        return _fake_popen(0)(cmd, **kwargs)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", "--force"])

    assert bootstrap.main() == 0
    assert ran, "--force must re-run a step whose output already exists"


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


def test_committed_report_does_not_look_pre_built(tmp_path):
    """
    results/report.html is committed to the repo; results/backtest_results.json
    is gitignored. On a fresh container the report therefore exists while its
    input does not, and an existence-only check would skip regeneration --
    shipping the repository's report instead of the container's own backtest.
    """
    report = tmp_path / "report.html"
    backtest = tmp_path / "backtest_results.json"
    report.write_text("old")
    backtest.write_text("new")

    import os
    import time as _time
    os.utime(report, (_time.time() - 600, _time.time() - 600))

    step = bootstrap.Step("report", ["r.py"], outputs=[report], newer_than=[backtest])

    assert not step.is_done(), "report is older than the backtest it summarises"

    os.utime(report, (_time.time() + 600, _time.time() + 600))
    assert step.is_done()


def test_missing_dependency_does_not_mark_step_stale(tmp_path):
    """No backtest yet means nothing to be stale against -- not a reason to
    claim the report is outdated."""
    report = tmp_path / "report.html"
    report.write_text("x")

    step = bootstrap.Step(
        "report", ["r.py"], outputs=[report], newer_than=[tmp_path / "absent.json"]
    )

    assert step.is_done()


def test_report_step_depends_on_the_backtest():
    steps = {s.name: s for s in bootstrap.build_steps(loader.load_config(), 6000, "crypto")}

    assert steps["report"].newer_than == [bootstrap.RESULTS_DIR / "backtest_results.json"]
