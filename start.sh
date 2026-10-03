#!/usr/bin/env sh
# Container entrypoint.
#
# Artifacts are gitignored, so a fresh container has no cached data, no
# models, and no backtest results. Build them here rather than at image-build
# time: the data should be current as of boot, and baking ~10k bars plus a
# trained model into the image would make it large and stale.
#
# Bootstrap runs in the BACKGROUND, not before uvicorn. Blocking on it would
# leave the port unbound for minutes (fetch + walk-forward training + backtest
# + report), which reads to a platform health check as a dead service. This
# way the dashboard comes up immediately and /api/health reports
# bootstrap.state ("running" with the current step, then "complete") so the UI
# can show real progress instead of an empty page.
#
# Completed steps are skipped on restart (see scripts/bootstrap.py), so a
# redeploy costs a few stat() calls, not a full rebuild.
set -eu

echo "[start] launching artifact bootstrap in background"
python scripts/bootstrap.py &

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
