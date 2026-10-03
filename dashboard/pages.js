// Shared renderer for every sub-page.
//
// Each pages/*.html is a thin shell with <body data-page="...">; this file
// decides what to draw. The alternative -- a hand-written script per page --
// is how the original repo ended up with sixteen files, twelve of them
// empty. Every renderer reads from the same API endpoints as the dashboard,
// and every one of them renders missing data as "—" rather than a plausible
// number.

const PAGES = {
  // ---------------------------------------------------------------- portfolio
  async portfolio() {
    const state = await safe(() => DASH.getJSON("/api/state/paper"), {});
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);

    const rows = (state.positions || []).map((p) => [
      p.symbol,
      `<span class="badge ${p.side === "LONG" ? "buy" : "sell"}">${p.side}</span>`,
      DASH_FMT.num(p.qty, 6),
      DASH_FMT.money(p.current),
      DASH_FMT.money(p.notional),
      DASH_FMT.money(p.unrealized),
    ]);

    return `
      ${banner(state, backtest)}
      <div class="cards">
        <div class="card"><h3>Equity</h3><p>${DASH_FMT.money(state.portfolio)}</p>
          <small>${state.running ? "Live paper session" : "No session running"}</small></div>
        <div class="card"><h3>Peak Equity</h3><p>${DASH_FMT.money(state.peak_equity)}</p>
          <small>Session high-water mark</small></div>
        <div class="card"><h3>Drawdown</h3><p class="${DASH_FMT.signClass(-(state.drawdown_pct || 0))}">${DASH_FMT.pct(state.drawdown_pct)}</p>
          <small>From peak</small></div>
        <div class="card"><h3>Exposure</h3><p>${DASH_FMT.pct(state.open_exposure_pct)}</p>
          <small>Of equity, gross</small></div>
      </div>
      <div class="panel" style="margin-top:30px;">
        <h3>Open Positions</h3>
        ${dashTable(
          [{ label: "Symbol" }, { label: "Side" }, { label: "Qty", num: true },
           { label: "Mark", num: true }, { label: "Notional", num: true },
           { label: "Unrealized", num: true }],
          rows,
          "No open positions."
        )}
      </div>
      <div class="panel" style="margin-top:20px;">
        <h3>Funding</h3>
        <p class="muted" style="line-height:1.6;">
          This platform does not hold or move money. The deposit, withdrawal and
          transaction pages are UI demonstrations with simulated balances — there is no
          payment processor connected and no real account behind them. Paper trading
          starts from a virtual ${DASH_FMT.money(100000, { decimals: 0 })} balance.
        </p>
      </div>`;
  },

  // ------------------------------------------------------------------ markets
  async markets() {
    const state = await safe(() => DASH.getJSON("/api/state/paper"), {});
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);
    const prices = state.latest_prices || {};
    const symbols = (backtest && backtest.symbols) || Object.keys(prices);

    const rows = symbols.map((s) => {
      const sig = (state.recent_signals || {})[s] || {};
      const conf = sig.confidence;
      return [
        s,
        DASH_FMT.money(prices[s]),
        sig.target_position === undefined || sig.target_position === null
          ? '<span class="muted">—</span>'
          : `<span class="badge ${sig.target_position > 0 ? "buy" : sig.target_position < 0 ? "sell" : ""}">${
              sig.target_position > 0 ? "LONG" : sig.target_position < 0 ? "SHORT" : "FLAT"
            }</span>`,
        DASH_FMT.num(conf, 4),
        `<span class="muted" style="font-size:12px;">${DASH_FMT.text(sig.reason)}</span>`,
      ];
    });

    return `
      ${banner(state, backtest)}
      <p class="muted" style="margin-bottom:18px;">
        Prices update while a paper session is running. Signals are the model's current
        target exposure per symbol and update on the same cycle.
      </p>
      ${dashTable(
        [{ label: "Symbol" }, { label: "Last price", num: true }, { label: "Signal" },
         { label: "Confidence", num: true }, { label: "Reason" }],
        rows,
        "No market data yet — start a paper session, or run the backtest pipeline."
      )}`;
  },

  // --------------------------------------------------------------- strategies
  async strategies() {
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);
    const summary = (backtest && backtest.training_summary) || {};

    const rows = Object.entries(summary).map(([symbol, s]) => [
      symbol,
      `<span class="badge ${s.is_real_data ? "real" : "synthetic"}">${s.is_real_data ? "real" : "synthetic"}</span>`,
      DASH_FMT.num(s.mean_auc, 4),
      DASH_FMT.num(s.n_labeled_bars, 0),
      DASH_FMT.num(s.n_folds_used, 0),
      DASH_FMT.num(((backtest.per_symbol || {})[symbol] || {}).num_trades, 0),
    ]);

    return `
      ${backtest ? dashProvenanceBanner(backtest.data_source) : ""}
      <div class="banner info">
        <b>How a signal is produced.</b> Each symbol has its own
        HistGradientBoosting model trained on cost-aware labels — the target is
        "did price move more than the round-trip cost", not "did price go up".
        A probability above <b>0.58</b> goes long, below <b>0.42</b> goes short, and
        everything between stays flat. Staying out is a position.
      </div>
      <div class="panel">
        <h3>Per-symbol model & activity</h3>
        ${dashTable(
          [{ label: "Symbol" }, { label: "Data" }, { label: "Mean AUC", num: true },
           { label: "Labeled bars", num: true }, { label: "Folds", num: true },
           { label: "Backtest fills", num: true }],
          rows,
          "No trained models found. Run scripts/train_model.py."
        )}
      </div>
      <div class="panel" style="margin-top:20px;">
        <h3>Reading AUC honestly</h3>
        <p class="muted" style="line-height:1.6;">
          Walk-forward AUC of 0.50 means the model found nothing. Values like 0.53–0.65
          are a real but modest edge — and an edge in <i>ranking</i> is not the same as
          profit after fees and slippage. The backtest applies both; that is the number
          that matters.
        </p>
      </div>`;
  },

  // -------------------------------------------------------------------- paper
  async paper() {
    const state = await safe(() => DASH.getJSON("/api/state/paper"), {});
    const status = await safe(() => DASH.getJSON("/api/paper/status"), {});
    const equity = await safe(() => DASH.getJSON("/api/paper/equity"), { points: [] });

    const points = equity.points || [];
    return `
      ${banner(state, null)}
      <div class="cards">
        <div class="card"><h3>Session</h3>
          <p style="font-size:22px;">${status.running ? "Running" : "Stopped"}</p>
          <small>${status.cycles ? status.cycles + " cycles completed" : "No cycles yet"}</small></div>
        <div class="card"><h3>Equity points</h3><p>${points.length}</p>
          <small>Recorded this session</small></div>
        <div class="card"><h3>Poll interval</h3>
          <p style="font-size:22px;">${DASH_FMT.text(status.poll_seconds)}${status.poll_seconds ? "s" : ""}</p>
          <small>Between cycles</small></div>
        <div class="card"><h3>Data source</h3>
          <p style="font-size:22px;">${DASH_FMT.text(status.data_source)}</p>
          <small>${DASH_FMT.text(status.timeframe)} bars</small></div>
      </div>
      <div class="panel" style="margin-top:30px;">
        <h3>Control</h3>
        <button class="action green" id="start-bot" ${status.running ? "disabled" : ""}>▶ Start</button>
        <button class="action red" id="stop-bot" ${status.running ? "" : "disabled"}>■ Stop</button>
        <p class="muted" id="action-msg" style="margin-top:12px;font-size:13px;"></p>
        ${status.last_error ? `<pre>${status.last_error}</pre>` : ""}
      </div>
      <div class="chart-card">
        <h3>Session Equity</h3>
        <p class="muted" style="margin-bottom:16px;font-size:13px;">
          ${points.length > 1 ? points.length + " cycles" : "Not enough points yet"}
        </p>
        <div style="height:320px;"><canvas id="equityChart"></canvas></div>
      </div>`;
  },

  // --------------------------------------------------------------------- live
  async live() {
    const state = await safe(() => DASH.getJSON("/api/state/live"), {});
    return `
      <div class="banner danger">
        <b>Live trading is disabled on this instance.</b>
        ${state.message ? state.message : ""}
      </div>
      <div class="panel">
        <h3>Why this page is empty on purpose</h3>
        <p class="muted" style="line-height:1.7;">
          This dashboard is public and has no authentication, by request. If it exposed a
          button that placed real orders, anyone who found the URL could trade the
          account. So real-money execution is gated in code, not in the UI:
          <code>execution/crypto_live.py</code> and <code>execution/forex_live.py</code>
          both refuse to place a single order unless
          <code>LIVE_TRADING_CONFIRMED=yes</code> is set in the server environment.
          No endpoint here can set it.
        </p>
        <h3 style="margin-top:24px;">Before you would ever enable it</h3>
        <p class="muted" style="line-height:1.7;">
          Backtest first and read the numbers. Then paper trade for days or weeks on a real
          feed — see the Paper Trading page. A strategy that looks fine in backtest can
          still fail live on feed hiccups, miscalibration on unfamiliar data, or a
          timezone assumption that was quietly wrong. Read
          <code>README.md § Going live safely</code>.
        </p>
      </div>`;
  },

  // ----------------------------------------------------------------- backtest
  async backtest() {
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);
    if (!backtest) {
      return emptyState(
        "No backtest results yet",
        "Run the pipeline: <code>python scripts/fetch_market_data.py</code> → " +
          "<code>python scripts/train_model.py</code> → <code>python scripts/run_backtest.py</code>."
      );
    }
    const perf = backtest.performance || {};
    const rows = Object.entries(perf).map(([k, v]) => [
      k.replace(/_/g, " "),
      typeof v === "number" ? DASH_FMT.num(v, 4) : DASH_FMT.text(v),
    ]);

    return `
      ${dashProvenanceBanner(backtest.data_source)}
      <div class="cards">
        <div class="card"><h3>Total return</h3>
          <p class="${DASH_FMT.signClass(perf.total_return_pct)}">${DASH_FMT.pct(perf.total_return_pct, 2, { sign: true })}</p>
          <small>After fees &amp; slippage</small></div>
        <div class="card"><h3>Sharpe</h3><p>${DASH_FMT.num(perf.sharpe, 2)}</p>
          <small>Daily-resampled</small></div>
        <div class="card"><h3>Max drawdown</h3><p>${DASH_FMT.pct(perf.max_drawdown_pct)}</p>
          <small>Kill-switch at 15%</small></div>
        <div class="card"><h3>Trades</h3><p>${DASH_FMT.num(perf.num_trades, 0)}</p>
          <small>${DASH_FMT.num(backtest.data_source && backtest.data_source.n_bars, 0)} bars</small></div>
      </div>
      <div class="panel" style="margin-top:30px;">
        <h3>Full performance</h3>
        ${dashTable([{ label: "Metric" }, { label: "Value", num: true }], rows, "No metrics.")}
      </div>
      <div class="panel" style="margin-top:20px;">
        <h3>Interactive report</h3>
        <p class="muted" style="margin-bottom:14px;line-height:1.6;">
          The full report includes the walk-forward validation timeline per symbol and the
          equity curve with halt markers.
        </p>
        <a class="action blue" href="/report" style="text-decoration:none;display:inline-block;">📑 Open full report</a>
      </div>`;
  },

  // -------------------------------------------------------------- performance
  async performance() {
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);
    if (!backtest) {
      return emptyState("No performance data yet", "Run scripts/run_backtest.py.");
    }
    const perf = backtest.performance || {};
    const ds = backtest.data_source || {};

    const groups = [
      ["Returns", ["total_return_pct", "annualized_return_pct", "n_days_observed"]],
      ["Risk-adjusted", ["sharpe", "sortino", "calmar"]],
      ["Drawdown", ["max_drawdown_pct", "max_drawdown_duration_days"]],
      ["Trade quality", ["win_rate_pct", "profit_factor", "avg_win", "avg_loss", "num_trades"]],
    ];

    const blocks = groups
      .map(([title, keys]) => {
        const rows = keys
          .filter((k) => k in perf)
          .map((k) => [
            k.replace(/_/g, " "),
            typeof perf[k] === "number" ? DASH_FMT.num(perf[k], 4) : DASH_FMT.text(perf[k]),
          ]);
        if (!rows.length) return "";
        return `<div class="panel" style="margin-bottom:20px;"><h3>${title}</h3>
          ${dashTable([{ label: "Metric" }, { label: "Value", num: true }], rows, "None.")}</div>`;
      })
      .join("");

    return `
      ${dashProvenanceBanner(ds)}
      <div class="banner info">
        Observation window: <b>${DASH_FMT.num(ds.n_bars, 0)}</b> bars of
        ${DASH_FMT.text(ds.timeframe)} data, ${DASH_FMT.shortTime(ds.start)} → ${DASH_FMT.shortTime(ds.end)}.
        Risk metrics computed from daily-resampled returns need months of data before they
        mean much — <code>n_days_observed</code> is reported for exactly that reason.
      </div>
      ${blocks}`;
  },

  // ------------------------------------------------------------------- models
  async models() {
    const health = await safe(() => DASH.getJSON("/api/health"), {});
    const backtest = await safe(() => DASH.getJSON("/api/results/backtest"), null);
    const summary = (backtest && backtest.training_summary) || {};
    const models = (health.artifacts && health.artifacts.trained_models) || [];

    const foldRows = [];
    Object.entries(summary).forEach(([symbol, s]) => {
      (s.folds || []).forEach((f) => {
        foldRows.push([
          symbol,
          f.fold_number,
          DASH_FMT.num(f.train_size, 0),
          DASH_FMT.num(f.test_size, 0),
          DASH_FMT.num(f.auc, 4),
          DASH_FMT.num(f.brier, 4),
          `<span class="muted" style="font-size:12px;">${DASH_FMT.shortTime(f.test_start)} → ${DASH_FMT.shortTime(f.test_end)}</span>`,
        ]);
      });
    });

    return `
      <div class="banner info">
        <b>Walk-forward validation with an embargo gap.</b> Financial series can't be
        shuffled — a random split leaks future information backward through correlated bars.
        Each fold trains on an expanding window, skips an embargo equal to the label horizon,
        then tests on the next chunk forward in time. The backtest only ever replays these
        out-of-fold predictions, never the final model on its own training data.
      </div>
      <div class="panel" style="margin-bottom:20px;">
        <h3>Registered models</h3>
        ${models.length
          ? `<p class="muted">${models.length} model file(s) in <code>models/registry/</code>: ${models.join(", ")}</p>`
          : '<p class="muted">No model files found. Run scripts/train_model.py.</p>'}
      </div>
      <div class="panel">
        <h3>Fold diagnostics</h3>
        ${dashTable(
          [{ label: "Symbol" }, { label: "Fold", num: true }, { label: "Train", num: true },
           { label: "Test", num: true }, { label: "AUC", num: true },
           { label: "Brier", num: true }, { label: "Test window" }],
          foldRows,
          "No fold data yet."
        )}
      </div>`;
  },

  // ------------------------------------------------------------------ reports
  async reports() {
    const health = await safe(() => DASH.getJSON("/api/health"), {});
    const a = health.artifacts || {};
    const rows = [
      ["Backtest results (JSON)", a.backtest_results, "/results/backtest_results.json"],
      ["Training summary (JSON)", a.training_summary, "/results/training_summary.json"],
      ["Interactive report (HTML)", a.report_html, "/report"],
    ].map(([name, present, href]) => [
      name,
      present
        ? '<span class="badge real">available</span>'
        : '<span class="badge synthetic">missing</span>',
      present ? `<a class="link" href="${href}">open</a>` : '<span class="muted">—</span>',
    ]);

    return `
      ${a.report_html ? "" : `<div class="banner warn">
        No report has been generated yet. Run <code>python scripts/build_report.py</code>.</div>`}
      <div class="panel">
        <h3>Generated artifacts</h3>
        ${dashTable([{ label: "Artifact" }, { label: "Status" }, { label: "Link" }], rows, "None.")}
      </div>
      <div class="panel" style="margin-top:20px;">
        <h3>Raw API</h3>
        <p class="muted" style="line-height:1.7;">
          The same data the dashboard reads is available directly:
          <a class="link" href="/api/results/backtest">/api/results/backtest</a>,
          <a class="link" href="/api/training/summary">/api/training/summary</a>,
          <a class="link" href="/api/state/paper">/api/state/paper</a>,
          <a class="link" href="/api/health">/api/health</a>.
        </p>
      </div>`;
  },

  // ------------------------------------------------------------- transactions
  async transactions() {
    const state = await safe(() => DASH.getJSON("/api/state/paper"), {});
    const log = (state.trade_log || []).slice().reverse();

    const rows = log.map((t) => [
      `<span class="muted" style="font-size:12px;">${DASH_FMT.shortTime(t.timestamp)}</span>`,
      DASH_FMT.text(t.symbol),
      `<span class="badge ${t.side === "buy" ? "buy" : "sell"}">${DASH_FMT.text(t.side)}</span>`,
      DASH_FMT.num(t.qty, 6),
      DASH_FMT.money(t.price),
      DASH_FMT.text(t.status),
    ]);

    return `
      <div class="banner info">
        Simulated fills from the paper-trading loop — the order log the orchestrator writes
        each cycle. These are not real transactions; no money moves.
      </div>
      ${dashTable(
        [{ label: "Time" }, { label: "Symbol" }, { label: "Side" }, { label: "Qty", num: true },
         { label: "Fill price", num: true }, { label: "Status" }],
        rows,
        state.running ? "No fills yet this session." : "No paper session has run yet."
      )}`;
  },

  // --------------------------------------------------------------------- logs
  async logs() {
    const state = await safe(() => DASH.getJSON("/api/state/paper"), {});
    const signals = state.recent_signals || {};
    const rows = Object.entries(signals).map(([symbol, s]) => [
      DASH_FMT.text(symbol),
      DASH_FMT.shortTime(s.timestamp),
      s.target_position > 0 ? "LONG" : s.target_position < 0 ? "SHORT" : "FLAT",
      DASH_FMT.num(s.confidence, 4),
      `<span class="muted" style="font-size:12px;">${DASH_FMT.text(s.reason)}</span>`,
    ]);

    return `
      <div class="panel" style="margin-bottom:20px;">
        <h3>Latest signals</h3>
        ${dashTable(
          [{ label: "Symbol" }, { label: "Time" }, { label: "Target" },
           { label: "Confidence", num: true }, { label: "Reason" }],
          rows,
          "No signals recorded — start a paper session."
        )}
      </div>
      <div class="panel">
        <h3>Session snapshot</h3>
        <pre>${state.running || state.portfolio ? JSON.stringify(state, null, 2) : "No session state on disk."}</pre>
      </div>`;
  },

  // ----------------------------------------------------------------- settings
  async settings() {
    const health = await safe(() => DASH.getJSON("/api/health"), {});
    return `
      <div class="banner warn">
        <b>Read-only.</b> This page reports the running configuration. Trading parameters
        live in <code>config/config.yaml</code> and are applied on the next process start —
        there is no writable settings endpoint, deliberately, on an unauthenticated server.
      </div>
      <div class="panel" style="margin-bottom:20px;">
        <h3>Instance status</h3>
        <pre>${JSON.stringify(health, null, 2)}</pre>
      </div>
      <div class="panel">
        <h3>Risk controls (applied in code, see strategy/risk.py)</h3>
        <p class="muted" style="line-height:1.8;">
          • Volatility-scaled position sizing — fixed % of equity per trade, sized off ATR<br>
          • Hard cap on single-position and total exposure<br>
          • Daily loss limit that blocks new entries for the rest of the day<br>
          • <b>Max drawdown kill-switch</b> that halts all trading until a human calls
          <code>reset_halt()</code> — intentionally not automatic<br>
          • A no-trade rebalance band, so a drifting confidence score doesn't pay fees on
          every bar
        </p>
      </div>`;
  },
};

// ---------------------------------------------------------------------------

async function safe(fn, fallback) {
  try {
    return await fn();
  } catch (_) {
    return fallback;
  }
}

function emptyState(title, body) {
  return `<div class="panel"><h3>${title}</h3>
    <p class="muted" style="line-height:1.7;margin-top:8px;">${body}</p></div>`;
}

function banner(state, backtest) {
  if (backtest && backtest.data_source) return dashProvenanceBanner(backtest.data_source);
  if (state && state.trading_halted) {
    return `<div class="banner danger"><b>Trading halted.</b>
      ${DASH_FMT.text(state.halt_reason)} The kill-switch stays engaged until a human
      resets it — that is deliberate.</div>`;
  }
  return "";
}

/** Wires the Start/Stop buttons on any page that renders them. */
function wireControls() {
  const start = document.getElementById("start-bot");
  const stop = document.getElementById("stop-bot");
  const msg = document.getElementById("action-msg");

  if (start) {
    start.addEventListener("click", async () => {
      msg.textContent = "Starting…";
      start.disabled = true;
      try {
        const res = await DASH.postJSON("/api/paper/start", {});
        msg.textContent = `Started (${res.symbols.join(", ")}, every ${res.poll_seconds}s).`;
        setTimeout(() => location.reload(), 900);
      } catch (err) {
        msg.textContent = `Could not start: ${err.message}`;
        start.disabled = false;
      }
    });
  }

  if (stop) {
    stop.addEventListener("click", async () => {
      msg.textContent = "Stopping…";
      stop.disabled = true;
      try {
        const res = await DASH.postJSON("/api/paper/stop", {});
        msg.textContent = res.stopped ? "Stopped." : res.message;
        setTimeout(() => location.reload(), 900);
      } catch (err) {
        msg.textContent = `Could not stop: ${err.message}`;
        stop.disabled = false;
      }
    });
  }
}

document.addEventListener("layout:ready", async () => {
  const page = document.body.dataset.page;
  const target = document.getElementById("page-content");
  if (!page || !target || !PAGES[page]) return;

  target.innerHTML = '<p class="spinner">Loading…</p>';
  try {
    target.innerHTML = await PAGES[page]();
  } catch (err) {
    target.innerHTML = `<div class="banner danger">Failed to load: ${err.message}</div>`;
    return;
  }
  wireControls();

  // The paper page draws a session equity chart; reuse the dashboard's
  // Chart.js instance if the library is loaded.
  if (page === "paper" && typeof Chart !== "undefined") {
    const points = await safe(async () => (await DASH.getJSON("/api/paper/equity")).points, []);
    const canvas = document.getElementById("equityChart");
    if (canvas && points.length > 1) {
      new Chart(canvas, {
        type: "line",
        data: {
          labels: points.map((p) => DASH_FMT.shortTime(p.timestamp)),
          datasets: [{
            label: "Paper equity",
            data: points.map((p) => p.equity),
            borderColor: "#4ade80",
            backgroundColor: "#4ade8022",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
            fill: true,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#94a3b8" } } },
          scales: {
            x: { ticks: { color: "#64748b", maxTicksLimit: 8 }, grid: { color: "#1e293b" } },
            y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
          },
        },
      });
    }
  }
});
