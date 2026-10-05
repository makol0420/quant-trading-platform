// Dashboard: real data from the API, or an explicit "nothing here yet" state.
//
// Two charts are possible: the walk-forward backtest equity curve (what the
// strategy would have done on history) and the live paper session's equity
// (what it is doing now). They are never blended into one line, because they
// measure different things over different periods.

let chart = null;
let refreshTimer = null;
let backtestCache = null;
let chartRendered = false;

const els = {};

/**
 * Resolves once layout.js has injected the sidebar and topbar.
 *
 * This page's scripts run at the end of <body>, so they execute while the
 * topbar is still an empty <header> -- layout.js fills it in asynchronously
 * and only then fires `layout:ready`. Reading the DOM before that point
 * returns null for anything the topbar owns, and the first null dereference
 * threw out of refresh(), taking the equity chart and the polling timer down
 * with it and leaving the page on its literal "Loading…" placeholders. The
 * contract is layout.js's; this is the page script honouring it instead of
 * racing it.
 */
function whenLayoutReady() {
  return new Promise((resolve) => {
    if (window.__qtpLayoutReady) return resolve();
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    document.addEventListener("layout:ready", done, { once: true });
    // A layout.js that 404s or wedges must not leave the dashboard blank.
    setTimeout(done, 3000);
  });
}

function initEls() {
  els.banner = document.getElementById("banner");
  els.portfolio = document.getElementById("portfolio");
  els.pnl = document.getElementById("pnl");
  els.trades = document.getElementById("trades");
  els.winrate = document.getElementById("winrate");
  els.portfolioNote = document.getElementById("portfolio-note");
  els.pnlNote = document.getElementById("pnl-note");
  els.tradesNote = document.getElementById("trades-note");
  els.winrateNote = document.getElementById("winrate-note");
  els.status = document.getElementById("status");
  els.chartTitle = document.getElementById("chart-title");
  els.chartNote = document.getElementById("chart-note");
  els.startBtn = document.getElementById("start-bot");
  els.stopBtn = document.getElementById("stop-bot");
  els.actionMsg = document.getElementById("action-msg");
  // No #session-state here: that pill lives in the topbar and is owned by
  // layout.js, which also refreshes it on pages that never load app.js.
}

/** Hourly downsample so a 20k-point equity curve stays a readable chart. */
function downsample(points, key, maxPoints = 400) {
  if (!points || points.length <= maxPoints) return points || [];
  const step = Math.ceil(points.length / maxPoints);
  const out = [];
  for (let i = 0; i < points.length; i += step) out.push(points[i]);
  if (out[out.length - 1] !== points[points.length - 1]) out.push(points[points.length - 1]);
  return out;
}

function renderChart(labels, values, label, color) {
  const ctx = document.getElementById("equityChart");
  if (!ctx) return;
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label,
          data: values,
          borderWidth: 2,
          borderColor: color,
          backgroundColor: color + "22",
          tension: 0.25,
          pointRadius: 0,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: true, labels: { color: "#94a3b8" } },
        tooltip: {
          callbacks: {
            label: (c) => `${c.dataset.label}: ${DASH_FMT.money(c.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 8, autoSkip: true },
          grid: { color: "#1e293b" },
        },
        y: {
          ticks: {
            color: "#64748b",
            callback: (v) => "$" + Number(v).toLocaleString(),
          },
          grid: { color: "#1e293b" },
        },
      },
    },
  });
}

function renderBacktestChart(backtest) {
  const raw = backtest.equity_curve || [];
  const points = downsample(raw, "equity");
  els.chartTitle.textContent = "Backtest Equity (walk-forward, out-of-fold)";
  els.chartNote.textContent =
    `${DASH_FMT.num(points.length, 0)} of ${DASH_FMT.num(raw.length, 0)} bars shown ` +
    `(hourly sample) · ${DASH_FMT.money(backtest.starting_equity, { decimals: 0 })} start`;
  renderChart(
    points.map((p) => DASH_FMT.shortTime(p.timestamp)),
    points.map((p) => p.equity),
    "Backtest equity",
    "#38bdf8"
  );
}

function renderSessionChart(points) {
  els.chartTitle.textContent = "Paper Session Equity (live)";
  els.chartNote.textContent = `${points.length} cycle(s) recorded since the session started`;
  renderChart(
    points.map((p) => DASH_FMT.shortTime(p.timestamp)),
    points.map((p) => p.equity),
    "Paper equity",
    "#4ade80"
  );
}

function setCard(el, noteEl, value, note, signValue) {
  el.textContent = value;
  el.className = DASH_FMT.signClass(signValue);
  if (noteEl) noteEl.textContent = note;
}

async function loadHealth() {
  try {
    const health = await DASH.getJSON("/api/health");
    if (!health.ready) {
      els.banner.innerHTML = `<div class="banner warn">
        <b>Pipeline artifacts missing on this instance.</b>
        No backtest results or trained models were found, so the figures below are empty by
        design rather than invented. Generate them with:
        <code>python scripts/fetch_market_data.py</code> →
        <code>python scripts/train_model.py</code> →
        <code>python scripts/run_backtest.py</code>.
      </div>`;
    }
    return health;
  } catch (err) {
    els.banner.innerHTML = `<div class="banner danger">
      Could not reach the API: ${err.message}</div>`;
    return null;
  }
}

async function loadBacktest() {
  try {
    const backtest = await DASH.getJSON("/api/results/backtest");
    backtestCache = backtest;
    return backtest;
  } catch (err) {
    if (err.status === 404) return null;
    throw err;
  }
}

async function loadState() {
  try {
    return await DASH.getJSON("/api/state/paper");
  } catch (err) {
    return { running: false, message: `Could not load state: ${err.message}` };
  }
}

function renderCards(state) {
  const running = !!state.running;

  setCard(
    els.portfolio, els.portfolioNote,
    DASH_FMT.money(state.portfolio, { decimals: 2 }),
    running ? "Live paper equity" : "No session running",
    null
  );
  setCard(
    els.pnl, els.pnlNote,
    DASH_FMT.money(state.daily_pl, { sign: true }),
    state.closed_trades ? `${state.closed_trades} closed trade(s)` : "No closed trades yet",
    state.daily_pl
  );
  setCard(
    els.trades, els.tradesNote,
    state.open_trades === null || state.open_trades === undefined ? "—" : state.open_trades,
    "Open positions",
    null
  );
  setCard(
    els.winrate, els.winrateNote,
    state.win_rate === null || state.win_rate === undefined ? "—" : state.win_rate + "%",
    state.closed_trades
      ? `Across ${state.closed_trades} closed trade(s)`
      : "Needs closed trades",
    null
  );
}

function renderPositions(state) {
  const rows = (state.positions || []).map((p) => [
    p.symbol,
    `<span class="badge ${p.side === "LONG" ? "buy" : "sell"}">${p.side}</span>`,
    DASH_FMT.num(p.qty, 6),
    DASH_FMT.money(p.current),
    DASH_FMT.money(p.notional),
  ]);
  document.getElementById("positions").innerHTML = dashTable(
    [
      { label: "Symbol" },
      { label: "Side" },
      { label: "Qty", num: true },
      { label: "Mark", num: true },
      { label: "Notional", num: true },
    ],
    rows,
    state.running
      ? "Flat — no open positions right now."
      : "No paper session has run yet."
  );
}

function renderStatus(state, session) {
  const halted = !!state.trading_halted;
  const running = !!state.running;

  if (halted && state.halt_reason) {
    els.status.innerHTML = `<span class="badge sell">Halted</span> ${state.halt_reason}`;
  } else if (!running) {
    els.status.textContent = state.message || "Idle.";
  } else {
    els.status.textContent = [
      `Equity ${DASH_FMT.money(state.portfolio)}`,
      `Peak ${DASH_FMT.money(state.peak_equity)}`,
      `Drawdown ${DASH_FMT.pct(state.drawdown_pct)}`,
      `Exposure ${DASH_FMT.pct(state.open_exposure_pct)}`,
      `Updated ${DASH_FMT.time(state.timestamp)}`,
    ].join("\n");
  }

  els.startBtn.disabled = running;
  els.stopBtn.disabled = !running;

  if (session && session.last_error) {
    els.status.textContent += "\n\nLast cycle error:\n" + session.last_error;
  }
}

async function refresh() {
  const [health, state] = await Promise.all([loadHealth(), loadState()]);
  const session = health ? health.paper_session : null;

  renderCards(state);
  renderPositions(state);
  renderStatus(state, session);

  // Live session equity takes precedence once it has points; otherwise show
  // the backtest curve so the chart is never blank-but-fake.
  let points = [];
  if (state.running) {
    try {
      points = (await DASH.getJSON("/api/paper/equity")).points || [];
    } catch (_) {
      points = [];
    }
  }

  if (points.length > 1) {
    renderSessionChart(points);
    chartRendered = true;
  } else {
    const backtest = backtestCache || (await loadBacktest());
    if (backtest) {
      if (!backtestCache) els.banner.innerHTML = dashProvenanceBanner(backtest.data_source);
      renderBacktestChart(backtest);
      chartRendered = true;
    } else {
      els.chartTitle.textContent = "Equity";
      els.chartNote.textContent = "No equity data available yet.";
    }
  }
}

async function startBot() {
  els.actionMsg.textContent = "Starting paper session…";
  els.startBtn.disabled = true;
  try {
    const res = await DASH.postJSON("/api/paper/start", {});
    els.actionMsg.textContent = res.running
      ? `Paper session started (${res.symbols.join(", ")} @ ${res.timeframe}, polling every ${res.poll_seconds}s).`
      : "Session did not start.";
  } catch (err) {
    els.actionMsg.textContent = `Could not start: ${err.message}`;
  }
  await refresh();
}

async function stopBot() {
  els.actionMsg.textContent = "Stopping…";
  try {
    const res = await DASH.postJSON("/api/paper/stop", {});
    els.actionMsg.textContent = res.stopped ? "Paper session stopped." : res.message;
  } catch (err) {
    els.actionMsg.textContent = `Could not stop: ${err.message}`;
  }
  await refresh();
}

/**
 * One polling pass. Never rejects, so a single bad cycle cannot kill the
 * interval that would have recovered from it.
 */
async function tick() {
  try {
    const state = await loadState();
    // Repaint everything while the page is still incomplete: a session is
    // running, or no equity curve has been drawn yet (the backtest artifacts
    // may still be building on a cold container, which 404s until they land).
    // Once the dashboard is whole and idle, only the status panel changes and
    // polling the rest would be pure load.
    if (state.running || !chartRendered) await refresh();
    else renderStatus(state, null);
  } catch (err) {
    console.error("[dashboard] refresh failed:", err);
    els.banner.innerHTML = `<div class="banner danger">Could not render the dashboard: ${err.message}</div>`;
  }
}

async function init() {
  await whenLayoutReady();
  initEls();
  if (els.startBtn) els.startBtn.addEventListener("click", startBot);
  if (els.stopBtn) els.stopBtn.addEventListener("click", stopBot);

  // The timer is registered BEFORE the first pass and independently of it.
  // The first refresh can fail for reasons that resolve themselves, and it
  // used to be the only thing standing between the page and its polling loop,
  // so any single error left the dashboard on "Loading…" permanently.
  refreshTimer = setInterval(tick, 5000);
  await tick();
}

init();
