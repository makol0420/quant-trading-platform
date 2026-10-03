// Shared helpers for the dashboard and the sub-pages.
//
// The formatting rule that matters: anything the API returns as null is
// rendered as "—", never as 0 or a guess. The previous dashboard showed a
// hardcoded 71% win rate and $125 daily P/L with nothing behind them, which
// is precisely the kind of invented number this platform's README argues
// against. A missing value must look missing.

const DASH = {
  async getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        if (body && body.detail) detail = body.detail;
      } catch (_) {
        /* non-JSON error body; keep the status line */
      }
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    return res.json();
  },

  async postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.detail || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  },
};

const DASH_FMT = {
  /** Money with an explicit sign option. null/undefined -> em dash. */
  money(value, { sign = false, decimals = 2 } = {}) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    const n = Number(value);
    const s = n < 0 ? "-" : sign ? "+" : "";
    return (
      s +
      "$" +
      Math.abs(n).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    );
  },

  num(value, decimals = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  },

  pct(value, decimals = 2, { sign = false } = {}) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    const n = Number(value);
    return (sign && n > 0 ? "+" : "") + n.toFixed(decimals) + "%";
  },

  ratioPct(value, decimals = 2) {
    // For values already expressed as 0..1 fractions.
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return (Number(value) * 100).toFixed(decimals) + "%";
  },

  text(value, fallback = "—") {
    if (value === null || value === undefined || value === "") return fallback;
    return String(value);
  },

  /** Color class for a signed number, for cards. */
  signClass(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    return Number(value) < 0 ? "neg" : Number(value) > 0 ? "pos" : "";
  },

  time(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
  },

  shortTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  },
};

/** HTML for a data-provenance badge, from a backtest payload's data_source. */
function dashProvenanceBadge(dataSource) {
  if (!dataSource) return "";
  return dataSource.is_real_data
    ? '<span class="badge real">Real market data</span>'
    : '<span class="badge synthetic">Synthetic data</span>';
}

/**
 * Banner describing where the numbers came from. Every page that shows
 * performance figures includes this, so a synthetic run can never be
 * mistaken for a real one.
 */
function dashProvenanceBanner(dataSource) {
  if (!dataSource) return "";
  const span = `${DASH_FMT.shortTime(dataSource.start)} → ${DASH_FMT.shortTime(dataSource.end)}`;
  const src = (dataSource.providers || []).join(", ") || "unknown";
  if (dataSource.is_real_data) {
    return `<div class="banner ok">${dashProvenanceBadge(dataSource)}
      Real exchange data via <b>${src}</b> — ${DASH_FMT.num(dataSource.n_bars, 0)} bars
      of ${DASH_FMT.text(dataSource.timeframe)} history (${span}). Fees and slippage are
      applied per side; predictions are walk-forward out-of-fold.</div>`;
  }
  return `<div class="banner warn">${dashProvenanceBadge(dataSource)}
    <b>Synthetic data.</b> These bars come from a regime-switching random walk
    (<code>data/providers/synthetic.py</code>), not a real market. Useful for
    verifying the pipeline; meaningless as a performance claim.</div>`;
}

/** Renders a table from headers + rows, or a muted note when there's no data. */
function dashTable(headers, rows, emptyMessage) {
  if (!rows || rows.length === 0) {
    return `<p class="muted">${emptyMessage || "No data."}</p>`;
  }
  const head = headers
    .map((h) => `<th class="${h.num ? "num" : ""}">${h.label}</th>`)
    .join("");
  const body = rows
    .map(
      (row) =>
        "<tr>" +
        row
          .map((cell, i) => `<td class="${headers[i].num ? "num" : ""}">${cell}</td>`)
          .join("") +
        "</tr>"
    )
    .join("");
  return `<div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/** Paints the shared sidebar/topbar active state after layout.js injects them. */
function dashMarkActiveNav() {
  const path = window.location.pathname.replace(/\/$/, "") || "/";
  document.querySelectorAll(".sidebar nav a").forEach((a) => {
    const href = a.getAttribute("href");
    if (!href) return;
    const norm = href.replace(/\/$/, "") || "/";
    if (norm === path) a.classList.add("active");
    else a.classList.remove("active");
  });
}
