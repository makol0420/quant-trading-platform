// Injects the shared sidebar and topbar into every page, so navigation is
// defined once (components/sidebar.html) rather than duplicated per page.
//
// The previous version fetched /components/* from an API that never mounted
// that path, so every sub-page silently rendered an empty nav. The mount
// exists now (see api/main.py), and failures are logged rather than swallowed.

async function loadComponent(id, file) {
  const el = document.getElementById(id);
  if (!el) return;
  try {
    const res = await fetch(file);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${file}`);
    el.innerHTML = await res.text();
  } catch (err) {
    console.error("[layout] component load failed:", err);
    el.innerHTML = `<div class="banner danger">Navigation failed to load (${err.message}).</div>`;
  }
}

/** Keeps the topbar status pill in sync with the actual paper session. */
async function refreshSessionPill() {
  const pill = document.getElementById("session-state");
  if (!pill || typeof DASH === "undefined") return;
  try {
    const status = await DASH.getJSON("/api/paper/status");
    if (status.running) {
      pill.textContent = `🟢 Paper Trading · ${status.cycles || 0} cycles`;
      pill.className = "status";
    } else {
      pill.textContent = "⚪ Paper Trading (stopped)";
      pill.className = "status off";
    }
  } catch (_) {
    pill.textContent = "⚪ API unreachable";
    pill.className = "status off";
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([
    loadComponent("sidebar", "/components/sidebar.html"),
    loadComponent("topbar", "/components/topbar.html"),
  ]);

  const title = document.body.dataset.title;
  const titleEl = document.getElementById("page-title");
  if (title && titleEl) titleEl.textContent = title;
  if (typeof dashMarkActiveNav === "function") dashMarkActiveNav();

  await refreshSessionPill();
  setInterval(refreshSessionPill, 10000);

  // Page scripts wait on this rather than racing the injected markup.
  document.dispatchEvent(new CustomEvent("layout:ready"));
});
