/* ═══════════════════════════════════════════════════════════════════════════
   Def Space Platform — API Client
   All calls go to FastAPI backend; falls back to demo mode if offline.
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = (() => {
  const h = window.location.hostname;
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000';
  if (h.includes('render.com') || h.includes('railway.app'))
    return `${window.location.protocol}//${h}`;
  return '';  // same origin (nginx proxy)
})();

/** Generic fetch wrapper */
async function api(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'API error');
  }
  return res.json();
}

const DS = {
  health:   ()           => api('GET', '/api/health'),
  projects: ()           => api('GET', '/api/projects'),

  // P1
  p1: {
    calculate:   (body) => api('POST', '/api/p1/calculate', body),
    history:     ()     => api('GET', '/api/p1/history'),
    sensitivity: ()     => api('GET', '/api/p1/sensitivity'),
  },

  // P2
  p2: {
    calculate:   (body) => api('POST', '/api/p2/calculate', body),
    history:     ()     => api('GET', '/api/p2/history'),
    modelInfo:   ()     => api('GET', '/api/p2/model-info'),
  },

  // P3
  p3: {
    analyse: (body) => api('POST', '/api/p3/analyse', body),
    log:     (body) => api('POST', '/api/p3/log', body),
    logs:    (type) => api('GET', `/api/p3/logs${type ? '?log_type='+type : ''}`),
    specs:   ()     => api('GET', '/api/p3/specs'),
  },

  // P4
  p4: {
    calculate: (body) => api('POST', '/api/p4/calculate', body),
    history:   ()     => api('GET', '/api/p4/history'),
    modelInfo: ()     => api('GET', '/api/p4/model-info'),
  },

  // P5
  p5: {
    calculate: (body) => api('POST', '/api/p5/calculate', body),
    history:   ()     => api('GET', '/api/p5/history'),
    modelInfo: ()     => api('GET', '/api/p5/model-info'),
  },

  // P6
  p6: {
    analyse: (body) => api('POST', '/api/p6/analyse', body),
    history: ()     => api('GET', '/api/p6/history'),
  },
};

/* ── Tab switching helper ─────────────────────────────────────────────────── */
function initTabs(containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;
  const tabs  = container.querySelectorAll('.tab');
  const panes = container.querySelectorAll('.tab-pane');

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      if (panes[i]) panes[i].classList.add('active');
    });
  });
  if (tabs[0]) tabs[0].classList.add('active');
  if (panes[0]) panes[0].classList.add('active');
}

/* ── Chart.js default theme ──────────────────────────────────────────────── */
function dsChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = '#6B7A99';
  Chart.defaults.borderColor = '#1E2840';
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.padding = 16;
}

/* ── Number formatters ───────────────────────────────────────────────────── */
const fmt = {
  n: (v, d=3) => (v == null ? '—' : Number(v).toFixed(d)),
  pct: (v)    => (v == null ? '—' : Number(v).toFixed(1) + '%'),
  int: (v)    => (v == null ? '—' : Math.round(v).toLocaleString()),
  ms:  (v)    => (v == null ? '—' : Number(v).toFixed(2) + ' m/s'),
};

/* ── Show / hide loader ──────────────────────────────────────────────────── */
function showLoader(btn, text = 'Computing…') {
  if (!btn) return;
  btn._origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>${text}`;
}
function hideLoader(btn) {
  if (!btn) return;
  btn.disabled = false;
  btn.innerHTML = btn._origText || 'Calculate';
}

/* ── Toast notification ──────────────────────────────────────────────────── */
function toast(msg, type = 'info') {
  const t = document.createElement('div');
  const bg = { info: '#3D7FFF', ok: '#22C55E', err: '#F04040', warn: '#F0A500' };
  t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${bg[type]||bg.info};color:#fff;
    padding:10px 18px;border-radius:6px;font-size:13px;
    box-shadow:0 4px 20px rgba(0,0,0,.6);
    animation:fadeUp .25s ease;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

/* ── Badge builder ───────────────────────────────────────────────────────── */
function badge(text, type = 'blue') {
  return `<span class="badge badge-${type}">${text}</span>`;
}

/* ── Stable / Unstable badge ─────────────────────────────────────────────── */
function stabilityBadge(isStable, grade) {
  const type = isStable ? 'green' : 'red';
  const icon = isStable ? '✓' : '✗';
  return badge(`${icon} ${grade}`, type);
}
