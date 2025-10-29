// ---- global symbol bus ----
window.SymbolBus = new EventTarget();
window.setSymbol = (symbol) => {
  const s = (symbol || '').trim().toUpperCase();
  if (!s) return;
  localStorage.setItem('gtp.activeSymbol', s);
  SymbolBus.dispatchEvent(new CustomEvent('symbol', { detail: { symbol: s } }));
};
window.getSymbol = () => localStorage.getItem('gtp.activeSymbol') || 'AAPL';

// Fire once on load so all modules initialize
window.addEventListener('DOMContentLoaded', () => setSymbol(getSymbol()));

SymbolBus.addEventListener('symbol', (e) => {
  const sym = e.detail.symbol;
  // update header labels
  const hdr = document.querySelector('[data-role="module-title"]');
  if (hdr) hdr.textContent = `${hdr.dataset.kind || 'Module'} — ${sym}`;

  // reload module data for "sym" here...
  // e.g., fetchLevel2(sym), fetchTimeAndSales(sym), reloadChart(sym), etc.
});

// watchlist row click → publish
function onWatchlistClick(sym) {
  setSymbol(sym);
}

// montage input (on Enter) → publish
document.getElementById('montage-symbol-input').addEventListener('keydown', (e)=>{
  if (e.key === 'Enter') setSymbol(e.target.value);
});

const BRIDGE = 'http://127.0.0.1:9101';  // FastAPI dashboard_api
const H = { 'Content-Type': 'application/json', 'X-API-Key': 'My_Super_Strong_Key_123' };

async function previewOrder(symbol, side, qty){
  const res = await fetch(`${BRIDGE}/api/order/preview`, {
    method: 'POST', headers: H, body: JSON.stringify({ symbol, side, qty })
  });
  if (!res.ok) throw new Error(`Preview failed: ${res.status}`);
  return await res.json();
}

async function placeOrder(symbol, side, qty){
  const res = await fetch(`${BRIDGE}/api/order/place`, {
    method: 'POST', headers: H, body: JSON.stringify({ symbol, side, qty })
  });
  if (!res.ok) throw new Error(`Order failed: ${res.status}`);
  return await res.json();
}

document.getElementById('btn-buy').addEventListener('click', async ()=>{
  const sym = getSymbol(); const qty = Number(document.getElementById('qty').value || 100);
  try {
    const prev = await previewOrder(sym, 'BUY', qty);
    // show preview UI if you have one…
    const out = await placeOrder(sym, 'BUY', qty);
    console.log('ORDER BUY:', out);
  } catch(e){ console.error(e); }
});

document.getElementById('btn-sell').addEventListener('click', async ()=>{
  const sym = getSymbol(); const qty = Number(document.getElementById('qty').value || 100);
  try {
    const prev = await previewOrder(sym, 'SELL', qty);
    const out = await placeOrder(sym, 'SELL', qty);
    console.log('ORDER SELL:', out);
  } catch(e){ console.error(e); }
});

// Trailing stop button — basic client rule -> call your bridge to place OCO/OCA in real impl
document.getElementById('btn-trailing-stop').addEventListener('click', async ()=>{
  const sym = getSymbol();
  // Example: just preview a SELL (you can extend with stop params on your API later)
  try { await previewOrder(sym, 'SELL', 100); } catch(e){ console.error(e); }
});



const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function pingStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    $('statusText').textContent = data.connected ? 'connected' : 'disconnected';
    $('status').className = 'status ' + (data.connected ? 'ok' : 'bad');
  } catch (e) {
    $('statusText').textContent = 'error';
    $('status').className = 'status bad';
  }
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}),
  });
  const txt = await res.text();
  try { return JSON.parse(txt); } catch { return {raw: txt}; }
}

function appendLog(el, obj) {
  const line = (typeof obj === 'string') ? obj : JSON.stringify(obj);
  el.textContent += (line + "\n");
  el.scrollTop = el.scrollHeight;
}

async function startMtf() {
  const symbol = $('mtfSymbol').value.trim() || 'SPY';
  const out = await postJSON('/api/bots/mtf/start', { symbol });
  appendLog($('mtfLog'), out);
}

async function stopMtf() {
  const out = await postJSON('/api/bots/mtf/stop', {});
  appendLog($('mtfLog'), out);
}

async function startWarrior() {
  const symbol = $('warriorSymbol').value.trim() || 'AAPL';
  const out = await postJSON('/api/bots/warrior/start', { symbol });
  appendLog($('warriorLog'), out);
}

async function stopWarrior() {
  const out = await postJSON('/api/bots/warrior/stop', {});
  appendLog($('warriorLog'), out);
}

async function tailSignals() {
  try {
    const res = await fetch('/api/signals/tail'); // optional endpoint to add
    const data = await res.text();
    $('signalsTail').textContent = data || '(no data)';
  } catch (e) {
    $('signalsTail').textContent = 'error fetching signals';
  }
}

let autoTailTimer = null;
function toggleAutoTail() {
  if (autoTailTimer) {
    clearInterval(autoTailTimer);
    autoTailTimer = null;
    $('btnAutoTail').classList.remove('active');
    $('btnAutoTail').textContent = 'Auto (3s)';
  } else {
    autoTailTimer = setInterval(tailSignals, 3000);
    $('btnAutoTail').classList.add('active');
    $('btnAutoTail').textContent = 'Auto: ON';
  }
}

function wireEvents() {
  $('btnMtfStart').addEventListener('click', startMtf);
  $('btnMtfStop').addEventListener('click', stopMtf);
  $('btnWarriorStart').addEventListener('click', startWarrior);
  $('btnWarriorStop').addEventListener('click', stopWarrior);
  $('btnTail').addEventListener('click', tailSignals);
  $('btnAutoTail').addEventListener('click', toggleAutoTail);
}

(async function init(){
  wireEvents();
  await pingStatus();
  // soft refresh of status every 10s
  setInterval(pingStatus, 10000);
})();