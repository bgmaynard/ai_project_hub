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