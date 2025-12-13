@echo off
title Warrior Trading Scanners
color 0E

echo ============================================================
echo   WARRIOR TRADING SCANNERS
echo   Sub-$8 + $2-20 Momentum + VWAP + Spread Monitor
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo [1/5] Checking API Server...
curl -s http://localhost:9100/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo     API Server not running. Starting it...
    start "API Server" cmd /k "python alpaca_dashboard_api.py"
    timeout /t 5 /nobreak > nul
)

echo [2/5] Starting Sub-$8 Penny Scanner...
start "Penny Scanner" cmd /c "python -c \"
from ai.penny_momentum_scanner import start_penny_scanner, PennyMover
import time, logging
logging.basicConfig(level=logging.INFO, format='%%(asctime)s | %%(message)s')

def alert(m):
    print(f'\\n*** PENNY MOVER: {m.symbol} @ ${m.price:.2f} | {m.signal} ***\\n')

scanner = start_penny_scanner(interval=5, on_mover=alert)
print(f'Scanning {len(scanner.watchlist)} stocks under $8...')
while True:
    time.sleep(10)
    signals = scanner.get_buy_signals()
    if signals: print(f'Buy signals: {[s[\"symbol\"] for s in signals]}')
\""

echo [3/5] Starting Warrior $2-20 Scanner...
start "Warrior Scanner" cmd /c "python -c \"
from ai.warrior_momentum_scanner import start_warrior_scanner, WarriorSetup
import time, logging
logging.basicConfig(level=logging.INFO, format='%%(asctime)s | %%(message)s')

def alert(s):
    print(f'\\n*** A+ SETUP: {s.symbol} @ ${s.price:.2f} | {s.setup_type} ***\\n')
    print(f'    Gap: {s.gap_pct:+.1f}%% | VWAP: {s.price_vs_vwap:+.1f}%% | Entry: {s.entry_zone}')

scanner = start_warrior_scanner(interval=10, on_setup=alert)
print(f'Scanning {len(scanner.watchlist)} stocks in $2-$20 range...')
while True:
    time.sleep(15)
    setups = scanner.get_a_setups()
    if setups: print(f'A-grade setups: {[s[\"symbol\"] for s in setups]}')
\""

echo [4/5] Starting VWAP Tracker...
start "VWAP Tracker" cmd /c "python -c \"
from ai.vwap_tracker import start_vwap_tracking, VWAPAlert
import time, requests, logging
logging.basicConfig(level=logging.INFO, format='%%(asctime)s | %%(message)s')

# Get positions to track
symbols = []
try:
    r = requests.get('http://localhost:9100/api/alpaca/positions', timeout=3)
    if r.status_code == 200:
        symbols = [p['symbol'] for p in r.json()]
except: pass

# Add common momentum stocks if no positions
if not symbols:
    symbols = ['BEAT', 'EDIT', 'SOFI', 'PLTR', 'NIO', 'LCID', 'RIVN']

def on_cross(alert):
    print(f'\\n*** VWAP {alert.alert_type.upper()}: {alert.message} ***\\n')

tracker = start_vwap_tracking(symbols=symbols, interval=2,
    on_cross_above=on_cross, on_cross_below=on_cross,
    on_reclaim=on_cross, on_rejection=on_cross)

print(f'Tracking VWAP for: {symbols}')
while True:
    time.sleep(10)
    buy_signals = tracker.get_buy_signals()
    if buy_signals: print(f'VWAP buy signals: {[s[\"symbol\"] for s in buy_signals]}')
\""

echo [5/5] Starting Spread Monitor...
start "Spread Monitor" cmd /c "python -c \"
from ai.spread_monitor import start_spread_monitoring, SpreadAlert
import time, logging
logging.basicConfig(level=logging.INFO, format='%%(asctime)s | %%(message)s')

def on_danger(alert):
    print(f'\\n*** DANGER: {alert.message} ***\\n')

def on_exit(alert):
    print(f'\\n*** AUTO-EXIT: {alert.message} ***\\n')

# Will auto-load positions
monitor = start_spread_monitoring(auto_exit=True, on_danger=on_danger, on_exit=on_exit)

print('Monitoring spreads... Will auto-exit on danger spreads')
while True:
    time.sleep(5)
    danger = monitor.get_danger_spreads()
    if danger: print(f'Wide spreads: {[d[\"symbol\"] for d in danger]}')
\""

echo.
echo ============================================================
echo   ALL SCANNERS STARTED
echo ============================================================
echo.
echo   Running:
echo     - Penny Scanner (sub-$8 momentum)
echo     - Warrior Scanner ($2-20 gap/VWAP plays)
echo     - VWAP Tracker (position monitoring)
echo     - Spread Monitor (auto-exit on wide spreads)
echo.
echo   Press any key to open dashboard...
pause > nul

start http://localhost:9100/ui/complete_platform.html
