@echo off
title News Trading Workflow
color 0B

echo ============================================================
echo   COMPLETE NEWS TRADING WORKFLOW
echo ============================================================
echo.
echo   Components:
echo     - Benzinga Breaking News (1s polling)
echo     - Spike Validation (filters algo noise)
echo     - Intelligent Watchlist (ML triggers)
echo     - Time and Sales (color-coded buy/sell)
echo     - Level 2 Order Book
echo     - Jacknife/Whipsaw Protection
echo.
echo ============================================================
echo.

cd /d C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2

echo [1/3] Checking API Server...
curl -s http://localhost:9100/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo     API Server not running. Starting it...
    start "API Server" cmd /k "python morpheus_trading_api.py"
    timeout /t 8 /nobreak > nul
    echo     API Server started.
) else (
    echo     API Server already running.
)

echo.
echo [2/3] Starting Trading Workflow...
echo.

start "News Trading Workflow" cmd /c "python -c \"
import sys
sys.path.insert(0, '.')
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s | %%(levelname)-7s | %%(message)s',
    datefmt='%%H:%%M:%%S'
)
logger = logging.getLogger()

print()
print('='*60)
print('  NEWS TRADING WORKFLOW')
print('='*60)
print()

# Callbacks
def on_news(signal):
    color = '\\033[92m' if signal.action == 'BUY' else '\\033[93m' if signal.action == 'WAIT' else '\\033[91m'
    reset = '\\033[0m'
    print(f'{color}')
    print(f'  NEWS SIGNAL: {signal.action} {signal.symbol}')
    print(f'  Headline: {signal.headline[:50]}...')
    print(f'  Entry: \${signal.entry_price:.2f} | Target: \${signal.target_price:.2f} | Stop: \${signal.stop_price:.2f}')
    if signal.validation:
        print(f'  Validation: {signal.validation.verdict} (score={signal.validation.overall_score})')
    print(f'{reset}')

def on_qualified(entry):
    print(f'\\033[92m')
    print(f'  QUALIFIED: {entry.symbol} @ \${entry.current_price:.2f}')
    print(f'  Source: {entry.discovery_source} | Catalyst: {entry.catalyst}')
    print(f'\\033[0m')

def on_trade(signal, result):
    print(f'\\033[96m')
    print(f'  TRADE EXECUTED: {signal.symbol}')
    print(f'  Size: {signal.position_size} shares @ \${signal.entry_price:.2f}')
    print(f'\\033[0m')

def on_jacknife(alert):
    print(f'\\033[91m')
    print(f'  JACKNIFE ALERT: {alert.message}')
    print(f'  Action: {alert.action_taken}')
    print(f'\\033[0m')

# Start workflow
from ai.trading_workflow import start_trading_workflow, stop_trading_workflow

# Momentum watchlist
watchlist = ['TSLA', 'NVDA', 'AMD', 'META', 'SOFI', 'PLTR', 'NIO', 'LCID']

print(f'Starting with watchlist: {watchlist}')
print()

workflow = start_trading_workflow(
    watchlist=watchlist,
    auto_trade=False,  # Set True to auto-execute trades
    require_validation=True,
    require_training=False,
    on_news=on_news,
    on_qualified=on_qualified,
    on_trade=on_trade,
    on_jacknife=on_jacknife
)

print()
print('Workflow running. Monitoring for:')
print('  - Breaking news (Benzinga)')
print('  - Spike validation')
print('  - Jacknife/reversal detection')
print()
print('Press Ctrl+C to stop')
print()
print('-'*60)

try:
    while True:
        time.sleep(15)
        status = workflow.get_status()

        print(f\"\\r[STATUS] News: {status['stats']['news_detected']} | \"\n              f\"Qualified: {status['stats']['stocks_qualified']} | \"\n              f\"Trades: {status['stats']['trades_executed']} | \"\n              f\"Whipsaws Avoided: {status['stats']['whipsaws_avoided']}\", end='')

        # Show watchlist
        wl = workflow.get_watchlist()
        if wl:
            print(f'\\n  Watchlist ({len(wl)}): {[w[\"symbol\"] for w in wl[:5]]}')

except KeyboardInterrupt:
    print('\\n\\nStopping workflow...')
    stop_trading_workflow()
    print('Done.')
\""

echo.
echo [3/3] Opening Dashboard...
timeout /t 3 /nobreak > nul
start http://localhost:9100/ui/complete_platform.html

echo.
echo ============================================================
echo   WORKFLOW STARTED
echo ============================================================
echo.
echo   API Endpoints Available:
echo     - GET  /api/workflow/status     - Workflow status
echo     - POST /api/workflow/start      - Start workflow
echo     - GET  /api/tape/trades/{sym}   - Time and Sales
echo     - GET  /api/level2/{sym}        - Level 2 Order Book
echo     - GET  /api/tape/flow/{sym}     - Buy/Sell Flow
echo     - GET  /api/watchlist/intelligent - Smart Watchlist
echo.
echo   Press any key to close this window...
pause > nul
