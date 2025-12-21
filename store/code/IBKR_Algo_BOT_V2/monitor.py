"""
MORPHEUS SYSTEM MONITOR
Run: python monitor.py
"""
import requests
import json
import sys

def main():
    print('='*60)
    print('            MORPHEUS SYSTEM MONITOR')
    print('='*60)
    print()

    # Core Status
    try:
        r = requests.get('http://localhost:9100/api/status', timeout=5)
        d = r.json()
        print('[CORE SYSTEM]')
        print(f'  Status:     {d["status"].upper()}')
        print(f'  Broker:     {d["broker"]} (Connected: {d["broker_connected"]})')
        print(f'  Paper Mode: {d["paper_mode"]}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # Scalper
    try:
        r = requests.get('http://localhost:9100/api/scanner/scalper/status', timeout=5)
        d = r.json()
        print('[HFT SCALPER]')
        print(f'  Running:    {"YES" if d["is_running"] else "NO"}')
        print(f'  Trading:    {"ENABLED" if d["enabled"] else "DISABLED"}')
        print(f'  Mode:       {"PAPER" if d["paper_mode"] else "LIVE"}')
        print(f'  Watchlist:  {d["watchlist_count"]} symbols')
        print(f'  Positions:  {d["open_positions"]}')
        print(f'  Trades:     {d["daily_trades"]} today')
        print(f'  Daily PnL:  ${d["daily_pnl"]:.2f}')
        print(f'  Risk/Trade: ${d["risk_management"]["risk_per_trade"]:.2f}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # News Trader
    try:
        r = requests.get('http://localhost:9100/api/scanner/news-trader/status', timeout=5)
        d = r.json()
        print('[NEWS TRADER]')
        print(f'  Enabled:    {"YES" if d["enabled"] else "NO"}')
        print(f'  Running:    {"YES" if d["is_running"] else "NO"}')
        if d.get('stats'):
            print(f'  News Recv:  {d["stats"]["news_received"]}')
            print(f'  Candidates: {d["stats"]["candidates_evaluated"]}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # Scalper Config (Filters)
    try:
        r = requests.get('http://localhost:9100/api/scanner/scalper/config', timeout=5)
        d = r.json().get('config', {})
        print('[ACTIVE FILTERS]')
        filters = [
            ('Chronos', d.get('use_chronos_filter', False)),
            ('Order Flow', d.get('use_order_flow_filter', False)),
            ('Regime Gate', d.get('use_regime_gating', False)),
            ('Scalp Fade', d.get('use_scalp_fade_filter', False)),
            ('ATR Stops', d.get('use_atr_stops', False)),
        ]
        for name, enabled in filters:
            status = "ON" if enabled else "OFF"
            print(f'  {name:12} {status}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # Watchlist with AI Signals
    try:
        r = requests.get('http://localhost:9100/api/worklist', timeout=5)
        d = r.json()
        print('[WATCHLIST]')
        print(f'  {"Symbol":<6} {"Price":>8} {"Change":>8} {"AI Signal":<16} {"Entry"}')
        print(f'  {"-"*6} {"-"*8} {"-"*8} {"-"*16} {"-"*6}')
        for s in d.get('data', []):
            signal = s.get('ai_signal', 'UNKNOWN')
            entry = s.get('entry_status', '?')
            print(f'  {s["symbol"]:<6} ${s["price"]:>7.2f} {s["change_percent"]:>+7.1f}% {signal:<16} {entry}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # Recent Trades
    try:
        r = requests.get('http://localhost:9100/api/scanner/scalper/trades', timeout=5)
        d = r.json()
        trades = d.get('trades', [])[:5]  # Last 5
        if trades:
            print('[RECENT TRADES]')
            for t in trades:
                pnl = t.get('pnl', 0)
                symbol = t.get('symbol', '?')
                exit_reason = t.get('exit_reason', '?')
                emoji = '+' if pnl >= 0 else ''
                print(f'  {symbol:<6} ${pnl:>+7.2f}  {exit_reason}')
    except Exception as e:
        print(f'  Error: {e}')
    print()

    # Stats
    try:
        r = requests.get('http://localhost:9100/api/scanner/scalper/stats', timeout=5)
        d = r.json()
        print('[LIFETIME STATS]')
        print(f'  Total Trades: {d["total_trades"]}')
        print(f'  Win Rate:     {d["win_rate"]:.1f}%')
        print(f'  Total PnL:    ${d["total_pnl"]:.2f}')
        print(f'  Avg Win:      ${d["avg_win"]:.2f}')
        print(f'  Avg Loss:     ${d["avg_loss"]:.2f}')
        print(f'  Profit Factor: {d["profit_factor"]:.2f}')
    except Exception as e:
        print(f'  Error: {e}')

    print()
    print('='*60)

if __name__ == '__main__':
    main()
