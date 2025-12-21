"""
AUTO SCAN & TRADE
=================
Automatically scans for momentum stocks and trades them.

1. Scans multiple sources for movers
2. Filters for scalping criteria
3. Adds top picks to watchlist
4. Starts scalper in paper mode
5. Monitors and trades automatically

Run: python auto_scan_trade.py

For pre-market: Run at 4:00 AM ET
"""
import requests
import time
import json
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

API_BASE = "http://localhost:9100"


def check_server() -> bool:
    """Check if Morpheus server is running"""
    try:
        r = requests.get(f"{API_BASE}/api/status", timeout=5)
        return r.status_code == 200
    except:
        return False


def get_yahoo_movers() -> List[Dict]:
    """Get movers from Yahoo Finance"""
    try:
        from yahooquery import Screener
        s = Screener()

        all_movers = []

        # Get gainers
        data = s.get_screeners(['day_gainers'], count=30)
        for q in data.get('day_gainers', {}).get('quotes', []):
            all_movers.append({
                'symbol': q.get('symbol'),
                'price': q.get('regularMarketPrice', 0),
                'change_pct': q.get('regularMarketChangePercent', 0),
                'volume': q.get('regularMarketVolume', 0),
                'source': 'yahoo'
            })

        # Get most active (might have momentum)
        data = s.get_screeners(['most_actives'], count=30)
        for q in data.get('most_actives', {}).get('quotes', []):
            sym = q.get('symbol')
            if sym not in [m['symbol'] for m in all_movers]:
                change = q.get('regularMarketChangePercent', 0)
                if change > 0:  # Only gainers
                    all_movers.append({
                        'symbol': sym,
                        'price': q.get('regularMarketPrice', 0),
                        'change_pct': change,
                        'volume': q.get('regularMarketVolume', 0),
                        'source': 'yahoo'
                    })

        return all_movers
    except Exception as e:
        log.error(f"Yahoo error: {e}")
        return []


def get_schwab_movers() -> List[Dict]:
    """Get movers from Schwab API"""
    try:
        r = requests.get(f"{API_BASE}/api/market/movers?direction=all", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get('gainers', [])
    except Exception as e:
        log.warning(f"Schwab movers not available: {e}")
    return []


def filter_scalp_candidates(movers: List[Dict]) -> List[Dict]:
    """Filter for scalping criteria"""
    candidates = []

    for m in movers:
        price = m.get('price', 0)
        change = m.get('change_pct', 0)
        volume = m.get('volume', 0)

        # Scalping criteria
        if 1.0 <= price <= 20.0:  # Price range
            if change >= 5.0:  # Min 5% gap
                if volume >= 500000:  # Min volume
                    m['score'] = int(change * 2 + (volume / 1000000))
                    candidates.append(m)

    # Sort by score
    return sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)


def get_default_watchlist_id() -> str:
    """Get the default watchlist ID"""
    try:
        r = requests.get(f"{API_BASE}/api/watchlists", timeout=5)
        if r.status_code == 200:
            data = r.json()
            watchlists = data.get('watchlists', [])
            for wl in watchlists:
                if wl.get('is_default'):
                    return wl.get('id')
            # Return first if no default
            if watchlists:
                return watchlists[0].get('id')
    except:
        pass
    return "default"


def add_symbols_to_watchlist(symbols: List[str]) -> bool:
    """Add multiple symbols to the default watchlist"""
    try:
        wl_id = get_default_watchlist_id()
        r = requests.post(
            f"{API_BASE}/api/watchlists/{wl_id}/symbols",
            json={"symbols": symbols},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        log.error(f"Error adding symbols: {e}")
        return False


def add_to_scalper_watchlist(symbols: List[str]) -> bool:
    """Add symbols directly to the scalper watchlist"""
    try:
        for sym in symbols:
            r = requests.post(
                f"{API_BASE}/api/scanner/scalper/watchlist/add/{sym}",
                timeout=5
            )
            if r.status_code == 200:
                log.info(f"  Added to scalper: {sym}")
        return True
    except Exception as e:
        log.error(f"Error adding to scalper: {e}")
        return False


def add_to_dashboard_worklist(symbols: List[str]) -> bool:
    """Add symbols to the dashboard worklist (what user sees on UI)"""
    try:
        for sym in symbols:
            r = requests.post(
                f"{API_BASE}/api/worklist/add",
                json={"symbol": sym},
                timeout=5
            )
            if r.status_code == 200:
                log.info(f"  Added to dashboard: {sym}")
            else:
                log.warning(f"  Failed to add {sym} to dashboard: {r.status_code}")
        return True
    except Exception as e:
        log.error(f"Error adding to dashboard: {e}")
        return False


def clear_watchlist() -> bool:
    """Clear scalper watchlist"""
    try:
        # Get current scalper watchlist
        r = requests.get(f"{API_BASE}/api/scanner/scalper/config", timeout=5)
        if r.status_code == 200:
            data = r.json()
            current = data.get('config', {}).get('watchlist', [])
            for sym in current:
                requests.delete(f"{API_BASE}/api/scanner/scalper/watchlist/{sym}", timeout=5)
        return True
    except:
        return False


def start_scalper(paper_mode: bool = True) -> bool:
    """Start the HFT scalper"""
    try:
        # Start monitoring
        r = requests.post(f"{API_BASE}/api/scanner/scalper/start", timeout=5)
        if r.status_code != 200:
            log.error("Failed to start scalper")
            return False

        # Enable trading
        mode = "true" if paper_mode else "false"
        r = requests.post(f"{API_BASE}/api/scanner/scalper/enable?paper_mode={mode}", timeout=5)
        return r.status_code == 200
    except Exception as e:
        log.error(f"Error starting scalper: {e}")
        return False


def get_scalper_status() -> Dict:
    """Get scalper status"""
    try:
        r = requests.get(f"{API_BASE}/api/scanner/scalper/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}


def run_scan_and_trade(max_symbols: int = 5, paper_mode: bool = True):
    """
    Main function: Scan for movers and start trading
    """
    print("=" * 60)
    print("         AUTO SCAN & TRADE")
    print(f"         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Check server
    log.info("Checking Morpheus server...")
    if not check_server():
        log.error("Server not running! Start with: python morpheus_trading_api.py")
        return False
    log.info("Server: OK")

    # Scan for movers
    log.info("Scanning for momentum stocks...")

    all_movers = []

    # Try Schwab first
    schwab = get_schwab_movers()
    if schwab:
        log.info(f"Schwab: {len(schwab)} movers")
        all_movers.extend(schwab)
    else:
        log.info("Schwab: No data (market closed?)")

    # Get Yahoo movers
    yahoo = get_yahoo_movers()
    if yahoo:
        log.info(f"Yahoo: {len(yahoo)} movers")
        # Add unique symbols
        existing = [m['symbol'] for m in all_movers]
        for m in yahoo:
            if m['symbol'] not in existing:
                all_movers.append(m)

    if not all_movers:
        log.warning("No movers found from any source")
        return False

    # Filter for scalping
    log.info("Filtering for scalp candidates...")
    candidates = filter_scalp_candidates(all_movers)

    if not candidates:
        log.warning("No stocks meet scalping criteria (Price $1-$20, Gap 5%+, Vol 500K+)")
        return False

    log.info(f"Found {len(candidates)} scalp candidates")

    # Show top picks
    print()
    print("TOP SCALP PICKS:")
    print("-" * 60)
    print(f"{'Symbol':<8} {'Price':>8} {'Change':>8} {'Volume':>12} {'Score':>6}")
    print("-" * 60)

    for c in candidates[:max_symbols]:
        print(f"{c['symbol']:<8} ${c['price']:>7.2f} {c['change_pct']:>+7.1f}% {c['volume']:>12,} {c['score']:>6}")

    print()

    # Clear old watchlist and add new picks
    log.info("Updating watchlists...")
    clear_watchlist()

    symbols_to_add = [c['symbol'] for c in candidates[:max_symbols]]

    # Add to BOTH watchlists so user sees them
    log.info("Adding to scalper watchlist (for trading)...")
    add_to_scalper_watchlist(symbols_to_add)

    log.info("Adding to dashboard worklist (for visibility)...")
    add_to_dashboard_worklist(symbols_to_add)

    added = symbols_to_add

    if not added:
        log.error("Failed to add any symbols to watchlist")
        return False

    # Start scalper
    mode_str = "PAPER" if paper_mode else "LIVE"
    log.info(f"Starting scalper in {mode_str} mode...")

    if start_scalper(paper_mode):
        log.info("Scalper started successfully!")
    else:
        log.error("Failed to start scalper")
        return False

    # Show status
    print()
    print("=" * 60)
    print("TRADING ACTIVE")
    print("=" * 60)
    print(f"  Mode:      {mode_str}")
    print(f"  Watchlist: {', '.join(added)}")
    print(f"  Criteria:  7% spike, 5x volume, 2% stop, 3% target")
    print()
    print("  Monitor with:")
    print("    python monitor.py")
    print("    curl http://localhost:9100/api/scanner/scalper/status")
    print()
    print("  Stop with:")
    print("    curl -X POST http://localhost:9100/api/scanner/scalper/stop")
    print("=" * 60)

    return True


def continuous_scan(interval_minutes: int = 5, paper_mode: bool = True):
    """
    Continuously scan and update watchlist
    """
    log.info(f"Starting continuous scan (every {interval_minutes} min)")
    log.info("Press Ctrl+C to stop")

    while True:
        try:
            run_scan_and_trade(max_symbols=5, paper_mode=paper_mode)

            # Wait for next scan
            log.info(f"Next scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            log.info("Stopped by user")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(60)  # Wait 1 min on error


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Continuous mode: scan every 5 minutes
        continuous_scan(interval_minutes=5, paper_mode=True)
    else:
        # One-time scan and trade
        run_scan_and_trade(max_symbols=5, paper_mode=True)
