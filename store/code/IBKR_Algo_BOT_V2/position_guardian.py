"""
Position Guardian
=================
Monitors ALL positions and enforces trailing stop protection.

Rules:
1. Track the HIGH WATER MARK for each position
2. If price drops from the high -> SELL immediately (using brain config thresholds)
3. After selling, watch for new momentum to re-enter
4. Works alongside any other trading bots

This runs as a separate process to protect ALL positions regardless of source.
Now loads trading rules from claude_trading_brain.json for persistent configuration.
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Try to load from Trading Brain (persistent config)
try:
    from ai.trading_brain import get_trading_brain
    brain = get_trading_brain()
    print("Trading Brain loaded - using persistent configuration")
except ImportError:
    brain = None
    print("Trading Brain not available - using defaults")

# Configuration - Load from brain config or use defaults
# MOMENTUM TRADING: Cut losers fast, only ride winners
def get_config():
    """Build config from brain or defaults - TIGHTENED FOR MOMENTUM ONLY"""
    if brain:
        return {
            # Percentage-based stops from brain (TIGHTENED)
            'TRAIL_PERCENT': brain.get_trailing_stop_percent() / 100,  # Convert from % to decimal
            'MIN_PROFIT_TO_TRAIL': 0.005,  # Start trailing after 0.5% profit (tighter)
            'HARD_STOP_PERCENT': brain.get_hard_stop_percent() / 100,
            'ZOMBIE_LOSS_PERCENT': brain.get_zombie_threshold() / 100,

            # Cent-based stops (for stocks < $1.00)
            'MIN_TRAIL_CENTS': 0.02,    # 2 cent trailing stop minimum
            'MIN_STOP_CENTS': 0.02,     # 2 cent hard stop minimum
            'MIN_ZOMBIE_CENTS': 0.04,   # 4 cent zombie threshold

            # Price tier thresholds
            'LOW_PRICE_THRESHOLD': 1.00, # Below this, use cent-based stops

            # SPREAD LIMIT from brain
            'MAX_SPREAD_CENTS': 0.03,   # $0.03 max spread - sell if wider (illiquid)

            # TIME-BASED EXITS - NO HOLD AND HOPE
            'STALE_POSITION_SECONDS': 300,  # 5 minutes - if not profitable, GET OUT
            'MAX_HOLD_SECONDS': 900,        # 15 minutes max hold for momentum plays

            'REENTRY_COOLDOWN': 300,    # 5 min cooldown before re-entering same symbol
            'CHECK_INTERVAL': 1.0,      # Check every 1 second
            'API_URL': 'http://localhost:9100/api/alpaca',
        }
    else:
        # Fallback defaults - AGGRESSIVE MOMENTUM SETTINGS
        return {
            'TRAIL_PERCENT': 0.01,       # 1% trailing (tightened from 1.5%)
            'MIN_PROFIT_TO_TRAIL': 0.005, # Start trailing at 0.5% profit
            'HARD_STOP_PERCENT': 0.01,   # 1% hard stop (tightened from 1.5%)
            'ZOMBIE_LOSS_PERCENT': 0.02, # 2% zombie threshold (tightened from 3%)
            'MIN_TRAIL_CENTS': 0.02,
            'MIN_STOP_CENTS': 0.02,
            'MIN_ZOMBIE_CENTS': 0.04,
            'LOW_PRICE_THRESHOLD': 1.00,
            'MAX_SPREAD_CENTS': 0.03,
            'STALE_POSITION_SECONDS': 300,  # 5 min - no progress = EXIT
            'MAX_HOLD_SECONDS': 900,        # 15 min max hold
            'REENTRY_COOLDOWN': 300,
            'CHECK_INTERVAL': 1.0,
            'API_URL': 'http://localhost:9100/api/alpaca',
        }

CONFIG = get_config()

# Persistent storage for high water marks
HWM_FILE = Path('store/position_guardian_hwm.json')


class PositionGuardian:
    """Guards all positions with trailing stops from their highs - MOMENTUM ONLY"""

    def __init__(self):
        self.api = CONFIG['API_URL']
        self.high_water_marks = {}  # symbol -> highest price seen
        self.entry_prices = {}      # symbol -> entry price
        self.entry_times = {}       # symbol -> timestamp when position was first seen
        self.cooldowns = {}         # symbol -> timestamp when can re-enter
        self.failed_sells = {}      # symbol -> timestamp of last failed sell attempt
        self.high_prices = {}       # symbol -> high price for fallback
        self.stats = {
            'protected_exits': 0,
            'total_saved': 0.0,
            'time_based_exits': 0,
        }
        self.load_hwm()

    def load_hwm(self):
        """Load high water marks from file"""
        if HWM_FILE.exists():
            try:
                data = json.loads(HWM_FILE.read_text())
                self.high_water_marks = data.get('hwm', {})
                self.entry_prices = data.get('entries', {})
                self.stats = data.get('stats', self.stats)
                self.log(f"Loaded HWM for {len(self.high_water_marks)} symbols")
            except:
                pass

    def save_hwm(self):
        """Save high water marks to file"""
        HWM_FILE.parent.mkdir(parents=True, exist_ok=True)
        HWM_FILE.write_text(json.dumps({
            'hwm': self.high_water_marks,
            'entries': self.entry_prices,
            'stats': self.stats,
            'updated': datetime.now().isoformat()
        }, indent=2))

    def log(self, msg):
        """Log with timestamp"""
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{ts} | GUARDIAN | {msg}")

    def get_positions(self):
        """Get all current positions"""
        try:
            r = requests.get(f"{self.api}/positions", timeout=5)
            if r.status_code == 200:
                data = r.json()
                # API returns {"positions": [...]} - extract the array
                if isinstance(data, dict):
                    return data.get('positions', [])
                return data if isinstance(data, list) else []
            return []
        except:
            return []

    def get_quote(self, symbol):
        """Get current price for a symbol"""
        try:
            r = requests.get(f"{self.api}/quote/{symbol}", timeout=2)
            if r.status_code == 200:
                data = r.json()
                return float(data.get('last', data.get('bid', 0)))
        except:
            pass
        return 0

    def get_spread(self, symbol):
        """Get bid/ask spread for a symbol"""
        try:
            r = requests.get(f"{self.api}/quote/{symbol}", timeout=2)
            if r.status_code == 200:
                data = r.json()
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                if bid > 0 and ask > 0:
                    return ask - bid
        except:
            pass
        return 0

    def get_open_orders(self):
        """Get all open orders to check for pending sells"""
        try:
            r = requests.get(f"{self.api}/orders", timeout=5)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return []

    def has_pending_order(self, symbol):
        """Check if symbol has a pending sell order"""
        orders = self.get_open_orders()
        for order in orders:
            if order.get('symbol') == symbol and order.get('side') == 'sell':
                return True
        return False

    def sell_position(self, symbol, qty, reason):
        """Sell a position using limit order at bid price for better fills"""
        try:
            # Check if there's already a pending sell order for this symbol
            if self.has_pending_order(symbol):
                self.log(f"SKIP {symbol}: pending sell order exists")
                # Set a cooldown to avoid checking too frequently
                self.failed_sells[symbol] = time.time()
                return False

            # Check if we recently failed to sell this symbol (wait 60 seconds)
            if symbol in self.failed_sells:
                if time.time() - self.failed_sells[symbol] < 60:
                    return False  # Silent skip, already logged

            # Get current bid price for limit order
            bid_price = None
            try:
                r = requests.get(f"{self.api}/quote/{symbol}", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    bid_price = float(data.get('bid', data.get('last', 0)))
            except:
                pass

            # Use limit order at bid for extended hours, market otherwise
            order_data = {
                'symbol': symbol,
                'quantity': int(abs(qty)),
                'action': 'sell',
                'time_in_force': 'day',
                'extended_hours': True
            }

            # ALWAYS use LIMIT orders - never market orders (per trading rules)
            if bid_price and bid_price > 0:
                # Use limit order at bid price (guaranteed fill)
                order_data['order_type'] = 'limit'
                order_data['limit_price'] = round(bid_price, 2)
                self.log(f"Placing LIMIT sell @ ${bid_price:.2f} (at bid)")
            else:
                # Fallback: Use current price * 0.99 for limit if no bid
                # Get current price from position if available
                current_price = self.high_prices.get(symbol, 1.0)
                fallback_limit = round(current_price * 0.99, 2)
                order_data['order_type'] = 'limit'
                order_data['limit_price'] = max(0.50, fallback_limit)  # Min $0.50
                self.log(f"Placing LIMIT sell @ ${fallback_limit:.2f} (fallback, no bid)")

            r = requests.post(f"{self.api}/place-order", json=order_data, timeout=5)
            response = r.json()

            if response.get('success'):
                self.log(f"SOLD {symbol} x{int(qty)} - {reason}")
                # Set cooldown
                self.cooldowns[symbol] = time.time() + CONFIG['REENTRY_COOLDOWN']
                # Clear failed sells tracking
                if symbol in self.failed_sells:
                    del self.failed_sells[symbol]
                return True
            else:
                # Check if it's an "insufficient qty" error (shares held for another order)
                detail = str(response.get('detail', ''))
                if 'insufficient qty' in detail or 'held_for_orders' in detail:
                    self.log(f"SKIP {symbol}: shares held for pending order")
                    self.failed_sells[symbol] = time.time()
                else:
                    self.log(f"Order failed: {response}")
                    self.failed_sells[symbol] = time.time()
        except Exception as e:
            self.log(f"Error selling {symbol}: {e}")
            self.failed_sells[symbol] = time.time()
        return False

    def check_positions(self):
        """Check all positions and enforce trailing stops"""
        positions = self.get_positions()

        # Track which symbols we currently hold
        current_symbols = set()

        for pos in positions:
            symbol = pos['symbol']
            qty = float(pos['quantity'])

            # Skip short positions
            if qty <= 0:
                continue

            current_symbols.add(symbol)

            entry_price = float(pos['avg_price'])
            current_price = float(pos['current_price'])

            # Initialize tracking if new position
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = entry_price
                self.high_water_marks[symbol] = current_price
                self.entry_times[symbol] = time.time()  # Track when we first saw this position
                self.high_prices[symbol] = current_price  # For fallback sell price
                self.log(f"Tracking NEW position: {symbol} @ ${entry_price:.2f}")

            # Update high water mark
            if current_price > self.high_water_marks.get(symbol, 0):
                old_hwm = self.high_water_marks.get(symbol, current_price)
                self.high_water_marks[symbol] = current_price
                if current_price > old_hwm * 1.005:  # Log significant new highs
                    self.log(f"NEW HIGH: {symbol} ${current_price:.2f} (was ${old_hwm:.2f})")

            # Calculate metrics
            hwm = self.high_water_marks[symbol]
            entry = self.entry_prices[symbol]

            pnl_from_entry = (current_price - entry) / entry
            drop_from_high = (hwm - current_price) / hwm

            # Only enforce trailing stop if we're in profit territory
            # OR if the drop from high is severe (>3% from ANY high)
            min_profit = CONFIG['MIN_PROFIT_TO_TRAIL']
            trail_pct = CONFIG['TRAIL_PERCENT']

            should_sell = False
            reason = ""

            # Rule 1: If in profit and drops 3% from high -> SELL
            if pnl_from_entry > min_profit and drop_from_high >= trail_pct:
                should_sell = True
                reason = f"TRAIL STOP: dropped {drop_from_high:.1%} from high ${hwm:.2f}"

            # Rule 2: If drops 3% from high even if not in profit (protect capital)
            elif drop_from_high >= trail_pct and hwm > entry * 1.01:
                # Only if we were at least 1% up at some point
                should_sell = True
                reason = f"PROTECT GAINS: was +{((hwm-entry)/entry):.1%}, now dropping"

            # Rule 3: Hard stop - never let it drop more than 5% from entry
            elif pnl_from_entry <= -CONFIG['HARD_STOP_PERCENT']:
                should_sell = True
                reason = f"HARD STOP: down {pnl_from_entry:.1%} from entry"

            # Rule 4: ZOMBIE KILLER - positions down >10% are dead, dump them
            # This catches orphan positions from crashes, bad trades, etc.
            elif pnl_from_entry <= -CONFIG['ZOMBIE_LOSS_PERCENT']:
                should_sell = True
                reason = f"ZOMBIE KILLER: position down {pnl_from_entry:.1%}, cutting losses"

            # Rule 5: WIDE SPREAD - illiquid stock, dump immediately
            # If spread > $0.03, you can't exit cleanly - get out now
            if not should_sell:
                spread = self.get_spread(symbol)
                max_spread = CONFIG['MAX_SPREAD_CENTS']
                if spread > max_spread:
                    should_sell = True
                    reason = f"WIDE SPREAD: ${spread:.2f} spread > ${max_spread:.2f} limit (illiquid)"

            # Rule 6: STALE POSITION - NO HOLD AND HOPE
            # If position is NOT profitable after 5 minutes, momentum failed - GET OUT
            if not should_sell and symbol in self.entry_times:
                position_age_seconds = time.time() - self.entry_times[symbol]
                stale_threshold = CONFIG.get('STALE_POSITION_SECONDS', 300)
                if position_age_seconds > stale_threshold and pnl_from_entry <= 0:
                    should_sell = True
                    reason = f"STALE POSITION: {position_age_seconds/60:.1f} min old, still at {pnl_from_entry:.1%} - MOMENTUM FAILED"
                    self.stats['time_based_exits'] += 1

            # Rule 7: MAX HOLD TIME - Momentum plays don't last forever
            # 15 minute max hold regardless of P/L (unless big winner)
            if not should_sell and symbol in self.entry_times:
                position_age_seconds = time.time() - self.entry_times[symbol]
                max_hold = CONFIG.get('MAX_HOLD_SECONDS', 900)
                # Only force exit if NOT a big winner (let winners run)
                if position_age_seconds > max_hold and pnl_from_entry < 0.03:  # < 3% gain
                    should_sell = True
                    reason = f"MAX HOLD TIME: {position_age_seconds/60:.1f} min > {max_hold/60:.0f} min limit - TIME TO EXIT"
                    self.stats['time_based_exits'] += 1

            if should_sell:
                self.log(f"EXIT SIGNAL: {symbol} @ ${current_price:.2f}")
                self.log(f"  Entry: ${entry:.2f} | High: ${hwm:.2f} | Now: ${current_price:.2f}")
                self.log(f"  Reason: {reason}")

                if self.sell_position(symbol, qty, reason):
                    self.stats['protected_exits'] += 1
                    # Calculate how much we saved by selling now vs holding
                    potential_loss_avoided = (hwm - current_price) * qty
                    self.stats['total_saved'] += potential_loss_avoided

                    # Clear tracking
                    if symbol in self.high_water_marks:
                        del self.high_water_marks[symbol]
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]
                    if symbol in self.entry_times:
                        del self.entry_times[symbol]
                    if symbol in self.high_prices:
                        del self.high_prices[symbol]

        # Clean up tracking for positions we no longer hold
        for symbol in list(self.high_water_marks.keys()):
            if symbol not in current_symbols:
                del self.high_water_marks[symbol]
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]
                if symbol in self.entry_times:
                    del self.entry_times[symbol]
                if symbol in self.high_prices:
                    del self.high_prices[symbol]

        # Save state periodically
        self.save_hwm()

    def startup_scan(self):
        """Scan all positions at startup and flag violations"""
        self.log("STARTUP SCAN - Checking all positions...")
        positions = self.get_positions()
        violations = []

        for pos in positions:
            symbol = pos['symbol']
            qty = float(pos['quantity'])
            if qty <= 0:
                continue

            entry = float(pos['avg_price'])
            current = float(pos['current_price'])
            pnl_pct = (current - entry) / entry

            # Flag any position in violation
            if pnl_pct <= -CONFIG['ZOMBIE_LOSS_PERCENT']:
                violations.append({
                    'symbol': symbol,
                    'qty': qty,
                    'entry': entry,
                    'current': current,
                    'pnl_pct': pnl_pct,
                    'reason': 'ZOMBIE (>10% loss)'
                })
            elif pnl_pct <= -CONFIG['HARD_STOP_PERCENT']:
                violations.append({
                    'symbol': symbol,
                    'qty': qty,
                    'entry': entry,
                    'current': current,
                    'pnl_pct': pnl_pct,
                    'reason': 'HARD STOP (>5% loss)'
                })

            # Initialize tracking for all positions
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = entry
                self.high_water_marks[symbol] = max(current, entry)
                self.log(f"  Tracking: {symbol} @ ${entry:.2f} ({pnl_pct:+.1%})")

        if violations:
            self.log(f"\n*** VIOLATIONS FOUND - {len(violations)} positions need immediate attention ***")
            for v in violations:
                self.log(f"  {v['symbol']}: ${v['current']:.2f} ({v['pnl_pct']:.1%}) - {v['reason']}")
                # Queue for immediate sale
                self.sell_position(v['symbol'], v['qty'], v['reason'])
        else:
            self.log("  All positions within limits")

        self.save_hwm()

    def run(self):
        """Main loop"""
        self.log("=" * 60)
        self.log("POSITION GUARDIAN STARTED - MOMENTUM ONLY MODE")
        self.log("=" * 60)
        self.log("RULES - NO HOLD AND HOPE:")
        self.log(f"  1. Trailing stop: {CONFIG['TRAIL_PERCENT']:.1%} from high")
        self.log(f"  2. Hard stop: {CONFIG['HARD_STOP_PERCENT']:.1%} loss from entry")
        self.log(f"  3. Zombie killer: {CONFIG['ZOMBIE_LOSS_PERCENT']:.1%} loss = immediate sell")
        self.log(f"  4. Max spread: ${CONFIG['MAX_SPREAD_CENTS']:.2f} - sell if wider")
        self.log(f"  5. STALE POSITION: {CONFIG.get('STALE_POSITION_SECONDS', 300)/60:.0f} min not profitable = EXIT")
        self.log(f"  6. MAX HOLD: {CONFIG.get('MAX_HOLD_SECONDS', 900)/60:.0f} min max (unless big winner)")
        self.log("=" * 60)

        # Immediate scan at startup
        self.startup_scan()

        cycle = 0
        while True:
            try:
                self.check_positions()

                cycle += 1
                if cycle % 60 == 0:  # Status every 60 seconds
                    positions = self.get_positions()
                    long_positions = [p for p in positions if float(p['quantity']) > 0]

                    if long_positions:
                        self.log(f"[{cycle}] Guarding {len(long_positions)} positions | "
                                f"Protected exits: {self.stats['protected_exits']}")
                        for p in long_positions:
                            sym = p['symbol']
                            hwm = self.high_water_marks.get(sym, 0)
                            curr = float(p['current_price'])
                            entry = self.entry_prices.get(sym, float(p['avg_price']))
                            pnl = (curr - entry) / entry * 100
                            drop = (hwm - curr) / hwm * 100 if hwm > 0 else 0
                            self.log(f"  {sym}: ${curr:.2f} | Entry: ${entry:.2f} ({pnl:+.1f}%) | "
                                    f"High: ${hwm:.2f} (drop: {drop:.1f}%)")
                    else:
                        self.log(f"[{cycle}] No positions to guard")

            except Exception as e:
                self.log(f"Error: {e}")

            time.sleep(CONFIG['CHECK_INTERVAL'])


if __name__ == '__main__':
    guardian = PositionGuardian()
    guardian.run()
