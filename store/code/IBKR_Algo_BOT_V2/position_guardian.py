"""
Position Guardian
=================
Monitors ALL positions and enforces trailing stop protection.

Rules:
1. Track the HIGH WATER MARK for each position
2. If price drops 3% from the high -> SELL immediately
3. After selling, watch for new momentum to re-enter
4. Works alongside any other trading bots

This runs as a separate process to protect ALL positions regardless of source.
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Configuration - Aligned with 2:1 Profit/Loss Ratio
# Optimized for low-priced momentum stocks ($0.20-$0.40 range)
CONFIG = {
    # Percentage-based stops (for stocks > $1.00)
    'TRAIL_PERCENT': 0.015,     # 1.5% trailing stop from high
    'MIN_PROFIT_TO_TRAIL': 0.01, # Start trailing after 1% profit
    'HARD_STOP_PERCENT': 0.015, # 1.5% hard stop from entry
    'ZOMBIE_LOSS_PERCENT': 0.03, # 3% loss = zombie position

    # Cent-based stops (for stocks < $1.00)
    'MIN_TRAIL_CENTS': 0.02,    # 2 cent trailing stop minimum
    'MIN_STOP_CENTS': 0.02,     # 2 cent hard stop minimum
    'MIN_ZOMBIE_CENTS': 0.04,   # 4 cent zombie threshold

    # Price tier thresholds
    'LOW_PRICE_THRESHOLD': 1.00, # Below this, use cent-based stops

    'REENTRY_COOLDOWN': 300,    # 5 min cooldown before re-entering same symbol
    'CHECK_INTERVAL': 1.0,      # Check every 1 second
    'API_URL': 'http://localhost:9100/api/alpaca',
}

# Persistent storage for high water marks
HWM_FILE = Path('store/position_guardian_hwm.json')


class PositionGuardian:
    """Guards all positions with trailing stops from their highs"""

    def __init__(self):
        self.api = CONFIG['API_URL']
        self.high_water_marks = {}  # symbol -> highest price seen
        self.entry_prices = {}      # symbol -> entry price
        self.cooldowns = {}         # symbol -> timestamp when can re-enter
        self.stats = {
            'protected_exits': 0,
            'total_saved': 0.0,
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
            return r.json() if r.status_code == 200 else []
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

    def sell_position(self, symbol, qty, reason):
        """Sell a position using limit order at bid price for better fills"""
        try:
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

            if bid_price and bid_price > 0:
                # Use limit order at bid price for extended hours
                order_data['order_type'] = 'limit'
                order_data['limit_price'] = round(bid_price, 2)
                self.log(f"Placing LIMIT sell @ ${bid_price:.2f}")
            else:
                # Fall back to market order
                order_data['order_type'] = 'market'
                self.log(f"Placing MARKET sell (no bid available)")

            r = requests.post(f"{self.api}/place-order", json=order_data, timeout=5)

            if r.json().get('success'):
                self.log(f"SOLD {symbol} x{int(qty)} - {reason}")
                # Set cooldown
                self.cooldowns[symbol] = time.time() + CONFIG['REENTRY_COOLDOWN']
                return True
            else:
                self.log(f"Order failed: {r.json()}")
        except Exception as e:
            self.log(f"Error selling {symbol}: {e}")
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

        # Clean up tracking for positions we no longer hold
        for symbol in list(self.high_water_marks.keys()):
            if symbol not in current_symbols:
                del self.high_water_marks[symbol]
                if symbol in self.entry_prices:
                    del self.entry_prices[symbol]

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
        self.log("POSITION GUARDIAN STARTED")
        self.log(f"Trailing stop: {CONFIG['TRAIL_PERCENT']:.0%} from high")
        self.log(f"Hard stop: {CONFIG['HARD_STOP_PERCENT']:.0%} loss from entry")
        self.log(f"Zombie killer: {CONFIG['ZOMBIE_LOSS_PERCENT']:.0%} loss = immediate sell")
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
