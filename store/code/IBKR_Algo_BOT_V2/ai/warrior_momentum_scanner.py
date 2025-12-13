"""
Warrior Trading Momentum Scanner
================================
Scans for $2-20 stocks with gap-up momentum - the Warrior Trading sweet spot.

Ross Cameron's criteria for momentum plays:
- Float under 100M (ideally under 20M)
- Relative volume 2x+
- Gap up 5%+ in premarket
- Fresh catalyst (news, earnings)
- Price $2-$20 range

This scanner focuses on:
1. Gap-ups showing continuation
2. VWAP reclaim plays
3. High relative volume breakouts
4. Multi-day runners on day 2+
"""

import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class WarriorSetup:
    """A Warrior Trading style momentum setup"""
    symbol: str
    price: float
    vwap: float
    price_vs_vwap: float  # % above/below VWAP
    gap_pct: float
    change_pct: float
    volume: int
    relative_volume: float
    float_category: str  # "micro", "small", "medium", "large"
    setup_type: str  # "gap_go", "vwap_reclaim", "breakout", "continuation"
    signal: str  # "A+", "A", "B", "C" quality rating
    entry_zone: Tuple[float, float]  # (low, high) suggested entry
    risk_reward: str
    reason: str
    detected_at: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "vwap": round(self.vwap, 2),
            "price_vs_vwap": f"{self.price_vs_vwap:+.1f}%",
            "gap_pct": round(self.gap_pct, 1),
            "change_pct": round(self.change_pct, 1),
            "volume": self.volume,
            "relative_volume": f"{self.relative_volume:.1f}x",
            "float_category": self.float_category,
            "setup_type": self.setup_type,
            "signal": self.signal,
            "entry_zone": f"${self.entry_zone[0]:.2f} - ${self.entry_zone[1]:.2f}",
            "risk_reward": self.risk_reward,
            "reason": self.reason,
            "detected_at": self.detected_at.isoformat()
        }


# Warrior Trading focus stocks - small cap momentum names
WARRIOR_WATCHLIST = [
    # Recent momentum runners (update weekly)
    'BEAT', 'DNA', 'SOUN', 'EDIT', 'BBGI', 'MAMA',
    # Biotech/Pharma (catalyst plays)
    'NVAX', 'SRNE', 'OCGN', 'VXRT', 'INO', 'SAVA', 'ATOS',
    # EV/Tech small caps
    'LCID', 'RIVN', 'GOEV', 'FFIE', 'NKLA', 'FSR',
    # Retail favorites with volume
    'SOFI', 'PLTR', 'CLOV', 'WISH', 'BB',
    # Clean energy momentum
    'PLUG', 'FCEL', 'BLNK', 'WKHS',
    # General small cap movers
    'OPEN', 'BYND', 'HOOD', 'AFRM', 'UPST',
    'COIN', 'MARA', 'RIOT', 'HIVE', 'BITF',
    # Recent IPOs and SPACs
    'VFS', 'IONQ', 'RKLB',
]


class WarriorMomentumScanner:
    """
    Scans for Warrior Trading style setups ($2-$20 momentum).
    Focuses on gap-ups, VWAP reclaims, and high relative volume.
    """

    def __init__(self, api_url: str = "http://localhost:9100/api/alpaca"):
        self.api_url = api_url

        # Warrior price range
        self.min_price = 2.00
        self.max_price = 20.00

        # Quality filters
        self.min_gap_pct = 3.0      # At least 3% gap
        self.min_rel_volume = 1.5   # 1.5x normal volume
        self.max_spread_pct = 2.0   # Max 2% spread

        # Watchlist
        self.watchlist = WARRIOR_WATCHLIST.copy()

        # Tracking
        self.price_history: Dict[str, List[Dict]] = defaultdict(list)
        self.prev_close: Dict[str, float] = {}
        self.vwap_data: Dict[str, Dict] = {}
        self.max_history = 30

        # Results
        self.setups: List[WarriorSetup] = []
        self.last_scan: Optional[datetime] = None

        # Background scanning
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.scan_interval = 10  # seconds

        # Callbacks
        self.on_setup_detected = None

        logger.info(f"WarriorMomentumScanner initialized - watching {len(self.watchlist)} stocks")

    def add_symbol(self, symbol: str):
        """Add a symbol to watchlist"""
        if symbol.upper() not in self.watchlist:
            self.watchlist.append(symbol.upper())
            logger.info(f"Added {symbol} to Warrior scanner watchlist")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from watchlist"""
        if symbol.upper() in self.watchlist:
            self.watchlist.remove(symbol.upper())

    def scan(self) -> List[WarriorSetup]:
        """Run a single scan for Warrior setups"""
        setups = []
        now = datetime.now()

        for symbol in self.watchlist:
            try:
                setup = self._analyze_symbol(symbol, now)
                if setup:
                    setups.append(setup)

                    # Callback for A+ setups
                    if setup.signal == 'A+' and self.on_setup_detected:
                        try:
                            self.on_setup_detected(setup)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by signal quality then relative volume
        quality_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3}
        setups.sort(key=lambda x: (quality_order.get(x.signal, 4), -x.relative_volume))

        self.setups = setups
        self.last_scan = now

        return setups

    def _analyze_symbol(self, symbol: str, now: datetime) -> Optional[WarriorSetup]:
        """Analyze a single symbol for Warrior setups"""

        # Get quote
        r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=3)
        if r.status_code != 200:
            return None

        quote = r.json()
        bid = float(quote.get('bid', 0))
        ask = float(quote.get('ask', 0))
        last = float(quote.get('last', 0))
        volume = int(quote.get('volume', 0) or 0)
        prev_close = float(quote.get('prev_close', 0) or quote.get('previous_close', 0) or 0)

        # Use last price, fallback to mid
        price = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        if price <= 0:
            return None

        # Store prev close for gap calc
        if prev_close > 0:
            self.prev_close[symbol] = prev_close
        else:
            prev_close = self.prev_close.get(symbol, price * 0.95)

        # Price filter - Warrior range
        if price > self.max_price or price < self.min_price:
            return None

        # Calculate spread
        spread_pct = 0
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / bid * 100
            if spread_pct > self.max_spread_pct:
                return None  # Skip wide spreads

        # Calculate gap %
        gap_pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0

        # Track price for momentum calculation
        self.price_history[symbol].append({
            'price': price,
            'volume': volume,
            'time': now
        })
        if len(self.price_history[symbol]) > self.max_history:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history:]

        # Calculate VWAP (simplified - cumulative)
        vwap = self._calculate_vwap(symbol, price, volume)
        price_vs_vwap = (price - vwap) / vwap * 100 if vwap > 0 else 0

        # Calculate relative volume
        rel_volume = self._estimate_relative_volume(volume)

        # Calculate change % from first reading
        history = self.price_history[symbol]
        change_pct = 0
        if len(history) >= 2:
            first_price = history[0]['price']
            change_pct = (price - first_price) / first_price * 100

        # Float category (simplified - based on typical behavior)
        float_category = self._estimate_float_category(symbol, rel_volume)

        # Determine setup type
        setup_type, signal, reason = self._classify_setup(
            price, vwap, price_vs_vwap, gap_pct, change_pct, rel_volume, float_category
        )

        # Only return actionable setups
        if signal not in ['A+', 'A', 'B']:
            return None

        # Calculate entry zone
        entry_low = max(price * 0.99, vwap) if price > vwap else price * 0.98
        entry_high = price * 1.01

        # Risk/reward estimate
        risk = abs(price - vwap) if price > vwap else price * 0.03
        reward = price * 0.05  # Target 5% move
        rr_ratio = reward / risk if risk > 0 else 0
        risk_reward = f"{rr_ratio:.1f}:1" if rr_ratio > 0 else "N/A"

        return WarriorSetup(
            symbol=symbol,
            price=price,
            vwap=vwap,
            price_vs_vwap=price_vs_vwap,
            gap_pct=gap_pct,
            change_pct=change_pct,
            volume=volume,
            relative_volume=rel_volume,
            float_category=float_category,
            setup_type=setup_type,
            signal=signal,
            entry_zone=(entry_low, entry_high),
            risk_reward=risk_reward,
            reason=reason,
            detected_at=now
        )

    def _calculate_vwap(self, symbol: str, price: float, volume: int) -> float:
        """Calculate approximate VWAP"""
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = {
                'cum_pv': 0,
                'cum_vol': 0,
                'vwap': price
            }

        data = self.vwap_data[symbol]

        if volume > 0:
            # Incremental VWAP update
            data['cum_pv'] += price * volume
            data['cum_vol'] += volume
            data['vwap'] = data['cum_pv'] / data['cum_vol'] if data['cum_vol'] > 0 else price

        return data['vwap']

    def _estimate_relative_volume(self, volume: int) -> float:
        """Estimate relative volume based on absolute volume"""
        # Simplified - in production, compare to 20-day average
        if volume >= 5000000:
            return 5.0
        elif volume >= 2000000:
            return 3.0
        elif volume >= 1000000:
            return 2.0
        elif volume >= 500000:
            return 1.5
        elif volume >= 100000:
            return 1.0
        else:
            return 0.5

    def _estimate_float_category(self, symbol: str, rel_volume: float) -> str:
        """Estimate float category based on typical behavior"""
        # In production, use actual float data
        # High relative volume often indicates low float
        if rel_volume >= 3.0:
            return "micro"  # Sub 10M float
        elif rel_volume >= 2.0:
            return "small"  # 10-50M float
        elif rel_volume >= 1.5:
            return "medium"  # 50-100M float
        else:
            return "large"  # 100M+ float

    def _classify_setup(self, price: float, vwap: float, price_vs_vwap: float,
                        gap_pct: float, change_pct: float, rel_volume: float,
                        float_category: str) -> Tuple[str, str, str]:
        """Classify the setup type and quality"""

        # Gap and Go - gapping up with continuation
        if gap_pct >= 10 and price > vwap and rel_volume >= 2.0:
            if float_category in ['micro', 'small']:
                return ('gap_go', 'A+', f"Strong gap ({gap_pct:.0f}%) + low float + above VWAP")
            else:
                return ('gap_go', 'A', f"Gap up {gap_pct:.0f}% with volume")

        # VWAP Reclaim - was below, now reclaiming
        if 0 < price_vs_vwap < 2.0 and gap_pct >= 5:
            return ('vwap_reclaim', 'A', f"Reclaiming VWAP on gap ({gap_pct:.0f}%)")

        # Breakout - strong momentum above VWAP
        if price_vs_vwap >= 3.0 and rel_volume >= 2.0 and change_pct > 0:
            return ('breakout', 'A', f"Breaking out +{price_vs_vwap:.1f}% above VWAP")

        # Continuation - already moving, still has momentum
        if change_pct >= 3.0 and price > vwap:
            if rel_volume >= 2.0:
                return ('continuation', 'A', f"Continuing +{change_pct:.1f}% with strong volume")
            else:
                return ('continuation', 'B', f"Momentum +{change_pct:.1f}%")

        # Moderate gap with volume
        if gap_pct >= 5 and rel_volume >= 1.5:
            return ('gap_go', 'B', f"Gap {gap_pct:.0f}% with decent volume")

        # Watching - some criteria met
        if gap_pct >= 3 or (price > vwap and change_pct > 0):
            return ('watch', 'C', "Building setup")

        return ('none', 'D', "No setup")

    def start_scanning(self, interval: int = 10):
        """Start background scanning"""
        if self.is_running:
            return

        self.is_running = True
        self.scan_interval = interval

        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

        logger.info(f"Warrior scanner started - scanning every {interval}s")

    def stop_scanning(self):
        """Stop background scanning"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Warrior scanner stopped")

    def _scan_loop(self):
        """Background scanning loop"""
        while self.is_running:
            try:
                setups = self.scan()

                # Log A+ and A setups
                top_setups = [s for s in setups if s.signal in ['A+', 'A']]
                if top_setups:
                    logger.warning(f"WARRIOR SETUPS: {[s.symbol for s in top_setups]}")
                    for s in top_setups[:3]:
                        logger.warning(
                            f"  [{s.signal}] {s.symbol}: ${s.price:.2f} | "
                            f"Gap: {s.gap_pct:+.1f}% | "
                            f"VWAP: {s.price_vs_vwap:+.1f}% | "
                            f"Vol: {s.relative_volume:.1f}x | "
                            f"{s.setup_type}"
                        )

            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            time.sleep(self.scan_interval)

    def get_top_setups(self, limit: int = 10) -> List[Dict]:
        """Get top setups from last scan"""
        return [s.to_dict() for s in self.setups[:limit]]

    def get_a_setups(self) -> List[Dict]:
        """Get A+ and A quality setups"""
        return [s.to_dict() for s in self.setups if s.signal in ['A+', 'A']]

    def get_gap_setups(self) -> List[Dict]:
        """Get gap and go setups"""
        return [s.to_dict() for s in self.setups if s.setup_type == 'gap_go']

    def get_vwap_reclaims(self) -> List[Dict]:
        """Get VWAP reclaim setups"""
        return [s.to_dict() for s in self.setups if s.setup_type == 'vwap_reclaim']

    def get_status(self) -> Dict:
        """Get scanner status"""
        setup_counts = {}
        for s in self.setups:
            setup_counts[s.signal] = setup_counts.get(s.signal, 0) + 1

        return {
            "is_running": self.is_running,
            "watchlist_size": len(self.watchlist),
            "price_range": f"${self.min_price:.2f} - ${self.max_price:.2f}",
            "min_gap": f"{self.min_gap_pct}%",
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "setups_found": len(self.setups),
            "by_quality": setup_counts,
            "a_plus_setups": len([s for s in self.setups if s.signal == 'A+']),
            "a_setups": len([s for s in self.setups if s.signal == 'A'])
        }

    def reset_vwap(self):
        """Reset VWAP data (call at market open)"""
        self.vwap_data.clear()
        self.price_history.clear()
        logger.info("VWAP data reset")


# Singleton
_scanner: Optional[WarriorMomentumScanner] = None


def get_warrior_scanner() -> WarriorMomentumScanner:
    """Get or create warrior scanner singleton"""
    global _scanner
    if _scanner is None:
        _scanner = WarriorMomentumScanner()
    return _scanner


def start_warrior_scanner(interval: int = 10, on_setup=None):
    """Start the warrior scanner"""
    scanner = get_warrior_scanner()
    if on_setup:
        scanner.on_setup_detected = on_setup
    scanner.start_scanning(interval)
    return scanner


def stop_warrior_scanner():
    """Stop the warrior scanner"""
    scanner = get_warrior_scanner()
    scanner.stop_scanning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_a_plus_setup(setup: WarriorSetup):
        print(f"\n*** A+ SETUP: {setup.symbol} @ ${setup.price:.2f} ***")
        print(f"    Type: {setup.setup_type}")
        print(f"    Gap: {setup.gap_pct:+.1f}% | VWAP: {setup.price_vs_vwap:+.1f}%")
        print(f"    Entry Zone: {setup.entry_zone}")
        print(f"    {setup.reason}\n")

    scanner = start_warrior_scanner(interval=10, on_setup=on_a_plus_setup)

    print("Warrior scanner running... Press Ctrl+C to stop")
    print(f"Watching {len(scanner.watchlist)} stocks in ${scanner.min_price}-${scanner.max_price} range")
    print("Looking for: Gap & Go, VWAP Reclaim, Breakout, Continuation setups")

    try:
        while True:
            time.sleep(15)
            a_setups = scanner.get_a_setups()
            if a_setups:
                print(f"\nA-quality setups: {[s['symbol'] for s in a_setups]}")
    except KeyboardInterrupt:
        stop_warrior_scanner()
        print("Stopped")
