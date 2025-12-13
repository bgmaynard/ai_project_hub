"""
Penny/Cheap Stock Momentum Scanner
==================================
Scans for sub-$8 stocks with momentum - perfect for small account scalping.

Target: Stocks under $8 with:
- High relative volume (2x+ normal)
- Price momentum (moving up)
- Tight spreads (tradeable)
- News catalyst (optional but preferred)

These are the "BEAT" type plays - cheap stocks that spike on momentum.
"""

import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class PennyMover:
    """A cheap stock showing momentum"""
    symbol: str
    price: float
    bid: float
    ask: float
    spread_pct: float
    volume: int
    relative_volume: float
    change_pct: float
    momentum_score: float
    signal: str  # "strong_buy", "buy", "watch", "avoid"
    reason: str
    detected_at: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": round(self.spread_pct, 2),
            "volume": self.volume,
            "relative_volume": round(self.relative_volume, 1),
            "change_pct": round(self.change_pct, 2),
            "momentum_score": round(self.momentum_score, 1),
            "signal": self.signal,
            "reason": self.reason,
            "detected_at": self.detected_at.isoformat()
        }


# Common penny/small cap stocks to scan
# These often have momentum plays
PENNY_WATCHLIST = [
    # Biotech/Pharma (FDA plays)
    'NVAX', 'SRNE', 'OCGN', 'VXRT', 'INO', 'ATHX', 'ATOS', 'SAVA',
    # EV/Tech small caps
    'NIO', 'LCID', 'RIVN', 'GOEV', 'FFIE', 'MULN', 'NKLA',
    # Meme/Retail favorites
    'AMC', 'BB', 'BBBY', 'WISH', 'CLOV', 'SOFI', 'PLTR',
    # Recent movers (update regularly)
    'BEAT', 'EDIT', 'MAMA', 'BBGI', 'DNA', 'SOUN',
    # General small caps
    'PLUG', 'FCEL', 'BLNK', 'WKHS', 'RIDE', 'FSR',
    # Add more as discovered
]


class PennyMomentumScanner:
    """
    Scans for cheap stocks (<$8) with momentum.
    Perfect for small account scalping.
    """

    def __init__(self, api_url: str = "http://localhost:9100/api/alpaca"):
        self.api_url = api_url

        # Price filter
        self.max_price = 8.00  # Under $8
        self.min_price = 0.10  # Above $0.10 (avoid sub-pennies)

        # Quality filters
        self.max_spread_pct = 3.0   # Max 3% spread for entry
        self.min_rel_volume = 1.5   # At least 1.5x normal volume

        # Watchlist
        self.watchlist = PENNY_WATCHLIST.copy()

        # Price tracking for momentum
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.max_history = 20

        # Results
        self.movers: List[PennyMover] = []
        self.last_scan: Optional[datetime] = None

        # Background scanning
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.scan_interval = 5  # seconds

        # Callbacks
        self.on_mover_detected = None

        logger.info(f"PennyMomentumScanner initialized - watching {len(self.watchlist)} stocks")

    def add_symbol(self, symbol: str):
        """Add a symbol to watchlist"""
        if symbol.upper() not in self.watchlist:
            self.watchlist.append(symbol.upper())
            logger.info(f"Added {symbol} to penny scanner watchlist")

    def remove_symbol(self, symbol: str):
        """Remove a symbol from watchlist"""
        if symbol.upper() in self.watchlist:
            self.watchlist.remove(symbol.upper())

    def scan(self) -> List[PennyMover]:
        """Run a single scan of all watchlist stocks"""
        movers = []
        now = datetime.now()

        for symbol in self.watchlist:
            try:
                # Get quote
                r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
                if r.status_code != 200:
                    continue

                quote = r.json()
                bid = float(quote.get('bid', 0))
                ask = float(quote.get('ask', 0))
                last = float(quote.get('last', 0))
                volume = int(quote.get('volume', 0) or 0)

                # Use last price, fallback to mid
                price = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                if price <= 0:
                    continue

                # Price filter
                if price > self.max_price or price < self.min_price:
                    continue

                # Calculate spread
                spread_pct = 0
                if bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / bid * 100

                # Skip wide spreads (not tradeable)
                if spread_pct > self.max_spread_pct:
                    continue

                # Track price history
                self.price_history[symbol].append(price)
                if len(self.price_history[symbol]) > self.max_history:
                    self.price_history[symbol] = self.price_history[symbol][-self.max_history:]

                # Calculate momentum
                prices = self.price_history[symbol]
                momentum_score = 0
                change_pct = 0

                if len(prices) >= 3:
                    # Short-term momentum (last 3 readings)
                    short_change = (prices[-1] - prices[-3]) / prices[-3] * 100
                    momentum_score = short_change * 10  # Scale it

                    # Change from first reading
                    change_pct = (prices[-1] - prices[0]) / prices[0] * 100

                # Estimate relative volume (simplified)
                # In production, compare to average daily volume
                rel_volume = 1.0
                if volume > 100000:
                    rel_volume = 2.0
                if volume > 500000:
                    rel_volume = 3.0
                if volume > 1000000:
                    rel_volume = 5.0

                # Generate signal
                signal, reason = self._generate_signal(
                    price, spread_pct, momentum_score, change_pct, rel_volume
                )

                # Only add if signal is actionable
                if signal in ['strong_buy', 'buy', 'watch']:
                    mover = PennyMover(
                        symbol=symbol,
                        price=price,
                        bid=bid,
                        ask=ask,
                        spread_pct=spread_pct,
                        volume=volume,
                        relative_volume=rel_volume,
                        change_pct=change_pct,
                        momentum_score=momentum_score,
                        signal=signal,
                        reason=reason,
                        detected_at=now
                    )
                    movers.append(mover)

                    # Trigger callback for strong signals
                    if signal == 'strong_buy' and self.on_mover_detected:
                        try:
                            self.on_mover_detected(mover)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        # Sort by momentum score
        movers.sort(key=lambda x: x.momentum_score, reverse=True)

        self.movers = movers
        self.last_scan = now

        return movers

    def _generate_signal(self, price: float, spread_pct: float,
                         momentum: float, change_pct: float,
                         rel_volume: float) -> tuple:
        """Generate trading signal"""

        # Strong buy: momentum + volume + tight spread
        if momentum > 5 and rel_volume >= 2.0 and spread_pct < 1.5:
            return 'strong_buy', f"Strong momentum ({momentum:.1f}) + high volume"

        # Buy: positive momentum + decent spread
        if momentum > 2 and spread_pct < 2.0:
            return 'buy', f"Positive momentum ({momentum:.1f})"

        # Watch: some momentum or volume
        if momentum > 0 or rel_volume >= 2.0:
            return 'watch', f"Building momentum" if momentum > 0 else "High volume"

        return 'avoid', "No momentum"

    def start_scanning(self, interval: int = 5):
        """Start background scanning"""
        if self.is_running:
            return

        self.is_running = True
        self.scan_interval = interval

        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

        logger.info(f"Penny scanner started - scanning every {interval}s")

    def stop_scanning(self):
        """Stop background scanning"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Penny scanner stopped")

    def _scan_loop(self):
        """Background scanning loop"""
        while self.is_running:
            try:
                movers = self.scan()

                # Log significant movers
                strong = [m for m in movers if m.signal == 'strong_buy']
                if strong:
                    logger.warning(f"PENNY MOVERS: {[m.symbol for m in strong]}")
                    for m in strong[:3]:
                        logger.warning(
                            f"  {m.symbol}: ${m.price:.2f} | "
                            f"Mom: {m.momentum_score:.1f} | "
                            f"Spread: {m.spread_pct:.1f}% | "
                            f"Vol: {m.relative_volume:.1f}x"
                        )

            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            time.sleep(self.scan_interval)

    def get_top_movers(self, limit: int = 10) -> List[Dict]:
        """Get top movers from last scan"""
        return [m.to_dict() for m in self.movers[:limit]]

    def get_buy_signals(self) -> List[Dict]:
        """Get stocks with buy signals"""
        return [m.to_dict() for m in self.movers if m.signal in ['strong_buy', 'buy']]

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "is_running": self.is_running,
            "watchlist_size": len(self.watchlist),
            "price_range": f"${self.min_price:.2f} - ${self.max_price:.2f}",
            "max_spread": f"{self.max_spread_pct}%",
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "movers_found": len(self.movers),
            "buy_signals": len([m for m in self.movers if m.signal in ['strong_buy', 'buy']])
        }


# Singleton
_scanner: Optional[PennyMomentumScanner] = None


def get_penny_scanner() -> PennyMomentumScanner:
    """Get or create penny scanner singleton"""
    global _scanner
    if _scanner is None:
        _scanner = PennyMomentumScanner()
    return _scanner


def start_penny_scanner(interval: int = 5, on_mover=None):
    """Start the penny scanner"""
    scanner = get_penny_scanner()
    if on_mover:
        scanner.on_mover_detected = on_mover
    scanner.start_scanning(interval)
    return scanner


def stop_penny_scanner():
    """Stop the penny scanner"""
    scanner = get_penny_scanner()
    scanner.stop_scanning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_strong_mover(mover: PennyMover):
        print(f"\n*** STRONG MOVER: {mover.symbol} @ ${mover.price:.2f} ***")
        print(f"    Signal: {mover.signal} - {mover.reason}\n")

    scanner = start_penny_scanner(interval=5, on_mover=on_strong_mover)

    print("Penny scanner running... Press Ctrl+C to stop")
    print(f"Watching {len(scanner.watchlist)} stocks under ${scanner.max_price}")

    try:
        while True:
            time.sleep(10)
            signals = scanner.get_buy_signals()
            if signals:
                print(f"\nBuy signals: {[s['symbol'] for s in signals]}")
    except KeyboardInterrupt:
        stop_penny_scanner()
        print("Stopped")
