"""
Unified Trade Scanner
=====================
Combines three data sources for comprehensive trade discovery:
1. WATCHLIST - Monitors symbols from the worklist for entry signals
2. NEWS - Breaking news with catalyst detection
3. MOMENTUM - High momentum stocks meeting tradeable criteria

TRADABILITY CRITERIA (all sources filtered through):
- Spread < 2% (for safe market orders)
- Price $0.50 - $20 (Warrior Trading sweet spot)
- Volume > 100K (minimum liquidity)
- Float < 10M preferred (low float momentum plays)

User insight: Pre-market movers often have counter-moves at market open,
high volatility means dump risk at any time.
"""

import requests
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import os

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    WATCHLIST = "watchlist"
    NEWS = "news"
    MOMENTUM = "momentum"


class SignalType(Enum):
    BREAKOUT = "breakout"           # Breaking resistance/VWAP
    MOMENTUM_SURGE = "momentum"     # Strong momentum building
    NEWS_CATALYST = "news"          # Breaking news trigger
    GAP_CONTINUATION = "gap"        # Gap continuing after open
    VWAP_RECLAIM = "vwap_reclaim"   # Reclaiming VWAP
    REVERSAL_WARNING = "reversal"   # Counter-move warning


@dataclass
class TradeSignal:
    """Unified trade signal from any source"""
    symbol: str
    source: SignalSource
    signal_type: SignalType
    price: float
    bid: float
    ask: float
    spread_pct: float
    volume: int
    change_pct: float
    confidence: float  # 0-1
    reason: str
    tradeable: bool
    rejection_reason: Optional[str] = None
    catalyst: Optional[str] = None
    momentum: float = 0.0
    vwap: Optional[float] = None
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "source": self.source.value,
            "signal_type": self.signal_type.value,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "spread_pct": round(self.spread_pct, 2),
            "volume": self.volume,
            "change_pct": round(self.change_pct, 2),
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "tradeable": self.tradeable,
            "rejection_reason": self.rejection_reason,
            "catalyst": self.catalyst,
            "momentum": round(self.momentum, 2),
            "vwap": self.vwap,
            "detected_at": self.detected_at.isoformat()
        }


class UnifiedTradeScanner:
    """
    Unified scanner combining watchlist, news, and momentum scanning.
    Filters everything through tradability criteria.
    """

    def __init__(self, api_url: str = "http://localhost:9100/api"):
        self.api_url = api_url
        self.alpaca_url = f"{api_url}/alpaca"

        # Tradability criteria (aligned with autonomous trader for consistency)
        self.min_price = 0.50
        self.max_price = 20.00
        self.min_volume = 100000
        self.max_spread_pct = 1.0  # Tightened from 2.0% - safer for market orders
        self.max_float = 10_000_000  # 10M preferred
        self.min_confidence = 0.60  # Aligned with autonomous trader

        # Scanning state
        self.is_running = False
        self._scan_thread: Optional[threading.Thread] = None
        self.scan_interval = 5  # seconds

        # Track price history for momentum calculation
        self.price_history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = 20

        # Results
        self.signals: List[TradeSignal] = []
        self.last_scan: Optional[datetime] = None

        # Callbacks
        self.on_tradeable_signal = None  # Called when tradeable signal found

        # Stats
        self.stats = {
            "scans": 0,
            "signals_found": 0,
            "tradeable": 0,
            "rejected": 0
        }

        logger.info("UnifiedTradeScanner initialized")
        logger.info(f"Tradability: ${self.min_price}-${self.max_price}, spread<{self.max_spread_pct}%, vol>{self.min_volume/1000:.0f}K, conf>={self.min_confidence:.0%}")

    def get_watchlist_symbols(self) -> List[str]:
        """Get symbols from the worklist API"""
        try:
            r = requests.get(f"{self.api_url}/worklist", timeout=5)
            if r.status_code == 200:
                data = r.json()
                symbols = [item['symbol'] for item in data.get('data', [])]
                return symbols
        except Exception as e:
            logger.debug(f"Watchlist fetch error: {e}")
        return []

    def get_position_symbols(self) -> List[str]:
        """Get symbols from current positions"""
        try:
            r = requests.get(f"{self.alpaca_url}/positions", timeout=5)
            if r.status_code == 200:
                positions = r.json()
                return [pos['symbol'] for pos in positions]
        except Exception as e:
            logger.debug(f"Position fetch error: {e}")
        return []

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a symbol"""
        try:
            r = requests.get(f"{self.alpaca_url}/quote/{symbol}", timeout=3)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def check_tradability(self, symbol: str, quote: Dict) -> tuple[bool, Optional[str]]:
        """Check if a stock meets tradability criteria"""
        bid = float(quote.get('bid', 0))
        ask = float(quote.get('ask', 0))
        last = float(quote.get('last', 0))
        volume = int(quote.get('volume', 0) or 0)

        price = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        if price <= 0:
            return False, "No price data"

        # Price range
        if price < self.min_price:
            return False, f"Price ${price:.2f} < ${self.min_price}"
        if price > self.max_price:
            return False, f"Price ${price:.2f} > ${self.max_price}"

        # Volume
        if volume < self.min_volume:
            return False, f"Volume {volume:,} < {self.min_volume:,}"

        # Spread
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / bid * 100
            if spread_pct > self.max_spread_pct:
                return False, f"Spread {spread_pct:.1f}% > {self.max_spread_pct}%"

        return True, None

    def calculate_momentum(self, symbol: str, current_price: float) -> float:
        """Calculate momentum from price history"""
        history = self.price_history.get(symbol, [])

        if len(history) < 3:
            return 0.0

        # Use last 5 prices or available
        prices = [h['price'] for h in history[-5:]]
        if len(prices) >= 2:
            return (current_price - prices[0]) / prices[0] * 100

        return 0.0

    def scan_watchlist(self) -> List[TradeSignal]:
        """Scan watchlist symbols for trade signals"""
        signals = []
        watchlist = self.get_watchlist_symbols()

        for symbol in watchlist:
            quote = self.get_quote(symbol)
            if not quote:
                continue

            tradeable, rejection = self.check_tradability(symbol, quote)

            bid = float(quote.get('bid', 0))
            ask = float(quote.get('ask', 0))
            last = float(quote.get('last', 0))
            volume = int(quote.get('volume', 0) or 0)
            price = last if last > 0 else (bid + ask) / 2
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 99
            change_pct = float(quote.get('change_percent', 0) or 0)

            # Track price
            self.price_history[symbol].append({
                'price': price,
                'time': datetime.now()
            })
            if len(self.price_history[symbol]) > self.max_history:
                self.price_history[symbol] = self.price_history[symbol][-self.max_history:]

            # Calculate momentum
            momentum = self.calculate_momentum(symbol, price)

            # Determine signal type
            signal_type = SignalType.MOMENTUM_SURGE
            confidence = 0.5
            reason = "Watchlist symbol"

            if change_pct >= 10:
                signal_type = SignalType.BREAKOUT
                confidence = 0.8
                reason = f"Strong gapper +{change_pct:.1f}%"
            elif momentum > 1.0:
                signal_type = SignalType.MOMENTUM_SURGE
                confidence = 0.7
                reason = f"Momentum building +{momentum:.1f}%"
            elif change_pct >= 5:
                signal_type = SignalType.GAP_CONTINUATION
                confidence = 0.6
                reason = f"Gap continuation +{change_pct:.1f}%"
            elif change_pct <= -5:
                signal_type = SignalType.REVERSAL_WARNING
                confidence = 0.3
                reason = f"Potential dump {change_pct:.1f}%"

            # Only create signal if there's something notable
            if abs(change_pct) >= 3 or abs(momentum) >= 0.5:
                signal = TradeSignal(
                    symbol=symbol,
                    source=SignalSource.WATCHLIST,
                    signal_type=signal_type,
                    price=price,
                    bid=bid,
                    ask=ask,
                    spread_pct=spread_pct,
                    volume=volume,
                    change_pct=change_pct,
                    confidence=confidence,
                    reason=reason,
                    tradeable=tradeable,
                    rejection_reason=rejection,
                    momentum=momentum
                )
                signals.append(signal)

        return signals

    def scan_momentum_universe(self, universe: List[str] = None) -> List[TradeSignal]:
        """Scan a universe of stocks for momentum signals"""
        signals = []

        # Default momentum universe
        if universe is None:
            universe = [
                'TSLA', 'NVDA', 'AMD', 'PLTR', 'SOFI', 'NIO', 'LCID',
                'MARA', 'RIOT', 'COIN', 'HOOD', 'GME', 'AMC',
                'PLUG', 'FCEL', 'MULN', 'FFIE', 'GOEV'
            ]

        for symbol in universe:
            quote = self.get_quote(symbol)
            if not quote:
                continue

            tradeable, rejection = self.check_tradability(symbol, quote)

            bid = float(quote.get('bid', 0))
            ask = float(quote.get('ask', 0))
            last = float(quote.get('last', 0))
            volume = int(quote.get('volume', 0) or 0)
            price = last if last > 0 else (bid + ask) / 2
            spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 99
            change_pct = float(quote.get('change_percent', 0) or 0)

            # Track price
            self.price_history[symbol].append({
                'price': price,
                'time': datetime.now()
            })
            if len(self.price_history[symbol]) > self.max_history:
                self.price_history[symbol] = self.price_history[symbol][-self.max_history:]

            momentum = self.calculate_momentum(symbol, price)

            # Only flag strong momentum
            if momentum > 1.5 or abs(change_pct) >= 5:
                signal_type = SignalType.BREAKOUT if momentum > 2 else SignalType.MOMENTUM_SURGE
                confidence = min(0.9, 0.5 + abs(momentum) / 10)

                signal = TradeSignal(
                    symbol=symbol,
                    source=SignalSource.MOMENTUM,
                    signal_type=signal_type,
                    price=price,
                    bid=bid,
                    ask=ask,
                    spread_pct=spread_pct,
                    volume=volume,
                    change_pct=change_pct,
                    confidence=confidence,
                    reason=f"High momentum +{momentum:.1f}%",
                    tradeable=tradeable,
                    rejection_reason=rejection,
                    momentum=momentum
                )
                signals.append(signal)

        return signals

    def integrate_news_signals(self) -> List[TradeSignal]:
        """Integrate signals from the news detector"""
        signals = []

        try:
            from ai.warrior_news_detector import get_news_detector

            detector = get_news_detector()
            news_alerts = detector.get_active_alerts()

            for alert in news_alerts:
                for symbol in alert.symbols:
                    quote = self.get_quote(symbol)
                    if not quote:
                        continue

                    tradeable, rejection = self.check_tradability(symbol, quote)

                    bid = float(quote.get('bid', 0))
                    ask = float(quote.get('ask', 0))
                    last = float(quote.get('last', 0))
                    volume = int(quote.get('volume', 0) or 0)
                    price = last if last > 0 else (bid + ask) / 2
                    spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 99
                    change_pct = float(quote.get('change_percent', 0) or 0)

                    # Confidence based on news urgency
                    urgency_confidence = {
                        'critical': 0.9,
                        'high': 0.75,
                        'medium': 0.5,
                        'low': 0.3
                    }
                    confidence = urgency_confidence.get(alert.urgency.value, 0.5)

                    signal = TradeSignal(
                        symbol=symbol,
                        source=SignalSource.NEWS,
                        signal_type=SignalType.NEWS_CATALYST,
                        price=price,
                        bid=bid,
                        ask=ask,
                        spread_pct=spread_pct,
                        volume=volume,
                        change_pct=change_pct,
                        confidence=confidence,
                        reason=f"NEWS: {alert.headline[:50]}...",
                        tradeable=tradeable,
                        rejection_reason=rejection,
                        catalyst=alert.headline
                    )
                    signals.append(signal)

        except Exception as e:
            logger.debug(f"News integration error: {e}")

        return signals

    def integrate_warrior_scanner(self) -> List[TradeSignal]:
        """Integrate signals from the Warrior Momentum Scanner"""
        signals = []

        try:
            from ai.warrior_momentum_scanner import get_warrior_scanner

            scanner = get_warrior_scanner()
            setups = scanner.get_a_setups()  # Get A+ and A setups

            for setup in setups:
                symbol = setup['symbol']
                quote = self.get_quote(symbol)
                if not quote:
                    continue

                tradeable, rejection = self.check_tradability(symbol, quote)

                bid = float(quote.get('bid', 0))
                ask = float(quote.get('ask', 0))
                price = setup.get('price', 0)
                volume = int(quote.get('volume', 0) or 0)
                spread_pct = ((ask - bid) / bid * 100) if bid > 0 else 99
                change_pct = setup.get('change_pct', 0)

                # Map setup type to signal type
                setup_type = setup.get('setup_type', 'watch')
                signal_type_map = {
                    'gap_go': SignalType.GAP_CONTINUATION,
                    'vwap_reclaim': SignalType.VWAP_RECLAIM,
                    'breakout': SignalType.BREAKOUT,
                    'continuation': SignalType.MOMENTUM_SURGE
                }
                signal_type = signal_type_map.get(setup_type, SignalType.MOMENTUM_SURGE)

                # Confidence from signal quality
                quality_confidence = {'A+': 0.9, 'A': 0.75, 'B': 0.5, 'C': 0.3}
                confidence = quality_confidence.get(setup.get('signal', 'B'), 0.5)

                signal = TradeSignal(
                    symbol=symbol,
                    source=SignalSource.MOMENTUM,
                    signal_type=signal_type,
                    price=price,
                    bid=bid,
                    ask=ask,
                    spread_pct=spread_pct,
                    volume=volume,
                    change_pct=change_pct,
                    confidence=confidence,
                    reason=setup.get('reason', 'Warrior setup'),
                    tradeable=tradeable,
                    rejection_reason=rejection,
                    vwap=setup.get('vwap')
                )
                signals.append(signal)

        except Exception as e:
            logger.debug(f"Warrior scanner integration error: {e}")

        return signals

    def scan(self) -> List[TradeSignal]:
        """Run a full scan from all sources"""
        all_signals = []

        # 1. Scan watchlist
        watchlist_signals = self.scan_watchlist()
        all_signals.extend(watchlist_signals)

        # 2. Integrate news signals
        news_signals = self.integrate_news_signals()
        all_signals.extend(news_signals)

        # 3. Integrate Warrior scanner
        warrior_signals = self.integrate_warrior_scanner()
        all_signals.extend(warrior_signals)

        # 4. Scan momentum universe (only symbols not already covered)
        covered_symbols = {s.symbol for s in all_signals}
        momentum_signals = self.scan_momentum_universe()
        for sig in momentum_signals:
            if sig.symbol not in covered_symbols:
                all_signals.append(sig)

        # Deduplicate by symbol, keeping highest confidence
        symbol_best: Dict[str, TradeSignal] = {}
        for signal in all_signals:
            if signal.symbol not in symbol_best:
                symbol_best[signal.symbol] = signal
            elif signal.confidence > symbol_best[signal.symbol].confidence:
                symbol_best[signal.symbol] = signal

        # Sort by tradeable first, then confidence
        self.signals = sorted(
            symbol_best.values(),
            key=lambda x: (x.tradeable, x.confidence),
            reverse=True
        )

        self.last_scan = datetime.now()

        # Update stats
        self.stats["scans"] += 1
        self.stats["signals_found"] += len(self.signals)
        tradeable_count = len([s for s in self.signals if s.tradeable])
        self.stats["tradeable"] += tradeable_count
        self.stats["rejected"] += len(self.signals) - tradeable_count

        # Trigger callback for tradeable signals
        tradeable_signals = [s for s in self.signals if s.tradeable and s.confidence >= self.min_confidence]
        if tradeable_signals and self.on_tradeable_signal:
            for sig in tradeable_signals:
                try:
                    self.on_tradeable_signal(sig)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        return self.signals

    def start_scanning(self, interval: int = 5):
        """Start background scanning"""
        if self.is_running:
            return

        self.is_running = True
        self.scan_interval = interval
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()

        logger.info(f"Unified scanner started - scanning every {interval}s")

    def stop_scanning(self):
        """Stop background scanning"""
        self.is_running = False
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        logger.info("Unified scanner stopped")

    def _scan_loop(self):
        """Background scanning loop"""
        while self.is_running:
            try:
                signals = self.scan()

                # Log tradeable signals
                tradeable = [s for s in signals if s.tradeable and s.confidence >= self.min_confidence]
                if tradeable:
                    logger.warning(f"TRADEABLE SIGNALS ({len(tradeable)}):")
                    for sig in tradeable[:5]:
                        logger.warning(
                            f"  [{sig.source.value}] {sig.symbol}: ${sig.price:.2f} | "
                            f"{sig.change_pct:+.1f}% | Conf: {sig.confidence:.0%} | "
                            f"{sig.reason[:40]}"
                        )

            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            time.sleep(self.scan_interval)

    def get_tradeable_signals(self, min_confidence: float = 0.5) -> List[Dict]:
        """Get only tradeable signals above confidence threshold"""
        return [
            s.to_dict() for s in self.signals
            if s.tradeable and s.confidence >= min_confidence
        ]

    def get_all_signals(self) -> List[Dict]:
        """Get all signals including rejected ones"""
        return [s.to_dict() for s in self.signals]

    def get_by_source(self, source: SignalSource) -> List[Dict]:
        """Get signals from a specific source"""
        return [s.to_dict() for s in self.signals if s.source == source]

    def get_status(self) -> Dict:
        """Get scanner status"""
        return {
            "is_running": self.is_running,
            "scan_interval": self.scan_interval,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "criteria": {
                "price_range": f"${self.min_price:.2f} - ${self.max_price:.2f}",
                "max_spread": f"{self.max_spread_pct}%",
                "min_volume": f"{self.min_volume:,}"
            },
            "signals": {
                "total": len(self.signals),
                "tradeable": len([s for s in self.signals if s.tradeable]),
                "by_source": {
                    "watchlist": len([s for s in self.signals if s.source == SignalSource.WATCHLIST]),
                    "news": len([s for s in self.signals if s.source == SignalSource.NEWS]),
                    "momentum": len([s for s in self.signals if s.source == SignalSource.MOMENTUM])
                }
            },
            "stats": self.stats
        }


# Singleton
_scanner: Optional[UnifiedTradeScanner] = None


def get_unified_scanner() -> UnifiedTradeScanner:
    """Get or create unified scanner singleton"""
    global _scanner
    if _scanner is None:
        _scanner = UnifiedTradeScanner()
    return _scanner


def start_unified_scanner(interval: int = 5, on_signal=None):
    """Start the unified scanner"""
    scanner = get_unified_scanner()
    if on_signal:
        scanner.on_tradeable_signal = on_signal
    scanner.start_scanning(interval)
    return scanner


def stop_unified_scanner():
    """Stop the unified scanner"""
    scanner = get_unified_scanner()
    scanner.stop_scanning()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    def on_signal(signal: TradeSignal):
        """Callback when tradeable signal found"""
        print(f"\n*** TRADEABLE SIGNAL ***")
        print(f"  {signal.symbol} @ ${signal.price:.2f}")
        print(f"  Source: {signal.source.value}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Spread: {signal.spread_pct:.1f}%")
        print(f"  Reason: {signal.reason}")
        print()

    print("=" * 60)
    print("  UNIFIED TRADE SCANNER")
    print("=" * 60)
    print("Sources: Watchlist | News | Momentum")
    print("Criteria: $0.50-$20 | Spread <2% | Vol >100K")
    print("=" * 60)

    scanner = start_unified_scanner(interval=5, on_signal=on_signal)

    try:
        while True:
            time.sleep(10)
            tradeable = scanner.get_tradeable_signals(min_confidence=0.5)
            if tradeable:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tradeable signals: {len(tradeable)}")
                for sig in tradeable[:5]:
                    print(f"  {sig['symbol']}: ${sig['price']:.2f} | {sig['change_pct']:+.1f}% | {sig['source']}")
    except KeyboardInterrupt:
        stop_unified_scanner()
        print("\nStopped")
