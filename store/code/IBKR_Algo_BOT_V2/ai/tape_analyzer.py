"""
Tape Analyzer - Ross Cameron Style Tape Reading
================================================
Implements tape reading techniques from Warrior Trading:
- Green/red flow analysis
- Large seller thinning detection
- First green print trigger
- Buy/sell pressure ratios

Based on: "The tape should be red, dropping on high volume,
the moment I see a green print I'll press the buy button"
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TapeEntry:
    """Single trade from time & sales"""
    timestamp: datetime
    price: float
    size: int
    side: str  # 'buy' (at ask), 'sell' (at bid), 'mid'
    exchange: str = ""

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'size': self.size,
            'side': self.side,
            'exchange': self.exchange
        }


@dataclass
class Level2Snapshot:
    """Level 2 bid/ask data"""
    timestamp: datetime
    bids: List[Dict]  # [{'price': X, 'size': Y}, ...]
    asks: List[Dict]

    def get_bid_total(self, levels: int = 5) -> int:
        return sum(b['size'] for b in self.bids[:levels])

    def get_ask_total(self, levels: int = 5) -> int:
        return sum(a['size'] for a in self.asks[:levels])


@dataclass
class TapeAnalysis:
    """Result of tape analysis"""
    symbol: str
    timestamp: datetime

    # Flow metrics
    green_volume: int = 0
    red_volume: int = 0
    total_volume: int = 0
    green_ratio: float = 0.0

    # Large order tracking
    large_seller_detected: bool = False
    seller_thinning: bool = False
    seller_original_size: int = 0
    seller_current_size: int = 0

    # Signal
    first_green_after_red: bool = False
    buy_pressure: float = 0.0

    # Recommendation
    signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'green_volume': self.green_volume,
            'red_volume': self.red_volume,
            'total_volume': self.total_volume,
            'green_ratio': round(self.green_ratio, 3),
            'large_seller_detected': self.large_seller_detected,
            'seller_thinning': self.seller_thinning,
            'first_green_after_red': self.first_green_after_red,
            'buy_pressure': round(self.buy_pressure, 3),
            'signal': self.signal,
            'confidence': round(self.confidence, 3)
        }


class TapeAnalyzer:
    """
    Analyzes time & sales data for entry/exit signals.

    Key Patterns Detected:
    1. Green/Red flow imbalance
    2. Large seller being absorbed (thinning)
    3. First green print on red tape (entry trigger)
    4. Irrational flush detection
    """

    def __init__(self, max_history: int = 500):
        self.tape_history: Dict[str, deque] = {}  # symbol -> trades
        self.level2_history: Dict[str, deque] = {}  # symbol -> snapshots
        self.max_history = max_history

        # Large order threshold (shares)
        self.large_order_threshold = 5000

        # Thinning detection
        self.thinning_decrease_pct = 0.70  # Must decrease 70% to be "thinning"

        logger.info("TapeAnalyzer initialized")

    def add_trade(self, symbol: str, trade: TapeEntry):
        """Add a trade to the tape history"""
        if symbol not in self.tape_history:
            self.tape_history[symbol] = deque(maxlen=self.max_history)

        self.tape_history[symbol].append(trade)

    def add_trades_batch(self, symbol: str, trades: List[Dict]):
        """Add multiple trades from API response"""
        for t in trades:
            entry = TapeEntry(
                timestamp=datetime.fromisoformat(t.get('timestamp', datetime.now().isoformat())),
                price=float(t.get('price', 0)),
                size=int(t.get('size', 0)),
                side=t.get('side', 'mid'),
                exchange=t.get('exchange', '')
            )
            self.add_trade(symbol, entry)

    def add_level2(self, symbol: str, snapshot: Level2Snapshot):
        """Add Level 2 snapshot"""
        if symbol not in self.level2_history:
            self.level2_history[symbol] = deque(maxlen=100)

        self.level2_history[symbol].append(snapshot)

    def analyze(self, symbol: str, lookback_seconds: int = 60) -> TapeAnalysis:
        """
        Analyze recent tape for trading signals.

        Args:
            symbol: Stock symbol
            lookback_seconds: How far back to analyze

        Returns:
            TapeAnalysis with signals and metrics
        """
        analysis = TapeAnalysis(
            symbol=symbol,
            timestamp=datetime.now()
        )

        if symbol not in self.tape_history or len(self.tape_history[symbol]) == 0:
            return analysis

        # Filter to lookback window
        cutoff = datetime.now() - timedelta(seconds=lookback_seconds)
        recent_trades = [
            t for t in self.tape_history[symbol]
            if t.timestamp > cutoff
        ]

        if not recent_trades:
            return analysis

        # Calculate green/red flow
        analysis.green_volume = sum(t.size for t in recent_trades if t.side == 'buy')
        analysis.red_volume = sum(t.size for t in recent_trades if t.side == 'sell')
        analysis.total_volume = analysis.green_volume + analysis.red_volume

        if analysis.total_volume > 0:
            analysis.green_ratio = analysis.green_volume / analysis.total_volume
            analysis.buy_pressure = analysis.green_ratio

        # Check for first green after red streak
        analysis.first_green_after_red = self._detect_first_green(recent_trades)

        # Check for large seller thinning
        if symbol in self.level2_history and len(self.level2_history[symbol]) >= 5:
            thinning_result = self._detect_seller_thinning(symbol)
            analysis.large_seller_detected = thinning_result['detected']
            analysis.seller_thinning = thinning_result['thinning']
            analysis.seller_original_size = thinning_result.get('original_size', 0)
            analysis.seller_current_size = thinning_result.get('current_size', 0)

        # Determine signal
        analysis.signal, analysis.confidence = self._calculate_signal(analysis)

        return analysis

    def _detect_first_green(self, trades: List[TapeEntry], red_streak_min: int = 5) -> bool:
        """
        Detect first green print after a red streak.

        "The tape should be red, dropping on high volume,
        the moment I see a green print I'll press the buy button"
        """
        if len(trades) < red_streak_min + 1:
            return False

        # Check if last trade is green
        if trades[-1].side != 'buy':
            return False

        # Check if prior trades were red streak
        red_count = 0
        for trade in trades[-red_streak_min-1:-1]:
            if trade.side == 'sell':
                red_count += 1

        return red_count >= red_streak_min * 0.7  # 70% red is "red tape"

    def _detect_seller_thinning(self, symbol: str) -> Dict:
        """
        Detect large seller being absorbed.

        Pattern: 25k -> 19k -> 15k -> 11k -> 5k (seller thinning)
        """
        snapshots = list(self.level2_history[symbol])[-10:]

        if len(snapshots) < 5:
            return {'detected': False, 'thinning': False}

        # Track ask size at best ask
        ask_sizes = []
        for snap in snapshots:
            if snap.asks:
                ask_sizes.append(snap.asks[0]['size'])

        if not ask_sizes:
            return {'detected': False, 'thinning': False}

        # Check for large seller
        max_size = max(ask_sizes)
        if max_size < self.large_order_threshold:
            return {'detected': False, 'thinning': False}

        # Check if thinning (consistently decreasing)
        current_size = ask_sizes[-1]
        decrease_pct = (max_size - current_size) / max_size

        # Check if sizes are trending down
        is_decreasing = all(
            ask_sizes[i] >= ask_sizes[i+1]
            for i in range(len(ask_sizes)-3, len(ask_sizes)-1)
        )

        return {
            'detected': True,
            'thinning': is_decreasing and decrease_pct >= self.thinning_decrease_pct,
            'original_size': max_size,
            'current_size': current_size,
            'decrease_pct': decrease_pct
        }

    def _calculate_signal(self, analysis: TapeAnalysis) -> Tuple[str, float]:
        """Calculate overall signal from analysis"""
        score = 0.0
        max_score = 0.0

        # Green ratio (max 30 points)
        max_score += 30
        if analysis.green_ratio > 0.6:
            score += 30
        elif analysis.green_ratio > 0.5:
            score += 15
        elif analysis.green_ratio < 0.4:
            score -= 20

        # First green after red (30 points)
        max_score += 30
        if analysis.first_green_after_red:
            score += 30

        # Seller thinning (40 points)
        max_score += 40
        if analysis.seller_thinning:
            score += 40
        elif analysis.large_seller_detected:
            score -= 10  # Large seller present but not thinning

        # Calculate confidence
        confidence = max(0, min(1, (score + max_score) / (2 * max_score)))

        # Determine signal
        if score >= 50:
            return "BULLISH", confidence
        elif score <= -20:
            return "BEARISH", confidence
        else:
            return "NEUTRAL", confidence

    def detect_irrational_flush(
        self,
        symbol: str,
        lookback_seconds: int = 30,
        min_drop_pct: float = 3.0
    ) -> Dict:
        """
        Detect irrational flush (fast panic drop).

        "The best dips are when... in almost what seems like a flash,
        it drops 50 cents or $1.00 or more per share.
        This to me is what I'd call 'irrational'"
        """
        if symbol not in self.tape_history:
            return {'detected': False}

        cutoff = datetime.now() - timedelta(seconds=lookback_seconds)
        recent = [t for t in self.tape_history[symbol] if t.timestamp > cutoff]

        if len(recent) < 10:
            return {'detected': False}

        # Find high and current
        high_price = max(t.price for t in recent[:len(recent)//2])  # High in first half
        current_price = recent[-1].price

        drop_pct = (high_price - current_price) / high_price * 100
        drop_dollars = high_price - current_price

        if drop_pct >= min_drop_pct or drop_dollars >= 0.50:
            return {
                'detected': True,
                'high_price': high_price,
                'current_price': current_price,
                'drop_pct': drop_pct,
                'drop_dollars': drop_dollars,
                'action': 'DIP_BUY_OPPORTUNITY'
            }

        return {'detected': False, 'drop_pct': drop_pct}

    def get_entry_signal(self, symbol: str) -> Dict:
        """
        Get entry signal based on tape analysis.

        Returns signal type and confidence for trading decision.
        """
        analysis = self.analyze(symbol)
        flush = self.detect_irrational_flush(symbol)

        signals = []

        # Check for first green (highest priority)
        if analysis.first_green_after_red:
            signals.append({
                'type': 'FIRST_GREEN_PRINT',
                'confidence': 0.8,
                'action': 'BUY_NOW',
                'reason': 'First green after red streak - shorts covering'
            })

        # Check for seller thinning
        if analysis.seller_thinning:
            signals.append({
                'type': 'SELLER_THINNING',
                'confidence': 0.85,
                'action': 'PREPARE_TO_BUY',
                'reason': f'Large seller absorbed: {analysis.seller_original_size} -> {analysis.seller_current_size}'
            })

        # Check for irrational flush
        if flush.get('detected'):
            signals.append({
                'type': 'IRRATIONAL_FLUSH',
                'confidence': 0.7,
                'action': 'DIP_BUY',
                'reason': f'Flash drop {flush["drop_pct"]:.1f}% in 30s'
            })

        # Check buy pressure
        if analysis.buy_pressure > 0.65:
            signals.append({
                'type': 'STRONG_BUY_PRESSURE',
                'confidence': 0.6,
                'action': 'BUY_SUPPORTED',
                'reason': f'Green ratio {analysis.green_ratio:.1%}'
            })

        if not signals:
            return {
                'symbol': symbol,
                'signal': 'NO_SIGNAL',
                'analysis': analysis.to_dict()
            }

        # Return strongest signal
        best_signal = max(signals, key=lambda x: x['confidence'])
        return {
            'symbol': symbol,
            'signal': best_signal['type'],
            'confidence': best_signal['confidence'],
            'action': best_signal['action'],
            'reason': best_signal['reason'],
            'all_signals': signals,
            'analysis': analysis.to_dict()
        }

    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            'symbols_tracked': list(self.tape_history.keys()),
            'tape_counts': {s: len(t) for s, t in self.tape_history.items()},
            'level2_counts': {s: len(l) for s, l in self.level2_history.items()},
            'large_order_threshold': self.large_order_threshold
        }


# Singleton instance
_tape_analyzer: Optional[TapeAnalyzer] = None


def get_tape_analyzer() -> TapeAnalyzer:
    """Get or create TapeAnalyzer instance"""
    global _tape_analyzer
    if _tape_analyzer is None:
        _tape_analyzer = TapeAnalyzer()
    return _tape_analyzer


# Convenience functions
async def analyze_tape(symbol: str) -> TapeAnalysis:
    """Analyze tape for a symbol"""
    return get_tape_analyzer().analyze(symbol)


async def get_tape_entry_signal(symbol: str) -> Dict:
    """Get entry signal from tape"""
    return get_tape_analyzer().get_entry_signal(symbol)
