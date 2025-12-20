"""
Order Flow Analyzer
====================
Analyzes bid/ask imbalance from Level 2 data to identify buying/selling pressure.

Buy pressure > 60% = Strong buyers = Good entry signal
Buy pressure < 40% = Strong sellers = Skip entry

Usage:
    from ai.order_flow_analyzer import get_order_flow_signal
    signal = await get_order_flow_signal("AAPL")
    if signal['recommendation'] == 'ENTER':
        # Execute trade
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowSignal:
    """Order flow analysis result"""
    symbol: str
    buy_pressure: float  # 0.0 to 1.0 (percentage as decimal)
    sell_pressure: float  # 0.0 to 1.0
    imbalance: float  # -1.0 (all sell) to +1.0 (all buy)
    spread_percent: float  # Bid-ask spread as percentage
    recommendation: str  # ENTER, SKIP, NEUTRAL
    confidence: float  # 0.0 to 1.0
    reason: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'buy_pressure': round(self.buy_pressure * 100, 1),  # As percentage
            'sell_pressure': round(self.sell_pressure * 100, 1),
            'imbalance': round(self.imbalance, 3),
            'spread_percent': round(self.spread_percent, 3),
            'recommendation': self.recommendation,
            'confidence': round(self.confidence, 2),
            'reason': self.reason,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'timestamp': self.timestamp
        }


class OrderFlowAnalyzer:
    """
    Analyzes order flow from bid/ask data to detect buying/selling pressure.

    Key Metrics:
    - Buy Pressure: bid_size / (bid_size + ask_size)
    - Imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    - Spread: (ask - bid) / mid_price

    Entry Criteria:
    - Buy pressure > 60% = strong buying → ENTER
    - Buy pressure < 40% = strong selling → SKIP
    - Spread < 1% = tight (good) | Spread > 2% = wide (caution)
    """

    def __init__(self):
        self.min_buy_pressure = 0.55  # Minimum to consider entry (55%)
        self.strong_buy_pressure = 0.65  # Strong signal (65%)
        self.max_spread_percent = 1.5  # Maximum spread to consider
        self.min_total_size = 100  # Minimum shares on book
        self.history: Dict[str, list] = {}  # Track recent readings per symbol
        self.history_max = 10  # Keep last 10 readings

    def analyze(self, quote: Dict) -> OrderFlowSignal:
        """
        Analyze order flow from a quote.

        Args:
            quote: Dict with bid, ask, bid_size, ask_size

        Returns:
            OrderFlowSignal with recommendation
        """
        symbol = quote.get('symbol', 'UNKNOWN')
        bid = float(quote.get('bid', 0) or 0)
        ask = float(quote.get('ask', 0) or 0)
        bid_size = int(quote.get('bid_size', 0) or 0)
        ask_size = int(quote.get('ask_size', 0) or 0)
        last = float(quote.get('last', 0) or quote.get('price', 0) or 0)

        # Calculate mid price
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else last

        # Calculate spread
        spread = ask - bid if bid > 0 and ask > 0 else 0
        spread_percent = (spread / mid_price * 100) if mid_price > 0 else 0

        # Calculate total size
        total_size = bid_size + ask_size

        # Handle edge cases
        if total_size < self.min_total_size:
            return OrderFlowSignal(
                symbol=symbol,
                buy_pressure=0.5,
                sell_pressure=0.5,
                imbalance=0.0,
                spread_percent=spread_percent,
                recommendation='NEUTRAL',
                confidence=0.0,
                reason=f"Insufficient book depth ({total_size} shares)",
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                timestamp=datetime.now().isoformat()
            )

        if spread_percent > self.max_spread_percent * 2:
            return OrderFlowSignal(
                symbol=symbol,
                buy_pressure=bid_size / total_size,
                sell_pressure=ask_size / total_size,
                imbalance=(bid_size - ask_size) / total_size,
                spread_percent=spread_percent,
                recommendation='SKIP',
                confidence=0.9,
                reason=f"Spread too wide ({spread_percent:.1f}%)",
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                timestamp=datetime.now().isoformat()
            )

        # Calculate buy/sell pressure
        buy_pressure = bid_size / total_size
        sell_pressure = ask_size / total_size
        imbalance = (bid_size - ask_size) / total_size  # -1 to +1

        # Store in history for trend analysis
        self._add_to_history(symbol, buy_pressure)

        # Determine recommendation
        recommendation, confidence, reason = self._get_recommendation(
            buy_pressure, sell_pressure, spread_percent, symbol
        )

        return OrderFlowSignal(
            symbol=symbol,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            imbalance=imbalance,
            spread_percent=spread_percent,
            recommendation=recommendation,
            confidence=confidence,
            reason=reason,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=datetime.now().isoformat()
        )

    def _add_to_history(self, symbol: str, buy_pressure: float):
        """Track recent buy pressure readings"""
        if symbol not in self.history:
            self.history[symbol] = []
        self.history[symbol].append(buy_pressure)
        if len(self.history[symbol]) > self.history_max:
            self.history[symbol] = self.history[symbol][-self.history_max:]

    def _get_trend(self, symbol: str) -> str:
        """Analyze buy pressure trend"""
        if symbol not in self.history or len(self.history[symbol]) < 3:
            return 'unknown'

        recent = self.history[symbol][-3:]
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            return 'improving'  # Buy pressure increasing
        elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            return 'weakening'  # Buy pressure decreasing
        return 'stable'

    def _get_recommendation(
        self,
        buy_pressure: float,
        sell_pressure: float,
        spread_percent: float,
        symbol: str
    ) -> Tuple[str, float, str]:
        """
        Determine entry recommendation.

        Returns:
            (recommendation, confidence, reason)
        """
        trend = self._get_trend(symbol)

        # Strong buy pressure
        if buy_pressure >= self.strong_buy_pressure:
            if spread_percent <= 0.5:
                return ('ENTER', 0.95, f"Strong buy pressure ({buy_pressure*100:.0f}%), tight spread")
            elif spread_percent <= 1.0:
                return ('ENTER', 0.85, f"Strong buy pressure ({buy_pressure*100:.0f}%)")
            else:
                return ('ENTER', 0.70, f"Strong buy pressure, but spread is wide ({spread_percent:.1f}%)")

        # Moderate buy pressure
        elif buy_pressure >= self.min_buy_pressure:
            if trend == 'improving':
                return ('ENTER', 0.75, f"Buy pressure improving ({buy_pressure*100:.0f}%)")
            elif spread_percent <= 0.5:
                return ('ENTER', 0.65, f"Moderate buy pressure ({buy_pressure*100:.0f}%), tight spread")
            else:
                return ('NEUTRAL', 0.50, f"Moderate buy pressure ({buy_pressure*100:.0f}%)")

        # Weak buy pressure (sellers dominate)
        elif buy_pressure < 0.45:
            if trend == 'weakening':
                return ('SKIP', 0.90, f"Sellers dominating ({sell_pressure*100:.0f}% sell), pressure weakening")
            else:
                return ('SKIP', 0.80, f"Sellers dominating ({sell_pressure*100:.0f}% sell)")

        # Neutral zone
        else:
            if spread_percent > self.max_spread_percent:
                return ('SKIP', 0.60, f"Neutral with wide spread ({spread_percent:.1f}%)")
            return ('NEUTRAL', 0.40, f"Balanced book ({buy_pressure*100:.0f}% buy)")

    def get_summary(self, symbol: str) -> Dict:
        """Get summary of recent order flow for a symbol"""
        if symbol not in self.history or not self.history[symbol]:
            return {'symbol': symbol, 'readings': 0, 'avg_buy_pressure': 0, 'trend': 'unknown'}

        readings = self.history[symbol]
        return {
            'symbol': symbol,
            'readings': len(readings),
            'avg_buy_pressure': round(sum(readings) / len(readings) * 100, 1),
            'latest_buy_pressure': round(readings[-1] * 100, 1),
            'trend': self._get_trend(symbol)
        }


# Singleton instance
_analyzer = None


def get_order_flow_analyzer() -> OrderFlowAnalyzer:
    """Get singleton order flow analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = OrderFlowAnalyzer()
    return _analyzer


async def get_order_flow_signal(symbol: str, quote: Dict = None) -> OrderFlowSignal:
    """
    Get order flow signal for a symbol.

    Args:
        symbol: Stock ticker
        quote: Optional pre-fetched quote (saves API call)

    Returns:
        OrderFlowSignal with buy pressure and recommendation
    """
    analyzer = get_order_flow_analyzer()

    # Fetch quote if not provided
    if quote is None:
        try:
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()
            quote = schwab.get_quote(symbol)
            if not quote:
                return OrderFlowSignal(
                    symbol=symbol,
                    buy_pressure=0.5,
                    sell_pressure=0.5,
                    imbalance=0.0,
                    spread_percent=0.0,
                    recommendation='NEUTRAL',
                    confidence=0.0,
                    reason="Failed to fetch quote",
                    bid=0, ask=0, bid_size=0, ask_size=0,
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return OrderFlowSignal(
                symbol=symbol,
                buy_pressure=0.5,
                sell_pressure=0.5,
                imbalance=0.0,
                spread_percent=0.0,
                recommendation='NEUTRAL',
                confidence=0.0,
                reason=f"Error: {str(e)}",
                bid=0, ask=0, bid_size=0, ask_size=0,
                timestamp=datetime.now().isoformat()
            )

    return analyzer.analyze(quote)


async def check_entry_allowed(symbol: str, quote: Dict = None) -> Tuple[bool, str]:
    """
    Quick check if entry is allowed based on order flow.

    Returns:
        (allowed, reason)
    """
    signal = await get_order_flow_signal(symbol, quote)

    if signal.recommendation == 'ENTER':
        return True, f"Buy pressure {signal.buy_pressure*100:.0f}%"
    elif signal.recommendation == 'SKIP':
        return False, signal.reason
    else:
        # NEUTRAL - could go either way, allow with caution
        return True, f"Neutral ({signal.buy_pressure*100:.0f}% buy)"


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test with mock data
        test_quotes = [
            {"symbol": "AAPL", "bid": 180.50, "ask": 180.55, "bid_size": 500, "ask_size": 200},  # Strong buy
            {"symbol": "TSLA", "bid": 250.00, "ask": 250.10, "bid_size": 100, "ask_size": 400},  # Strong sell
            {"symbol": "NVDA", "bid": 500.00, "ask": 500.05, "bid_size": 300, "ask_size": 300},  # Neutral
            {"symbol": "SPY", "bid": 450.00, "ask": 450.02, "bid_size": 1000, "ask_size": 500},  # Buy pressure
        ]

        print("\nOrder Flow Analysis Test")
        print("=" * 60)

        for quote in test_quotes:
            signal = await get_order_flow_signal(quote['symbol'], quote)
            print(f"\n{quote['symbol']}:")
            print(f"  Bid: ${signal.bid:.2f} x {signal.bid_size}")
            print(f"  Ask: ${signal.ask:.2f} x {signal.ask_size}")
            print(f"  Buy Pressure: {signal.buy_pressure*100:.1f}%")
            print(f"  Spread: {signal.spread_percent:.2f}%")
            print(f"  Recommendation: {signal.recommendation} (confidence: {signal.confidence:.0%})")
            print(f"  Reason: {signal.reason}")

    asyncio.run(test())
