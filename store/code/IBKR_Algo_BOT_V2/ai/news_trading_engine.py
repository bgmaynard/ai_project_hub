"""
News Trading Engine
====================
Combines Benzinga fast news + spike validation into a complete trading system.

Flow:
1. Benzinga detects breaking news
2. Immediately record pre-news price
3. Wait 5-15 seconds for algos to spike
4. Validate the spike (volume, price action, spread, news quality)
5. If validated -> BUY
6. If fading -> WAIT or AVOID
7. Monitor position with spread/VWAP tracking

This is the complete news-based momentum trading system.
"""

import requests
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import pytz

# Import our modules
from .benzinga_fast_news import get_fast_news, BreakingNews
from .news_spike_validator import get_spike_validator, SpikeValidation
from .spread_monitor import get_spread_monitor
from .vwap_tracker import get_vwap_tracker

logger = logging.getLogger(__name__)


@dataclass
class NewsTradeSignal:
    """Complete trade signal from news"""
    symbol: str
    action: str  # BUY, WAIT, AVOID

    # News info
    headline: str
    catalyst: str
    news_confidence: float

    # Validation info
    validation: Optional[SpikeValidation]

    # Entry info
    entry_price: float
    target_price: float
    stop_price: float
    position_size: int

    # Risk info
    spread_pct: float
    spread_safe: bool

    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "headline": self.headline[:60],
            "catalyst": self.catalyst,
            "news_confidence": f"{self.news_confidence:.0%}",
            "entry": f"${self.entry_price:.2f}",
            "target": f"${self.target_price:.2f}",
            "stop": f"${self.stop_price:.2f}",
            "size": self.position_size,
            "spread": f"{self.spread_pct:.2f}%",
            "spread_safe": self.spread_safe,
            "validation_verdict": self.validation.verdict if self.validation else "N/A",
            "validation_score": self.validation.overall_score if self.validation else 0,
            "timestamp": self.timestamp.isoformat()
        }


class NewsTradingEngine:
    """
    Complete news-based trading system.
    Detects news -> Validates spike -> Generates trade signals.
    """

    def __init__(self, api_url: str = "http://localhost:9100/api/alpaca"):
        self.api_url = api_url
        self.et_tz = pytz.timezone('US/Eastern')

        # Get our component modules
        self.news_scanner = get_fast_news()
        self.spike_validator = get_spike_validator()
        self.spread_monitor = get_spread_monitor()
        self.vwap_tracker = get_vwap_tracker()

        # Trading parameters
        self.max_position_value = 1000  # Max $ per position
        self.min_news_confidence = 0.6
        self.min_validation_score = 60
        self.validation_wait_seconds = 8  # Wait before validating

        # Risk parameters
        self.default_stop_pct = 3.0  # 3% stop loss
        self.default_target_pct = 5.0  # 5% profit target
        self.max_spread_pct = 2.0

        # Signals
        self.signals: List[NewsTradeSignal] = []
        self.max_signals = 100

        # Active trades
        self.active_trades: Dict[str, Dict] = {}

        # Callbacks
        self.on_trade_signal: Optional[Callable] = None
        self.on_trade_executed: Optional[Callable] = None

        # Control
        self.is_running = False
        self.auto_trade = False  # Set True to auto-execute trades

        # Track what news we've processed
        self.processed_news: set = set()

        logger.info("NewsTradingEngine initialized")

    def start(self, watchlist: List[str] = None, auto_trade: bool = False):
        """Start the news trading engine"""
        self.auto_trade = auto_trade

        # Set up news scanner callback
        self.news_scanner.on_breaking_news = self._on_news_alert
        self.news_scanner.on_buy_signal = self._on_news_buy_signal

        # Start the news scanner
        self.news_scanner.start(watchlist)

        # Start spread monitor with our positions
        self.spread_monitor.start_monitoring()

        self.is_running = True
        logger.info(f"NewsTradingEngine STARTED - auto_trade={auto_trade}")

    def stop(self):
        """Stop the engine"""
        self.is_running = False
        self.news_scanner.stop()
        self.spread_monitor.stop_monitoring()
        self.vwap_tracker.stop_monitoring()
        logger.info("NewsTradingEngine STOPPED")

    def _on_news_alert(self, news: BreakingNews):
        """Called when any breaking news is detected"""
        # Record pre-news prices for all symbols mentioned
        for symbol in news.symbols:
            try:
                r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
                if r.status_code == 200:
                    q = r.json()
                    price = float(q.get('last', q.get('ask', 0)))
                    if price > 0:
                        self.spike_validator.record_pre_news_price(symbol, price)
                        logger.info(f"Recorded pre-news price for {symbol}: ${price:.2f}")
            except:
                pass

    def _on_news_buy_signal(self, news: BreakingNews):
        """Called when news scanner detects a potential buy"""
        news_id = news.id

        # Skip if already processed
        if news_id in self.processed_news:
            return
        self.processed_news.add(news_id)

        # Process each symbol
        for symbol in news.symbols[:3]:  # Max 3 symbols per news
            threading.Thread(
                target=self._process_news_signal,
                args=(symbol, news),
                daemon=True
            ).start()

    def _process_news_signal(self, symbol: str, news: BreakingNews):
        """Process a news signal - validate and generate trade signal"""
        try:
            logger.info(f"Processing news signal for {symbol}: {news.headline[:50]}...")

            # Wait for algos to spike the price
            logger.info(f"Waiting {self.validation_wait_seconds}s for spike to develop...")
            time.sleep(self.validation_wait_seconds)

            # Validate the spike
            validation = self.spike_validator.validate_spike_sync(
                symbol=symbol,
                headline=news.headline,
                catalyst_type=news.catalyst_type,
                wait_seconds=5  # Additional wait during validation
            )

            # Log validation result
            logger.info(
                f"Validation for {symbol}: {validation.verdict} "
                f"(score={validation.overall_score}, confidence={validation.confidence:.0%})"
            )

            # Generate trade signal
            signal = self._generate_trade_signal(symbol, news, validation)

            if signal:
                self.signals.append(signal)
                if len(self.signals) > self.max_signals:
                    self.signals = self.signals[-self.max_signals:]

                # Log signal
                logger.warning(
                    f"\n{'='*50}\n"
                    f"NEWS TRADE SIGNAL: {signal.action}\n"
                    f"Symbol: {signal.symbol}\n"
                    f"News: {signal.headline[:60]}\n"
                    f"Entry: ${signal.entry_price:.2f} | Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}\n"
                    f"Size: {signal.position_size} shares\n"
                    f"{'='*50}"
                )

                # Callback
                if self.on_trade_signal:
                    self.on_trade_signal(signal)

                # Auto-execute if enabled
                if self.auto_trade and signal.action == "BUY":
                    self._execute_trade(signal)

        except Exception as e:
            logger.error(f"Error processing news signal for {symbol}: {e}")

    def _generate_trade_signal(self, symbol: str, news: BreakingNews,
                              validation: SpikeValidation) -> Optional[NewsTradeSignal]:
        """Generate a trade signal from news and validation"""
        now = datetime.now(self.et_tz)

        # Get current quote for entry price
        try:
            r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
            if r.status_code != 200:
                return None

            q = r.json()
            current_price = float(q.get('last', 0)) or float(q.get('ask', 0))
            bid = float(q.get('bid', 0))
            ask = float(q.get('ask', 0))

            if current_price <= 0:
                return None

            spread = ask - bid if ask > bid else 0
            spread_pct = (spread / bid * 100) if bid > 0 else 0
            spread_safe = spread_pct < self.max_spread_pct

        except:
            return None

        # Determine action based on validation
        if validation.verdict == "BUY_NOW":
            action = "BUY"
        elif validation.verdict == "WAIT_PULLBACK":
            action = "WAIT"
        else:
            action = "AVOID"

        # Additional checks
        if not spread_safe:
            action = "AVOID"

        if validation.overall_score < self.min_validation_score:
            if action == "BUY":
                action = "WAIT"

        if news.confidence < self.min_news_confidence:
            if action == "BUY":
                action = "WAIT"

        # Calculate position size
        position_size = max(1, int(self.max_position_value / current_price))

        # Calculate targets and stops
        if validation.spike_pct > 0:
            # Use spike as reference for targets
            stop_price = current_price * (1 - self.default_stop_pct / 100)
            target_price = current_price * (1 + self.default_target_pct / 100)
        else:
            stop_price = current_price * 0.97
            target_price = current_price * 1.05

        return NewsTradeSignal(
            symbol=symbol,
            action=action,
            headline=news.headline,
            catalyst=news.catalyst_type,
            news_confidence=news.confidence,
            validation=validation,
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size=position_size,
            spread_pct=spread_pct,
            spread_safe=spread_safe,
            timestamp=now
        )

    def _execute_trade(self, signal: NewsTradeSignal):
        """Execute a trade based on signal"""
        try:
            order = {
                'symbol': signal.symbol,
                'quantity': signal.position_size,
                'action': 'buy',
                'order_type': 'market',
                'time_in_force': 'day',
                'extended_hours': True
            }

            r = requests.post(f"{self.api_url}/place-order", json=order, timeout=5)
            result = r.json()

            if result.get('success'):
                logger.warning(f"TRADE EXECUTED: Bought {signal.position_size} {signal.symbol}")

                # Track the trade
                self.active_trades[signal.symbol] = {
                    'entry': signal.entry_price,
                    'target': signal.target_price,
                    'stop': signal.stop_price,
                    'size': signal.position_size,
                    'signal': signal.to_dict(),
                    'timestamp': datetime.now(self.et_tz).isoformat()
                }

                # Add to spread monitoring
                self.spread_monitor.add_position(signal.symbol)

                # Add to VWAP tracking
                self.vwap_tracker.add_symbol(signal.symbol)

                if self.on_trade_executed:
                    self.on_trade_executed(signal, result)

            else:
                logger.error(f"Trade failed: {result}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def get_signals(self, limit: int = 20, action: str = None) -> List[Dict]:
        """Get recent signals"""
        signals = self.signals[-limit:]
        if action:
            signals = [s for s in signals if s.action == action]
        return [s.to_dict() for s in signals]

    def get_buy_signals(self) -> List[Dict]:
        """Get recent BUY signals"""
        return self.get_signals(action="BUY")

    def get_active_trades(self) -> Dict:
        """Get active trades"""
        return self.active_trades

    def get_status(self) -> Dict:
        """Get engine status"""
        return {
            "is_running": self.is_running,
            "auto_trade": self.auto_trade,
            "news_scanner_running": self.news_scanner.is_running,
            "news_detected": self.news_scanner.news_detected,
            "signals_generated": len(self.signals),
            "active_trades": len(self.active_trades),
            "processed_news": len(self.processed_news)
        }


# Singleton
_engine: Optional[NewsTradingEngine] = None


def get_news_trading_engine() -> NewsTradingEngine:
    """Get or create news trading engine singleton"""
    global _engine
    if _engine is None:
        _engine = NewsTradingEngine()
    return _engine


def start_news_trading(watchlist: List[str] = None, auto_trade: bool = False,
                      on_signal: Callable = None) -> NewsTradingEngine:
    """Start the news trading engine"""
    engine = get_news_trading_engine()

    if on_signal:
        engine.on_trade_signal = on_signal

    engine.start(watchlist, auto_trade)
    return engine


def stop_news_trading():
    """Stop the news trading engine"""
    engine = get_news_trading_engine()
    engine.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_signal(signal):
        print(f"\n*** SIGNAL: {signal.action} {signal.symbol} @ ${signal.entry_price:.2f} ***")
        print(f"    News: {signal.headline[:50]}...")
        print(f"    Validation: {signal.validation.verdict if signal.validation else 'N/A'}")

    # Start with some momentum stocks
    watchlist = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'META']

    engine = start_news_trading(
        watchlist=watchlist,
        auto_trade=False,  # Set True to actually trade
        on_signal=on_signal
    )

    print(f"News Trading Engine running...")
    print(f"Watching: {watchlist}")
    print(f"Auto-trade: {engine.auto_trade}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(10)
            status = engine.get_status()
            print(f"\nStatus: {status}")

            signals = engine.get_buy_signals()
            if signals:
                print(f"BUY signals: {[s['symbol'] for s in signals]}")

    except KeyboardInterrupt:
        stop_news_trading()
        print("\nStopped")
