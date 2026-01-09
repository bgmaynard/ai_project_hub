"""
News-to-Trade Pipeline
=======================
Automated workflow: Breaking News -> Symbol Filter -> Strategy Validation ->
Watchlist -> Analysis -> Trade Signal

Designed for Day Trade Dash style trading.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeCandidate:
    """A stock that passed news + strategy validation"""

    symbol: str
    headline: str
    catalyst_type: str
    sentiment: str
    news_time: datetime
    detected_time: datetime

    # Validation results
    price: float = 0.0
    float_shares: float = 0.0
    volume: int = 0
    change_percent: float = 0.0

    # Analysis results
    ai_signal: str = ""
    ai_confidence: float = 0.0
    pattern: str = ""

    # Technical Entry Metrics (Added 12/16/2024)
    vwap: float = 0.0
    vwap_extension: float = 0.0  # % above/below VWAP
    high_of_day: float = 0.0
    percent_from_hod: float = 0.0  # How far below HOD
    relative_volume: float = 0.0  # Current vol / avg vol
    macd_crossover: bool = False  # Recent bullish MACD crossover
    pullback_detected: bool = False  # Pulled back from recent high

    # Entry Quality Score (0-100)
    entry_quality_score: float = 0.0
    entry_warnings: List[str] = field(default_factory=list)

    # Status
    status: str = "pending"  # pending, analyzing, ready, traded, expired
    added_to_watchlist: bool = False
    analysis_complete: bool = False

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "headline": self.headline[:100],
            "catalyst_type": self.catalyst_type,
            "sentiment": self.sentiment,
            "news_time": self.news_time.isoformat() if self.news_time else None,
            "detected_time": self.detected_time.isoformat(),
            "price": self.price,
            "float_shares": self.float_shares,
            "volume": self.volume,
            "change_percent": self.change_percent,
            "ai_signal": self.ai_signal,
            "ai_confidence": self.ai_confidence,
            "pattern": self.pattern,
            # Technical entry metrics
            "vwap": self.vwap,
            "vwap_extension": self.vwap_extension,
            "high_of_day": self.high_of_day,
            "percent_from_hod": self.percent_from_hod,
            "relative_volume": self.relative_volume,
            "macd_crossover": self.macd_crossover,
            "pullback_detected": self.pullback_detected,
            "entry_quality_score": self.entry_quality_score,
            "entry_warnings": self.entry_warnings,
            "status": self.status,
            "added_to_watchlist": self.added_to_watchlist,
            "analysis_complete": self.analysis_complete,
        }


@dataclass
class PipelineConfig:
    """Configuration for the news trade pipeline"""

    # Symbol filters
    min_price: float = 1.00
    max_price: float = 20.00
    max_float: float = 50_000_000  # 50M shares
    min_volume: int = 100_000
    min_change_percent: float = 5.0

    # Catalyst filters
    required_catalysts: List[str] = field(
        default_factory=lambda: [
            "fda",
            "merger",
            "earnings_beat",
            "upgrade",
            "fda_reject",
            "earnings_miss",
            "downgrade",
        ]
    )

    # Analysis thresholds
    min_ai_confidence: float = 0.5

    # ==========================================================================
    # TECHNICAL ENTRY FILTERS (Added based on trade analysis 12/16/2024)
    # ==========================================================================
    # These filters help avoid chasing extended moves (ASNS, AZI style losses)
    # and favor confirmed momentum entries (AMCI style wins)

    # VWAP Filter: Only enter if price is within X% of VWAP
    # Prevents chasing stocks extended far above VWAP
    max_vwap_extension: float = 15.0  # 15% above VWAP max

    # Momentum Confirmation: Require MACD crossover in last N bars
    require_macd_crossover: bool = True
    macd_lookback_bars: int = 5

    # Entry relative to high of day
    # Prevents buying at the tippy top of a spike
    max_percent_from_hod: float = 5.0  # Entry should be within 5% of HOD (not AT HOD)
    min_percent_from_hod: float = 1.0  # Don't buy exactly at HOD (waiting for pullback)

    # Volume confirmation: Current volume should be above average
    min_relative_volume: float = 2.0  # 2x average volume

    # Pullback filter: Prefer entries on pullbacks, not straight spikes
    prefer_pullback_entry: bool = True
    pullback_percent: float = 3.0  # Wait for 3% pullback from recent high

    # Risk/Reward minimum
    min_risk_reward: float = 2.0  # Target 2:1 R/R minimum
    # ==========================================================================

    # Timing
    max_news_age_seconds: int = 300  # 5 minutes
    analysis_timeout_seconds: int = 30

    # Auto-actions
    auto_add_to_watchlist: bool = True
    auto_run_analysis: bool = True
    auto_alert: bool = True

    # Watchlist monitoring
    monitor_watchlist: bool = True
    watchlist_poll_interval: int = 5  # seconds

    # Trading hours (EST)
    pre_market_start: int = 4  # 4 AM
    market_close: int = 16  # 4 PM


class NewsTradePipeline:
    """
    Main pipeline that connects:
    1. Breaking news detection (BenzingaFastNews)
    2. Symbol filtering (price, float, volume)
    3. Strategy validation
    4. Watchlist management
    5. Analysis pipeline (AI, backtest, predict)
    6. Trade alerts
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Components (lazy loaded)
        self._fast_news = None
        self._news_monitor = None
        self._market_data = None

        # State
        self.is_running = False
        self.candidates: Dict[str, TradeCandidate] = {}
        self.alerts: List[Dict] = []
        self.known_watchlist: set = set()  # Track known symbols to detect new adds
        self.stats = {
            "news_received": 0,
            "symbols_filtered": 0,
            "validation_passed": 0,
            "added_to_watchlist": 0,
            "manual_adds_detected": 0,
            "analysis_run": 0,
            "alerts_sent": 0,
            "started_at": None,
        }

        # Callbacks
        self.on_candidate: Optional[Callable] = None
        self.on_alert: Optional[Callable] = None

        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info("NewsTradePipeline initialized")

    def _get_fast_news(self):
        """Get fast news scanner (lazy load)"""
        if self._fast_news is None:
            try:
                from ai.benzinga_fast_news import get_fast_news

                self._fast_news = get_fast_news()
            except ImportError:
                logger.warning("BenzingaFastNews not available")
        return self._fast_news

    def _get_news_monitor(self):
        """Get news monitor (lazy load)"""
        if self._news_monitor is None:
            try:
                from ai.news_feed_monitor import get_news_monitor

                self._news_monitor = get_news_monitor()
            except ImportError:
                logger.warning("NewsFeedMonitor not available")
        return self._news_monitor

    async def _get_market_data(self, symbol: str) -> Dict:
        """Get market data for a symbol"""
        try:
            # Try unified market data first
            from unified_market_data import get_market_data

            md = get_market_data()
            quote = await md.get_quote(symbol)
            if quote:
                return quote
        except:
            pass

        try:
            # Fallback to direct Schwab
            from schwab_market_data import get_schwab_market_data

            md = get_schwab_market_data()
            quote = await md.get_quote(symbol)
            if quote:
                return quote
        except:
            pass

        return {}

    async def _get_float_data(self, symbol: str) -> float:
        """Get float shares for a symbol"""
        try:
            from ai.fundamental_analysis import get_stock_float

            result = await get_stock_float(symbol)
            if result and "float_shares" in result:
                return result["float_shares"] or 0
        except:
            pass

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("floatShares", 0) or 0
        except:
            pass

        return 0

    async def _add_to_watchlist(self, symbol: str) -> bool:
        """Add symbol to watchlist"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:9100/api/worklist",
                    json={"symbol": symbol},
                    timeout=5.0,
                )
                return response.status_code == 200
        except:
            pass

        try:
            # Direct approach
            from ai.intelligent_watchlist import get_intelligent_watchlist

            wl = get_intelligent_watchlist()
            wl.add_symbol(symbol)
            return True
        except:
            pass

        return False

    async def _run_analysis(self, symbol: str) -> Dict:
        """Run AI analysis on a symbol"""
        results = {"ai_signal": "", "ai_confidence": 0.0, "pattern": ""}

        try:
            # AI Prediction
            from ai.ai_predictor import get_predictor

            predictor = get_predictor()
            prediction = await asyncio.wait_for(
                asyncio.to_thread(predictor.predict, symbol),
                timeout=self.config.analysis_timeout_seconds,
            )
            if prediction:
                results["ai_signal"] = prediction.get("signal", "")
                results["ai_confidence"] = prediction.get("confidence", 0)
        except Exception as e:
            logger.debug(f"AI prediction error for {symbol}: {e}")

        try:
            # Chart patterns
            from ai.chart_patterns import get_pattern_detector

            detector = get_pattern_detector()
            patterns = await asyncio.wait_for(
                asyncio.to_thread(detector.analyze, symbol),
                timeout=self.config.analysis_timeout_seconds,
            )
            if patterns and patterns.get("patterns"):
                results["pattern"] = patterns["patterns"][0].get("pattern", "")
        except Exception as e:
            logger.debug(f"Pattern detection error for {symbol}: {e}")

        return results

    def _on_breaking_news(self, alert):
        """Callback when breaking news is detected"""
        self.stats["news_received"] += 1

        # Process in background
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_news_alert(alert), self._loop
            )

    async def _process_news_alert(self, alert):
        """Process a breaking news alert through the pipeline"""
        try:
            symbols = alert.symbols if hasattr(alert, "symbols") else []
            if not symbols:
                return

            for symbol in symbols:
                # Skip if already processing
                if symbol in self.candidates:
                    continue

                self.stats["symbols_filtered"] += 1

                # Create candidate
                candidate = TradeCandidate(
                    symbol=symbol,
                    headline=(
                        alert.headline if hasattr(alert, "headline") else str(alert)
                    ),
                    catalyst_type=(
                        alert.catalyst_type
                        if hasattr(alert, "catalyst_type")
                        else "unknown"
                    ),
                    sentiment=(
                        alert.sentiment if hasattr(alert, "sentiment") else "neutral"
                    ),
                    news_time=(
                        alert.published_at
                        if hasattr(alert, "published_at")
                        else datetime.now()
                    ),
                    detected_time=datetime.now(),
                )

                # Validate catalyst type
                if self.config.required_catalysts:
                    if candidate.catalyst_type not in self.config.required_catalysts:
                        logger.debug(
                            f"{symbol}: Catalyst {candidate.catalyst_type} not in required list"
                        )
                        continue

                # Get market data
                quote = await self._get_market_data(symbol)
                if quote:
                    candidate.price = quote.get("price", 0) or quote.get("last", 0)
                    candidate.volume = quote.get("volume", 0)
                    candidate.change_percent = quote.get("change_percent", 0)

                # Validate price range
                if (
                    candidate.price < self.config.min_price
                    or candidate.price > self.config.max_price
                ):
                    logger.debug(f"{symbol}: Price ${candidate.price} out of range")
                    continue

                # Validate volume
                if candidate.volume < self.config.min_volume:
                    logger.debug(f"{symbol}: Volume {candidate.volume} below minimum")
                    continue

                # Validate change percent
                if abs(candidate.change_percent) < self.config.min_change_percent:
                    logger.debug(
                        f"{symbol}: Change {candidate.change_percent}% below threshold"
                    )
                    continue

                # Get float data
                candidate.float_shares = await self._get_float_data(symbol)

                # Validate float (0 means unknown, allow it)
                if (
                    candidate.float_shares > 0
                    and candidate.float_shares > self.config.max_float
                ):
                    logger.debug(
                        f"{symbol}: Float {candidate.float_shares:,.0f} above maximum"
                    )
                    continue

                # Passed validation!
                self.stats["validation_passed"] += 1
                candidate.status = "validated"
                self.candidates[symbol] = candidate

                logger.info(
                    f"CANDIDATE: {symbol} - {candidate.catalyst_type} - ${candidate.price:.2f} - {candidate.headline[:50]}..."
                )

                # Auto add to watchlist
                if self.config.auto_add_to_watchlist:
                    success = await self._add_to_watchlist(symbol)
                    candidate.added_to_watchlist = success
                    if success:
                        self.stats["added_to_watchlist"] += 1
                        logger.info(f"  Added {symbol} to watchlist")

                # Auto run analysis
                if self.config.auto_run_analysis:
                    candidate.status = "analyzing"
                    analysis = await self._run_analysis(symbol)
                    candidate.ai_signal = analysis.get("ai_signal", "")
                    candidate.ai_confidence = analysis.get("ai_confidence", 0)
                    candidate.pattern = analysis.get("pattern", "")
                    candidate.analysis_complete = True
                    candidate.status = "ready"
                    self.stats["analysis_run"] += 1

                # Generate alert
                if self.config.auto_alert:
                    alert_data = self._generate_alert(candidate)
                    self.alerts.append(alert_data)
                    self.stats["alerts_sent"] += 1

                    if self.on_alert:
                        try:
                            self.on_alert(alert_data)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")

                # Candidate callback
                if self.on_candidate:
                    try:
                        self.on_candidate(candidate)
                    except Exception as e:
                        logger.error(f"Candidate callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing news alert: {e}")

    def _generate_alert(self, candidate: TradeCandidate) -> Dict:
        """Generate a trade alert from a candidate"""

        # Determine action
        action = "WATCH"
        if candidate.sentiment == "bullish":
            if candidate.ai_confidence >= self.config.min_ai_confidence:
                action = "BUY"
            else:
                action = "WATCH LONG"
        elif candidate.sentiment == "bearish":
            action = "AVOID" if candidate.ai_signal == "BEARISH" else "WATCH SHORT"

        alert = {
            "timestamp": datetime.now().isoformat(),
            "symbol": candidate.symbol,
            "action": action,
            "headline": candidate.headline[:100],
            "catalyst": candidate.catalyst_type,
            "sentiment": candidate.sentiment,
            "price": candidate.price,
            "change_percent": candidate.change_percent,
            "float": candidate.float_shares,
            "volume": candidate.volume,
            "ai_signal": candidate.ai_signal,
            "ai_confidence": candidate.ai_confidence,
            "pattern": candidate.pattern,
            "priority": (
                "HIGH"
                if candidate.catalyst_type in ["fda", "merger", "earnings_beat"]
                else "MEDIUM"
            ),
        }

        # Log the alert
        logger.warning(
            f"\n{'='*60}\n"
            f"TRADE ALERT: {action} {candidate.symbol}\n"
            f"Price: ${candidate.price:.2f} ({candidate.change_percent:+.1f}%)\n"
            f"Catalyst: {candidate.catalyst_type} | Sentiment: {candidate.sentiment}\n"
            f"Float: {candidate.float_shares:,.0f} | Volume: {candidate.volume:,}\n"
            f"AI: {candidate.ai_signal} ({candidate.ai_confidence:.0%})\n"
            f"News: {candidate.headline[:80]}...\n"
            f"{'='*60}"
        )

        return alert

    def start(self, watchlist: List[str] = None):
        """Start the news trade pipeline"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return

        self.is_running = True
        self.stats["started_at"] = datetime.now().isoformat()

        # Start fast news scanner with our callback
        fast_news = self._get_fast_news()
        if fast_news:
            fast_news.on_breaking_news = self._on_breaking_news
            fast_news.start(watchlist)

        # Start background processing thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"NewsTradePipeline STARTED - watching {len(watchlist or [])} symbols"
        )

    def stop(self):
        """Stop the pipeline"""
        self.is_running = False

        # Stop fast news
        fast_news = self._get_fast_news()
        if fast_news:
            fast_news.stop()

        # Stop thread
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("NewsTradePipeline STOPPED")

    def _run_loop(self):
        """Background event loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._background_tasks())
        except Exception as e:
            logger.error(f"Pipeline loop error: {e}")
        finally:
            self._loop.close()

    async def _background_tasks(self):
        """Background tasks - cleanup, watchlist monitoring"""
        last_watchlist_check = 0

        while self.is_running:
            try:
                now = datetime.now()

                # Monitor watchlist for manual adds
                if self.config.monitor_watchlist:
                    if (
                        time.time() - last_watchlist_check
                        >= self.config.watchlist_poll_interval
                    ):
                        await self._check_watchlist_changes()
                        last_watchlist_check = time.time()

                # Clean up old candidates
                expired = []
                for symbol, candidate in self.candidates.items():
                    age = (now - candidate.detected_time).total_seconds()
                    if age > self.config.max_news_age_seconds * 2:
                        expired.append(symbol)

                for symbol in expired:
                    del self.candidates[symbol]

                await asyncio.sleep(1)  # Check more frequently

            except Exception as e:
                logger.error(f"Background task error: {e}")
                await asyncio.sleep(5)

    async def _check_watchlist_changes(self):
        """Check for new symbols added to watchlist manually"""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:9100/api/worklist", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    current_symbols = set()

                    for item in data.get("data", []):
                        symbol = item.get("symbol")
                        if symbol:
                            current_symbols.add(symbol)

                    # Find new symbols (added manually)
                    new_symbols = current_symbols - self.known_watchlist

                    for symbol in new_symbols:
                        # Skip if already a candidate
                        if symbol not in self.candidates:
                            logger.info(f"MANUAL ADD DETECTED: {symbol}")
                            self.stats["manual_adds_detected"] += 1
                            await self._process_manual_add(symbol)

                    # Update known watchlist
                    self.known_watchlist = current_symbols

        except Exception as e:
            logger.debug(f"Watchlist check error: {e}")

    async def _process_manual_add(self, symbol: str):
        """Process a manually added symbol through the pipeline"""
        try:
            # Create candidate from manual add
            candidate = TradeCandidate(
                symbol=symbol,
                headline=f"Manual add - {symbol}",
                catalyst_type="manual",
                sentiment="neutral",
                news_time=datetime.now(),
                detected_time=datetime.now(),
            )

            # Get market data
            quote = await self._get_market_data(symbol)
            if quote:
                candidate.price = quote.get("price", 0) or quote.get("last", 0)
                candidate.volume = quote.get("volume", 0)
                candidate.change_percent = quote.get("change_percent", 0)

            # Get float
            candidate.float_shares = await self._get_float_data(symbol)

            # Store candidate
            candidate.status = "validated"
            candidate.added_to_watchlist = True
            self.candidates[symbol] = candidate

            # Run analysis
            if self.config.auto_run_analysis:
                candidate.status = "analyzing"
                analysis = await self._run_analysis(symbol)
                candidate.ai_signal = analysis.get("ai_signal", "")
                candidate.ai_confidence = analysis.get("ai_confidence", 0)
                candidate.pattern = analysis.get("pattern", "")
                candidate.analysis_complete = True
                candidate.status = "ready"
                self.stats["analysis_run"] += 1

                logger.info(
                    f"MANUAL ADD PROCESSED: {symbol} - "
                    f"${candidate.price:.2f} ({candidate.change_percent:+.1f}%) - "
                    f"AI: {candidate.ai_signal} ({candidate.ai_confidence:.0%})"
                )

            # Generate alert if significant
            if self.config.auto_alert and abs(candidate.change_percent) >= 5:
                alert_data = self._generate_alert(candidate)
                self.alerts.append(alert_data)
                self.stats["alerts_sent"] += 1

                if self.on_alert:
                    try:
                        self.on_alert(alert_data)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing manual add {symbol}: {e}")

    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            "is_running": self.is_running,
            "config": {
                "min_price": self.config.min_price,
                "max_price": self.config.max_price,
                "max_float": self.config.max_float,
                "min_volume": self.config.min_volume,
                "min_change_percent": self.config.min_change_percent,
                "auto_add_to_watchlist": self.config.auto_add_to_watchlist,
                "auto_run_analysis": self.config.auto_run_analysis,
                "auto_alert": self.config.auto_alert,
            },
            "stats": self.stats,
            "active_candidates": len(self.candidates),
            "recent_alerts": len(self.alerts),
        }

    def get_candidates(self) -> List[Dict]:
        """Get current trade candidates"""
        return [c.to_dict() for c in self.candidates.values()]

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts[-limit:]

    def update_config(self, **kwargs):
        """Update pipeline configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")


# Singleton instance
_pipeline: Optional[NewsTradePipeline] = None


def get_news_trade_pipeline() -> NewsTradePipeline:
    """Get or create the news trade pipeline singleton"""
    global _pipeline
    if _pipeline is None:
        _pipeline = NewsTradePipeline()
    return _pipeline


def start_news_trade_pipeline(
    watchlist: List[str] = None, config: Dict = None
) -> NewsTradePipeline:
    """Start the news trade pipeline"""
    pipeline = get_news_trade_pipeline()

    if config:
        pipeline.update_config(**config)

    pipeline.start(watchlist)
    return pipeline


def stop_news_trade_pipeline():
    """Stop the news trade pipeline"""
    pipeline = get_news_trade_pipeline()
    pipeline.stop()


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_alert(alert):
        print(f"\n*** ALERT: {alert['action']} {alert['symbol']} ***\n")

    pipeline = start_news_trade_pipeline(
        watchlist=["AAPL", "TSLA", "NVDA"],
        config={"min_price": 1.0, "max_price": 50.0, "min_change_percent": 3.0},
    )

    pipeline.on_alert = on_alert

    print("Pipeline running... Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(10)
            status = pipeline.get_status()
            print(f"Stats: {status['stats']}")
    except KeyboardInterrupt:
        stop_news_trade_pipeline()
        print("Stopped")
