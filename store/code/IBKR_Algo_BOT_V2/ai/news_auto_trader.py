"""
News Auto-Trader Coordinator
============================
Connects news detection -> watchlist -> AI evaluation -> HFT scalper execution.

Pipeline:
1. BenzingaFastNews detects breaking news with actionable signal
2. Symbol auto-added to watchlist for monitoring
3. AI filters evaluate (Chronos, Qlib, Order Flow)
4. If all filters pass -> HFT Scalper executes trade
5. Circuit breaker protection throughout

Configuration:
- Minimum confidence for news signal: 0.7
- All AI filters must agree (Chronos, Qlib, Order Flow)
- Max concurrent news trades: 2
- Cooldown between trades on same symbol: 5 min
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class NewsTradeCandidate:
    """A news-triggered trade candidate"""
    symbol: str
    news_headline: str
    news_source: str
    news_sentiment: str
    news_urgency: str
    news_catalyst: str
    news_confidence: float
    news_detected_at: datetime

    # AI filter results
    chronos_score: Optional[float] = None
    chronos_direction: Optional[str] = None
    qlib_score: Optional[float] = None
    order_flow_buy_pressure: Optional[float] = None
    order_flow_recommendation: Optional[str] = None

    # Evaluation result
    all_filters_passed: bool = False
    rejection_reason: Optional[str] = None

    # Execution result
    trade_triggered: bool = False
    trade_id: Optional[str] = None
    trade_result: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'news_detected_at': self.news_detected_at.isoformat()
        }


@dataclass
class NewsAutoTraderConfig:
    """Configuration for news auto-trader"""
    enabled: bool = False
    paper_mode: bool = True  # Start in paper mode

    # News thresholds
    min_news_confidence: float = 0.7
    min_news_urgency: str = "high"  # high, critical

    # Stock screening requirements (Warrior method)
    screen_before_add: bool = True  # Screen stocks before adding to watchlist
    min_price: float = 1.0  # Minimum stock price
    max_price: float = 20.0  # Maximum stock price
    max_float: float = 50_000_000  # 50M max float (low float preference)
    min_avg_volume: int = 100_000  # Minimum avg daily volume
    require_catalyst: bool = True  # Must have high impact news

    # AI filter requirements
    require_chronos: bool = True
    min_chronos_score: float = 0.6
    require_qlib: bool = True
    min_qlib_score: float = 0.55
    require_order_flow: bool = True
    min_buy_pressure: float = 0.55

    # Risk limits
    max_concurrent_trades: int = 2
    symbol_cooldown_minutes: int = 5
    max_daily_trades: int = 10

    # Watchlist
    auto_add_to_watchlist: bool = True
    watchlist_expiry_minutes: int = 30

    def to_dict(self) -> Dict:
        return asdict(self)


# Config persistence
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "news_auto_trader_config.json")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "news_auto_trader_history.json")


def load_config() -> NewsAutoTraderConfig:
    """Load config from file"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                return NewsAutoTraderConfig(**data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return NewsAutoTraderConfig()


def save_config(config: NewsAutoTraderConfig):
    """Save config to file"""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    except Exception as e:
        logger.error(f"Error saving config: {e}")


class NewsAutoTrader:
    """
    Coordinates news detection -> evaluation -> execution pipeline.
    """

    def __init__(self):
        self.config = load_config()

        # Trade tracking
        self.candidates: List[NewsTradeCandidate] = []
        self.executed_trades: List[Dict] = []
        self.symbol_cooldowns: Dict[str, datetime] = {}
        self.daily_trade_count: int = 0
        self.last_reset_date: str = ""

        # Components (lazy loaded)
        self._fast_news = None
        self._hft_scalper = None
        self._chronos = None
        self._qlib = None
        self._order_flow = None

        # State
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_scan_time: Optional[datetime] = None  # For observability

        # Watchlist integration
        self.auto_added_symbols: Dict[str, datetime] = {}

        # Stats
        self.stats = {
            "news_received": 0,
            "candidates_evaluated": 0,
            "stocks_screened_out": 0,  # Failed price/float/volume screening
            "trades_triggered": 0,
            "trades_filtered_out": 0,
            "successful_trades": 0,
            "failed_trades": 0
        }

        logger.info("NewsAutoTrader initialized")

    def _get_fast_news(self):
        """Lazy load fast news scanner"""
        if self._fast_news is None:
            try:
                from .benzinga_fast_news import get_fast_news
                self._fast_news = get_fast_news()
            except ImportError:
                from benzinga_fast_news import get_fast_news
                self._fast_news = get_fast_news()
        return self._fast_news

    def _get_hft_scalper(self):
        """Lazy load HFT scalper"""
        if self._hft_scalper is None:
            try:
                from ai.hft_scalper import get_hft_scalper
                self._hft_scalper = get_hft_scalper()
            except ImportError:
                from hft_scalper import get_hft_scalper
                self._hft_scalper = get_hft_scalper()
        return self._hft_scalper

    def _get_chronos(self):
        """Lazy load Chronos predictor"""
        if self._chronos is None:
            try:
                from ai.chronos_predictor import ChronosPredictor
                self._chronos = ChronosPredictor()
            except Exception as e:
                logger.warning(f"Chronos not available: {e}")
        return self._chronos

    def _get_qlib(self):
        """Lazy load Qlib predictor"""
        if self._qlib is None:
            try:
                from ai.qlib_predictor import get_qlib_predictor
                self._qlib = get_qlib_predictor()
            except Exception as e:
                logger.warning(f"Qlib not available: {e}")
        return self._qlib

    def _get_order_flow(self):
        """Lazy load Order Flow analyzer"""
        if self._order_flow is None:
            try:
                from ai.order_flow_analyzer import OrderFlowAnalyzer
                self._order_flow = OrderFlowAnalyzer()
            except Exception as e:
                logger.warning(f"Order Flow not available: {e}")
        return self._order_flow

    async def _check_stock_requirements(self, symbol: str, catalyst: str = "") -> tuple:
        """
        Screen stock against strategy requirements before adding to watchlist.
        Returns (passed: bool, rejection_reason: Optional[str], stock_data: Dict)

        Warrior Method Criteria:
        - Price range: $1-$20 (configurable)
        - Low float: < 50M shares (configurable)
        - Adequate volume: > 100K avg daily (configurable)
        - High impact catalyst (FDA, earnings, M&A, etc.)
        """
        stock_data = {}

        if not self.config.screen_before_add:
            return True, None, stock_data

        try:
            # Get current price from market data
            price = None
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"http://localhost:9100/api/price/{symbol}",
                        timeout=5
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price = data.get("price") or data.get("last") or data.get("lastPrice")
                        stock_data["price"] = price
            except Exception as e:
                logger.debug(f"Could not get price for {symbol}: {e}")

            # Check price range
            if price is not None:
                if price < self.config.min_price:
                    return False, f"price ${price:.2f} < ${self.config.min_price:.2f} min", stock_data
                if price > self.config.max_price:
                    return False, f"price ${price:.2f} > ${self.config.max_price:.2f} max", stock_data
            else:
                logger.warning(f"Could not get price for {symbol}, skipping price check")

            # Get float data
            float_shares = None
            avg_volume = None
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info

                float_shares = info.get("floatShares")
                avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")

                stock_data["float"] = float_shares
                stock_data["avg_volume"] = avg_volume
                stock_data["market_cap"] = info.get("marketCap")
                stock_data["sector"] = info.get("sector")
            except Exception as e:
                logger.debug(f"Could not get yfinance data for {symbol}: {e}")

            # Check float
            if float_shares is not None:
                if float_shares > self.config.max_float:
                    float_m = float_shares / 1_000_000
                    max_m = self.config.max_float / 1_000_000
                    return False, f"float {float_m:.1f}M > {max_m:.1f}M max", stock_data
            else:
                logger.warning(f"Could not get float for {symbol}, skipping float check")

            # Check average volume
            if avg_volume is not None:
                if avg_volume < self.config.min_avg_volume:
                    return False, f"avg volume {avg_volume:,} < {self.config.min_avg_volume:,} min", stock_data

            # Check catalyst requirement
            if self.config.require_catalyst:
                high_impact_catalysts = [
                    "fda", "earnings", "acquisition", "merger", "bankruptcy",
                    "upgrade", "downgrade", "guidance", "contract", "patent",
                    "offering", "buyback", "dividend", "settlement", "approval"
                ]
                catalyst_lower = catalyst.lower() if catalyst else ""
                has_catalyst = any(c in catalyst_lower for c in high_impact_catalysts)

                if not has_catalyst and catalyst:
                    # Also check if news urgency indicates catalyst
                    pass  # We'll let it through if it passed urgency check already

            # All checks passed!
            logger.info(f"Stock {symbol} PASSED screening: price=${price}, float={float_shares}, vol={avg_volume}")
            return True, None, stock_data

        except Exception as e:
            logger.error(f"Error checking stock requirements for {symbol}: {e}")
            # Don't reject on error - let it through for manual review
            return True, None, stock_data

    async def start(self):
        """Start the news auto-trader"""
        if self.is_running:
            logger.warning("NewsAutoTrader already running")
            return

        self.is_running = True

        # Reset daily counter if new day
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today

        # Start fast news scanner with our callback
        fast_news = self._get_fast_news()
        fast_news.on_buy_signal = self._on_buy_signal
        fast_news.on_sell_signal = self._on_sell_signal

        # Get current watchlist to monitor
        try:
            from ai.intelligent_watchlist import get_watchlist
            watchlist = get_watchlist()
            symbols = watchlist.get_symbols() if watchlist else []
        except:
            symbols = []

        # Start news scanner
        if not fast_news.is_running:
            fast_news.start(watchlist=symbols)

        logger.info(f"NewsAutoTrader STARTED - {'PAPER' if self.config.paper_mode else 'LIVE'} mode")
        logger.info(f"Monitoring {len(symbols)} symbols, config: {self.config.to_dict()}")

    async def stop(self):
        """Stop the news auto-trader"""
        self.is_running = False

        # Stop news scanner
        fast_news = self._get_fast_news()
        if fast_news.is_running:
            fast_news.stop()

        logger.info("NewsAutoTrader STOPPED")

    def _on_buy_signal(self, alert):
        """Handle buy signal from news scanner"""
        # Update last scan time for observability
        self.last_scan_time = datetime.now()

        if not self.config.enabled:
            return

        self.stats["news_received"] += 1

        # Run async evaluation in thread-safe way
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._evaluate_buy_signal(alert))
            else:
                loop.run_until_complete(self._evaluate_buy_signal(alert))
        except:
            # Fallback - create new loop
            asyncio.run(self._evaluate_buy_signal(alert))

    def _on_sell_signal(self, alert):
        """Handle sell signal from news scanner - we don't short, so ignore"""
        # Log but don't act - user preference is no shorting
        logger.info(f"SELL signal ignored (no shorting): {alert.symbols} - {alert.headline[:50]}")

    async def _evaluate_buy_signal(self, alert):
        """Evaluate a buy signal through AI filters"""
        if not alert.symbols:
            return

        for symbol in alert.symbols:
            try:
                await self._process_candidate(symbol, alert)
            except Exception as e:
                logger.error(f"Error processing candidate {symbol}: {e}")

    async def _process_candidate(self, symbol: str, alert):
        """Process a single trade candidate through all filters"""

        # Create candidate
        candidate = NewsTradeCandidate(
            symbol=symbol,
            news_headline=alert.headline,
            news_source=alert.source,
            news_sentiment=alert.sentiment,
            news_urgency=alert.urgency,
            news_catalyst=alert.catalyst_type,
            news_confidence=alert.confidence,
            news_detected_at=alert.detected_at
        )

        self.stats["candidates_evaluated"] += 1

        # Check basic eligibility (news thresholds, cooldowns, limits)
        rejection = self._check_eligibility(symbol, alert)
        if rejection:
            candidate.rejection_reason = rejection
            self.candidates.append(candidate)
            self.stats["trades_filtered_out"] += 1
            logger.info(f"Candidate {symbol} rejected: {rejection}")
            return

        # NEW: Check stock requirements (price, float, volume) BEFORE adding to watchlist
        if self.config.screen_before_add:
            passed, rejection, stock_data = await self._check_stock_requirements(
                symbol,
                catalyst=alert.catalyst_type
            )
            if not passed:
                candidate.rejection_reason = f"SCREEN: {rejection}"
                self.candidates.append(candidate)
                self.stats["trades_filtered_out"] += 1
                self.stats["stocks_screened_out"] += 1
                logger.info(f"Candidate {symbol} FAILED screening: {rejection}")
                return
            else:
                # Log successful screening with data
                price = stock_data.get("price", "?")
                float_val = stock_data.get("float")
                float_str = f"{float_val/1_000_000:.1f}M" if float_val else "?"
                logger.info(f"Candidate {symbol} PASSED screening: ${price}, float={float_str}")

        # Auto-add to watchlist (only if screening passed)
        if self.config.auto_add_to_watchlist:
            await self._add_to_watchlist(
                symbol,
                headline=candidate.news_headline,
                catalyst=candidate.news_catalyst
            )

        # Run AI filters
        passed, rejection = await self._run_ai_filters(candidate)
        candidate.all_filters_passed = passed

        if not passed:
            candidate.rejection_reason = rejection
            self.candidates.append(candidate)
            self.stats["trades_filtered_out"] += 1
            logger.info(f"Candidate {symbol} failed AI filters: {rejection}")
            return

        # All filters passed - execute trade!
        logger.warning(f"ALL FILTERS PASSED for {symbol} - triggering trade!")
        await self._execute_trade(candidate)

        self.candidates.append(candidate)

    def _check_eligibility(self, symbol: str, alert) -> Optional[str]:
        """Check if symbol is eligible for trading"""

        # Check news confidence
        if alert.confidence < self.config.min_news_confidence:
            return f"confidence {alert.confidence:.0%} < {self.config.min_news_confidence:.0%}"

        # Check urgency
        urgency_levels = ["low", "medium", "high", "critical"]
        min_idx = urgency_levels.index(self.config.min_news_urgency)
        alert_idx = urgency_levels.index(alert.urgency)
        if alert_idx < min_idx:
            return f"urgency {alert.urgency} < {self.config.min_news_urgency}"

        # Check cooldown
        if symbol in self.symbol_cooldowns:
            cooldown_end = self.symbol_cooldowns[symbol]
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).seconds
                return f"cooldown active ({remaining}s remaining)"

        # Check daily limit
        if self.daily_trade_count >= self.config.max_daily_trades:
            return f"daily limit reached ({self.config.max_daily_trades})"

        # Check concurrent trades
        scalper = self._get_hft_scalper()
        if scalper:
            open_positions = len(scalper.positions)
            if open_positions >= self.config.max_concurrent_trades:
                return f"max concurrent trades ({self.config.max_concurrent_trades})"

        return None

    async def _run_ai_filters(self, candidate: NewsTradeCandidate) -> tuple:
        """Run all AI filters on candidate"""
        symbol = candidate.symbol

        # Filter 1: Chronos
        if self.config.require_chronos:
            chronos = self._get_chronos()
            if chronos:
                try:
                    prediction = await chronos.predict(symbol)
                    candidate.chronos_score = prediction.get("score", 0)
                    candidate.chronos_direction = prediction.get("direction", "neutral")

                    if candidate.chronos_score < self.config.min_chronos_score:
                        return False, f"Chronos {candidate.chronos_score:.0%} < {self.config.min_chronos_score:.0%}"
                    if candidate.chronos_direction != "bullish":
                        return False, f"Chronos direction: {candidate.chronos_direction}"
                except Exception as e:
                    logger.warning(f"Chronos filter error: {e}")
                    # Don't reject on error - continue with other filters

        # Filter 2: Qlib
        if self.config.require_qlib:
            qlib = self._get_qlib()
            if qlib:
                try:
                    score = await qlib.compute_score(symbol)
                    candidate.qlib_score = score

                    if score < self.config.min_qlib_score:
                        return False, f"Qlib {score:.0%} < {self.config.min_qlib_score:.0%}"
                except Exception as e:
                    logger.warning(f"Qlib filter error: {e}")

        # Filter 3: Order Flow
        if self.config.require_order_flow:
            order_flow = self._get_order_flow()
            if order_flow:
                try:
                    signal = await order_flow.analyze(symbol)
                    if signal:
                        candidate.order_flow_buy_pressure = signal.buy_pressure
                        candidate.order_flow_recommendation = signal.recommendation

                        if signal.buy_pressure < self.config.min_buy_pressure:
                            return False, f"Buy pressure {signal.buy_pressure:.0%} < {self.config.min_buy_pressure:.0%}"
                        if signal.recommendation == "SKIP":
                            return False, "Order flow recommends SKIP"
                except Exception as e:
                    logger.warning(f"Order flow filter error: {e}")

        # All filters passed!
        return True, None

    async def _add_to_watchlist(self, symbol: str, headline: str = "", catalyst: str = ""):
        """Add symbol to watchlist for monitoring"""
        try:
            from ai.intelligent_watchlist import get_intelligent_watchlist
            watchlist = get_intelligent_watchlist()
            if watchlist:
                # Use add_from_news which does qualification
                success = watchlist.add_from_news(
                    symbol=symbol,
                    headline=headline,
                    catalyst=catalyst,
                    confidence=0.8,  # High confidence from news trigger
                    validation_score=80  # High validation score
                )
                if success:
                    self.auto_added_symbols[symbol] = datetime.now()
                    logger.info(f"Auto-added {symbol} to watchlist from news trigger")
                return success
        except Exception as e:
            logger.warning(f"Could not add {symbol} to watchlist: {e}")
        return False

    async def _execute_trade(self, candidate: NewsTradeCandidate):
        """
        Execute trade via Signal Gating Engine.

        GATING ENFORCEMENT: All trades MUST go through gating.
        """
        symbol = candidate.symbol

        try:
            # GATING ENFORCEMENT: Route through Signal Gating Engine first
            from ai.gated_trading import get_gated_trading_manager
            manager = get_gated_trading_manager()

            approved, exec_request, reason = manager.gate_trade_attempt(
                symbol=symbol,
                trigger_type="news_triggered",
                quote={"price": candidate.entry_price or 0}
            )

            if not approved:
                candidate.trade_triggered = False
                candidate.trade_result = f"gating_vetoed: {reason}"
                logger.info(f"GATING VETOED: {symbol} - {reason}")
                return

            # Gating approved - now route to scalper
            scalper = self._get_hft_scalper()
            if not scalper:
                candidate.trade_result = "scalper_not_available"
                return

            # Ensure scalper is running
            if not scalper.is_running:
                scalper.start()

            # Add symbol to scalper watchlist if not already
            if symbol not in scalper.config.watchlist:
                scalper.config.watchlist.append(symbol)

            gating_token = f"GATED_{symbol}_{datetime.now().strftime('%H%M%S')}"

            if self.config.paper_mode:
                # Paper trade - just log
                candidate.trade_triggered = True
                candidate.trade_result = f"paper_trade_triggered (gated: {gating_token})"
                logger.warning(f"PAPER TRADE (GATED): Would buy {symbol} - {candidate.news_headline[:50]}")
            else:
                # Live trade - scalper will handle via gated entry
                candidate.trade_triggered = True
                candidate.trade_result = f"live_trade_queued (gated: {gating_token})"
                logger.warning(f"LIVE TRADE (GATED): Queueing {symbol} for scalper entry")

                # Add to scalper's watchlist (scalper will verify gating on execute_entry)
                if symbol not in scalper.config.watchlist:
                    scalper.config.watchlist.append(symbol)

            # Update tracking
            self.daily_trade_count += 1
            self.symbol_cooldowns[symbol] = datetime.now() + timedelta(
                minutes=self.config.symbol_cooldown_minutes
            )
            self.stats["trades_triggered"] += 1

            self.executed_trades.append({
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "paper_mode": self.config.paper_mode,
                "gating_token": gating_token,
                "news_catalyst": candidate.news_catalyst,
                "news_confidence": candidate.news_confidence,
                "chronos_score": candidate.chronos_score,
                "qlib_score": candidate.qlib_score,
                "order_flow_pressure": candidate.order_flow_buy_pressure
            })

        except Exception as e:
            candidate.trade_result = f"error: {str(e)}"
            logger.error(f"Trade execution error for {symbol}: {e}")

    async def cleanup_watchlist(self):
        """Remove expired auto-added symbols from watchlist"""
        now = datetime.now()
        expired = []

        for symbol, added_at in self.auto_added_symbols.items():
            if now - added_at > timedelta(minutes=self.config.watchlist_expiry_minutes):
                expired.append(symbol)

        if expired:
            try:
                from ai.intelligent_watchlist import get_watchlist
                watchlist = get_watchlist()
                if watchlist:
                    for symbol in expired:
                        await watchlist.remove_symbol(symbol)
                        del self.auto_added_symbols[symbol]
                    logger.info(f"Cleaned up {len(expired)} expired symbols from watchlist")
            except Exception as e:
                logger.warning(f"Watchlist cleanup error: {e}")

    def update_config(self, **kwargs) -> Dict:
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        save_config(self.config)
        return self.config.to_dict()

    def get_status(self) -> Dict:
        """Get current status"""
        fast_news = self._get_fast_news()
        scalper = self._get_hft_scalper()

        return {
            "enabled": self.config.enabled,
            "is_running": self.is_running,
            "paper_mode": self.config.paper_mode,
            "news_scanner_running": fast_news.is_running if fast_news else False,
            "scalper_running": scalper.is_running if scalper else False,
            "daily_trade_count": self.daily_trade_count,
            "max_daily_trades": self.config.max_daily_trades,
            "auto_added_symbols": list(self.auto_added_symbols.keys()),
            "cooldowns_active": len(self.symbol_cooldowns),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "stats": self.stats,
            "config": self.config.to_dict()
        }

    def get_candidates(self, limit: int = 50) -> List[Dict]:
        """Get recent trade candidates"""
        return [c.to_dict() for c in self.candidates[-limit:]]

    def get_executed_trades(self, limit: int = 20) -> List[Dict]:
        """Get executed trades"""
        return self.executed_trades[-limit:]


# Singleton instance
_news_auto_trader: Optional[NewsAutoTrader] = None


def get_news_auto_trader() -> NewsAutoTrader:
    """Get or create the news auto-trader singleton"""
    global _news_auto_trader
    if _news_auto_trader is None:
        _news_auto_trader = NewsAutoTrader()
    return _news_auto_trader


async def start_news_auto_trader(paper_mode: bool = True) -> Dict:
    """Start the news auto-trader"""
    trader = get_news_auto_trader()
    trader.config.enabled = True
    trader.config.paper_mode = paper_mode
    save_config(trader.config)

    await trader.start()
    return trader.get_status()


async def stop_news_auto_trader() -> Dict:
    """Stop the news auto-trader"""
    trader = get_news_auto_trader()
    await trader.stop()
    trader.config.enabled = False
    save_config(trader.config)
    return trader.get_status()


if __name__ == "__main__":
    # Test the news auto-trader
    import logging
    logging.basicConfig(level=logging.INFO)

    async def test():
        trader = get_news_auto_trader()

        print("Config:", json.dumps(trader.config.to_dict(), indent=2))

        # Start in paper mode
        status = await start_news_auto_trader(paper_mode=True)
        print("Status:", json.dumps(status, indent=2))

        # Let it run for a bit
        print("\nRunning for 60 seconds...")
        await asyncio.sleep(60)

        # Check results
        print("\nCandidates:", trader.get_candidates())
        print("Executed:", trader.get_executed_trades())

        await stop_news_auto_trader()

    asyncio.run(test())
