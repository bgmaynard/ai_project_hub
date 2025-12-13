"""
Autonomous Trading Bot
=====================

Complete autonomous trading system that combines:
- AlphaFusion V2 (prediction engine)
- Trading Engine (risk management & execution)
- IBKR integration

Author: AI Trading Bot Team
Version: 2.0
"""

from __future__ import annotations
from typing import Dict, Optional, List
from dataclasses import dataclass
import time
import asyncio
import logging

from .alpha_fusion_v2 import (
    AlphaFusionEngine,
    MarketData,
    TradeLabel,
    market_data_from_ibkr
)
from .trading_engine import (
    TradingEngine,
    TradingSignal,
    RiskLimits,
    Side,
    OrderType
)

# Optional sentiment analyzer
try:
    from .warrior_sentiment_analyzer import get_sentiment_analyzer, WarriorSentimentAnalyzer
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

# Indicator Weight Calculator for weighted signals
try:
    from .indicator_weight_calculator import get_weight_calculator
    HAS_INDICATOR_WEIGHTS = True
except ImportError:
    HAS_INDICATOR_WEIGHTS = False

# Momentum Scanner for autonomous stock discovery
try:
    from .momentum_scanner import get_momentum_scanner, CRITERIA as SCANNER_CRITERIA
    HAS_MOMENTUM_SCANNER = True
except ImportError:
    HAS_MOMENTUM_SCANNER = False

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Bot configuration - MOMENTUM ONLY, NO HOLD AND HOPE"""
    # Account settings
    account_size: float = 50000.0

    # AlphaFusion settings
    horizon_seconds: float = 2.0
    learning_rate: float = 0.01
    k_neighbors: int = 20

    # Risk limits - TIGHTENED FOR MOMENTUM TRADING
    max_position_size_usd: float = 5000.0
    max_positions: int = 5
    daily_loss_limit_usd: float = 500.0
    min_probability_threshold: float = 0.60
    max_spread_pct: float = 0.005
    stop_loss_pct: float = 0.01       # TIGHTENED: 1% stop loss (was 2%)
    take_profit_pct: float = 0.03     # TIGHTENED: 3% take profit (was 4%)
    trailing_stop_pct: float = 0.01   # TIGHTENED: 1% trail (was 1.5%)

    # TIME-BASED EXITS - NO HOLD AND HOPE
    stale_position_seconds: int = 300  # 5 min - if not profitable, GET OUT
    max_hold_seconds: int = 900        # 15 min max hold for momentum plays

    # Trading settings
    enabled: bool = False  # Start disabled for safety
    watchlist: List[str] = None

    # Scanner settings for autonomous stock discovery
    scanner_enabled: bool = True           # Enable autonomous scanning
    scanner_interval_seconds: int = 60     # Scan every 60 seconds
    scanner_max_watchlist: int = 10        # Max stocks to track from scanner
    scanner_min_score: float = 60.0        # Min momentum score to consider

    def __post_init__(self):
        if self.watchlist is None:
            self.watchlist = []


class AutonomousTrader:
    """
    Autonomous trading bot

    Usage:
        bot = AutonomousTrader(config=config, ib_connection=ib)
        await bot.start()
        # Bot runs autonomously
        await bot.stop()
    """

    def __init__(self,
                 config: BotConfig,
                 ib_connection=None):

        self.config = config
        self.ib = ib_connection

        # Core engines
        self.alpha_fusion = AlphaFusionEngine(
            horizon_seconds=config.horizon_seconds,
            learning_rate=config.learning_rate,
            k_neighbors=config.k_neighbors
        )

        risk_limits = RiskLimits(
            max_position_size_usd=config.max_position_size_usd,
            max_positions=config.max_positions,
            daily_loss_limit_usd=config.daily_loss_limit_usd,
            min_probability_threshold=config.min_probability_threshold,
            max_spread_pct=config.max_spread_pct,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct
        )

        self.trading_engine = TradingEngine(
            account_size=config.account_size,
            risk_limits=risk_limits
        )

        # Sentiment analyzer (optional)
        self.sentiment_analyzer: Optional[WarriorSentimentAnalyzer] = None
        if HAS_SENTIMENT:
            try:
                self.sentiment_analyzer = get_sentiment_analyzer()
                logger.info("✓ Sentiment analyzer integrated")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")

        # Indicator weight calculator (optional)
        self.weight_calculator = None
        if HAS_INDICATOR_WEIGHTS:
            try:
                self.weight_calculator = get_weight_calculator()
                logger.info("✓ Indicator weight calculator integrated")
            except Exception as e:
                logger.warning(f"Could not load indicator weight calculator: {e}")

        # Momentum scanner for autonomous stock discovery
        self.scanner = None
        if HAS_MOMENTUM_SCANNER and config.scanner_enabled:
            try:
                self.scanner = get_momentum_scanner()
                logger.info("✓ Momentum scanner integrated for autonomous stock discovery")
            except Exception as e:
                logger.warning(f"Could not load momentum scanner: {e}")

        # Scanner state
        self.last_scanner_refresh = 0
        self.scanner_discovered_stocks: Dict[str, Dict] = {}  # symbol -> {score, catalyst, timestamp}

        # Sentiment cache (avoid over-fetching)
        self.sentiment_cache: Dict[str, tuple] = {}  # symbol -> (timestamp, score)
        self.sentiment_cache_ttl = 300  # 5 minutes

        # State
        self.running = False
        self.last_predictions: Dict[str, any] = {}
        self.pending_orders: Dict[str, any] = {}
        self.position_entry_times: Dict[str, float] = {}  # symbol -> timestamp for time-based exits

        # Statistics
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        self.total_trades_rejected = 0

        logger.info("Autonomous Trader initialized")

    async def start(self):
        """Start the autonomous trading bot"""
        if self.running:
            logger.warning("Bot already running")
            return

        if not self.config.enabled:
            logger.warning("Bot is disabled in config")
            return

        logger.info("Starting Autonomous Trader...")
        self.running = True

        # Start main trading loop
        asyncio.create_task(self._trading_loop())

        logger.info("✓ Autonomous Trader started")

    async def stop(self):
        """Stop the autonomous trading bot"""
        logger.info("Stopping Autonomous Trader...")
        self.running = False

        # Close all positions
        await self._close_all_positions()

        # Cleanup sentiment analyzer
        if self.sentiment_analyzer:
            try:
                await self.sentiment_analyzer.close()
            except Exception as e:
                logger.warning(f"Error closing sentiment analyzer: {e}")

        logger.info("✓ Autonomous Trader stopped")

    async def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Refresh watchlist from scanner periodically
                await self._refresh_watchlist_from_scanner()

                # Process each symbol in watchlist (both configured and scanner-discovered)
                all_symbols = set(self.config.watchlist) | set(self.scanner_discovered_stocks.keys())
                for symbol in all_symbols:
                    await self._process_symbol(symbol)

                # Check exits for open positions
                await self._check_exits()

                # Update pending labels
                await self._update_labels()

                # Small sleep to avoid hammering
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _refresh_watchlist_from_scanner(self):
        """
        Refresh the watchlist with top momentum stocks from the scanner.
        This is the autonomous stock discovery mechanism.
        """
        if not self.scanner or not self.config.scanner_enabled:
            return

        # Check if it's time to refresh
        now = time.time()
        if now - self.last_scanner_refresh < self.config.scanner_interval_seconds:
            return

        self.last_scanner_refresh = now

        try:
            # Run scanner to find momentum stocks
            logger.info("Running autonomous scanner for stock discovery...")
            result = self.scanner.run_full_scan()

            if not result.get('success'):
                logger.warning("Scanner scan failed")
                return

            # Get top stocks meeting criteria
            top_stocks = result.get('top_stocks', [])
            added_count = 0
            removed_count = 0

            # Clear stale discoveries (keep only high-scoring ones)
            stale_symbols = []
            for sym, data in self.scanner_discovered_stocks.items():
                # Remove if score dropped below threshold or been tracked > 1 hour
                age_hours = (now - data.get('timestamp', 0)) / 3600
                if data.get('score', 0) < self.config.scanner_min_score or age_hours > 1:
                    stale_symbols.append(sym)

            for sym in stale_symbols:
                del self.scanner_discovered_stocks[sym]
                removed_count += 1
                logger.debug(f"Removed stale discovery: {sym}")

            # Add new high-scoring stocks
            for stock in top_stocks:
                symbol = stock.get('symbol', '')
                score = stock.get('momentum_score', 0)
                catalyst = stock.get('catalyst', 'SCANNER')

                # Skip if already in config watchlist
                if symbol in self.config.watchlist:
                    continue

                # Skip if below minimum score
                if score < self.config.scanner_min_score:
                    continue

                # Skip if we've hit max watchlist size
                if len(self.scanner_discovered_stocks) >= self.config.scanner_max_watchlist:
                    # Only replace if new stock has higher score than lowest
                    if self.scanner_discovered_stocks:
                        min_sym = min(self.scanner_discovered_stocks.keys(),
                                    key=lambda s: self.scanner_discovered_stocks[s].get('score', 0))
                        min_score = self.scanner_discovered_stocks[min_sym].get('score', 0)
                        if score > min_score:
                            del self.scanner_discovered_stocks[min_sym]
                            removed_count += 1
                        else:
                            continue

                # Add to discovered stocks
                if symbol not in self.scanner_discovered_stocks:
                    self.scanner_discovered_stocks[symbol] = {
                        'score': score,
                        'catalyst': catalyst,
                        'timestamp': now,
                        'price': stock.get('price', 0),
                        'change_pct': stock.get('change_pct', 0),
                        'rvol': stock.get('rvol', 0),
                    }
                    added_count += 1
                    logger.info(f"SCANNER DISCOVERY: {symbol} (Score: {score:.0f}, "
                               f"Change: {stock.get('change_pct', 0):+.1f}%, Catalyst: {catalyst})")
                else:
                    # Update existing
                    self.scanner_discovered_stocks[symbol]['score'] = score
                    self.scanner_discovered_stocks[symbol]['price'] = stock.get('price', 0)

            if added_count > 0 or removed_count > 0:
                logger.info(f"Scanner refresh: +{added_count} added, -{removed_count} removed, "
                           f"tracking {len(self.scanner_discovered_stocks)} discoveries")

        except Exception as e:
            logger.error(f"Error refreshing watchlist from scanner: {e}")

    async def _get_sentiment_score(self, symbol: str) -> float:
        """
        Get sentiment score for a symbol with caching.
        Returns a score between -1.0 and 1.0, or 0.0 if unavailable.
        """
        if not self.sentiment_analyzer:
            return 0.0

        # Check cache
        if symbol in self.sentiment_cache:
            cached_time, cached_score = self.sentiment_cache[symbol]
            if time.time() - cached_time < self.sentiment_cache_ttl:
                return cached_score

        try:
            # Fetch fresh sentiment (use 6 hours for trading context)
            sentiment_result = await self.sentiment_analyzer.analyze_symbol(
                symbol=symbol,
                hours=6,
                sources=['news', 'twitter', 'reddit']
            )

            # Calculate weighted sentiment score
            # Weight by confidence and number of signals
            if sentiment_result.signals_count > 0:
                score = sentiment_result.overall_score * sentiment_result.overall_confidence

                # Boost for breaking news
                if sentiment_result.breaking_news:
                    boost = sentiment_result.breaking_news.severity * 0.3
                    if sentiment_result.breaking_news.alert_type == 'positive':
                        score = min(score + boost, 1.0)
                    elif sentiment_result.breaking_news.alert_type == 'negative':
                        score = max(score - boost, -1.0)
                    logger.info(f"Breaking news for {symbol}: {sentiment_result.breaking_news.headline[:80]}")

                # Cache the result
                self.sentiment_cache[symbol] = (time.time(), score)

                logger.debug(f"Sentiment for {symbol}: {score:+.3f} ({sentiment_result.signals_count} signals)")
                return score
            else:
                self.sentiment_cache[symbol] = (time.time(), 0.0)
                return 0.0

        except Exception as e:
            logger.warning(f"Error fetching sentiment for {symbol}: {e}")
            return 0.0

    def _get_weighted_indicator_signal(self, symbol: str, market_data: MarketData) -> Dict:
        """
        Get weighted trading signal from indicator weight calculator.
        Combines multiple indicator signals weighted by historical accuracy.

        Returns a dict with:
            - direction: 'BUY', 'SELL', or 'HOLD'
            - composite_score: -1.0 to 1.0 (negative = bearish)
            - confidence: 0.0 to 1.0
            - contributing_indicators: list of active signals
        """
        if not self.weight_calculator:
            return {
                'direction': 'HOLD',
                'composite_score': 0.0,
                'confidence': 0.0,
                'contributing_indicators': []
            }

        try:
            # Build signals from various indicators/patterns
            signals = []

            # RSI signal
            rsi = market_data.indicators.get('rsi', 50) if hasattr(market_data, 'indicators') else 50
            if rsi < 30:
                signals.append({'indicator': 'rsi_oversold', 'direction': 'BUY', 'confidence': min(0.9, (30 - rsi) / 30)})
            elif rsi > 70:
                signals.append({'indicator': 'rsi_overbought', 'direction': 'SELL', 'confidence': min(0.9, (rsi - 70) / 30)})

            # MACD signal (if available)
            if hasattr(market_data, 'indicators') and 'macd_histogram' in market_data.indicators:
                macd_hist = market_data.indicators['macd_histogram']
                if macd_hist > 0:
                    signals.append({'indicator': 'macd_bullish', 'direction': 'BUY', 'confidence': min(0.7, abs(macd_hist) / 2)})
                elif macd_hist < 0:
                    signals.append({'indicator': 'macd_bearish', 'direction': 'SELL', 'confidence': min(0.7, abs(macd_hist) / 2)})

            # Volume surge signal
            volume_ratio = market_data.volume_ratio if hasattr(market_data, 'volume_ratio') else 1.0
            if volume_ratio > 2.0:
                # High volume often confirms trend
                signals.append({'indicator': 'volume_surge', 'direction': 'BUY', 'confidence': min(0.6, (volume_ratio - 2) / 5)})

            # VWAP reclaim signal (if price above vwap after being below)
            if hasattr(market_data, 'vwap') and market_data.vwap > 0:
                vwap_distance = (market_data.last - market_data.vwap) / market_data.vwap
                if 0 < vwap_distance < 0.02:  # Just reclaimed VWAP
                    signals.append({'indicator': 'vwap_reclaim', 'direction': 'BUY', 'confidence': 0.6})
                elif -0.02 < vwap_distance < 0:  # Just lost VWAP
                    signals.append({'indicator': 'vwap_breakdown', 'direction': 'SELL', 'confidence': 0.6})

            # Calculate weighted composite signal
            if signals:
                result = self.weight_calculator.calculate_weighted_signal(signals)
                logger.debug(f"Weighted signal for {symbol}: {result['direction']} (score={result['composite_score']:.3f})")
                return result
            else:
                return {
                    'direction': 'HOLD',
                    'composite_score': 0.0,
                    'confidence': 0.0,
                    'contributing_indicators': []
                }

        except Exception as e:
            logger.warning(f"Error calculating weighted signal for {symbol}: {e}")
            return {
                'direction': 'HOLD',
                'composite_score': 0.0,
                'confidence': 0.0,
                'contributing_indicators': []
            }

    def _record_prediction(self, symbol: str, direction: str, confidence: float,
                          price: float, prediction_data: Dict):
        """Record prediction for later verification against actual outcome."""
        if not self.weight_calculator:
            return None

        try:
            # Determine target/stop based on direction
            if direction == 'BUY':
                target_price = price * 1.02  # 2% target
                stop_price = price * 0.99    # 1% stop
            elif direction == 'SELL':
                target_price = price * 0.98  # 2% target (inverse)
                stop_price = price * 1.01    # 1% stop (inverse)
            else:
                return None

            # Record with composite indicator
            prediction_id = self.weight_calculator.record_prediction(
                symbol=symbol,
                indicator='composite_ai',  # Combined AI + weighted indicators
                direction=direction,
                confidence=confidence,
                price=price,
                target_price=target_price,
                stop_price=stop_price,
                timeframe='1day'
            )

            logger.debug(f"Recorded prediction {prediction_id} for {symbol}: {direction} @ ${price:.2f}")
            return prediction_id

        except Exception as e:
            logger.warning(f"Error recording prediction for {symbol}: {e}")
            return None

    async def _process_symbol(self, symbol: str):
        """Process one symbol through the pipeline"""
        try:
            # Get current market data from IBKR
            ticker = await self._get_ticker(symbol)
            if ticker is None:
                return

            market_data = market_data_from_ibkr(ticker)

            # Get sentiment score from integrated analyzer
            sentiment = await self._get_sentiment_score(symbol)

            # Get weighted indicator signal (pattern-based, accuracy-weighted)
            weighted_signal = self._get_weighted_indicator_signal(symbol, market_data)

            # Combine sentiment + weighted indicator signal
            # Weighted signal score ranges -1 to 1, sentiment ranges -1 to 1
            combined_sentiment = sentiment * 0.4 + weighted_signal['composite_score'] * 0.6

            # Generate prediction with combined sentiment
            prediction = self.alpha_fusion.predict(market_data, combined_sentiment)

            # Adjust prediction confidence based on indicator agreement
            confidence_boost = 0
            if weighted_signal['direction'] != 'HOLD':
                # Boost confidence if weighted indicators strongly agree
                if weighted_signal['confidence'] > 0.6:
                    confidence_boost = 0.1
                elif weighted_signal['confidence'] > 0.4:
                    confidence_boost = 0.05

            adjusted_p_final = min(1.0, prediction.p_final + confidence_boost)

            # Store prediction with weighted signal data
            self.last_predictions[symbol] = {
                'timestamp': prediction.timestamp,
                'p_final': adjusted_p_final,
                'p_original': prediction.p_final,
                'reliability': prediction.reliability,
                'features': prediction.features,
                'weighted_signal': {
                    'direction': weighted_signal['direction'],
                    'composite_score': weighted_signal['composite_score'],
                    'confidence': weighted_signal['confidence'],
                    'indicators': weighted_signal.get('contributing_indicators', [])
                },
                'combined_sentiment': combined_sentiment
            }

            # Check if we should trade
            signal = self.trading_engine.evaluate_signal(
                symbol=symbol,
                p_final=adjusted_p_final,
                reliability=prediction.reliability,
                bid=market_data.bid,
                ask=market_data.ask,
                spread_pct=market_data.spread_pct,
                expected_slippage=self.alpha_fusion.slippage_tracker.get_expected_slippage(),
                fill_probability=self.alpha_fusion.slippage_tracker.get_limit_fill_rate()
            )

            if signal:
                self.total_signals_generated += 1
                logger.info(f"Signal generated for {symbol}: {signal.side.value} {signal.quantity} @ {signal.confidence:.3f}")
                logger.info(f"  Weighted indicator: {weighted_signal['direction']} ({weighted_signal['confidence']:.2f})")

                # Record prediction for tracking accuracy
                price = market_data.last if market_data.last else (market_data.bid + market_data.ask) / 2
                self._record_prediction(
                    symbol=symbol,
                    direction=signal.side.value,
                    confidence=signal.confidence,
                    price=price,
                    prediction_data=self.last_predictions[symbol]
                )

                # Execute the trade
                success = await self._execute_signal(signal, market_data)

                if success:
                    self.total_trades_executed += 1
                else:
                    self.total_trades_rejected += 1

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def _execute_signal(self, signal: TradingSignal, market_data: MarketData) -> bool:
        """Execute a trading signal"""
        try:
            if self.ib is None:
                logger.warning("No IB connection - cannot execute")
                return False

            # Create IBKR contract
            from ib_insync import Stock, MarketOrder, LimitOrder

            contract = Stock(signal.symbol, 'SMART', 'USD')

            # Create order
            if signal.order_type == OrderType.MARKET:
                order = MarketOrder(
                    signal.side.value,
                    signal.quantity
                )
            elif signal.order_type == OrderType.LIMIT:
                order = LimitOrder(
                    signal.side.value,
                    signal.quantity,
                    signal.limit_price
                )
            else:
                logger.error(f"Unsupported order type: {signal.order_type}")
                return False

            # Place order
            trade = self.ib.placeOrder(contract, order)

            # Wait for fill (with timeout)
            filled = False
            for _ in range(10):  # Wait up to 5 seconds
                await asyncio.sleep(0.5)
                if trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break

            if trade.orderStatus.status == 'Filled':
                filled_price = trade.orderStatus.avgFillPrice

                # Record execution in slippage tracker
                intended_price = signal.limit_price if signal.limit_price else market_data.ask if signal.side == Side.BUY else market_data.bid

                self.alpha_fusion.record_execution(
                    order_type=signal.order_type.value,
                    intended_price=intended_price,
                    filled_price=filled_price,
                    side=signal.side.value
                )

                # Add position to trading engine
                quantity_signed = signal.quantity if signal.side == Side.BUY else -signal.quantity
                self.trading_engine.add_position(
                    symbol=signal.symbol,
                    quantity=quantity_signed,
                    entry_price=filled_price
                )

                logger.info(f"✓ Order filled: {signal.symbol} {signal.side.value} {signal.quantity} @ ${filled_price:.2f}")

                filled = True

            elif trade.orderStatus.status == 'Cancelled':
                logger.warning(f"✗ Order cancelled: {signal.symbol}")
                self.alpha_fusion.record_execution(
                    order_type=signal.order_type.value,
                    intended_price=signal.limit_price or 0,
                    filled_price=None,
                    side=signal.side.value
                )

            return filled

        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            return False

    async def _check_exits(self):
        """Check if any open positions should be exited - MOMENTUM ONLY, NO HOLD AND HOPE"""
        for symbol in list(self.trading_engine.positions.keys()):
            try:
                # Get current price
                ticker = await self._get_ticker(symbol)
                if ticker is None:
                    continue

                current_price = float(ticker.last) if ticker.last == ticker.last else None
                if current_price is None:
                    continue

                # Track entry time if not already tracked
                if symbol not in self.position_entry_times:
                    self.position_entry_times[symbol] = time.time()

                # Update position
                self.trading_engine.update_position(symbol, current_price)

                # Check for exit signal from trading engine
                exit_signal = self.trading_engine.check_exits(symbol, current_price)

                # TIME-BASED EXIT CHECKS - NO HOLD AND HOPE
                if not exit_signal:
                    position = self.trading_engine.positions.get(symbol)
                    if position:
                        entry_price = position.entry_price
                        pnl_pct = (current_price - entry_price) / entry_price
                        position_age = time.time() - self.position_entry_times.get(symbol, time.time())

                        # Rule 1: STALE POSITION - not profitable after 5 min = GET OUT
                        if position_age > self.config.stale_position_seconds and pnl_pct <= 0:
                            logger.warning(f"STALE POSITION: {symbol} {position_age/60:.1f}min old, {pnl_pct:.1%} - MOMENTUM FAILED")
                            from .trading_engine import Side, OrderType, TradingSignal
                            exit_signal = TradingSignal(
                                timestamp=time.time(),
                                symbol=symbol,
                                side=Side.SELL if position.quantity > 0 else Side.BUY,
                                order_type=OrderType.MARKET,
                                quantity=abs(position.quantity),
                                reason=f"STALE: {position_age/60:.1f}min not profitable - MOMENTUM FAILED"
                            )

                        # Rule 2: MAX HOLD TIME - 15 min max (unless big winner >3%)
                        elif position_age > self.config.max_hold_seconds and pnl_pct < 0.03:
                            logger.warning(f"MAX HOLD: {symbol} {position_age/60:.1f}min old - TIME TO EXIT")
                            from .trading_engine import Side, OrderType, TradingSignal
                            exit_signal = TradingSignal(
                                timestamp=time.time(),
                                symbol=symbol,
                                side=Side.SELL if position.quantity > 0 else Side.BUY,
                                order_type=OrderType.MARKET,
                                quantity=abs(position.quantity),
                                reason=f"MAX HOLD: {position_age/60:.1f}min exceeds limit"
                            )

                if exit_signal:
                    logger.info(f"Exit signal: {symbol} - {exit_signal.reason}")

                    # Execute exit
                    market_data = market_data_from_ibkr(ticker)
                    success = await self._execute_signal(exit_signal, market_data)

                    if success:
                        # Close position in trading engine
                        self.trading_engine.close_position(symbol, current_price)
                        # Clear entry time tracking
                        if symbol in self.position_entry_times:
                            del self.position_entry_times[symbol]
                        logger.info(f"✓ Position closed: {symbol}")

            except Exception as e:
                logger.error(f"Error checking exits for {symbol}: {e}")

    async def _update_labels(self):
        """Update model with realized outcomes after horizon"""
        # This would check pending predictions and create labels
        # For now, simplified - real implementation would track entry times
        # and check if horizon has elapsed, then compute label
        pass

    async def _close_all_positions(self):
        """Emergency close all positions"""
        logger.info("Closing all positions...")

        for symbol in list(self.trading_engine.positions.keys()):
            try:
                position = self.trading_engine.positions[symbol]

                # Get current price
                ticker = await self._get_ticker(symbol)
                if ticker is None:
                    continue

                current_price = float(ticker.last) if ticker.last == ticker.last else position.current_price

                # Create exit signal
                from .trading_engine import Side, OrderType, TradingSignal

                exit_signal = TradingSignal(
                    timestamp=time.time(),
                    symbol=symbol,
                    side=Side.SELL if position.quantity > 0 else Side.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(position.quantity),
                    reason="Emergency close"
                )

                # Execute
                market_data = market_data_from_ibkr(ticker)
                await self._execute_signal(exit_signal, market_data)

                # Close in engine
                self.trading_engine.close_position(symbol, current_price)

            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")

    async def _get_ticker(self, symbol: str):
        """Get ticker data from IBKR"""
        if self.ib is None:
            return None

        try:
            from ib_insync import Stock
            contract = Stock(symbol, 'SMART', 'USD')

            # Get qualified contract
            contracts = await self.ib.qualifyContractsAsync(contract)
            if not contracts:
                return None

            # Request market data if not already streaming
            ticker = self.ib.reqMktData(contracts[0], '', False, False)

            # Wait a bit for data
            await asyncio.sleep(0.1)

            return ticker

        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None

    def get_status(self) -> Dict:
        """Get bot status"""
        alpha_stats = self.alpha_fusion.get_stats()
        trading_stats = self.trading_engine.get_stats()

        return {
            "running": self.running,
            "enabled": self.config.enabled,
            "watchlist": self.config.watchlist,
            "total_signals_generated": self.total_signals_generated,
            "total_trades_executed": self.total_trades_executed,
            "total_trades_rejected": self.total_trades_rejected,
            "alpha_fusion": alpha_stats,
            "trading_engine": trading_stats,
            "last_predictions": self.last_predictions
        }

    def update_config(self, **kwargs):
        """Update bot configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")

    def enable(self):
        """Enable trading"""
        self.config.enabled = True
        self.trading_engine.trading_enabled = True
        logger.info("✓ Trading enabled")

    def disable(self):
        """Disable trading"""
        self.config.enabled = False
        self.trading_engine.trading_enabled = False
        logger.info("✗ Trading disabled")

    def save_state(self, filepath: str):
        """Save bot state"""
        self.alpha_fusion.save_state(filepath)
        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load bot state"""
        self.alpha_fusion.load_state(filepath)
        logger.info(f"State loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create config
        config = BotConfig(
            account_size=50000.0,
            watchlist=['AAPL', 'MSFT', 'GOOGL'],
            max_position_size_usd=5000.0,
            daily_loss_limit_usd=500.0,
            min_probability_threshold=0.65,
            enabled=False  # Start disabled for testing
        )

        # Create bot (without IBKR connection for demo)
        bot = AutonomousTrader(config=config, ib_connection=None)

        print("Bot Status:")
        status = bot.get_status()
        for key, value in status.items():
            if key not in ['alpha_fusion', 'trading_engine', 'last_predictions']:
                print(f"  {key}: {value}")

    asyncio.run(main())
