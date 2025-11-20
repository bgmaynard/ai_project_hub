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

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Bot configuration"""
    # Account settings
    account_size: float = 50000.0

    # AlphaFusion settings
    horizon_seconds: float = 2.0
    learning_rate: float = 0.01
    k_neighbors: int = 20

    # Risk limits
    max_position_size_usd: float = 5000.0
    max_positions: int = 5
    daily_loss_limit_usd: float = 500.0
    min_probability_threshold: float = 0.60
    max_spread_pct: float = 0.005
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    trailing_stop_pct: float = 0.015

    # Trading settings
    enabled: bool = False  # Start disabled for safety
    watchlist: List[str] = None

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

        # State
        self.running = False
        self.last_predictions: Dict[str, any] = {}
        self.pending_orders: Dict[str, any] = {}

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

        logger.info("✓ Autonomous Trader stopped")

    async def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Process each symbol in watchlist
                for symbol in self.config.watchlist:
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

    async def _process_symbol(self, symbol: str):
        """Process one symbol through the pipeline"""
        try:
            # Get current market data from IBKR
            ticker = await self._get_ticker(symbol)
            if ticker is None:
                return

            market_data = market_data_from_ibkr(ticker)

            # Get sentiment (placeholder - would integrate real sentiment)
            sentiment = 0.0  # TODO: Integrate sentiment feeds

            # Generate prediction
            prediction = self.alpha_fusion.predict(market_data, sentiment)

            # Store prediction
            self.last_predictions[symbol] = {
                'timestamp': prediction.timestamp,
                'p_final': prediction.p_final,
                'reliability': prediction.reliability,
                'features': prediction.features
            }

            # Check if we should trade
            signal = self.trading_engine.evaluate_signal(
                symbol=symbol,
                p_final=prediction.p_final,
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
        """Check if any open positions should be exited"""
        for symbol in list(self.trading_engine.positions.keys()):
            try:
                # Get current price
                ticker = await self._get_ticker(symbol)
                if ticker is None:
                    continue

                current_price = float(ticker.last) if ticker.last == ticker.last else None
                if current_price is None:
                    continue

                # Update position
                self.trading_engine.update_position(symbol, current_price)

                # Check for exit signal
                exit_signal = self.trading_engine.check_exits(symbol, current_price)

                if exit_signal:
                    logger.info(f"Exit signal: {symbol} - {exit_signal.reason}")

                    # Execute exit
                    market_data = market_data_from_ibkr(ticker)
                    success = await self._execute_signal(exit_signal, market_data)

                    if success:
                        # Close position in trading engine
                        self.trading_engine.close_position(symbol, current_price)
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
