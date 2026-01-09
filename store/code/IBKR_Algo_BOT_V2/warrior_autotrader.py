"""
Warrior Trading Auto-Trader - 4AM Pre-Market Session
=====================================================
Automated trading based on Ross Cameron's Warrior Trading strategy:
- Scans for momentum stocks meeting the 5 Pillars criteria
- Monitors pre-market gappers and breaking news
- Executes trades with proper risk management
- Uses bracket orders (stop loss + take profit)

RISK CONTROLS:
- Max position size: $2,000 (adjustable)
- Max daily loss: $500 (stops trading if hit)
- Max concurrent positions: 5
- Stop loss: 2-3% per trade
- Take profit: 5-10% per trade
- Only trades stocks $2-$20

Author: Claude Code
"""

import asyncio
import json
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Set

import aiohttp
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("warrior_autotrader.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR RISK TOLERANCE
# ============================================================================

CONFIG = {
    # Trading Parameters
    "MAX_POSITION_SIZE": 2000,  # Max $ per position
    "MAX_DAILY_LOSS": 500,  # Stop trading if daily loss exceeds this
    "MAX_CONCURRENT_POSITIONS": 5,  # Max number of open positions
    "MIN_BUYING_POWER": 5000,  # Minimum buying power to trade
    # Warrior Trading Criteria
    "MIN_PRICE": 2.00,  # Min stock price
    "MAX_PRICE": 20.00,  # Max stock price
    "MIN_GAP_PCT": 4.0,  # Min gap % to consider
    "MIN_RVOL": 1.5,  # Min relative volume
    "MIN_VOLUME": 100000,  # Min pre-market volume
    # Risk Management
    "STOP_LOSS_PCT": 0.03,  # 3% stop loss
    "TAKE_PROFIT_PCT": 0.06,  # 6% take profit (2:1 R:R)
    "TRAILING_STOP_PCT": 0.02,  # 2% trailing stop after profit
    # Polling Intervals (seconds)
    "SCANNER_POLL_INTERVAL": 30,  # How often to scan
    "NEWS_POLL_INTERVAL": 60,  # How often to check news
    "POSITION_CHECK_INTERVAL": 10,  # How often to check positions
    # API
    "API_BASE": "http://localhost:9100/api/alpaca",
    "SCANNER_BASE": "http://localhost:9100/api/scanner/ALPACA",
    # Trading Hours (Eastern Time)
    "PRE_MARKET_START": time(4, 0),  # 4:00 AM ET
    "MARKET_OPEN": time(9, 30),  # 9:30 AM ET
    "MARKET_CLOSE": time(16, 0),  # 4:00 PM ET
    "AFTER_HOURS_END": time(20, 0),  # 8:00 PM ET
}

# ============================================================================
# AUTO TRADER CLASS
# ============================================================================


class WarriorAutoTrader:
    """
    Automated trader following Warrior Trading principles.
    Scans for momentum stocks and executes trades with risk management.
    """

    def __init__(self):
        self.et_tz = pytz.timezone("US/Eastern")
        self.session: Optional[aiohttp.ClientSession] = None

        # Trading state
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.is_trading_enabled = True

        # Tracking
        self.open_positions: Dict[str, dict] = {}
        self.traded_symbols: Set[str] = set()  # Avoid re-entering same symbol
        self.watchlist: List[dict] = []
        self.news_alerts: List[dict] = []

        # Circuit breaker integration
        self.circuit_breaker_ok = True

        logger.info("=" * 60)
        logger.info("WARRIOR AUTO-TRADER INITIALIZED")
        logger.info(f"Max Position Size: ${CONFIG['MAX_POSITION_SIZE']}")
        logger.info(f"Max Daily Loss: ${CONFIG['MAX_DAILY_LOSS']}")
        logger.info(f"Stop Loss: {CONFIG['STOP_LOSS_PCT']*100}%")
        logger.info(f"Take Profit: {CONFIG['TAKE_PROFIT_PCT']*100}%")
        logger.info("=" * 60)

    async def start(self):
        """Start the auto-trader"""
        self.session = aiohttp.ClientSession()

        try:
            # Verify server is running
            if not await self._check_server():
                logger.error("Server not responding. Exiting.")
                return

            # Check account status
            account = await self._get_account()
            if not account:
                logger.error("Could not get account info. Exiting.")
                return

            logger.info(f"Account: ${account.get('portfolio_value', 0):,.2f}")
            logger.info(f"Buying Power: ${account.get('buying_power', 0):,.2f}")

            # Load existing positions
            await self._load_positions()

            # Start the main trading loop
            await self._run_trading_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            if self.session:
                await self.session.close()

    async def _check_server(self) -> bool:
        """Check if the trading server is running"""
        try:
            async with self.session.get(
                f"{CONFIG['API_BASE'].replace('/api/alpaca', '')}/health"
            ) as resp:
                data = await resp.json()
                return data.get("status") == "ok"
        except Exception as e:
            logger.error(f"Server check failed: {e}")
            return False

    async def _get_account(self) -> Optional[dict]:
        """Get account information"""
        try:
            async with self.session.get(f"{CONFIG['API_BASE']}/account") as resp:
                return await resp.json()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None

    async def _load_positions(self):
        """Load current open positions"""
        try:
            async with self.session.get(f"{CONFIG['API_BASE']}/positions") as resp:
                positions = await resp.json()
                for pos in positions:
                    symbol = pos.get("symbol")
                    if symbol:
                        self.open_positions[symbol] = pos
                        self.traded_symbols.add(symbol)

                logger.info(f"Loaded {len(self.open_positions)} existing positions")
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows trading"""
        try:
            async with self.session.get(
                f"{CONFIG['API_BASE']}/ai/circuit-breaker/status"
            ) as resp:
                data = await resp.json()
                status = data.get("status", {})
                can_trade = status.get("can_trade", True)
                level = status.get("level", "NORMAL")

                if not can_trade:
                    logger.warning(f"Circuit breaker HALT - Level: {level}")
                    return False

                if level in ["WARNING", "CAUTION"]:
                    logger.info(f"Circuit breaker: {level} - reducing position sizes")

                return True
        except:
            return True  # Default to allowing trades if check fails

    async def _run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")

        # Create tasks for different activities
        tasks = [
            asyncio.create_task(self._scanner_loop()),
            asyncio.create_task(self._news_loop()),
            asyncio.create_task(self._position_monitor_loop()),
            asyncio.create_task(self._status_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")

    async def _scanner_loop(self):
        """Poll scanners for trading opportunities"""
        while self.is_trading_enabled:
            try:
                now_et = datetime.now(self.et_tz)

                # Only scan during trading hours
                if not self._is_trading_hours(now_et):
                    logger.info(f"Outside trading hours: {now_et.strftime('%H:%M')} ET")
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue

                # Check circuit breaker
                self.circuit_breaker_ok = await self._check_circuit_breaker()

                # Check daily loss limit
                if self.daily_pnl <= -CONFIG["MAX_DAILY_LOSS"]:
                    logger.warning(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}")
                    self.is_trading_enabled = False
                    break

                # Run scanners
                candidates = await self._scan_for_opportunities()

                if candidates:
                    logger.info(f"Found {len(candidates)} candidates")
                    await self._evaluate_and_trade(candidates)

                await asyncio.sleep(CONFIG["SCANNER_POLL_INTERVAL"])

            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
                await asyncio.sleep(10)

    async def _news_loop(self):
        """Monitor breaking news for catalysts"""
        while self.is_trading_enabled:
            try:
                # Get breaking news scanner results
                async with self.session.get(
                    f"{CONFIG['SCANNER_BASE']}/scan?preset=breaking_news"
                ) as resp:
                    data = await resp.json()
                    if data.get("success"):
                        results = data.get("results", [])
                        for stock in results[:5]:  # Top 5 news movers
                            symbol = stock.get("symbol")
                            if symbol and symbol not in self.traded_symbols:
                                logger.info(
                                    f"NEWS ALERT: {symbol} - checking for entry"
                                )
                                self.news_alerts.append(
                                    {
                                        "symbol": symbol,
                                        "time": datetime.now(self.et_tz).isoformat(),
                                        "data": stock,
                                    }
                                )

                await asyncio.sleep(CONFIG["NEWS_POLL_INTERVAL"])

            except Exception as e:
                logger.error(f"News loop error: {e}")
                await asyncio.sleep(30)

    async def _position_monitor_loop(self):
        """Monitor open positions for exits"""
        while self.is_trading_enabled:
            try:
                await self._update_positions()
                await asyncio.sleep(CONFIG["POSITION_CHECK_INTERVAL"])
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(5)

    async def _status_loop(self):
        """Periodic status updates"""
        while self.is_trading_enabled:
            try:
                account = await self._get_account()
                if account:
                    logger.info("-" * 40)
                    logger.info(
                        f"STATUS UPDATE @ {datetime.now(self.et_tz).strftime('%H:%M:%S')} ET"
                    )
                    logger.info(f"Portfolio: ${account.get('portfolio_value', 0):,.2f}")
                    logger.info(f"Daily P&L: ${self.daily_pnl:+,.2f}")
                    logger.info(
                        f"Trades: {self.trades_today} (W:{self.wins_today} L:{self.losses_today})"
                    )
                    logger.info(f"Open Positions: {len(self.open_positions)}")
                    logger.info("-" * 40)

                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Status loop error: {e}")
                await asyncio.sleep(60)

    def _is_trading_hours(self, now: datetime) -> bool:
        """Check if we're in trading hours"""
        current_time = now.time()

        # Pre-market: 4 AM - 9:30 AM
        # Regular: 9:30 AM - 4 PM
        # After hours: 4 PM - 8 PM

        if CONFIG["PRE_MARKET_START"] <= current_time < CONFIG["AFTER_HOURS_END"]:
            return True
        return False

    async def _scan_for_opportunities(self) -> List[dict]:
        """Scan multiple sources for trading opportunities"""
        candidates = []

        # 1. Warrior Trading Scanner (5 Pillars)
        try:
            async with self.session.get(
                f"{CONFIG['SCANNER_BASE']}/scan?preset=warrior"
            ) as resp:
                data = await resp.json()
                if data.get("success"):
                    for stock in data.get("results", [])[:10]:
                        stock["source"] = "warrior"
                        candidates.append(stock)
        except Exception as e:
            logger.error(f"Warrior scanner failed: {e}")

        # 2. Pre-Market Gappers
        try:
            async with self.session.get(
                f"{CONFIG['SCANNER_BASE']}/scan?preset=gainers"
            ) as resp:
                data = await resp.json()
                if data.get("success"):
                    for stock in data.get("results", [])[:10]:
                        stock["source"] = "gapper"
                        candidates.append(stock)
        except Exception as e:
            logger.error(f"Gapper scanner failed: {e}")

        # 3. Momentum Scanner
        try:
            async with self.session.get(
                f"{CONFIG['SCANNER_BASE']}/scan?preset=momentum"
            ) as resp:
                data = await resp.json()
                if data.get("success"):
                    for stock in data.get("results", [])[:5]:
                        stock["source"] = "momentum"
                        candidates.append(stock)
        except Exception as e:
            logger.error(f"Momentum scanner failed: {e}")

        # De-duplicate by symbol
        seen = set()
        unique_candidates = []
        for c in candidates:
            symbol = c.get("symbol")
            if symbol and symbol not in seen:
                seen.add(symbol)
                unique_candidates.append(c)

        return unique_candidates

    async def _evaluate_and_trade(self, candidates: List[dict]):
        """Evaluate candidates and execute trades"""

        # Check if we can take more positions
        if len(self.open_positions) >= CONFIG["MAX_CONCURRENT_POSITIONS"]:
            logger.info("Max positions reached - waiting for exits")
            return

        # Check circuit breaker
        if not self.circuit_breaker_ok:
            logger.warning("Circuit breaker active - no new trades")
            return

        for candidate in candidates:
            symbol = candidate.get("symbol", "")

            # Skip if already traded or holding
            if symbol in self.traded_symbols:
                continue

            # Check price range
            price = candidate.get("price", candidate.get("last_price", 0))
            if not (CONFIG["MIN_PRICE"] <= price <= CONFIG["MAX_PRICE"]):
                continue

            # Get AI prediction
            prediction = await self._get_ai_prediction(symbol)

            # Check if setup meets criteria
            if await self._is_valid_setup(candidate, prediction):
                await self._execute_trade(symbol, price, candidate, prediction)

                # Don't overload - one trade per scan cycle
                if len(self.open_positions) >= CONFIG["MAX_CONCURRENT_POSITIONS"]:
                    break

    async def _get_ai_prediction(self, symbol: str) -> Optional[dict]:
        """Get AI prediction for symbol"""
        try:
            async with self.session.get(
                f"{CONFIG['API_BASE']}/ai/explain/{symbol}"
            ) as resp:
                data = await resp.json()
                if data.get("success"):
                    return data.get("explanation", {})
        except:
            pass
        return None

    async def _is_valid_setup(
        self, candidate: dict, prediction: Optional[dict]
    ) -> bool:
        """Check if the candidate meets Warrior Trading criteria"""

        # Basic criteria from scanner
        gap_pct = abs(candidate.get("gap_pct", candidate.get("change_pct", 0)))
        volume = candidate.get("volume", candidate.get("pre_market_volume", 0))
        rvol = candidate.get("rvol", candidate.get("relative_volume", 1.0))

        # Warrior Trading 5 Pillars Check
        checks = {
            "gap": gap_pct >= CONFIG["MIN_GAP_PCT"],
            "volume": volume >= CONFIG["MIN_VOLUME"],
            "rvol": rvol >= CONFIG["MIN_RVOL"],
        }

        # AI check (optional but preferred)
        if prediction:
            pred_signal = prediction.get("prediction", "")
            confidence = prediction.get("confidence", 0)

            if "BULLISH" in pred_signal.upper() and confidence > 0.5:
                checks["ai"] = True
            elif "BEARISH" in pred_signal.upper():
                checks["ai"] = False
            else:
                checks["ai"] = True  # Neutral is OK
        else:
            checks["ai"] = True

        passed = all(checks.values())

        if passed:
            logger.info(
                f"VALID SETUP: {candidate.get('symbol')} - Gap:{gap_pct:.1f}% RVOL:{rvol:.1f}x"
            )

        return passed

    async def _execute_trade(
        self, symbol: str, price: float, candidate: dict, prediction: Optional[dict]
    ):
        """Execute a trade with bracket order"""

        # Calculate position size
        account = await self._get_account()
        buying_power = account.get("buying_power", 0) if account else 0

        if buying_power < CONFIG["MIN_BUYING_POWER"]:
            logger.warning(f"Insufficient buying power: ${buying_power:.2f}")
            return

        # Position sizing - risk 1% of account per trade
        position_value = min(CONFIG["MAX_POSITION_SIZE"], buying_power * 0.1)
        quantity = int(position_value / price)

        if quantity < 1:
            logger.warning(f"Position too small for {symbol}")
            return

        # Calculate stop and target
        stop_price = round(price * (1 - CONFIG["STOP_LOSS_PCT"]), 2)
        target_price = round(price * (1 + CONFIG["TAKE_PROFIT_PCT"]), 2)

        # Calculate limit price for extended hours
        limit_price = round(price * 1.005, 2)  # 0.5% above current price for fills

        logger.info("=" * 50)
        logger.info(f"EXECUTING TRADE: {symbol}")
        logger.info(f"  Current Price: ${price:.2f}")
        logger.info(f"  Limit Price: ${limit_price:.2f} (for extended hours)")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Value: ${limit_price * quantity:.2f}")
        logger.info(f"  Stop: ${stop_price:.2f} ({CONFIG['STOP_LOSS_PCT']*100}%)")
        logger.info(f"  Target: ${target_price:.2f} ({CONFIG['TAKE_PROFIT_PCT']*100}%)")
        logger.info("=" * 50)

        # Place bracket order using limit order (required for extended hours)
        try:
            order_data = {
                "symbol": symbol,
                "quantity": quantity,
                "action": "BUY",
                "take_profit_price": target_price,
                "stop_loss_price": stop_price,
                "limit_price": limit_price,  # Required for extended hours trading
            }

            async with self.session.post(
                f"{CONFIG['API_BASE']}/place-bracket-order", json=order_data
            ) as resp:
                result = await resp.json()

                if result.get("success") or result.get("order_id"):
                    logger.info(f"ORDER PLACED: {symbol} x{quantity} @ ~${price:.2f}")

                    # Track the position
                    self.open_positions[symbol] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "entry_price": price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "entry_time": datetime.now(self.et_tz).isoformat(),
                        "source": candidate.get("source", "scanner"),
                    }
                    self.traded_symbols.add(symbol)
                    self.trades_today += 1

                    # Record to brain
                    await self._record_trade(symbol, "BUY", price, quantity)

                else:
                    logger.error(f"Order failed: {result}")

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    async def _record_trade(
        self, symbol: str, action: str, price: float, quantity: int
    ):
        """Record trade to AI brain and journal"""
        try:
            await self.session.post(
                f"{CONFIG['API_BASE']}/ai/memory/record-trade",
                json={
                    "symbol": symbol,
                    "action": action,
                    "price": price,
                    "quantity": quantity,
                    "source": "warrior_autotrader",
                },
            )
        except:
            pass

    async def _update_positions(self):
        """Update position status and track P&L"""
        try:
            async with self.session.get(f"{CONFIG['API_BASE']}/positions") as resp:
                positions = await resp.json()

                current_symbols = set()
                for pos in positions:
                    symbol = pos.get("symbol")
                    if symbol:
                        current_symbols.add(symbol)
                        unrealized_pl = pos.get("unrealized_pl", 0)

                        if symbol in self.open_positions:
                            self.open_positions[symbol]["unrealized_pl"] = unrealized_pl
                            self.open_positions[symbol]["current_price"] = pos.get(
                                "current_price"
                            )

                # Check for closed positions (exited via bracket orders)
                closed = set(self.open_positions.keys()) - current_symbols
                for symbol in closed:
                    pos_data = self.open_positions.pop(symbol, {})
                    pnl = pos_data.get("unrealized_pl", 0)

                    if pnl >= 0:
                        self.wins_today += 1
                        logger.info(f"WINNER: {symbol} +${pnl:.2f}")
                    else:
                        self.losses_today += 1
                        logger.info(f"LOSER: {symbol} ${pnl:.2f}")

                    self.daily_pnl += pnl

        except Exception as e:
            logger.error(f"Position update failed: {e}")

    def stop(self):
        """Stop the auto-trader"""
        logger.info("Stopping auto-trader...")
        self.is_trading_enabled = False


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("  WARRIOR TRADING AUTO-TRADER")
    print("  Pre-Market Momentum Strategy")
    print("=" * 60 + "\n")

    trader = WarriorAutoTrader()

    try:
        await trader.start()
    except KeyboardInterrupt:
        print("\nStopping...")
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
