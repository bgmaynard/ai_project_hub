"""
Unified Broker Interface for Morpheus Trading Bot
==================================================
Single interface for all broker operations.
Broker: Schwab

Version: 2.2.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BrokerType(str, Enum):
    """Supported brokers"""

    SCHWAB = "schwab"


class OrderSide(str, Enum):
    """Order side"""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order types"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderResult:
    """Standardized order result"""

    success: bool
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0
    price: Optional[float] = None
    status: str = ""
    broker: str = ""
    error: Optional[str] = None
    timestamp: str = ""


class UnifiedBroker:
    """
    Unified broker interface for Schwab

    Usage:
        broker = UnifiedBroker()
        result = broker.place_limit_order("AAPL", OrderSide.BUY, 100, 150.00)
    """

    def __init__(self, prefer_schwab: bool = True):
        self.prefer_schwab = prefer_schwab
        self._schwab = None
        self._active_broker = None
        self._initialize()

    def _initialize(self):
        """Initialize Schwab broker connection"""
        try:
            from schwab_trading import (get_schwab_trading,
                                        is_schwab_trading_available)

            if is_schwab_trading_available():
                self._schwab = get_schwab_trading()
                if self._schwab:
                    self._active_broker = BrokerType.SCHWAB
                    logger.info("UnifiedBroker: Schwab broker connected")
        except Exception as e:
            logger.warning(f"Schwab initialization failed: {e}")

        if not self._active_broker:
            logger.error("UnifiedBroker: Schwab broker not available!")

    @property
    def broker_name(self) -> str:
        """Get active broker name"""
        return self._active_broker.value if self._active_broker else "none"

    @property
    def is_connected(self) -> bool:
        """Check if connected to any broker"""
        return self._active_broker is not None

    def get_account(self) -> Optional[Dict]:
        """Get account information"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.get_account_info()
            except Exception as e:
                logger.error(f"Schwab account error: {e}")

        return None

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.get_positions()
            except Exception as e:
                logger.error(f"Schwab positions error: {e}")

        return []

    def get_orders(self, status: str = "all") -> List[Dict]:
        """Get orders"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.get_orders()
            except Exception as e:
                logger.error(f"Schwab orders error: {e}")

        return []

    def place_limit_order(
        self, symbol: str, side: OrderSide, quantity: int, price: float
    ) -> OrderResult:
        """Place a limit order"""
        timestamp = datetime.now().isoformat()

        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                result = self._schwab.place_limit_order(
                    symbol=symbol, quantity=quantity, side=side.value, limit_price=price
                )
                if result and result.get("success"):
                    return OrderResult(
                        success=True,
                        order_id=result.get("order_id"),
                        symbol=symbol,
                        side=side.value,
                        quantity=quantity,
                        price=price,
                        status="PENDING",
                        broker="Schwab",
                        timestamp=timestamp,
                    )
                else:
                    return OrderResult(
                        success=False,
                        symbol=symbol,
                        side=side.value,
                        quantity=quantity,
                        price=price,
                        broker="Schwab",
                        error=result.get("error", "Unknown error"),
                        timestamp=timestamp,
                    )
            except Exception as e:
                logger.error(f"Schwab order error: {e}")
                # Don't fall back for orders - return error
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    price=price,
                    broker="Schwab",
                    error=str(e),
                    timestamp=timestamp,
                )

        return OrderResult(
            success=False,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=price,
            broker="none",
            error="No broker available",
            timestamp=timestamp,
        )

    def place_market_order(
        self, symbol: str, side: OrderSide, quantity: int
    ) -> OrderResult:
        """
        Place a market order
        WARNING: Market orders are discouraged - use limit orders instead!
        """
        logger.warning(
            f"Market order requested for {symbol} - consider using limit orders!"
        )
        timestamp = datetime.now().isoformat()

        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                result = self._schwab.place_market_order(
                    symbol=symbol, side=side.value, quantity=quantity
                )
                if result and result.get("success"):
                    return OrderResult(
                        success=True,
                        order_id=result.get("order_id"),
                        symbol=symbol,
                        side=side.value,
                        quantity=quantity,
                        status="PENDING",
                        broker="Schwab",
                        timestamp=timestamp,
                    )
            except Exception as e:
                logger.error(f"Schwab market order error: {e}")

        return OrderResult(
            success=False,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            broker=self.broker_name,
            error="Market orders should use limit orders instead",
            timestamp=timestamp,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Schwab cancel error: {e}")

        return False

    def close_position(self, symbol: str) -> OrderResult:
        """Close a position by selling all shares"""
        timestamp = datetime.now().isoformat()
        symbol = symbol.upper()

        # Get current positions
        positions = self.get_positions()
        position = next(
            (p for p in positions if p.get("symbol", "").upper() == symbol), None
        )

        if not position:
            return OrderResult(
                success=False,
                symbol=symbol,
                broker=self.broker_name,
                error=f"No position found for {symbol}",
                timestamp=timestamp,
            )

        quantity = abs(int(float(position.get("qty", position.get("quantity", 0)))))
        if quantity <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                broker=self.broker_name,
                error=f"Invalid quantity for {symbol}",
                timestamp=timestamp,
            )

        # Place sell order to close position
        return self.place_limit_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=float(
                position.get("current_price", position.get("market_value", 0))
                / quantity
            ),
        )

    def close_all_positions(self) -> List[OrderResult]:
        """Close all open positions"""
        results = []
        positions = self.get_positions()

        for position in positions:
            symbol = position.get("symbol", "")
            if symbol:
                result = self.close_position(symbol)
                results.append(result)

        return results

    def _get_smart_limit_price(self, symbol: str, side: str) -> Optional[float]:
        """
        Get intelligent limit price for smart orders.
        Uses bid for sells (slightly below market) and ask for buys (slightly above market).
        Adds a small buffer for better fill rates.
        """
        try:
            # Try Schwab market data first
            from schwab_market_data import (SchwabMarketData,
                                            is_schwab_available)

            if is_schwab_available():
                schwab_data = SchwabMarketData()
                quote = schwab_data.get_quote(symbol)
                if quote:
                    bid = quote.get("bid", quote.get("bidPrice", 0))
                    ask = quote.get("ask", quote.get("askPrice", 0))
                    last = quote.get("last", quote.get("lastPrice", 0))

                    if side.upper() == "BUY":
                        # For buys, use ask + 0.02% buffer for better fills
                        if ask and ask > 0:
                            return round(ask * 1.0002, 2)
                        elif last and last > 0:
                            return round(last * 1.001, 2)  # 0.1% above last
                    else:
                        # For sells, use bid - 0.02% buffer for faster fills
                        if bid and bid > 0:
                            return round(bid * 0.9998, 2)
                        elif last and last > 0:
                            return round(last * 0.999, 2)  # 0.1% below last
        except Exception as e:
            logger.debug(f"Schwab quote failed for {symbol}: {e}")

        return None

    def _validate_order(
        self, symbol: str, quantity: int, side: str, limit_price: float
    ) -> Optional[str]:
        """
        Validate order before submission.
        Returns error message if invalid, None if valid.
        """
        # Basic validation
        if quantity <= 0:
            return f"Invalid quantity: {quantity}"
        if limit_price <= 0:
            return f"Invalid limit price: {limit_price}"
        if len(symbol) == 0:
            return "Symbol is required"

        # Position size validation
        order_value = quantity * limit_price
        if order_value > 100000:  # Max single order value
            logger.warning(f"Large order: {symbol} ${order_value:.2f}")
            # Don't block but warn

        # Check buying power for buys
        if side.upper() == "BUY":
            try:
                account = self.get_account()
                if account:
                    buying_power = float(
                        account.get("buying_power", account.get("buyingPower", 0))
                    )
                    if buying_power > 0 and order_value > buying_power:
                        return f"Insufficient buying power: ${buying_power:.2f} < ${order_value:.2f}"
            except Exception as e:
                logger.warning(f"Could not validate buying power: {e}")

        return None  # Valid

    def place_smart_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: float = None,
        emergency: bool = False,
    ) -> OrderResult:
        """
        Smart order placement - uses best available execution method.

        Features:
        - Auto-fetches quote data if no limit_price provided
        - Validates order before submission
        - Routes to Schwab broker
        - Uses intelligent bid/ask pricing for better fills
        """
        timestamp = datetime.now().isoformat()
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        symbol = symbol.upper()

        # Get price if not provided
        effective_price = limit_price
        if not effective_price:
            effective_price = self._get_smart_limit_price(symbol, side)
            if effective_price:
                logger.info(
                    f"Smart order: Using calculated price ${effective_price:.2f} for {symbol}"
                )
            else:
                logger.warning(f"Smart order: Could not get quote for {symbol}")
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    broker=self.broker_name,
                    error="Could not determine limit price - no market data available",
                    timestamp=timestamp,
                )

        # Validate order (skip in emergency mode for faster execution)
        if not emergency:
            validation_error = self._validate_order(
                symbol, quantity, side, effective_price
            )
            if validation_error:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=effective_price,
                    broker=self.broker_name,
                    error=validation_error,
                    timestamp=timestamp,
                )

        # Use Schwab limit order
        return self.place_limit_order(symbol, order_side, quantity, effective_price)

    def get_status(self) -> Dict:
        """Get broker status"""
        return {
            "active_broker": self.broker_name,
            "schwab_available": self._schwab is not None,
            "connected": self.is_connected,
            "broker": "Schwab",
            "name": "Morpheus Trading Bot",
        }


# Singleton instance
_unified_broker: Optional[UnifiedBroker] = None


def get_unified_broker() -> UnifiedBroker:
    """Get singleton unified broker instance"""
    global _unified_broker
    if _unified_broker is None:
        _unified_broker = UnifiedBroker()
    return _unified_broker
