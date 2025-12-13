"""
Unified Broker Interface for Morpheus Trading Bot
==================================================
Single interface for all broker operations.
Primary: Schwab
Fallback: Alpaca (paper trading only)

Version: 2.1.0
"""

import logging
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class BrokerType(str, Enum):
    """Supported brokers"""
    SCHWAB = "schwab"
    ALPACA = "alpaca"


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
    Unified broker interface - abstracts Schwab and Alpaca

    Usage:
        broker = UnifiedBroker()
        result = broker.place_limit_order("AAPL", OrderSide.BUY, 100, 150.00)
    """

    def __init__(self, prefer_schwab: bool = True):
        self.prefer_schwab = prefer_schwab
        self._schwab = None
        self._alpaca = None
        self._active_broker = None
        self._initialize()

    def _initialize(self):
        """Initialize broker connections"""
        # Try Schwab first (primary)
        if self.prefer_schwab:
            try:
                from schwab_trading import get_schwab_trading, is_schwab_trading_available
                if is_schwab_trading_available():
                    self._schwab = get_schwab_trading()
                    if self._schwab:
                        self._active_broker = BrokerType.SCHWAB
                        logger.info("UnifiedBroker: Using Schwab as primary broker")
            except Exception as e:
                logger.warning(f"Schwab initialization failed: {e}")

        # Fallback to Alpaca
        if not self._active_broker:
            try:
                from alpaca_integration import get_alpaca_connector
                self._alpaca = get_alpaca_connector()
                if self._alpaca and self._alpaca.is_connected():
                    self._active_broker = BrokerType.ALPACA
                    logger.info("UnifiedBroker: Using Alpaca as fallback broker")
            except Exception as e:
                logger.warning(f"Alpaca initialization failed: {e}")

        if not self._active_broker:
            logger.error("UnifiedBroker: No broker available!")

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

        if self._alpaca:
            try:
                return self._alpaca.get_account()
            except Exception as e:
                logger.error(f"Alpaca account error: {e}")

        return None

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.get_positions()
            except Exception as e:
                logger.error(f"Schwab positions error: {e}")

        if self._alpaca:
            try:
                return self._alpaca.get_positions()
            except Exception as e:
                logger.error(f"Alpaca positions error: {e}")

        return []

    def get_orders(self, status: str = "all") -> List[Dict]:
        """Get orders"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.get_orders()
            except Exception as e:
                logger.error(f"Schwab orders error: {e}")

        if self._alpaca:
            try:
                return self._alpaca.get_orders(status)
            except Exception as e:
                logger.error(f"Alpaca orders error: {e}")

        return []

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float
    ) -> OrderResult:
        """Place a limit order"""
        timestamp = datetime.now().isoformat()

        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                result = self._schwab.place_limit_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side.value,
                    limit_price=price
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
                        timestamp=timestamp
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
                        timestamp=timestamp
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
                    timestamp=timestamp
                )

        # Alpaca fallback (paper trading only)
        if self._alpaca:
            try:
                if side == OrderSide.BUY:
                    result = self._alpaca.place_limit_order(symbol, quantity, price)
                else:
                    result = self._alpaca.sell_limit(symbol, quantity, price)

                return OrderResult(
                    success=True,
                    order_id=result.get("order_id") if result else None,
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    price=price,
                    status="PENDING",
                    broker="Alpaca (paper)",
                    timestamp=timestamp
                )
            except Exception as e:
                return OrderResult(
                    success=False,
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity,
                    price=price,
                    broker="Alpaca",
                    error=str(e),
                    timestamp=timestamp
                )

        return OrderResult(
            success=False,
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            price=price,
            broker="none",
            error="No broker available",
            timestamp=timestamp
        )

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int
    ) -> OrderResult:
        """
        Place a market order
        WARNING: Market orders are discouraged - use limit orders instead!
        """
        logger.warning(f"Market order requested for {symbol} - consider using limit orders!")
        timestamp = datetime.now().isoformat()

        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                result = self._schwab.place_market_order(
                    symbol=symbol,
                    side=side.value,
                    quantity=quantity
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
                        timestamp=timestamp
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
            timestamp=timestamp
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if self._active_broker == BrokerType.SCHWAB and self._schwab:
            try:
                return self._schwab.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Schwab cancel error: {e}")

        if self._alpaca:
            try:
                return self._alpaca.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Alpaca cancel error: {e}")

        return False

    def close_position(self, symbol: str) -> OrderResult:
        """Close a position by selling all shares"""
        timestamp = datetime.now().isoformat()
        symbol = symbol.upper()

        # Get current positions
        positions = self.get_positions()
        position = next((p for p in positions if p.get("symbol", "").upper() == symbol), None)

        if not position:
            return OrderResult(
                success=False,
                symbol=symbol,
                broker=self.broker_name,
                error=f"No position found for {symbol}",
                timestamp=timestamp
            )

        quantity = abs(int(float(position.get("qty", position.get("quantity", 0)))))
        if quantity <= 0:
            return OrderResult(
                success=False,
                symbol=symbol,
                broker=self.broker_name,
                error=f"Invalid quantity for {symbol}",
                timestamp=timestamp
            )

        # Place sell order to close position
        return self.place_limit_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=float(position.get("current_price", position.get("market_value", 0)) / quantity)
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

    def place_smart_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: float = None,
        emergency: bool = False
    ) -> OrderResult:
        """
        Smart order placement - uses best available execution method.

        For Schwab: Routes to limit order with current quote price
        For Alpaca: Uses Alpaca's native smart order with bid/ask logic
        """
        timestamp = datetime.now().isoformat()
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        # Try Alpaca's native smart order first if available (for paper trading)
        if self._alpaca and self._active_broker == BrokerType.ALPACA:
            try:
                result = self._alpaca.place_smart_order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    limit_price=limit_price,
                    emergency=emergency
                )
                if result and result.get("order_id"):
                    return OrderResult(
                        success=True,
                        order_id=result.get("order_id"),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=result.get("limit_price"),
                        status=result.get("status", "PENDING"),
                        broker="Alpaca (paper)",
                        timestamp=timestamp
                    )
            except Exception as e:
                logger.warning(f"Alpaca smart order failed: {e}")

        # For Schwab or fallback: use limit order with provided price
        if limit_price:
            return self.place_limit_order(symbol, order_side, quantity, limit_price)

        # No price provided - log warning and return error
        logger.warning(f"Smart order for {symbol}: No limit price provided and no quote available")
        return OrderResult(
            success=False,
            symbol=symbol,
            side=side,
            quantity=quantity,
            broker=self.broker_name,
            error="Limit price required for smart orders",
            timestamp=timestamp
        )

    def get_status(self) -> Dict:
        """Get broker status"""
        return {
            "active_broker": self.broker_name,
            "schwab_available": self._schwab is not None,
            "alpaca_available": self._alpaca is not None,
            "connected": self.is_connected,
            "primary": "Schwab",
            "fallback": "Alpaca (paper trading)",
            "name": "Morpheus Trading Bot"
        }


# Singleton instance
_unified_broker: Optional[UnifiedBroker] = None


def get_unified_broker() -> UnifiedBroker:
    """Get singleton unified broker instance"""
    global _unified_broker
    if _unified_broker is None:
        _unified_broker = UnifiedBroker()
    return _unified_broker
