"""
Alpaca Integration for AI Trading Dashboard
Drop-in replacement for IBKR connectivity
"""
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest,
    TakeProfitRequest, StopLossRequest, TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import pytz

load_dotenv()

class AlpacaConnector:
    """Alpaca connection matching IBKR interface with auto-reconnect"""

    # Connection health settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes

    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')

        # Trading client only (market data handled by AlpacaMarketData class)
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=True
        )

        # Note: Data client removed - use AlpacaMarketData class instead
        # This avoids duplicate client instantiation
        self._data_client = None  # Lazy loaded only if needed

        self.connected = False
        self._last_health_check = None
        self._consecutive_failures = 0
        self._connect()

    def _connect(self):
        """Connect to Alpaca with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                account = self.trading_client.get_account()
                self.connected = True
                self._consecutive_failures = 0
                self._last_health_check = datetime.now()
                print(f"[OK] Connected to Alpaca: {account.account_number}")
                return True
            except Exception as e:
                self._consecutive_failures += 1
                print(f"[WARN] Alpaca connection attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))  # Exponential backoff

        print(f"[FAIL] Alpaca connection failed after {self.MAX_RETRIES} attempts")
        self.connected = False
        return False

    def _reconnect(self):
        """Attempt to reconnect to Alpaca"""
        print("[INFO] Attempting Alpaca reconnection...")
        # Recreate trading client
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=True
        )
        self._data_client = None
        return self._connect()

    def _check_health(self) -> bool:
        """Check connection health, reconnect if needed"""
        # Skip if recently checked
        if self._last_health_check:
            elapsed = (datetime.now() - self._last_health_check).total_seconds()
            if elapsed < self.HEALTH_CHECK_INTERVAL:
                return self.connected

        try:
            # Quick health check - get account
            self.trading_client.get_account()
            self.connected = True
            self._consecutive_failures = 0
            self._last_health_check = datetime.now()
            return True
        except Exception as e:
            print(f"[WARN] Alpaca health check failed: {e}")
            self._consecutive_failures += 1
            self.connected = False

            # Try to reconnect if multiple failures
            if self._consecutive_failures >= 2:
                return self._reconnect()
            return False

    def is_connected(self) -> bool:
        """Check connection status with health verification"""
        # Do periodic health check
        if self._last_health_check:
            elapsed = (datetime.now() - self._last_health_check).total_seconds()
            if elapsed > self.HEALTH_CHECK_INTERVAL:
                self._check_health()
        return self.connected

    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if needed"""
        if not self.connected:
            return self._reconnect()
        return self._check_health()

    def get_market_session(self) -> Dict:
        """
        Determine current market session.

        Returns:
            Dict with session info:
            - session: "pre_market", "regular", "after_hours", "closed"
            - is_extended_hours: True if pre-market or after-hours
            - can_trade: True if any trading is possible
            - use_limit_orders: True if should use limit orders (extended hours)

        Market Hours (Eastern Time):
        - Pre-market: 4:00 AM - 9:30 AM
        - Regular: 9:30 AM - 4:00 PM
        - After-hours: 4:00 PM - 8:00 PM
        - Closed: 8:00 PM - 4:00 AM
        """
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        current_time = now_et.time()

        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return {
                "session": "closed",
                "is_extended_hours": False,
                "can_trade": False,
                "use_limit_orders": False,
                "current_time_et": str(current_time),
                "reason": "Weekend - market closed"
            }

        # Define market hours
        from datetime import time as dt_time
        pre_market_start = dt_time(4, 0)
        regular_open = dt_time(9, 30)
        regular_close = dt_time(16, 0)
        after_hours_end = dt_time(20, 0)

        if pre_market_start <= current_time < regular_open:
            return {
                "session": "pre_market",
                "is_extended_hours": True,
                "can_trade": True,
                "use_limit_orders": True,  # MUST use limit orders
                "current_time_et": str(current_time),
                "reason": "Pre-market - limit orders only"
            }
        elif regular_open <= current_time < regular_close:
            return {
                "session": "regular",
                "is_extended_hours": False,
                "can_trade": True,
                "use_limit_orders": False,  # Can use market orders
                "current_time_et": str(current_time),
                "reason": "Regular hours - all order types"
            }
        elif regular_close <= current_time < after_hours_end:
            return {
                "session": "after_hours",
                "is_extended_hours": True,
                "can_trade": True,
                "use_limit_orders": True,  # MUST use limit orders
                "current_time_et": str(current_time),
                "reason": "After-hours - limit orders only"
            }
        else:
            return {
                "session": "closed",
                "is_extended_hours": False,
                "can_trade": False,
                "use_limit_orders": False,
                "current_time_et": str(current_time),
                "reason": "Market closed"
            }

    def get_account(self) -> Dict:
        """Get account information with daily P/L"""
        account = self.trading_client.get_account()

        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity > 0 else 0

        return {
            "account_id": account.account_number,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": equity,
            "last_equity": last_equity,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "currency": account.currency,
            "status": str(account.status),
            "pattern_day_trader": account.pattern_day_trader
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = self.trading_client.get_all_positions()
        
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "quantity": float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": pos.side
            })
        
        return result
    
    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        """Get orders with limit for performance"""
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(status=status, limit=limit)
        orders = self.trading_client.get_orders(filter=request)

        result = []
        for order in orders:
            # Extract clean enum values (e.g., "OrderSide.BUY" -> "buy")
            side_str = str(order.side).split('.')[-1].lower() if order.side else ""
            type_str = str(order.type).split('.')[-1].lower() if order.type else ""
            status_str = str(order.status).split('.')[-1].lower() if order.status else ""

            result.append({
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": side_str,
                "quantity": float(order.qty),
                "order_type": type_str,
                "status": status_str,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "submitted_at": str(order.submitted_at),
                "filled_at": str(order.filled_at) if order.filled_at else None
            })

        return result
    
    def place_market_order(self, symbol: str, quantity: int, side: str) -> Dict:
        """Place market order"""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty)
        }
    
    def place_limit_order(self, symbol: str, quantity: int, side: str, limit_price: float) -> Dict:
        """Place limit order"""
        # SAFETY CHECK: Ensure limit price is valid (not $0.01 or below)
        if limit_price is None or limit_price < 0.50:
            raise ValueError(f"Invalid limit price ${limit_price:.2f if limit_price else 0:.2f} for {symbol} - minimum limit price is $0.50")

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "limit_price": float(order.limit_price)
        }

    def place_stop_order(self, symbol: str, quantity: int, side: str, stop_price: float) -> Dict:
        """Place stop order"""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        order_request = StopOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            stop_price=stop_price
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "stop_price": float(order.stop_price)
        }

    def place_stop_limit_order(self, symbol: str, quantity: int, side: str,
                               stop_price: float, limit_price: float) -> Dict:
        """Place stop-limit order"""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        order_request = StopLimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            stop_price=stop_price,
            limit_price=limit_price
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "stop_price": float(order.stop_price),
            "limit_price": float(order.limit_price)
        }

    def place_bracket_order(self, symbol: str, quantity: int, side: str,
                            take_profit_price: float, stop_loss_price: float,
                            limit_price: float = None) -> Dict:
        """
        Place a bracket order with automatic take profit and stop loss.

        A bracket order consists of three orders:
        1. Entry order (market or limit)
        2. Take profit order (limit sell/buy)
        3. Stop loss order (stop sell/buy)

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "BUY" or "SELL"
            take_profit_price: Price to take profit
            stop_loss_price: Price to cut losses
            limit_price: Optional limit price for entry (None = market order)

        Returns:
            Order details including all bracket leg IDs
        """
        # SAFETY CHECK: Ensure limit price is valid (not $0.01 or below)
        if limit_price is not None and limit_price < 0.50:
            raise ValueError(f"Invalid limit price ${limit_price:.2f} for {symbol} - minimum limit price is $0.50")

        # SAFETY CHECK: Ensure take profit and stop loss are valid
        if take_profit_price < 0.50:
            raise ValueError(f"Invalid take profit price ${take_profit_price:.2f} for {symbol} - minimum price is $0.50")
        if stop_loss_price < 0.50:
            raise ValueError(f"Invalid stop loss price ${stop_loss_price:.2f} for {symbol} - minimum price is $0.50")

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        # Create the bracket order request
        if limit_price:
            # Bracket with limit entry
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )
        else:
            # Bracket with market entry
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "order_class": "bracket",
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "entry_price": float(order.limit_price) if order.limit_price else "market",
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "legs": order.legs if hasattr(order, 'legs') else None
        }

    def place_oco_order(self, symbol: str, quantity: int, side: str,
                        take_profit_price: float, stop_loss_price: float) -> Dict:
        """
        Place an OCO (One-Cancels-Other) order.

        OCO order creates two orders where if one fills, the other is cancelled.
        Typically used to set both a profit target and stop loss for an existing position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "SELL" to close a long position, "BUY" to close a short
            take_profit_price: Limit price for profit taking
            stop_loss_price: Stop price for loss cutting

        Returns:
            Order details including both leg IDs
        """
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        # OCO order with limit (take profit) and stop (stop loss)
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=take_profit_price,
            order_class=OrderClass.OCO,
            stop_loss=StopLossRequest(stop_price=stop_loss_price)
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "order_class": "oco",
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "legs": order.legs if hasattr(order, 'legs') else None
        }

    def place_trailing_stop_order(self, symbol: str, quantity: int, side: str,
                                   trail_percent: float = None, trail_price: float = None) -> Dict:
        """
        Place a trailing stop order.

        A trailing stop follows the price up (for sells) or down (for buys)
        by a specified amount or percentage.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "SELL" for long positions, "BUY" for short positions
            trail_percent: Trailing percentage (e.g., 5.0 for 5%)
            trail_price: Trailing dollar amount (alternative to percent)

        Returns:
            Order details
        """
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        if trail_percent:
            order_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                trail_percent=trail_percent
            )
        elif trail_price:
            order_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                trail_price=trail_price
            )
        else:
            raise ValueError("Either trail_percent or trail_price must be specified")

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "order_class": "trailing_stop",
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "trail_percent": trail_percent,
            "trail_price": trail_price
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            print(f"Cancel order failed: {e}")
            return False

    def place_smart_order(self, symbol: str, quantity: int, side: str,
                          limit_price: float = None, emergency: bool = False,
                          momentum: bool = False) -> Dict:
        """
        SMART ORDER - LIMIT ORDERS ONLY (except emergency liquidation).

        ORDER PRICING RULES:
        ====================
        - BUY: Use ASK price (hit the ask to get filled fast)
        - SELL normal exit: Use BID price (guaranteed fill, protect against gaps)
        - SELL momentum: Use ASK price (try for better price while riding momentum up)
        - SELL emergency: Use BID price + market order during regular hours

        MARKET ORDERS ARE NEVER USED except for emergency=True during regular hours.
        This protects against getting stuck in bad fills from gaps/volatility.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: "BUY" or "SELL"
            limit_price: Optional limit price (auto-calculated from quote if not provided)
            emergency: If True and SELL, uses BID + market order during regular hours
            momentum: If True and SELL, uses ASK price for better exit while riding momentum

        Returns:
            Order details with session info
        """
        session = self.get_market_session()
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        # ============================================================
        # GET PRICE FROM CENTRALIZED DATA BUS (SCHWAB ONLY!)
        # ============================================================
        from market_data_bus import get_market_data_bus
        data_bus = get_market_data_bus()

        if limit_price is None:
            quote = data_bus.get_quote(symbol)
            if not quote:
                raise ValueError(f"No valid quote data from data bus for {symbol}. Cannot place order.")

            bid = quote.get("bid", 0) or 0
            ask = quote.get("ask", 0) or 0

            if bid <= 0 and ask <= 0:
                raise ValueError(f"No valid bid/ask data for {symbol}. bid={bid}, ask={ask}")

            # PRICING RULES (per trading rules):
            # - BUY: Always use ASK (hit the ask to get filled fast)
            # - SELL normal exit: Use BID (guaranteed fill, protect against gaps)
            # - SELL momentum: Use ASK (better exit while riding momentum up)
            # - SELL emergency: Use BID (get out NOW, accept the hit)
            if side.upper() == "BUY":
                limit_price = ask if ask > 0 else bid * 1.02
                price_type = "ASK"
            elif emergency:
                # EMERGENCY EXIT - Use BID to guarantee fill
                limit_price = bid if bid > 0 else ask * 0.98
                price_type = "BID (EMERGENCY)"
            elif momentum:
                # MOMENTUM SELL - Use ASK for better exit while price is rising
                limit_price = ask if ask > 0 else bid * 1.01
                price_type = "ASK (MOMENTUM)"
            else:
                # NORMAL SELL EXIT - Use BID to guarantee fill
                limit_price = bid if bid > 0 else ask * 0.99
                price_type = "BID (NORMAL EXIT)"

            # Round to 2 decimal places
            limit_price = round(limit_price, 2)

            logger.info(f"Smart order {side} {symbol}: ${limit_price:.2f} ({price_type}) | bid=${bid:.2f} ask=${ask:.2f} | source={quote.get('source', 'schwab')}")

        # SAFETY CHECK: Ensure limit price is valid
        if limit_price is None or limit_price < 0.50:
            raise ValueError(f"Invalid limit price ${limit_price:.2f} for {symbol}")

        # ============================================================
        # EMERGENCY EXIT DURING REGULAR HOURS = MARKET ORDER
        # ============================================================
        if emergency and side.upper() == "SELL" and session["session"] == "regular":
            # Use market order for fastest exit during regular hours
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_request)
            logger.warning(f"🚨 EMERGENCY MARKET SELL {symbol} x{quantity}")

            return {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "status": str(order.status).split('.')[-1].lower(),
                "side": str(order.side).split('.')[-1].lower(),
                "quantity": float(order.qty),
                "order_type": "market",
                "limit_price": None,
                "extended_hours": False,
                "emergency": True,
                "session": session["session"],
                "session_info": "EMERGENCY MARKET EXIT"
            }

        # ============================================================
        # DEFAULT: LIMIT ORDER WITH EXTENDED HOURS (works any session)
        # ============================================================
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            extended_hours=True  # KEY: Always enable extended hours
        )

        order = self.trading_client.submit_order(order_request)

        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "status": str(order.status).split('.')[-1].lower(),
            "side": str(order.side).split('.')[-1].lower(),
            "quantity": float(order.qty),
            "order_type": "limit",
            "limit_price": float(order.limit_price),
            "extended_hours": True,
            "emergency": emergency,
            "session": session["session"],
            "session_info": session["reason"]
        }

    def close_position_smart(self, symbol: str) -> Dict:
        """
        Smart position close that handles extended hours.

        During extended hours, uses limit order at current bid.
        During regular hours, uses market order.
        """
        session = self.get_market_session()

        # Get current position
        try:
            positions = self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)

            if not position:
                return {"success": False, "error": f"No position found for {symbol}"}

            quantity = int(abs(position["quantity"]))
            side = "SELL" if position["quantity"] > 0 else "BUY"

            if session["is_extended_hours"]:
                # Extended hours - use smart limit order
                return self.place_smart_order(symbol, quantity, side)
            else:
                # Regular hours - use direct close
                self.trading_client.close_position(symbol)
                return {
                    "success": True,
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": side,
                    "method": "direct_close",
                    "session": session["session"]
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_and_replace_extended(self, symbol: str) -> Dict:
        """
        Cancel existing orders for a symbol and place extended hours orders.

        Use this to convert pending market orders to extended hours limit orders.
        """
        results = {"cancelled": [], "new_orders": [], "errors": []}

        try:
            # Get all open orders for symbol
            orders = self.get_orders(status="open")
            symbol_orders = [o for o in orders if o["symbol"] == symbol]

            # Cancel existing orders
            for order in symbol_orders:
                if self.cancel_order(order["order_id"]):
                    results["cancelled"].append(order["order_id"])
                else:
                    results["errors"].append(f"Failed to cancel {order['order_id']}")

            # Get current position to determine what orders are needed
            positions = self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)

            if position:
                # Re-submit as smart order if we have a position
                quantity = int(abs(position["quantity"]))
                side = "SELL" if position["quantity"] > 0 else "BUY"

                new_order = self.place_smart_order(symbol, quantity, side)
                results["new_orders"].append(new_order)

            results["success"] = True

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results
    
    def _get_data_client(self):
        """Lazy load data client only when needed"""
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                self.api_key,
                self.secret_key
            )
        return self._data_client

    def get_quote(self, symbol: str) -> Dict:
        """Get latest quote for symbol"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._get_data_client().get_stock_latest_quote(request)
            
            quote = quotes[symbol]
            
            return {
                "symbol": symbol,
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                "bid_size": int(quote.bid_size),
                "ask_size": int(quote.ask_size)
            }
        except Exception as e:
            print(f"Get quote failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "bid": 0,
                "ask": 0,
                "last": 0,
                "bid_size": 0,
                "ask_size": 0
            }
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            self.trading_client.close_position(symbol)
            return True
        except Exception as e:
            print(f"Close position failed: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            return True
        except Exception as e:
            print(f"Close all positions failed: {e}")
            return False
    
    def get_clock(self) -> Dict:
        """Get market clock status"""
        try:
            clock = self.trading_client.get_clock()
            return {
                "timestamp": str(clock.timestamp),
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close)
            }
        except Exception as e:
            print(f"Get clock failed: {e}")
            # Return default closed state on error
            return {
                "timestamp": str(datetime.now()),
                "is_open": False,
                "next_open": None,
                "next_close": None
            }

# Global instance
alpaca_connector = None

def get_alpaca_connector():
    """Get or create Alpaca connector instance"""
    global alpaca_connector
    
    if alpaca_connector is None:
        alpaca_connector = AlpacaConnector()
    
    return alpaca_connector
