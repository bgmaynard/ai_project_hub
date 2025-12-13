"""
Schwab/ThinkOrSwim Trading Integration
Order placement and account management via Schwab API
"""
import os
import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

# Token file location
TOKEN_FILE = Path(__file__).parent / "schwab_token.json"

# Schwab API base URL
SCHWAB_TRADER_BASE = "https://api.schwabapi.com/trader/v1"

# Import token management from market data module
import schwab_market_data as _schwab_md

def _ensure_token():
    """Ensure token is valid"""
    return _schwab_md._ensure_token()

def _refresh_token():
    """Refresh the token"""
    return _schwab_md._refresh_token()

def _get_token_data():
    """Get current token data"""
    return _schwab_md._token_data


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class OrderDuration(str, Enum):
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    GTD = "GOOD_TILL_DATE"
    FOK = "FILL_OR_KILL"
    IOC = "IMMEDIATE_OR_CANCEL"


class OrderStatus(str, Enum):
    PENDING = "PENDING_ACTIVATION"
    QUEUED = "QUEUED"
    WORKING = "WORKING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


def _make_trading_request(method: str, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[Dict]:
    """Make authenticated request to Schwab Trader API"""
    access_token = _ensure_token()
    if not access_token:
        logger.error("No valid access token for trading")
        return None

    try:
        # Base headers for all requests
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }

        url = f"{SCHWAB_TRADER_BASE}{endpoint}"

        if method.upper() == "GET":
            response = httpx.get(url, headers=headers, params=params, timeout=30.0)
        elif method.upper() == "POST":
            response = httpx.post(url, headers=headers, json=data, timeout=30.0)
        elif method.upper() == "PUT":
            response = httpx.put(url, headers=headers, json=data, timeout=30.0)
        elif method.upper() == "DELETE":
            response = httpx.delete(url, headers=headers, timeout=30.0)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None

        if response.status_code in [200, 201]:
            if response.content:
                return response.json()
            return {"success": True}
        elif response.status_code == 401:
            # Token might be invalid, try refresh
            logger.warning("Got 401, attempting token refresh...")
            if _refresh_token():
                headers["Authorization"] = f"Bearer {_get_token_data().get('access_token')}"
                if method.upper() == "GET":
                    response = httpx.get(url, headers=headers, params=params, timeout=30.0)
                elif method.upper() == "POST":
                    response = httpx.post(url, headers=headers, json=data, timeout=30.0)
                elif method.upper() == "DELETE":
                    response = httpx.delete(url, headers=headers, timeout=30.0)

                if response.status_code in [200, 201]:
                    if response.content:
                        return response.json()
                    return {"success": True}
            logger.error(f"Request failed after refresh: {response.status_code}")
            return None
        elif response.status_code == 204:
            # No content (success for DELETE)
            return {"success": True}
        else:
            logger.error(f"Schwab Trader API error: {response.status_code} - {response.text[:500]}")
            return {"error": response.text, "status_code": response.status_code}

    except Exception as e:
        logger.error(f"Schwab Trader API request error: {e}")
        return None


class SchwabTrading:
    """Schwab/ThinkOrSwim trading client"""

    def __init__(self):
        """Initialize Schwab trading client"""
        self._accounts: List[Dict] = []
        self._selected_account: Optional[str] = None
        self._account_hash: Optional[str] = None
        self._load_accounts()

    def _load_accounts(self):
        """Load available Schwab accounts"""
        try:
            data = _make_trading_request("GET", "/accounts/accountNumbers")
            if data and isinstance(data, list):
                self._accounts = data
                logger.info(f"Loaded {len(self._accounts)} Schwab account(s)")

                # Auto-select first account if only one
                if len(self._accounts) == 1:
                    self._selected_account = self._accounts[0].get('accountNumber')
                    self._account_hash = self._accounts[0].get('hashValue')
                    logger.info(f"Auto-selected account: {self._selected_account}")
            else:
                logger.warning("No accounts found or error loading accounts")
        except Exception as e:
            logger.error(f"Error loading Schwab accounts: {e}")

    def get_accounts(self) -> List[Dict]:
        """Get list of available accounts"""
        if not self._accounts:
            self._load_accounts()

        # Return account info without sensitive hash
        return [
            {
                "account_number": acc.get('accountNumber'),
                "selected": acc.get('accountNumber') == self._selected_account
            }
            for acc in self._accounts
        ]

    def select_account(self, account_number: str) -> bool:
        """Select an account for trading"""
        for acc in self._accounts:
            if acc.get('accountNumber') == account_number:
                self._selected_account = account_number
                self._account_hash = acc.get('hashValue')
                logger.info(f"Selected account: {account_number}")
                return True
        logger.error(f"Account not found: {account_number}")
        return False

    def get_selected_account(self) -> Optional[str]:
        """Get currently selected account"""
        return self._selected_account

    def _ensure_account(self) -> bool:
        """Ensure an account is selected"""
        if not self._selected_account or not self._account_hash:
            if self._accounts:
                self._selected_account = self._accounts[0].get('accountNumber')
                self._account_hash = self._accounts[0].get('hashValue')
                return True
            logger.error("No Schwab account selected")
            return False
        return True

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self._ensure_account():
            return None

        data = _make_trading_request("GET", f"/accounts/{self._account_hash}", params={"fields": "positions"})

        if not data:
            return None

        account = data.get('securitiesAccount', {})
        balances = account.get('currentBalances', {})
        initial_balances = account.get('initialBalances', {})

        # Calculate daily P/L - compare current account value to start of day
        positions = account.get('positions', [])

        # Method 1: From positions (unrealized P/L on open positions)
        positions_daily_pl = sum(float(pos.get('currentDayProfitLoss', 0) or 0) for pos in positions)

        # Method 2: From account value change (includes closed trades)
        current_value = float(balances.get('liquidationValue', 0) or 0)
        initial_value = float(initial_balances.get('accountValue', 0) or 0)

        # If we have initial value, use the difference (more accurate for closed trades)
        if initial_value > 0:
            daily_pl = current_value - initial_value
        else:
            # Fallback to positions P/L
            daily_pl = positions_daily_pl

        # Calculate percentage
        base_value = initial_value if initial_value > 0 else current_value
        daily_pl_pct = (daily_pl / base_value * 100) if base_value > 0 else 0

        return {
            "account_number": account.get('accountNumber'),
            "type": account.get('type'),
            # Cash and buying power
            "cash": float(balances.get('cashBalance', 0) or 0),
            "buying_power": float(balances.get('buyingPower', 0) or 0),
            "day_trading_buying_power": float(balances.get('dayTradingBuyingPower', 0) or 0),
            "available_funds": float(balances.get('availableFunds', 0) or 0),
            # Account value
            "equity": float(balances.get('equity', 0) or 0),
            "market_value": float(balances.get('liquidationValue', 0) or 0),
            "long_market_value": float(balances.get('longMarketValue', 0) or 0),
            "short_market_value": float(balances.get('shortMarketValue', 0) or 0),
            # Settled vs unsettled
            "cash_available_for_trading": float(balances.get('cashAvailableForTrading', 0) or 0),
            "cash_available_for_withdrawal": float(balances.get('cashAvailableForWithdrawal', 0) or 0),
            "unsettled_cash": float(balances.get('unsettledCash', 0) or 0),
            # Margin info
            "margin_balance": float(balances.get('marginBalance', 0) or 0),
            "maintenance_requirement": float(balances.get('maintenanceRequirement', 0) or 0),
            "reg_t_call": float(balances.get('regTCall', 0) or 0),
            # Day trading status
            "is_day_trader": account.get('isDayTrader', False),
            "round_trips": account.get('roundTrips', 0),
            # Daily P/L
            "daily_pl": daily_pl,
            "daily_pl_pct": daily_pl_pct,
            "initial_account_value": initial_value,
            "current_account_value": current_value,
            # Counts
            "positions_count": len(positions),
            "source": "schwab"
        }

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        if not self._ensure_account():
            return []

        data = _make_trading_request("GET", f"/accounts/{self._account_hash}", params={"fields": "positions"})

        if not data:
            return []

        positions = data.get('securitiesAccount', {}).get('positions', [])

        result = []
        for pos in positions:
            instrument = pos.get('instrument', {})
            result.append({
                "symbol": instrument.get('symbol'),
                "asset_type": instrument.get('assetType'),
                "quantity": float(pos.get('longQuantity', 0) or 0) - float(pos.get('shortQuantity', 0) or 0),
                "avg_price": float(pos.get('averagePrice', 0) or 0),
                "current_price": float(pos.get('currentDayProfitLossPercentage', 0) or 0),  # Not directly available
                "market_value": float(pos.get('marketValue', 0) or 0),
                "cost_basis": float(pos.get('longQuantity', 0) or 0) * float(pos.get('averagePrice', 0) or 0),
                "unrealized_pl": float(pos.get('currentDayProfitLoss', 0) or 0),
                "unrealized_pl_pct": float(pos.get('currentDayProfitLossPercentage', 0) or 0),
                "source": "schwab"
            })

        return result

    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders for the account"""
        if not self._ensure_account():
            return []

        params = {
            "fromEnteredTime": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00.000Z"),
            "toEnteredTime": datetime.now().strftime("%Y-%m-%dT23:59:59.999Z")
        }

        if status:
            params["status"] = status

        data = _make_trading_request("GET", f"/accounts/{self._account_hash}/orders", params=params)

        if not data or not isinstance(data, list):
            return []

        result = []
        for order in data:
            legs = order.get('orderLegCollection', [])
            symbol = legs[0].get('instrument', {}).get('symbol') if legs else 'N/A'

            result.append({
                "order_id": str(order.get('orderId')),
                "symbol": symbol,
                "side": legs[0].get('instruction') if legs else 'N/A',
                "quantity": float(order.get('quantity', 0)),
                "filled_qty": float(order.get('filledQuantity', 0)),
                "price": float(order.get('price', 0) or order.get('stopPrice', 0) or 0),
                "order_type": order.get('orderType'),
                "status": order.get('status'),
                "duration": order.get('duration'),
                "entered_time": order.get('enteredTime'),
                "close_time": order.get('closeTime'),
                "source": "schwab"
            })

        return result

    def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        duration: str = "DAY",
        extended_hours: bool = False
    ) -> Dict:
        """
        Place an order

        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            side: BUY, SELL, BUY_TO_COVER, SELL_SHORT
            order_type: MARKET, LIMIT, STOP, STOP_LIMIT
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop price (required for STOP and STOP_LIMIT)
            duration: DAY, GOOD_TILL_CANCEL, etc.
            extended_hours: If True, allows pre-market and after-hours trading

        Returns:
            Order result dictionary
        """
        if not self._ensure_account():
            return {"error": "No account selected"}

        # Session type: NORMAL = regular hours only, SEAMLESS = extended hours
        session = "SEAMLESS" if extended_hours else "NORMAL"

        # Build order payload
        order = {
            "orderType": order_type.upper(),
            "session": session,
            "duration": duration.upper(),
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": side.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol.upper(),
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        # Add price fields based on order type
        if order_type.upper() in ["LIMIT", "STOP_LIMIT"]:
            if limit_price is None:
                return {"error": "Limit price required for LIMIT/STOP_LIMIT orders"}
            order["price"] = round(limit_price, 2)

        if order_type.upper() in ["STOP", "STOP_LIMIT"]:
            if stop_price is None:
                return {"error": "Stop price required for STOP/STOP_LIMIT orders"}
            order["stopPrice"] = round(stop_price, 2)

        logger.info(f"Placing order: {side} {quantity} {symbol} @ {order_type}")

        result = _make_trading_request("POST", f"/accounts/{self._account_hash}/orders", data=order)

        if result and "error" not in result:
            return {
                "success": True,
                "message": f"Order placed: {side} {quantity} {symbol}",
                "order": order,
                "source": "schwab"
            }

        return result or {"error": "Order placement failed"}

    def place_market_order(self, symbol: str, quantity: int, side: str, extended_hours: bool = False) -> Dict:
        """Place a market order"""
        return self.place_order(symbol, quantity, side, "MARKET", extended_hours=extended_hours)

    def place_limit_order(self, symbol: str, quantity: int, side: str, limit_price: float, extended_hours: bool = False) -> Dict:
        """Place a limit order"""
        return self.place_order(symbol, quantity, side, "LIMIT", limit_price=limit_price, extended_hours=extended_hours)

    def place_stop_order(self, symbol: str, quantity: int, side: str, stop_price: float, extended_hours: bool = False) -> Dict:
        """Place a stop order"""
        return self.place_order(symbol, quantity, side, "STOP", stop_price=stop_price, extended_hours=extended_hours)

    def place_stop_limit_order(self, symbol: str, quantity: int, side: str, stop_price: float, limit_price: float, extended_hours: bool = False) -> Dict:
        """Place a stop-limit order"""
        return self.place_order(symbol, quantity, side, "STOP_LIMIT", limit_price=limit_price, stop_price=stop_price, extended_hours=extended_hours)

    def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        limit_price: Optional[float] = None
    ) -> Dict:
        """
        Place a bracket order (entry + take profit + stop loss)

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            side: BUY or SELL
            take_profit_price: Take profit limit price
            stop_loss_price: Stop loss price
            limit_price: Entry limit price (None for market)
        """
        if not self._ensure_account():
            return {"error": "No account selected"}

        # Determine exit side
        exit_side = "SELL" if side.upper() == "BUY" else "BUY"

        # Build bracket order
        order = {
            "orderStrategyType": "TRIGGER",
            "session": "NORMAL",
            "duration": "DAY",
            "orderType": "LIMIT" if limit_price else "MARKET",
            "orderLegCollection": [
                {
                    "instruction": side.upper(),
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol.upper(),
                        "assetType": "EQUITY"
                    }
                }
            ],
            "childOrderStrategies": [
                {
                    "orderStrategyType": "OCO",
                    "childOrderStrategies": [
                        {
                            "orderStrategyType": "SINGLE",
                            "session": "NORMAL",
                            "duration": "GOOD_TILL_CANCEL",
                            "orderType": "LIMIT",
                            "price": round(take_profit_price, 2),
                            "orderLegCollection": [
                                {
                                    "instruction": exit_side,
                                    "quantity": quantity,
                                    "instrument": {
                                        "symbol": symbol.upper(),
                                        "assetType": "EQUITY"
                                    }
                                }
                            ]
                        },
                        {
                            "orderStrategyType": "SINGLE",
                            "session": "NORMAL",
                            "duration": "GOOD_TILL_CANCEL",
                            "orderType": "STOP",
                            "stopPrice": round(stop_loss_price, 2),
                            "orderLegCollection": [
                                {
                                    "instruction": exit_side,
                                    "quantity": quantity,
                                    "instrument": {
                                        "symbol": symbol.upper(),
                                        "assetType": "EQUITY"
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        if limit_price:
            order["price"] = round(limit_price, 2)

        logger.info(f"Placing bracket order: {side} {quantity} {symbol} TP={take_profit_price} SL={stop_loss_price}")

        result = _make_trading_request("POST", f"/accounts/{self._account_hash}/orders", data=order)

        if result and "error" not in result:
            return {
                "success": True,
                "message": f"Bracket order placed: {side} {quantity} {symbol}",
                "take_profit": take_profit_price,
                "stop_loss": stop_loss_price,
                "source": "schwab"
            }

        return result or {"error": "Bracket order placement failed"}

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        if not self._ensure_account():
            return {"error": "No account selected"}

        result = _make_trading_request("DELETE", f"/accounts/{self._account_hash}/orders/{order_id}")

        if result and "error" not in result:
            return {
                "success": True,
                "message": f"Order {order_id} canceled",
                "source": "schwab"
            }

        return result or {"error": f"Failed to cancel order {order_id}"}

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        orders = self.get_orders(status="WORKING")

        canceled = 0
        errors = []

        for order in orders:
            result = self.cancel_order(order['order_id'])
            if result.get('success'):
                canceled += 1
            else:
                errors.append(f"{order['order_id']}: {result.get('error', 'Unknown error')}")

        return {
            "success": len(errors) == 0,
            "canceled": canceled,
            "errors": errors,
            "source": "schwab"
        }


# Global instance
_schwab_trading: Optional[SchwabTrading] = None


def get_schwab_trading() -> Optional[SchwabTrading]:
    """Get or create the global Schwab trading instance"""
    global _schwab_trading

    # Check if token is available
    if not _ensure_token():
        return None

    if _schwab_trading is None:
        _schwab_trading = SchwabTrading()

    return _schwab_trading


def is_schwab_trading_available() -> bool:
    """Check if Schwab trading is available"""
    trading = get_schwab_trading()
    return trading is not None and len(trading.get_accounts()) > 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Schwab Trading...")
    print("=" * 50)

    trading = get_schwab_trading()

    if trading:
        # Test accounts
        print("\n--- Accounts ---")
        accounts = trading.get_accounts()
        for acc in accounts:
            selected = " [SELECTED]" if acc['selected'] else ""
            print(f"  {acc['account_number']}{selected}")

        # Test account info
        print("\n--- Account Info ---")
        info = trading.get_account_info()
        if info:
            print(f"  Account: {info.get('account_number')}")
            print(f"  Type: {info.get('type')}")
            print(f"  Buying Power: ${info.get('buying_power', 0):,.2f}")
            print(f"  Cash: ${info.get('cash', 0):,.2f}")
            print(f"  Equity: ${info.get('equity', 0):,.2f}")
            print(f"  Day Trader: {info.get('is_day_trader')}")
        else:
            print("  Failed to get account info")

        # Test positions
        print("\n--- Positions ---")
        positions = trading.get_positions()
        if positions:
            for pos in positions:
                print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_price']:.2f}")
        else:
            print("  No positions")

        # Test orders
        print("\n--- Recent Orders ---")
        orders = trading.get_orders()
        if orders:
            for order in orders[:5]:
                print(f"  {order['order_id']}: {order['side']} {order['quantity']} {order['symbol']} - {order['status']}")
        else:
            print("  No recent orders")

    else:
        print("Schwab trading not available")

    print("\n" + "=" * 50)
    print("Test complete!")
