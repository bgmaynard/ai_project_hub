"""
Alpaca MCP Server Integration
=============================
Integrates Alpaca's official MCP server with our Claude AI Intelligence module
for faster, more direct trading operations and market data access.

This module wraps the alpaca-mcp-server functions and exposes them as tools
that Claude Sonnet 4.5 can call in parallel for comprehensive trading analysis.

Author: AI Trading Bot Team
Version: 1.0
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Check if alpaca-mcp-server is available
MCP_AVAILABLE = False
try:
    from alpaca_mcp_server import server as mcp_server
    MCP_AVAILABLE = True
    logger.info("[OK] Alpaca MCP Server module loaded")
except ImportError as e:
    logger.warning(f"[WARN] Alpaca MCP Server not available: {e}")


class AlpacaMCPClient:
    """
    Client wrapper for Alpaca MCP Server functions.
    Provides direct access to Alpaca's trading and market data APIs
    through the MCP protocol for faster execution.
    """

    def __init__(self):
        """Initialize the Alpaca MCP Client"""
        self.available = MCP_AVAILABLE
        self.initialized = False

        if not self.available:
            logger.warning("Alpaca MCP Server not installed. Install with: pip install alpaca-mcp-server")
            return

        # Verify API keys are set
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")

        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not found in environment")
            return

        # Initialize MCP server clients
        try:
            mcp_server._ensure_clients()
            self.initialized = True
            logger.info("[OK] Alpaca MCP Client initialized successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize MCP clients: {e}")

    def get_status(self) -> Dict:
        """Get MCP client status"""
        return {
            "mcp_available": self.available,
            "initialized": self.initialized,
            "api_key_set": bool(self.api_key),
            "functions_available": self._get_available_functions() if self.available else []
        }

    def _get_available_functions(self) -> List[str]:
        """Get list of available MCP functions"""
        if not self.available:
            return []

        return [
            # Account & Portfolio
            "get_account_info",
            "get_all_positions",
            "get_open_position",
            "get_portfolio_history",

            # Orders
            "place_stock_order",
            "place_crypto_order",
            "place_option_market_order",
            "get_orders",
            "cancel_order_by_id",
            "cancel_all_orders",

            # Positions
            "close_position",
            "close_all_positions",

            # Market Data - Stocks
            "get_stock_latest_quote",
            "get_stock_latest_bar",
            "get_stock_latest_trade",
            "get_stock_bars",
            "get_stock_snapshot",
            "get_stock_quotes",
            "get_stock_trades",

            # Market Data - Crypto
            "get_crypto_latest_quote",
            "get_crypto_latest_bar",
            "get_crypto_bars",
            "get_crypto_snapshot",

            # Market Data - Options
            "get_option_contracts",
            "get_option_latest_quote",
            "get_option_snapshot",

            # Assets & Calendar
            "get_all_assets",
            "get_asset",
            "get_calendar",
            "get_clock",

            # Watchlists
            "get_watchlists",
            "get_watchlist_by_id",
            "create_watchlist",
            "update_watchlist_by_id",
            "delete_watchlist_by_id",

            # Corporate Actions
            "get_corporate_actions"
        ]

    # ========================================================================
    # ACCOUNT & PORTFOLIO FUNCTIONS
    # ========================================================================

    def get_account(self) -> Dict:
        """Get account information"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.get_account_info()
            # Handle async if needed
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.initialized:
            return [{"error": "MCP client not initialized"}]

        try:
            result = mcp_server.get_all_positions()
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return [{"error": str(e)}]

    def get_position(self, symbol: str) -> Dict:
        """Get position for a specific symbol"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.get_open_position(symbol_or_asset_id=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> Dict:
        """Get portfolio history"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.get_portfolio_history(period=period, timeframe=timeframe)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return {"error": str(e)}

    # ========================================================================
    # ORDER FUNCTIONS
    # ========================================================================

    def place_order(self, symbol: str, qty: float, side: str,
                    order_type: str = "market", time_in_force: str = "day",
                    limit_price: float = None, stop_price: float = None) -> Dict:
        """Place a stock order"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.place_stock_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict]:
        """Get orders with optional status filter"""
        if not self.initialized:
            return [{"error": "MCP client not initialized"}]

        try:
            result = mcp_server.get_orders(status=status, limit=limit)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return [{"error": str(e)}]

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order by ID"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.cancel_order_by_id(order_id=order_id)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return {"error": str(e)}

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.cancel_all_orders()
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return {"error": str(e)}

    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================

    def close_position(self, symbol: str, qty: float = None, percentage: float = None) -> Dict:
        """Close a position (fully or partially)"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.close_position(
                symbol_or_asset_id=symbol,
                qty=qty,
                percentage=percentage
            )
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e)}

    def close_all_positions(self, cancel_orders: bool = True) -> Dict:
        """Close all positions"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.close_all_positions(cancel_orders=cancel_orders)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {"error": str(e)}

    # ========================================================================
    # MARKET DATA - STOCKS
    # ========================================================================

    def get_quote(self, symbol: str) -> Dict:
        """Get latest quote for a symbol"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_stock_latest_quote(symbol=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_latest_bar(self, symbol: str) -> Dict:
        """Get latest bar for a symbol"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_stock_latest_bar(symbol=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting latest bar for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 start: str = None, end: str = None, limit: int = 100) -> Dict:
        """Get historical bars for a symbol"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            # Default to last month if no start specified
            if not start:
                start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            if not end:
                end = datetime.now().strftime("%Y-%m-%d")

            result = mcp_server.get_stock_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit
            )
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_snapshot(self, symbol: str) -> Dict:
        """Get full snapshot for a symbol (quote, bar, trade)"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_stock_snapshot(symbol=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting snapshot for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_multiple_snapshots(self, symbols: List[str]) -> Dict:
        """Get snapshots for multiple symbols in parallel"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(symbols), 10)) as executor:
            futures = {executor.submit(self.get_snapshot, s): s for s in symbols}
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    results[symbol] = {"error": str(e)}

        return results

    # ========================================================================
    # MARKET DATA - CRYPTO
    # ========================================================================

    def get_crypto_quote(self, symbol: str) -> Dict:
        """Get latest crypto quote"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_crypto_latest_quote(symbol=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting crypto quote for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    def get_crypto_bars(self, symbol: str, timeframe: str = "1Day",
                        start: str = None, end: str = None, limit: int = 100) -> Dict:
        """Get crypto historical bars"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            if not start:
                start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            if not end:
                end = datetime.now().strftime("%Y-%m-%d")

            result = mcp_server.get_crypto_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit
            )
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting crypto bars for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    # ========================================================================
    # OPTIONS
    # ========================================================================

    def get_option_contracts(self, underlying_symbol: str,
                              expiration_date: str = None,
                              strike_price: float = None,
                              contract_type: str = None) -> Dict:
        """Get option contracts for an underlying symbol"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.get_option_contracts(
                underlying_symbol=underlying_symbol,
                expiration_date=expiration_date,
                strike_price_gte=strike_price * 0.9 if strike_price else None,
                strike_price_lte=strike_price * 1.1 if strike_price else None,
                type=contract_type
            )
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting option contracts: {e}")
            return {"error": str(e)}

    def get_option_quote(self, symbol: str) -> Dict:
        """Get option quote (symbol is the option contract symbol)"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_option_latest_quote(symbol=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting option quote for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    # ========================================================================
    # MARKET INFO
    # ========================================================================

    def get_clock(self) -> Dict:
        """Get market clock (open/close times, is_open)"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.get_clock()
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting clock: {e}")
            return {"error": str(e)}

    def get_calendar(self, start: str = None, end: str = None) -> List[Dict]:
        """Get market calendar"""
        if not self.initialized:
            return [{"error": "MCP client not initialized"}]

        try:
            if not start:
                start = datetime.now().strftime("%Y-%m-%d")
            if not end:
                end = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

            result = mcp_server.get_calendar(start=start, end=end)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting calendar: {e}")
            return [{"error": str(e)}]

    def get_asset(self, symbol: str) -> Dict:
        """Get asset information"""
        if not self.initialized:
            return {"error": "MCP client not initialized", "symbol": symbol}

        try:
            result = mcp_server.get_asset(symbol_or_asset_id=symbol)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}

    # ========================================================================
    # WATCHLISTS
    # ========================================================================

    def get_watchlists(self) -> List[Dict]:
        """Get all watchlists"""
        if not self.initialized:
            return [{"error": "MCP client not initialized"}]

        try:
            result = mcp_server.get_watchlists()
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error getting watchlists: {e}")
            return [{"error": str(e)}]

    def create_watchlist(self, name: str, symbols: List[str]) -> Dict:
        """Create a new watchlist"""
        if not self.initialized:
            return {"error": "MCP client not initialized"}

        try:
            result = mcp_server.create_watchlist(name=name, symbols=symbols)
            if hasattr(result, '__await__'):
                import asyncio
                result = asyncio.get_event_loop().run_until_complete(result)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Error creating watchlist: {e}")
            return {"error": str(e)}

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _parse_result(self, result: Any) -> Any:
        """Parse MCP result to dictionary"""
        if result is None:
            return {"data": None}

        if isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return {"data": result}

        if isinstance(result, (dict, list)):
            return result

        # Handle Alpaca model objects
        if hasattr(result, '__dict__'):
            return self._model_to_dict(result)

        return {"data": str(result)}

    def _model_to_dict(self, obj: Any) -> Dict:
        """Convert Alpaca model object to dictionary"""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return {"data": str(obj)}


# ============================================================================
# TOOL DEFINITIONS FOR CLAUDE
# ============================================================================

def get_mcp_tools() -> List[Dict]:
    """
    Get tool definitions for Claude to use with Alpaca MCP.
    These tools provide direct access to Alpaca's trading infrastructure.
    """
    return [
        {
            "name": "alpaca_get_account",
            "description": "Get Alpaca account info including equity, buying power, cash, and account status",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "alpaca_get_positions",
            "description": "Get all open positions with current P&L, market value, and quantity",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "alpaca_get_quote",
            "description": "Get real-time quote for a stock symbol (bid, ask, last price)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., AAPL, TSLA)"
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "alpaca_get_snapshot",
            "description": "Get complete market snapshot for a symbol (quote, latest bar, latest trade)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "alpaca_get_bars",
            "description": "Get historical OHLCV bars for a symbol",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)",
                        "default": "1Day"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of bars to return",
                        "default": 100
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "alpaca_place_order",
            "description": "Place a stock order (market, limit, stop, etc.)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "qty": {
                        "type": "number",
                        "description": "Number of shares"
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Order side"
                    },
                    "order_type": {
                        "type": "string",
                        "enum": ["market", "limit", "stop", "stop_limit", "trailing_stop"],
                        "default": "market"
                    },
                    "limit_price": {
                        "type": "number",
                        "description": "Limit price (required for limit orders)"
                    },
                    "stop_price": {
                        "type": "number",
                        "description": "Stop price (required for stop orders)"
                    }
                },
                "required": ["symbol", "qty", "side"]
            }
        },
        {
            "name": "alpaca_get_orders",
            "description": "Get orders with optional status filter",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "default": "open"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50
                    }
                },
                "required": []
            }
        },
        {
            "name": "alpaca_cancel_order",
            "description": "Cancel a specific order by ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to cancel"
                    }
                },
                "required": ["order_id"]
            }
        },
        {
            "name": "alpaca_close_position",
            "description": "Close a position (fully or partially)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol to close"
                    },
                    "qty": {
                        "type": "number",
                        "description": "Quantity to close (optional, closes all if not specified)"
                    },
                    "percentage": {
                        "type": "number",
                        "description": "Percentage of position to close (0-100)"
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "alpaca_get_clock",
            "description": "Get market clock (is market open, next open/close times)",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "alpaca_get_portfolio_history",
            "description": "Get portfolio equity history over time",
            "input_schema": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Period (1D, 1W, 1M, 3M, 1A, all)",
                        "default": "1M"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe (1Min, 5Min, 15Min, 1H, 1D)",
                        "default": "1D"
                    }
                },
                "required": []
            }
        },
        {
            "name": "alpaca_get_crypto_quote",
            "description": "Get cryptocurrency quote",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto symbol (e.g., BTC/USD, ETH/USD)"
                    }
                },
                "required": ["symbol"]
            }
        }
    ]


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_mcp_client: Optional[AlpacaMCPClient] = None


def get_mcp_client() -> AlpacaMCPClient:
    """Get or create the MCP client singleton"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = AlpacaMCPClient()
    return _mcp_client


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = AlpacaMCPClient()
    print(f"\nMCP Client Status: {json.dumps(client.get_status(), indent=2)}")

    if client.initialized:
        print("\nTesting get_clock()...")
        clock = client.get_clock()
        print(f"Clock: {json.dumps(clock, indent=2, default=str)}")

        print("\nTesting get_account()...")
        account = client.get_account()
        print(f"Account: {json.dumps(account, indent=2, default=str)}")
