"""
TradingView Integration for Alpaca Trading Hub
Webhook receiver for TradingView alerts, scanners, and signals
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca_integration import get_alpaca_connector
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from watchlist_manager import get_watchlist_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tradingview", tags=["TradingView"])

# ============================================================================
# REAL-TIME PRICE CACHE (from TradingView webhooks)
# ============================================================================
# Stores real-time prices received from TradingView alerts
# Format: {"SYMBOL": {"price": 123.45, "bid": 123.40, "ask": 123.50, "volume": 1000, "timestamp": datetime}}
_tv_price_cache: Dict[str, Dict] = {}
_TV_PRICE_CACHE_TTL = 60  # Consider TV prices valid for 60 seconds


def get_tv_price(symbol: str) -> Optional[Dict]:
    """Get cached TradingView price if fresh"""
    symbol = symbol.upper()
    if symbol in _tv_price_cache:
        cached = _tv_price_cache[symbol]
        age = (datetime.now() - cached.get("timestamp", datetime.min)).total_seconds()
        if age < _TV_PRICE_CACHE_TTL:
            return cached
    return None


def set_tv_price(
    symbol: str,
    price: float,
    bid: float = None,
    ask: float = None,
    volume: int = None,
    high: float = None,
    low: float = None,
    open_price: float = None,
):
    """Store TradingView price in cache"""
    symbol = symbol.upper()
    _tv_price_cache[symbol] = {
        "symbol": symbol,
        "price": price,
        "bid": bid or price * 0.999,
        "ask": ask or price * 1.001,
        "volume": volume or 0,
        "high": high or price,
        "low": low or price,
        "open": open_price or price,
        "timestamp": datetime.now(),
        "source": "tradingview",
    }
    logger.info(f"TV Price Update: {symbol} = ${price:.2f}")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class TradingViewAlert(BaseModel):
    """TradingView alert webhook payload"""

    ticker: str
    action: Optional[str] = None  # "buy", "sell", "add_watchlist", "remove_watchlist"
    price: Optional[float] = None
    strategy: Optional[str] = None
    signal: Optional[str] = None
    time: Optional[str] = None
    exchange: Optional[str] = "NASDAQ"
    interval: Optional[str] = None
    volume: Optional[float] = None

    # Scanner-specific fields
    scanner_name: Optional[str] = None
    score: Optional[float] = None
    indicators: Optional[Dict] = None


class TradingViewScannerBatch(BaseModel):
    """Batch of scanner results from TradingView"""

    scanner_name: str
    symbols: List[str]
    timestamp: Optional[str] = None
    replace_watchlist: bool = False  # If true, replace entire watchlist


class TradingViewOrder(BaseModel):
    """TradingView order webhook"""

    ticker: str
    action: str  # "buy" or "sell"
    quantity: Optional[int] = 1
    order_type: str = "market"  # "market" or "limit"
    limit_price: Optional[float] = None
    strategy: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TradingViewPrice(BaseModel):
    """TradingView price update webhook"""

    ticker: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    time: Optional[str] = None


# ============================================================================
# WEBHOOK ENDPOINTS
# ============================================================================


@router.post("/webhook/alert")
async def tradingview_alert_webhook(alert: TradingViewAlert):
    """
    Receive TradingView alerts

    Example TradingView alert message:
    {
      "ticker": "{{ticker}}",
      "action": "add_watchlist",
      "price": "{{close}}",
      "strategy": "Momentum Scanner",
      "time": "{{time}}"
    }
    """
    try:
        logger.info(f"TradingView Alert: {alert.ticker} - {alert.action}")

        symbol = alert.ticker.upper()
        action = alert.action.lower() if alert.action else None

        # Handle different alert actions
        if action == "add_watchlist" or action == "add":
            return await add_symbol_to_watchlist(symbol, alert)

        elif action == "remove_watchlist" or action == "remove":
            return await remove_symbol_from_watchlist(symbol)

        elif action in ["buy", "sell"]:
            return {
                "success": True,
                "message": f"Signal received for {symbol}: {action}",
                "note": "Use /webhook/order for automated trading",
            }

        else:
            # Generic alert - log it
            return {
                "success": True,
                "message": f"Alert received for {symbol}",
                "ticker": symbol,
                "price": alert.price,
                "strategy": alert.strategy,
            }

    except Exception as e:
        logger.error(f"TradingView alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/scanner")
async def tradingview_scanner_webhook(batch: TradingViewScannerBatch):
    """
    Receive batch scanner results from TradingView

    Example payload:
    {
      "scanner_name": "Momentum Stocks",
      "symbols": ["NVDA", "TSLA", "AMD", "PLTR"],
      "timestamp": "2025-11-21 14:30:00",
      "replace_watchlist": false
    }
    """
    try:
        logger.info(
            f"TradingView Scanner: {batch.scanner_name} - {len(batch.symbols)} symbols"
        )

        watchlist_mgr = get_watchlist_manager()

        # Get or create scanner-specific watchlist
        scanner_watchlist_name = f"TV: {batch.scanner_name}"
        scanner_watchlist = watchlist_mgr.get_watchlist_by_name(scanner_watchlist_name)

        if not scanner_watchlist:
            # Create new watchlist for this scanner
            scanner_watchlist = watchlist_mgr.create_watchlist(
                scanner_watchlist_name, batch.symbols
            )
            action = "created"
        elif batch.replace_watchlist:
            # Replace entire watchlist
            watchlist_mgr.update_watchlist(
                scanner_watchlist["watchlist_id"], symbols=batch.symbols
            )
            action = "replaced"
        else:
            # Add to existing watchlist (merge)
            watchlist_mgr.add_symbols(scanner_watchlist["watchlist_id"], batch.symbols)
            action = "updated"

        return {
            "success": True,
            "message": f"Scanner watchlist {action}",
            "watchlist_name": scanner_watchlist_name,
            "symbols_count": len(batch.symbols),
            "action": action,
        }

    except Exception as e:
        logger.error(f"TradingView scanner error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/price")
async def tradingview_price_webhook(price_data: TradingViewPrice):
    """
    Receive real-time price updates from TradingView

    Set up a TradingView alert on any symbol with message:
    {
      "ticker": "{{ticker}}",
      "price": {{close}},
      "bid": {{bid}},
      "ask": {{ask}},
      "volume": {{volume}},
      "high": {{high}},
      "low": {{low}},
      "open": {{open}},
      "time": "{{time}}"
    }

    Set alert to trigger "Every bar close" on 1-minute chart for near real-time updates.
    """
    try:
        symbol = price_data.ticker.upper()

        # Store in cache
        set_tv_price(
            symbol=symbol,
            price=price_data.price,
            bid=price_data.bid,
            ask=price_data.ask,
            volume=price_data.volume,
            high=price_data.high,
            low=price_data.low,
            open_price=price_data.open,
        )

        return {
            "success": True,
            "message": f"Price updated for {symbol}",
            "symbol": symbol,
            "price": price_data.price,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"TradingView price webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}")
async def get_tradingview_price(symbol: str):
    """Get cached TradingView price for a symbol"""
    cached = get_tv_price(symbol)
    if cached:
        return {"success": True, "source": "tradingview", "data": cached}
    return {
        "success": False,
        "message": f"No TradingView price cached for {symbol}",
        "hint": "Set up a TradingView alert to push prices to /api/tradingview/webhook/price",
    }


@router.get("/prices")
async def get_all_tradingview_prices():
    """Get all cached TradingView prices"""
    fresh_prices = {}
    for symbol, data in _tv_price_cache.items():
        age = (datetime.now() - data.get("timestamp", datetime.min)).total_seconds()
        if age < _TV_PRICE_CACHE_TTL:
            fresh_prices[symbol] = data

    return {"success": True, "count": len(fresh_prices), "prices": fresh_prices}


@router.post("/webhook/order")
async def tradingview_order_webhook(order: TradingViewOrder):
    """
    Execute orders from TradingView alerts

    Example TradingView alert message:
    {
      "ticker": "{{ticker}}",
      "action": "{{strategy.order.action}}",
      "quantity": 100,
      "order_type": "market",
      "strategy": "My Strategy"
    }

    IMPORTANT: Only enable this in production after thorough testing!
    """
    try:
        logger.info(
            f"TradingView Order: {order.action} {order.quantity} {order.ticker}"
        )

        # Get Alpaca connector
        connector = get_alpaca_connector()

        if not connector.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to Alpaca")

        # Validate action
        if order.action.lower() not in ["buy", "sell"]:
            raise HTTPException(
                status_code=400, detail="Invalid action. Must be 'buy' or 'sell'"
            )

        # Prepare order
        symbol = order.ticker.upper()
        side = OrderSide.BUY if order.action.lower() == "buy" else OrderSide.SELL

        # Create order based on type
        if order.order_type.lower() == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type.lower() == "limit":
            if not order.limit_price:
                raise HTTPException(
                    status_code=400, detail="Limit price required for limit orders"
                )

            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid order type")

        # Submit order to Alpaca
        result = connector.client.submit_order(order_request)

        logger.info(f"Order submitted: {result.id} - {side} {order.quantity} {symbol}")

        return {
            "success": True,
            "message": f"Order submitted: {side} {order.quantity} {symbol}",
            "order_id": result.id,
            "status": result.status,
            "symbol": symbol,
            "quantity": order.quantity,
            "side": str(side),
            "strategy": order.strategy,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TradingView order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def add_symbol_to_watchlist(symbol: str, alert: TradingViewAlert):
    """Add symbol to default watchlist"""
    try:
        watchlist_mgr = get_watchlist_manager()
        default_watchlist = watchlist_mgr.get_default_watchlist()

        # Add symbol
        watchlist_mgr.add_symbols(default_watchlist["watchlist_id"], [symbol])

        logger.info(f"Added {symbol} to watchlist from TradingView")

        return {
            "success": True,
            "message": f"Added {symbol} to watchlist",
            "symbol": symbol,
            "price": alert.price,
            "strategy": alert.strategy,
            "watchlist": "Default",
        }

    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def remove_symbol_from_watchlist(symbol: str):
    """Remove symbol from default watchlist"""
    try:
        watchlist_mgr = get_watchlist_manager()
        default_watchlist = watchlist_mgr.get_default_watchlist()

        # Remove symbol
        watchlist_mgr.remove_symbols(default_watchlist["watchlist_id"], [symbol])

        logger.info(f"Removed {symbol} from watchlist")

        return {
            "success": True,
            "message": f"Removed {symbol} from watchlist",
            "symbol": symbol,
        }

    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================


@router.get("/webhook-url")
async def get_webhook_url():
    """Get webhook URL for TradingView configuration"""
    return {
        "webhook_urls": {
            "alerts": "http://localhost:9100/api/tradingview/webhook/alert",
            "scanner": "http://localhost:9100/api/tradingview/webhook/scanner",
            "orders": "http://localhost:9100/api/tradingview/webhook/order",
        },
        "example_alert_message": {
            "ticker": "{{ticker}}",
            "action": "add_watchlist",
            "price": "{{close}}",
            "strategy": "Momentum Scanner",
            "time": "{{time}}",
        },
        "example_order_message": {
            "ticker": "{{ticker}}",
            "action": "buy",
            "quantity": 100,
            "order_type": "market",
            "strategy": "{{strategy.order.comment}}",
        },
    }


@router.get("/status")
async def tradingview_integration_status():
    """Check TradingView integration status"""
    try:
        connector = get_alpaca_connector()
        watchlist_mgr = get_watchlist_manager()

        # Get all TradingView watchlists
        all_watchlists = watchlist_mgr.get_all_watchlists()
        tv_watchlists = [w for w in all_watchlists if w["name"].startswith("TV:")]

        return {
            "status": "operational",
            "alpaca_connected": connector.is_connected(),
            "tradingview_watchlists": len(tv_watchlists),
            "watchlists": [w["name"] for w in tv_watchlists],
            "webhook_endpoints": {
                "alerts": "/api/tradingview/webhook/alert",
                "scanner": "/api/tradingview/webhook/scanner",
                "orders": "/api/tradingview/webhook/order",
            },
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# TRADINGVIEW DESKTOP WATCHLIST SYNC
# ============================================================================

# Exchange mapping for proper TradingView symbol format
EXCHANGE_MAPPING = {
    # ETFs - typically AMEX
    "SPY": "AMEX",
    "QQQ": "NASDAQ",
    "DIA": "AMEX",
    "IWM": "AMEX",
    "XLF": "AMEX",
    "XLE": "AMEX",
    "XLK": "AMEX",
    "XLV": "AMEX",
    "GLD": "AMEX",
    "SLV": "AMEX",
    "TLT": "NASDAQ",
    "VXX": "CBOE",
    "UVXY": "AMEX",
    "SQQQ": "NASDAQ",
    "TQQQ": "NASDAQ",
    # Major NYSE stocks
    "JPM": "NYSE",
    "BAC": "NYSE",
    "WFC": "NYSE",
    "C": "NYSE",
    "GS": "NYSE",
    "MS": "NYSE",
    "V": "NYSE",
    "MA": "NYSE",
    "DIS": "NYSE",
    "KO": "NYSE",
    "PEP": "NASDAQ",
    "WMT": "NYSE",
    "HD": "NYSE",
    "MCD": "NYSE",
    "NKE": "NYSE",
    "BA": "NYSE",
    "CAT": "NYSE",
    "GE": "NYSE",
    "MMM": "NYSE",
    "IBM": "NYSE",
    "JNJ": "NYSE",
    "PFE": "NYSE",
    "MRK": "NYSE",
    "UNH": "NYSE",
    "CVX": "NYSE",
    "XOM": "NYSE",
    "BRK.A": "NYSE",
    "BRK.B": "NYSE",
    # NASDAQ tech stocks (default for most tech)
    "AAPL": "NASDAQ",
    "MSFT": "NASDAQ",
    "GOOGL": "NASDAQ",
    "GOOG": "NASDAQ",
    "AMZN": "NASDAQ",
    "META": "NASDAQ",
    "TSLA": "NASDAQ",
    "NVDA": "NASDAQ",
    "AMD": "NASDAQ",
    "INTC": "NASDAQ",
    "NFLX": "NASDAQ",
    "ADBE": "NASDAQ",
    "CRM": "NYSE",
    "ORCL": "NYSE",
    "CSCO": "NASDAQ",
    "AVGO": "NASDAQ",
    "QCOM": "NASDAQ",
    "TXN": "NASDAQ",
    "MU": "NASDAQ",
    "AMAT": "NASDAQ",
    "PLTR": "NYSE",
    "COIN": "NASDAQ",
    "MARA": "NASDAQ",
    "RIOT": "NASDAQ",
    "SOFI": "NASDAQ",
    "HOOD": "NASDAQ",
    "DKNG": "NASDAQ",
    "ROKU": "NASDAQ",
}


def get_tv_exchange(symbol: str) -> str:
    """Get TradingView exchange for a symbol"""
    symbol = symbol.upper().strip()
    return EXCHANGE_MAPPING.get(symbol, "NASDAQ")  # Default to NASDAQ


def format_tv_symbol(symbol: str) -> str:
    """Format symbol for TradingView (EXCHANGE:SYMBOL)"""
    symbol = symbol.upper().strip()
    exchange = get_tv_exchange(symbol)
    return f"{exchange}:{symbol}"


@router.get("/desktop/watchlist-export")
async def export_watchlist_for_tv_desktop(
    watchlist_id: Optional[str] = None, format: str = "txt"
):
    """
    Export watchlist in TradingView Desktop compatible format.

    TradingView Desktop can import watchlists from:
    1. Text file with one symbol per line (EXCHANGE:SYMBOL format)
    2. Comma-separated list

    Parameters:
    - watchlist_id: Optional specific watchlist (default: current/default)
    - format: "txt" (one per line), "csv" (comma-separated), or "json"

    Usage:
    1. Call this endpoint to get formatted symbols
    2. Copy to clipboard or save to file
    3. In TradingView Desktop: Right-click watchlist → Import List
    """
    try:
        watchlist_mgr = get_watchlist_manager()

        # Get watchlist
        if watchlist_id:
            watchlist = watchlist_mgr.get_watchlist(watchlist_id)
        else:
            watchlist = watchlist_mgr.get_default_watchlist()

        if not watchlist:
            return {"success": False, "error": "Watchlist not found"}

        symbols = watchlist.get("symbols", [])

        # Format symbols for TradingView
        tv_symbols = [format_tv_symbol(s) for s in symbols]

        if format == "txt":
            # One symbol per line - best for TV Desktop import
            content = "\n".join(tv_symbols)
            return {
                "success": True,
                "watchlist_name": watchlist["name"],
                "symbol_count": len(symbols),
                "format": "txt",
                "content": content,
                "instructions": "Copy content and paste into TradingView Desktop watchlist import",
            }

        elif format == "csv":
            # Comma-separated - good for quick copy
            content = ",".join(tv_symbols)
            return {
                "success": True,
                "watchlist_name": watchlist["name"],
                "symbol_count": len(symbols),
                "format": "csv",
                "content": content,
            }

        elif format == "json":
            # JSON format with full details
            return {
                "success": True,
                "watchlist_name": watchlist["name"],
                "watchlist_id": watchlist.get("watchlist_id"),
                "symbol_count": len(symbols),
                "format": "json",
                "symbols": symbols,
                "tv_formatted": tv_symbols,
                "tv_import_text": "\n".join(tv_symbols),
            }

        else:
            return {"success": False, "error": f"Unknown format: {format}"}

    except Exception as e:
        logger.error(f"Watchlist export error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/desktop/all-watchlists-export")
async def export_all_watchlists():
    """
    Export all watchlists for TradingView Desktop.

    Returns all watchlists with TV-formatted symbols.
    """
    try:
        watchlist_mgr = get_watchlist_manager()
        all_watchlists = watchlist_mgr.get_all_watchlists()

        result = []
        for wl in all_watchlists:
            symbols = wl.get("symbols", [])
            tv_symbols = [format_tv_symbol(s) for s in symbols]

            result.append(
                {
                    "name": wl["name"],
                    "id": wl.get("watchlist_id"),
                    "symbol_count": len(symbols),
                    "symbols": symbols,
                    "tv_formatted": tv_symbols,
                    "tv_import_text": "\n".join(tv_symbols),
                }
            )

        return {"success": True, "watchlist_count": len(result), "watchlists": result}

    except Exception as e:
        logger.error(f"All watchlists export error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/desktop/sync-url/{watchlist_id}")
async def get_tv_desktop_sync_url(watchlist_id: str, interval: str = "D"):
    """
    Generate TradingView Desktop URLs to open all symbols in a watchlist.

    Returns URLs that can be clicked to open each symbol in TV Desktop.

    Parameters:
    - watchlist_id: The watchlist to sync
    - interval: Default chart interval (1, 5, 15, 60, D, W)
    """
    try:
        watchlist_mgr = get_watchlist_manager()
        watchlist = watchlist_mgr.get_watchlist(watchlist_id)

        if not watchlist:
            return {"success": False, "error": "Watchlist not found"}

        symbols = watchlist.get("symbols", [])

        # Generate TV Desktop URLs
        urls = []
        for symbol in symbols:
            tv_symbol = format_tv_symbol(symbol)
            url = f"tradingview://chart?symbol={tv_symbol}&interval={interval}"
            urls.append({"symbol": symbol, "tv_symbol": tv_symbol, "url": url})

        return {
            "success": True,
            "watchlist_name": watchlist["name"],
            "interval": interval,
            "symbol_count": len(symbols),
            "urls": urls,
            "open_all_script": f"// Open all in TV Desktop (use browser console)\n"
            + "\n".join([f"window.open('{u['url']}');" for u in urls[:8]]),
        }

    except Exception as e:
        logger.error(f"Sync URL generation error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/desktop/clipboard-export")
async def generate_clipboard_export(data: Dict):
    """
    Generate watchlist export for clipboard copy.

    POST body:
    {
        "symbols": ["AAPL", "TSLA", "NVDA"],
        "format": "tv"  // "tv" for TradingView format, "plain" for just symbols
    }
    """
    try:
        symbols = data.get("symbols", [])
        format_type = data.get("format", "tv")

        if not symbols:
            return {"success": False, "error": "No symbols provided"}

        if format_type == "tv":
            # TradingView format with exchange prefix
            formatted = [format_tv_symbol(s) for s in symbols]
        else:
            # Plain symbols
            formatted = [s.upper() for s in symbols]

        return {
            "success": True,
            "symbol_count": len(formatted),
            "format": format_type,
            "text": "\n".join(formatted),
            "csv": ",".join(formatted),
        }

    except Exception as e:
        logger.error(f"Clipboard export error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/desktop/open-symbol/{symbol}")
async def get_tv_desktop_open_url(symbol: str, interval: str = "D"):
    """
    Get TradingView Desktop URL for a single symbol.

    Returns the tradingview:// protocol URL to open in desktop app.
    """
    tv_symbol = format_tv_symbol(symbol)
    url = f"tradingview://chart?symbol={tv_symbol}&interval={interval}"

    return {
        "success": True,
        "symbol": symbol.upper(),
        "tv_symbol": tv_symbol,
        "interval": interval,
        "url": url,
        "web_fallback": f"https://www.tradingview.com/chart/?symbol={tv_symbol}&interval={interval}",
    }
