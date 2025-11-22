"""
Compatibility Routes for Legacy UI
Maps old IBKR endpoints to new Alpaca endpoints
"""
from fastapi import APIRouter, HTTPException
from alpaca_market_data import get_alpaca_market_data
from alpaca_integration import get_alpaca_connector

router = APIRouter(tags=["Compatibility"])

# ============================================================================
# PRICE & MARKET DATA COMPATIBILITY
# ============================================================================

@router.get("/api/price/{symbol}")
async def get_price_compat(symbol: str):
    """Legacy price endpoint - maps to Alpaca quote"""
    try:
        market_data = get_alpaca_market_data()
        quote = market_data.get_latest_quote(symbol.upper())

        if not quote:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Return in format expected by UI
        return {
            "symbol": symbol,
            "price": quote["last"],
            "bid": quote["bid"],
            "ask": quote["ask"],
            "volume": quote.get("bid_size", 0) + quote.get("ask_size", 0),
            "change": 0.0,  # Not available from quote
            "change_percent": 0.0,
            "timestamp": quote.get("timestamp", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/level2/{symbol}")
async def get_level2_compat(symbol: str):
    """Legacy Level 2 endpoint - return basic bid/ask"""
    try:
        market_data = get_alpaca_market_data()
        quote = market_data.get_latest_quote(symbol.upper())

        if not quote:
            return {"bids": [], "asks": []}

        # Simple bid/ask structure
        return {
            "symbol": symbol,
            "bids": [{"price": quote["bid"], "size": quote.get("bid_size", 0)}],
            "asks": [{"price": quote["ask"], "size": quote.get("ask_size", 0)}]
        }
    except Exception as e:
        return {"bids": [], "asks": []}


@router.get("/api/timesales/{symbol}")
async def get_timesales_compat(symbol: str, limit: int = 50):
    """Legacy time & sales endpoint - return empty for now"""
    # Time & sales data not available from Alpaca quotes
    # Would need WebSocket stream for real-time trades
    return {
        "symbol": symbol,
        "trades": []
    }


# ============================================================================
# ACCOUNT COMPATIBILITY
# ============================================================================

@router.get("/api/account")
async def get_account_compat():
    """Legacy account endpoint - maps to Alpaca account"""
    try:
        connector = get_alpaca_connector()

        if not connector.is_connected():
            raise HTTPException(status_code=503, detail="Not connected to Alpaca")

        account = connector.get_account()

        # Return in format expected by UI
        return {
            "account_id": account["account_id"],
            "balance": account["portfolio_value"],
            "buying_power": account["buying_power"],
            "cash": account["cash"],
            "equity": account["equity"],
            "positions_value": account["portfolio_value"] - account["cash"],
            "currency": account["currency"],
            "status": account["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WORKLIST COMPATIBILITY
# ============================================================================

@router.get("/api/worklist")
async def get_worklist_compat():
    """Legacy worklist endpoint - return default watchlist"""
    default_watchlist = [
        "SPY", "QQQ", "AAPL", "MSFT", "GOOGL",
        "AMZN", "TSLA", "NVDA", "META", "AMD"
    ]

    try:
        market_data = get_alpaca_market_data()

        symbols = []
        for symbol in default_watchlist:
            try:
                quote = market_data.get_latest_quote(symbol)
                if quote:
                    symbols.append({
                        "symbol": symbol,
                        "name": symbol,
                        "price": quote["last"],
                        "change": 0.0,
                        "change_percent": 0.0,
                        "volume": quote.get("bid_size", 0) + quote.get("ask_size", 0),
                        "bid": quote["bid"],
                        "ask": quote["ask"]
                    })
            except:
                symbols.append({
                    "symbol": symbol,
                    "name": symbol,
                    "price": 0.0,
                    "change": 0.0,
                    "change_percent": 0.0,
                    "volume": 0,
                    "bid": 0.0,
                    "ask": 0.0
                })

        # Return just the array of symbols (not wrapped in an object)
        return symbols
    except Exception as e:
        # Return array of symbols with default values
        return [
            {
                "symbol": s,
                "name": s,
                "price": 0,
                "change": 0,
                "change_percent": 0,
                "volume": 0,
                "bid": 0,
                "ask": 0
            }
            for s in default_watchlist
        ]


@router.post("/api/worklist/add")
async def add_to_worklist_compat(data: dict):
    """Legacy add to worklist - placeholder"""
    return {"success": True, "message": "Symbol added to worklist"}


@router.delete("/api/worklist/{symbol}")
async def remove_from_worklist_compat(symbol: str):
    """Legacy remove from worklist - placeholder"""
    return {"success": True, "message": f"Symbol {symbol} removed"}


# ============================================================================
# SCANNER COMPATIBILITY
# ============================================================================

@router.get("/api/scanner/ALPACA/presets")
async def get_scanner_presets_compat():
    """Legacy scanner presets"""
    return {
        "presets": [
            {
                "name": "Top Gainers",
                "description": "Stocks with highest gains",
                "filters": {}
            },
            {
                "name": "High Volume",
                "description": "Stocks with unusual volume",
                "filters": {}
            }
        ]
    }


@router.post("/api/scanner/ALPACA/scan")
async def run_scanner_compat(data: dict):
    """Legacy scanner - return sample results"""
    return {
        "results": [
            {"symbol": "AAPL", "score": 85, "price": 180.0, "volume": 1000000},
            {"symbol": "TSLA", "score": 78, "price": 250.0, "volume": 800000},
            {"symbol": "NVDA", "score": 72, "price": 500.0, "volume": 600000}
        ]
    }


@router.get("/api/scanner/results")
async def get_scanner_results_compat():
    """Legacy scanner results"""
    return {
        "results": []
    }


# ============================================================================
# REAL-TIME SUBSCRIPTION COMPATIBILITY
# ============================================================================

@router.post("/api/subscribe")
async def subscribe_compat(data: dict):
    """Legacy subscribe endpoint - placeholder for real-time updates"""
    # Real-time streaming would require WebSocket implementation
    # For now, return success to prevent UI errors
    return {
        "success": True,
        "message": "Subscription acknowledged (polling mode)",
        "symbol": data.get("symbol", ""),
        "mode": "polling"
    }


@router.post("/api/unsubscribe")
async def unsubscribe_compat(data: dict):
    """Legacy unsubscribe endpoint"""
    return {
        "success": True,
        "message": "Unsubscribed",
        "symbol": data.get("symbol", "")
    }


# ============================================================================
# STATUS/HEALTH COMPATIBILITY
# ============================================================================

@router.get("/health")
async def health_compat():
    """Legacy health endpoint - redirect to /api/health"""
    connector = get_alpaca_connector()

    return {
        "status": "ok" if connector.is_connected() else "error",
        "broker": "Alpaca",
        "connected": connector.is_connected()
    }


# ============================================================================
# AI/CLAUDE COMPATIBILITY
# ============================================================================

@router.get("/api/claude/status")
async def claude_status_compat():
    """Claude AI status endpoint"""
    from ai.alpaca_ai_predictor import get_alpaca_predictor

    try:
        predictor = get_alpaca_predictor()
        has_model = predictor.model is not None

        return {
            "available": has_model,
            "status": "online" if has_model else "offline",
            "model_loaded": has_model,
            "accuracy": predictor.accuracy if has_model else 0.0,
            "features": len(predictor.feature_names) if has_model else 0
        }
    except:
        return {
            "available": False,
            "status": "offline",
            "model_loaded": False
        }


@router.get("/api/ai/status")
async def ai_status_compat():
    """AI status endpoint - alias for claude status"""
    return await claude_status_compat()


# ============================================================================
# BOT CONTROL COMPATIBILITY (PLACEHOLDERS)
# ============================================================================

@router.post("/api/bot/init")
async def bot_init_compat(data: dict):
    """Legacy bot init"""
    return {"success": True, "message": "Bot initialized"}


@router.post("/api/bot/start")
async def bot_start_compat():
    """Legacy bot start"""
    return {"success": True, "message": "Bot started", "status": "running"}


@router.post("/api/bot/stop")
async def bot_stop_compat():
    """Legacy bot stop"""
    return {"success": True, "message": "Bot stopped", "status": "stopped"}


@router.post("/api/bot/enable")
async def bot_enable_compat():
    """Legacy bot enable"""
    return {"success": True, "message": "Bot enabled"}


@router.post("/api/bot/disable")
async def bot_disable_compat():
    """Legacy bot disable"""
    return {"success": True, "message": "Bot disabled"}


@router.get("/api/bot/status")
async def bot_status_compat():
    """Legacy bot status"""
    return {
        "status": "stopped",
        "enabled": False,
        "trades_today": 0,
        "pnl_today": 0.0
    }


@router.post("/api/bot/config")
async def bot_config_compat(data: dict):
    """Legacy bot config"""
    return {"success": True, "message": "Configuration updated"}
