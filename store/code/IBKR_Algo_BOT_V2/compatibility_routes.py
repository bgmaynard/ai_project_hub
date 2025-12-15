"""
Compatibility Routes for Legacy UI
Maps old IBKR endpoints to new broker endpoints
Primary broker: Schwab (as of v2.1.0)
"""
import asyncio
import time
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

# Worklist cache with request coalescing
_worklist_cache = {"data": None, "timestamp": 0, "lock": None}
_WORKLIST_CACHE_TTL = 5.0  # Cache for 5 seconds

# Unified market data (Schwab)
try:
    from unified_market_data import get_unified_market_data
    HAS_UNIFIED = True
except ImportError:
    HAS_UNIFIED = False

# Unified broker (Schwab)
try:
    from unified_broker import get_unified_broker
    HAS_BROKER = True
except ImportError:
    HAS_BROKER = False

# Watchlist manager
try:
    from watchlist_manager import get_watchlist_manager
    HAS_WATCHLIST = True
except ImportError:
    HAS_WATCHLIST = False

# Schwab trading integration (primary broker)
try:
    from schwab_trading import get_schwab_trading, is_schwab_trading_available
    HAS_SCHWAB = True
except ImportError:
    HAS_SCHWAB = False

router = APIRouter(tags=["Compatibility"])

# ============================================================================
# PRICE & MARKET DATA COMPATIBILITY
# ============================================================================

@router.get("/api/price/{symbol}")
async def get_price_compat(symbol: str):
    """Price endpoint - uses Schwab"""
    try:
        quote = None

        # Use unified market data (Schwab)
        if HAS_UNIFIED:
            try:
                unified = get_unified_market_data()
                quote = unified.get_quote(symbol.upper())
            except Exception:
                pass

        if not quote:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Return in format expected by UI
        return {
            "symbol": symbol.upper(),
            "price": quote.get("last", quote.get("mid", 0)),
            "bid": quote.get("bid", 0),
            "ask": quote.get("ask", 0),
            "volume": quote.get("volume", quote.get("bid_size", 0) + quote.get("ask_size", 0)),
            "high": quote.get("high", 0),
            "low": quote.get("low", 0),
            "open": quote.get("open", 0),
            "close": quote.get("close", 0),
            "change": quote.get("change", 0),
            "change_percent": quote.get("change_percent", 0),
            "timestamp": quote.get("timestamp", ""),
            "source": quote.get("source", "unknown")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/level2/{symbol}")
async def get_level2_compat(symbol: str):
    """Level 2 endpoint - uses Schwab"""
    try:
        quote = None

        # Use unified market data (Schwab)
        if HAS_UNIFIED:
            try:
                unified = get_unified_market_data()
                quote = unified.get_quote(symbol.upper())
            except Exception:
                pass

        if not quote:
            return {"bids": [], "asks": []}

        # Simple bid/ask structure with source info
        return {
            "symbol": symbol.upper(),
            "bids": [{"price": quote.get("bid", 0), "size": quote.get("bid_size", 100)}],
            "asks": [{"price": quote.get("ask", 0), "size": quote.get("ask_size", 100)}],
            "source": quote.get("source", "unknown")
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
    """Legacy account endpoint - maps to Schwab account"""
    try:
        if not HAS_BROKER:
            raise HTTPException(status_code=503, detail="Broker not available")

        broker = get_unified_broker()
        if not broker or not broker.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to broker")

        account = broker.get_account()

        # Return in format expected by UI
        return {
            "account_id": account.get("account_number", "Unknown"),
            "balance": account.get("market_value", 0),
            "buying_power": account.get("buying_power", 0),
            "cash": account.get("cash", 0),
            "equity": account.get("market_value", 0),
            "positions_value": account.get("long_market_value", 0),
            "currency": "USD",
            "status": "active"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WORKLIST COMPATIBILITY
# ============================================================================

@router.get("/api/worklist")
async def get_worklist_compat():
    """Worklist endpoint - return saved watchlist with quotes (cached + coalesced)"""
    global _worklist_cache

    # Check cache first (instant return if fresh)
    now = time.time()
    if _worklist_cache["data"] and (now - _worklist_cache["timestamp"]) < _WORKLIST_CACHE_TTL:
        logger.info(f"[WORKLIST] Cache HIT - returning cached data (age: {(now - _worklist_cache['timestamp'])*1000:.0f}ms)")
        return _worklist_cache["data"]

    # Initialize lock if needed
    if _worklist_cache["lock"] is None:
        _worklist_cache["lock"] = asyncio.Lock()

    # Use lock to coalesce concurrent requests
    async with _worklist_cache["lock"]:
        # Double-check cache (another request may have populated it while waiting)
        now = time.time()
        if _worklist_cache["data"] and (now - _worklist_cache["timestamp"]) < _WORKLIST_CACHE_TTL:
            return _worklist_cache["data"]

        try:
            start_total = time.time()

            # Get watchlist from manager
            watchlist_symbols = []
            if HAS_WATCHLIST:
                mgr = get_watchlist_manager()
                watchlist = mgr.get_default_watchlist()
                watchlist_symbols = watchlist.get("symbols", [])

            t1 = time.time()
            logger.info(f"[WORKLIST] Got {len(watchlist_symbols)} symbols in {(t1-start_total)*1000:.0f}ms")

            if not watchlist_symbols:
                watchlist_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL",
                                     "AMZN", "TSLA", "NVDA", "META", "AMD"]

            symbols = []
            if HAS_UNIFIED:
                unified = get_unified_market_data()
                # Use batch quote fetching for speed (single API call vs sequential)
                try:
                    t2 = time.time()
                    quotes = unified.get_quotes(watchlist_symbols)
                    t3 = time.time()
                    logger.info(f"[WORKLIST] Batch fetch {len(watchlist_symbols)} quotes in {(t3-t2)*1000:.0f}ms")

                    for symbol in watchlist_symbols:
                        quote = quotes.get(symbol, {})
                        symbols.append({
                            "symbol": symbol,
                            "name": symbol,
                            "price": quote.get("last", 0),
                            "change": quote.get("change", 0),
                            "change_percent": quote.get("change_percent", 0),
                            "volume": quote.get("volume", 0),
                            "bid": quote.get("bid", 0),
                            "ask": quote.get("ask", 0)
                        })
                except Exception as batch_err:
                    logger.warning(f"[WORKLIST] Batch fetch failed: {batch_err}, falling back to sequential")
                    # Fallback to sequential if batch fails
                    seq_start = time.time()
                    for symbol in watchlist_symbols:
                        try:
                            quote = unified.get_quote(symbol)
                            if quote:
                                symbols.append({
                                    "symbol": symbol,
                                    "name": symbol,
                                    "price": quote.get("last", 0),
                                    "change": quote.get("change", 0),
                                    "change_percent": quote.get("change_percent", 0),
                                    "volume": quote.get("volume", 0),
                                    "bid": quote.get("bid", 0),
                                    "ask": quote.get("ask", 0)
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
                    logger.warning(f"[WORKLIST] Sequential fetch took {(time.time()-seq_start)*1000:.0f}ms")
            else:
                symbols = [{"symbol": s, "name": s, "price": 0, "change": 0, "change_percent": 0, "volume": 0, "bid": 0, "ask": 0} for s in watchlist_symbols]

            # Cache the result
            result = {"success": True, "data": symbols, "count": len(symbols)}
            _worklist_cache["data"] = result
            _worklist_cache["timestamp"] = time.time()
            logger.info(f"[WORKLIST] TOTAL request time: {(time.time()-start_total)*1000:.0f}ms for {len(symbols)} symbols")
            return result
        except Exception as e:
            return {"success": True, "data": [{"symbol": s, "name": s, "price": 0, "change": 0, "change_percent": 0, "volume": 0, "bid": 0, "ask": 0} for s in ["SPY", "QQQ", "AAPL"]], "count": 3}


@router.post("/api/worklist/add")
async def add_to_worklist_compat(data: dict):
    """Add symbol to worklist"""
    global _worklist_cache
    try:
        symbol = data.get("symbol", "").upper()
        if not symbol:
            return {"success": False, "message": "No symbol provided"}

        if HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            watchlist_id = watchlist.get("watchlist_id")
            if watchlist_id:
                mgr.add_symbols(watchlist_id, [symbol])
                # Invalidate cache so next request fetches fresh data
                _worklist_cache["data"] = None
                _worklist_cache["timestamp"] = 0
                return {"success": True, "message": f"Symbol {symbol} added to worklist"}

        return {"success": False, "message": "Watchlist manager not available"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.delete("/api/worklist/{symbol}")
async def remove_from_worklist_compat(symbol: str):
    """Remove symbol from worklist"""
    global _worklist_cache
    try:
        symbol = symbol.upper()
        if HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            watchlist_id = watchlist.get("watchlist_id")
            if watchlist_id:
                mgr.remove_symbols(watchlist_id, [symbol])
                # Invalidate cache so next request fetches fresh data
                _worklist_cache["data"] = None
                _worklist_cache["timestamp"] = 0
                return {"success": True, "message": f"Symbol {symbol} removed from worklist"}

        return {"success": False, "message": "Watchlist manager not available"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ============================================================================
# WATCHLIST (SINGULAR) COMPATIBILITY - Maps to /api/watchlists
# ============================================================================

@router.get("/api/watchlist/default")
async def get_watchlist_default():
    """Get default watchlist with quotes"""
    try:
        if HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            symbols = watchlist.get("symbols", [])

            # Add quote data for each symbol
            enriched_symbols = []
            if HAS_UNIFIED:
                unified = get_unified_market_data()
                for symbol in symbols:
                    try:
                        quote = unified.get_quote(symbol)
                        enriched_symbols.append({
                            "symbol": symbol,
                            "price": quote.get("last", 0) if quote else 0,
                            "change": quote.get("change", 0) if quote else 0,
                            "change_percent": quote.get("change_percent", 0) if quote else 0,
                            "volume": quote.get("volume", 0) if quote else 0,
                            "bid": quote.get("bid", 0) if quote else 0,
                            "ask": quote.get("ask", 0) if quote else 0
                        })
                    except:
                        enriched_symbols.append({"symbol": symbol, "price": 0, "change": 0, "change_percent": 0, "volume": 0, "bid": 0, "ask": 0})
            else:
                enriched_symbols = [{"symbol": s, "price": 0} for s in symbols]

            return {
                "watchlist_id": watchlist.get("watchlist_id"),
                "name": watchlist.get("name", "Default"),
                "symbols": enriched_symbols,
                "count": len(enriched_symbols)
            }
        return {"symbols": [], "count": 0}
    except Exception as e:
        return {"symbols": [], "count": 0, "error": str(e)}


# ============================================================================
# WATCHLIST-AI COMPATIBILITY - For AI working list
# ============================================================================

# Store for AI working list (in-memory, persists during session)
_ai_working_list = []

@router.get("/api/watchlist-ai/working-list")
async def get_ai_working_list():
    """Get AI working list"""
    global _ai_working_list
    try:
        # If empty, initialize from default watchlist
        if not _ai_working_list and HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            _ai_working_list = watchlist.get("symbols", [])[:20]  # Limit to 20

        # Enrich with quotes
        enriched = []
        if HAS_UNIFIED:
            unified = get_unified_market_data()
            for symbol in _ai_working_list:
                try:
                    quote = unified.get_quote(symbol)
                    enriched.append({
                        "symbol": symbol,
                        "price": quote.get("last", 0) if quote else 0,
                        "change": quote.get("change", 0) if quote else 0,
                        "change_percent": quote.get("change_percent", 0) if quote else 0,
                        "volume": quote.get("volume", 0) if quote else 0
                    })
                except:
                    enriched.append({"symbol": symbol, "price": 0, "change": 0, "change_percent": 0, "volume": 0})
        else:
            enriched = [{"symbol": s, "price": 0} for s in _ai_working_list]

        return {
            "symbols": enriched,
            "count": len(enriched),
            "name": "AI_Working_List"
        }
    except Exception as e:
        return {"symbols": [], "count": 0, "error": str(e)}


@router.post("/api/watchlist-ai/add")
async def add_to_ai_working_list(data: dict):
    """Add symbol to AI working list"""
    global _ai_working_list
    try:
        symbol = data.get("symbol", "").upper()
        if symbol and symbol not in _ai_working_list:
            _ai_working_list.append(symbol)
            # Also add to persistent watchlist
            if HAS_WATCHLIST:
                mgr = get_watchlist_manager()
                watchlist = mgr.get_default_watchlist()
                watchlist_id = watchlist.get("watchlist_id")
                if watchlist_id:
                    mgr.add_symbols(watchlist_id, [symbol])
            return {"success": True, "message": f"Added {symbol} to AI working list"}
        return {"success": False, "message": "Symbol already in list or invalid"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.delete("/api/watchlist-ai/remove/{symbol}")
async def remove_from_ai_working_list(symbol: str):
    """Remove symbol from AI working list"""
    global _ai_working_list
    try:
        symbol = symbol.upper()
        if symbol in _ai_working_list:
            _ai_working_list.remove(symbol)
            # Also remove from persistent watchlist
            if HAS_WATCHLIST:
                mgr = get_watchlist_manager()
                watchlist = mgr.get_default_watchlist()
                watchlist_id = watchlist.get("watchlist_id")
                if watchlist_id:
                    mgr.remove_symbols(watchlist_id, [symbol])
            return {"success": True, "message": f"Removed {symbol}"}
        return {"success": False, "message": "Symbol not in list"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/api/watchlist-ai/sync")
async def sync_ai_working_list(watchlist_name: str = "AI_Working_List"):
    """Sync AI working list with persistent storage"""
    global _ai_working_list
    try:
        if HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            _ai_working_list = watchlist.get("symbols", [])[:20]
            return {"success": True, "synced": len(_ai_working_list), "symbols": _ai_working_list}
        return {"success": False, "message": "Watchlist manager not available"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/api/watchlist-ai/train")
async def train_ai_on_watchlist(data: dict = None):
    """Trigger AI training on working list"""
    global _ai_working_list
    try:
        symbols = _ai_working_list if _ai_working_list else []
        return {
            "success": True,
            "message": f"Training queued for {len(symbols)} symbols",
            "symbols": symbols,
            "status": "queued"
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


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
    """Scanner results - returns watchlist symbols as fallback"""
    try:
        symbols = []
        if HAS_WATCHLIST:
            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()
            symbols = watchlist.get("symbols", [])
        return {
            "results": [{"symbol": s, "score": 50} for s in symbols[:20]],
            "symbols": symbols[:20],
            "count": len(symbols)
        }
    except:
        return {"results": [], "symbols": [], "count": 0}


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
    """Health endpoint - Schwab"""
    # Check Schwab
    if HAS_SCHWAB:
        try:
            if is_schwab_trading_available():
                schwab = get_schwab_trading()
                if schwab:
                    return {
                        "status": "ok",
                        "broker": "Schwab",
                        "connected": True,
                        "name": "Morpheus Trading Bot"
                    }
        except Exception:
            pass

    return {
        "status": "ok",
        "broker": "Schwab",
        "connected": HAS_BROKER,
        "name": "Morpheus Trading Bot"
    }


# ============================================================================
# BROKER SWITCH COMPATIBILITY
# ============================================================================

@router.post("/api/broker/switch/{broker_name}")
async def switch_broker_compat(broker_name: str):
    """Switch broker - Schwab is the only broker now"""
    broker_name = broker_name.lower()
    if broker_name == "schwab":
        return {
            "success": True,
            "broker": "Schwab",
            "message": "Schwab is the active broker",
            "connected": HAS_SCHWAB
        }
    else:
        return {
            "success": False,
            "broker": broker_name,
            "message": f"Broker '{broker_name}' not available. Schwab is the only supported broker.",
            "connected": False
        }


@router.get("/api/broker/status")
async def get_broker_status():
    """Get broker connection status"""
    return {
        "active_broker": "Schwab",
        "schwab_connected": HAS_SCHWAB,
        "available_brokers": ["Schwab"],
        "name": "Morpheus Trading Bot"
    }


# ============================================================================
# ASSET STATUS COMPATIBILITY
# ============================================================================

@router.get("/api/schwab/asset/{symbol}/status")
async def get_asset_status(symbol: str):
    """Get asset status (shortable, HTB, halted) for a symbol"""
    symbol = symbol.upper()
    try:
        # Default status - assume stocks are tradeable
        status = {
            "success": True,
            "symbol": symbol,
            "shortable": True,
            "easy_to_borrow": True,
            "htb": False,  # Hard to borrow
            "halted": False,
            "tradeable": True,
            "marginable": True,
            "source": "default"
        }

        # Try to get real status from Schwab if available
        if HAS_SCHWAB:
            try:
                schwab = get_schwab_trading()
                if schwab and hasattr(schwab, 'get_instrument_info'):
                    info = schwab.get_instrument_info(symbol)
                    if info:
                        status.update({
                            "shortable": info.get("shortable", True),
                            "easy_to_borrow": info.get("easy_to_borrow", True),
                            "htb": not info.get("easy_to_borrow", True),
                            "halted": info.get("halted", False),
                            "tradeable": info.get("tradeable", True),
                            "source": "schwab"
                        })
            except Exception:
                pass  # Use default status

        return status
    except Exception as e:
        # Return safe defaults on any error
        return {
            "success": True,
            "symbol": symbol,
            "shortable": True,
            "easy_to_borrow": True,
            "htb": False,
            "halted": False,
            "tradeable": True,
            "marginable": True,
            "source": "default",
            "note": "Using defaults"
        }


# ============================================================================
# AI/CLAUDE COMPATIBILITY
# ============================================================================

@router.get("/api/claude/status")
async def claude_status_compat():
    """Claude AI status endpoint"""
    try:
        from ai.ai_predictor import get_predictor
        predictor = get_predictor()
        has_model = predictor is not None and predictor.model is not None

        return {
            "available": has_model,
            "status": "online" if has_model else "offline",
            "model_loaded": has_model,
            "accuracy": getattr(predictor, 'accuracy', 0.0) if predictor else 0.0,
            "features": len(getattr(predictor, 'feature_names', [])) if predictor else 0
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


# ============================================================================
# PIPELINE & BACKTEST COMPATIBILITY
# ============================================================================

@router.post("/api/pipeline/train/advanced")
async def pipeline_train_advanced(data: dict = None):
    """Advanced training pipeline - placeholder"""
    return {
        "success": True,
        "status": "not_configured",
        "message": "Advanced training pipeline not configured. Use /api/ai/train for basic training."
    }


@router.post("/api/schwab/ai/backtest/run")
async def schwab_backtest_run(data: dict = None):
    """Backtest via Schwab path - redirects to main backtest endpoint"""
    from datetime import datetime, timedelta

    try:
        from backtester import get_backtester
        backtester = get_backtester()

        if not data:
            return {"success": False, "error": "No backtest parameters provided"}

        symbols = data.get("symbols", data.get("symbol", ["SPY"]))
        if isinstance(symbols, str):
            symbols = [symbols]

        # Convert days to start_date/end_date
        # Need at least 180 days for feature calculation (50 day indicators + 30 day warmup + trading)
        days = max(data.get("days", 180), 180)  # Minimum 180 days for valid backtest
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Use provided dates if available
        start_date = data.get("start_date", start_date)
        end_date = data.get("end_date", end_date)

        result = backtester.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=data.get("initial_capital", 10000)
        )
        # Convert dataclass to dict if needed
        if hasattr(result, '__dict__'):
            from dataclasses import asdict
            result = asdict(result)
        return {"success": True, "data": result}
    except ImportError:
        return {
            "success": False,
            "error": "Backtester module not available",
            "message": "Use /api/backtest/run instead"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/schwab/ai/backtest/quick")
async def schwab_backtest_quick(data: dict = None):
    """Quick backtest - shorter timeframe for rapid testing"""
    from datetime import datetime, timedelta

    try:
        from backtester import get_backtester
        backtester = get_backtester()

        if not data:
            data = {}

        symbols = data.get("symbols", data.get("symbol", ["SPY"]))
        if isinstance(symbols, str):
            symbols = [symbols]

        # Quick backtest needs minimum 180 days for feature warmup
        days = max(data.get("days", 180), 180)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        result = backtester.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=data.get("initial_capital", 10000)
        )
        # Convert dataclass to dict if needed
        if hasattr(result, '__dict__'):
            from dataclasses import asdict
            result = asdict(result)
        return {"success": True, "data": result, "type": "quick"}
    except ImportError:
        return {
            "success": False,
            "error": "Backtester module not available",
            "message": "Quick backtest requires backtester module"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
