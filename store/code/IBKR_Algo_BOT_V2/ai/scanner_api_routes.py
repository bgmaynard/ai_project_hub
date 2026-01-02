"""
Claude AI Stock Scanner API Routes
==================================
API endpoints for the AI-powered stock scanner.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .claude_stock_scanner import get_stock_scanner, ClaudeStockScanner
from .claude_watchlist_manager import get_watchlist_manager
from .penny_momentum_scanner import get_penny_scanner, start_penny_scanner, stop_penny_scanner
from .warrior_momentum_scanner import get_warrior_scanner, start_warrior_scanner, stop_warrior_scanner
from .finviz_momentum_scanner import get_finviz_scanner, start_finviz_scanner, stop_finviz_scanner, is_finviz_scanner_running

# Alpaca Momentum Scanner (uses Alpaca API directly)
try:
    from .momentum_scanner import get_momentum_scanner, CRITERIA, SCANNER_PRESETS
    HAS_ALPACA_SCANNER = True
except ImportError as e:
    HAS_ALPACA_SCANNER = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Alpaca momentum scanner not available: {e}")

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scanner")


# ============ Pydantic Models ============

class ScanRequest(BaseModel):
    preset_id: str
    max_results: int = 50
    add_to_watchlist: bool = False


class CustomScanRequest(BaseModel):
    name: str = "Custom Scan"
    min_price: float = 5
    max_price: float = 500
    min_change_percent: float = 2
    min_volume_ratio: float = 1.5
    max_results: int = 50


# ============ Endpoints ============

@router.get("/info")
async def get_scanner_info():
    """Get scanner system information"""
    scanner = get_stock_scanner()

    return {
        "status": "active",
        "ai_available": scanner.ai_available,
        "universe_loaded": scanner.universe_loaded,
        "universe_size": len(scanner.stock_universe),
        "presets_count": len(scanner.presets),
        "last_scan": scanner.last_scan_time.isoformat() if scanner.last_scan_time else None,
        "features": [
            "Multiple scanner presets (Momentum, Breakout, Gap, etc.)",
            "Full NASDAQ/NYSE universe scanning",
            "Claude AI analysis and selection",
            "Automatic watchlist integration",
            "Real-time market data"
        ]
    }


@router.get("/presets")
async def get_presets():
    """Get all scanner presets"""
    scanner = get_stock_scanner()
    presets = scanner.get_presets()

    return {
        "count": len(presets),
        "presets": presets
    }


@router.post("/run/{preset_id}")
async def run_scan(preset_id: str, max_results: int = 50, add_to_watchlist: bool = False):
    """Run a scanner with specific preset"""
    scanner = get_stock_scanner()

    if preset_id not in scanner.presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")

    # Run the scan
    results = await scanner.run_scan(preset_id, max_results)

    # Optionally analyze with Claude and add to watchlist
    claude_analysis = None
    if add_to_watchlist and results:
        claude_analysis = await scanner.claude_analyze_scan_results(preset_id, add_to_watchlist=True)

    return {
        "status": "success",
        "preset": preset_id,
        "count": len(results),
        "results": [r.to_dict() for r in results],
        "claude_analysis": claude_analysis
    }


@router.get("/results/{preset_id}")
async def get_results(preset_id: str):
    """Get latest scan results for a preset"""
    scanner = get_stock_scanner()

    results = scanner.get_results(preset_id)

    return {
        "preset": preset_id,
        "count": len(results),
        "results": results
    }


@router.post("/run-all")
async def run_all_scans():
    """Run all enabled scanner presets"""
    scanner = get_stock_scanner()

    all_results = await scanner.run_all_scans()

    summary = {}
    for preset_id, results in all_results.items():
        summary[preset_id] = {
            "count": len(results),
            "top_3": [
                {"symbol": r.symbol, "score": r.score, "change": r.change_percent}
                for r in results[:3]
            ]
        }

    return {
        "status": "success",
        "scans_run": len(all_results),
        "summary": summary
    }


@router.post("/analyze/{preset_id}")
async def analyze_with_claude(preset_id: str, add_to_watchlist: bool = True):
    """Have Claude AI analyze scan results"""
    scanner = get_stock_scanner()

    if preset_id not in scanner.scan_results:
        raise HTTPException(status_code=404, detail="No scan results. Run a scan first.")

    analysis = await scanner.claude_analyze_scan_results(preset_id, add_to_watchlist)

    return analysis


@router.post("/momentum-workflow")
async def run_momentum_workflow():
    """
    Complete momentum workflow:
    1. Run momentum scanner
    2. Have Claude select best stocks
    3. Add to watchlist
    4. Train AI models
    """
    scanner = get_stock_scanner()
    manager = get_watchlist_manager()

    results = {
        "timestamp": datetime.now().isoformat(),
        "steps": {}
    }

    # Step 1: Run momentum scan
    logger.info("[SCANNER] Step 1: Running momentum scan...")
    scan_results = await scanner.run_scan("momentum_warrior", max_results=30)
    results["steps"]["scan"] = {
        "found": len(scan_results),
        "top_movers": [
            {"symbol": r.symbol, "change": r.change_percent, "score": r.score}
            for r in scan_results[:10]
        ]
    }

    if not scan_results:
        results["status"] = "no_results"
        return results

    # Step 2: Claude analysis and selection
    logger.info("[SCANNER] Step 2: Claude AI analysis...")
    analysis = await scanner.claude_analyze_scan_results("momentum_warrior", add_to_watchlist=True)
    results["steps"]["claude_analysis"] = analysis

    # Step 3: Get updated working list
    working_list = manager.get_working_list()
    results["steps"]["working_list"] = {
        "count": len(working_list),
        "symbols": [e["symbol"] for e in working_list if e.get("active")]
    }

    # Step 4: Sync to platform watchlist
    sync_result = await manager.sync_to_platform_watchlist("Scanner_Results")
    results["steps"]["sync"] = sync_result

    results["status"] = "success"
    return results


@router.get("/quick-scan/{scan_type}")
async def quick_scan(scan_type: str, limit: int = 10):
    """
    Quick scan shortcuts:
    - momentum: High momentum stocks
    - gaps: Gap up/down stocks
    - volume: Volume explosion
    - breakout: Breakout candidates
    - oversold: Oversold bounces
    """
    scanner = get_stock_scanner()

    preset_map = {
        "momentum": "momentum_warrior",
        "gaps": "gap_and_go",
        "volume": "volume_explosion",
        "breakout": "breakout_hunter",
        "oversold": "oversold_bounce",
        "after_hours": "after_hours_movers",
        "new_highs": "new_52_week_high"
    }

    if scan_type not in preset_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scan type. Available: {list(preset_map.keys())}"
        )

    preset_id = preset_map[scan_type]
    results = await scanner.run_scan(preset_id, max_results=limit)

    return {
        "scan_type": scan_type,
        "count": len(results),
        "stocks": [
            {
                "symbol": r.symbol,
                "price": r.price,
                "change": r.change_percent,
                "volume_ratio": r.volume_ratio,
                "score": r.score,
                "signals": r.signals
            }
            for r in results
        ]
    }


@router.post("/custom")
async def run_custom_scan(request: CustomScanRequest):
    """Run a custom scan with user-defined criteria"""
    scanner = get_stock_scanner()

    # Create temporary custom preset
    from .claude_stock_scanner import ScannerPreset, ScannerType

    custom_preset = ScannerPreset(
        id="custom_temp",
        name=request.name,
        scanner_type=ScannerType.CUSTOM,
        description="User-defined custom scan",
        criteria={
            "min_price": request.min_price,
            "max_price": request.max_price,
            "min_change_percent": request.min_change_percent,
            "min_volume_ratio": request.min_volume_ratio,
        }
    )

    # Add to presets temporarily
    scanner.presets["custom_temp"] = custom_preset

    # Run scan
    results = await scanner.run_scan("custom_temp", max_results=request.max_results)

    # Remove temp preset
    del scanner.presets["custom_temp"]

    return {
        "status": "success",
        "name": request.name,
        "count": len(results),
        "results": [r.to_dict() for r in results]
    }


@router.get("/universe/load")
async def load_universe():
    """Load the full stock universe"""
    scanner = get_stock_scanner()
    count = await scanner.load_stock_universe()

    return {
        "status": "success",
        "stocks_loaded": count
    }


# ============ Penny Scanner Endpoints ============

@router.get("/penny/status")
async def penny_scanner_status():
    """Get penny scanner status"""
    scanner = get_penny_scanner()
    return scanner.get_status()


@router.post("/penny/start")
async def start_penny_scanning(interval: int = 5):
    """Start the penny momentum scanner"""
    scanner = start_penny_scanner(interval=interval)
    return {
        "status": "started",
        "interval": interval,
        "watchlist_size": len(scanner.watchlist)
    }


@router.post("/penny/stop")
async def stop_penny_scanning():
    """Stop the penny scanner"""
    stop_penny_scanner()
    return {"status": "stopped"}


@router.get("/penny/movers")
async def get_penny_movers(limit: int = 10):
    """Get current penny movers"""
    scanner = get_penny_scanner()
    return {
        "count": len(scanner.movers),
        "movers": scanner.get_top_movers(limit)
    }


@router.get("/penny/signals")
async def get_penny_buy_signals():
    """Get penny stocks with buy signals"""
    scanner = get_penny_scanner()
    return {
        "signals": scanner.get_buy_signals()
    }


@router.post("/penny/scan")
async def run_penny_scan():
    """Run a single penny scan"""
    scanner = get_penny_scanner()
    movers = scanner.scan()
    return {
        "count": len(movers),
        "movers": [m.to_dict() for m in movers[:10]]
    }


@router.post("/penny/add/{symbol}")
async def add_penny_symbol(symbol: str):
    """Add symbol to penny watchlist"""
    scanner = get_penny_scanner()
    scanner.add_symbol(symbol)
    return {"status": "added", "symbol": symbol.upper()}


# ============ Warrior Scanner Endpoints ============

@router.get("/warrior/status")
async def warrior_scanner_status():
    """Get Warrior scanner status"""
    scanner = get_warrior_scanner()
    return scanner.get_status()


@router.post("/warrior/start")
async def start_warrior_scanning(interval: int = 10):
    """Start the Warrior momentum scanner"""
    scanner = start_warrior_scanner(interval=interval)
    return {
        "status": "started",
        "interval": interval,
        "watchlist_size": len(scanner.watchlist),
        "price_range": f"${scanner.min_price} - ${scanner.max_price}"
    }


@router.post("/warrior/stop")
async def stop_warrior_scanning():
    """Stop the Warrior scanner"""
    stop_warrior_scanner()
    return {"status": "stopped"}


@router.get("/warrior/setups")
async def get_warrior_setups(limit: int = 10):
    """Get current Warrior setups"""
    scanner = get_warrior_scanner()
    return {
        "count": len(scanner.setups),
        "setups": scanner.get_top_setups(limit)
    }


@router.get("/warrior/a-grade")
async def get_warrior_a_setups():
    """Get A+ and A grade Warrior setups"""
    scanner = get_warrior_scanner()
    return {
        "setups": scanner.get_a_setups()
    }


@router.get("/warrior/gaps")
async def get_warrior_gaps():
    """Get gap and go setups"""
    scanner = get_warrior_scanner()
    return {
        "setups": scanner.get_gap_setups()
    }


@router.get("/warrior/vwap-reclaims")
async def get_warrior_vwap_reclaims():
    """Get VWAP reclaim setups"""
    scanner = get_warrior_scanner()
    return {
        "setups": scanner.get_vwap_reclaims()
    }


@router.post("/warrior/scan")
async def run_warrior_scan():
    """Run a single Warrior scan"""
    scanner = get_warrior_scanner()
    setups = scanner.scan()
    return {
        "count": len(setups),
        "setups": [s.to_dict() for s in setups[:10]]
    }


@router.post("/warrior/add/{symbol}")
async def add_warrior_symbol(symbol: str):
    """Add symbol to Warrior watchlist"""
    scanner = get_warrior_scanner()
    scanner.add_symbol(symbol)
    return {"status": "added", "symbol": symbol.upper()}


@router.post("/warrior/reset-vwap")
async def reset_warrior_vwap():
    """Reset VWAP data (call at market open)"""
    scanner = get_warrior_scanner()
    scanner.reset_vwap()
    return {"status": "vwap_reset"}


# ============ Alpaca Momentum Scanner Endpoints ============

class CheckSymbolsRequest(BaseModel):
    symbols: List[str]


class UpdateCriteriaRequest(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    max_spread_pct: Optional[float] = None
    max_float: Optional[int] = None
    min_gap_pct: Optional[float] = None
    min_momentum_pct: Optional[float] = None
    min_rvol: Optional[float] = None


@router.get("/ALPACA/status")
async def alpaca_scanner_status():
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca broker removed. Use Warrior scanner instead.",
        "redirect": "/api/scanner/warrior/status",
        "alternative_endpoints": [
            "/api/scanner/warrior/status",
            "/api/scanner/warrior/scan",
            "/api/scanner/warrior/gaps"
        ]
    }


@router.get("/ALPACA/presets")
async def alpaca_scanner_presets():
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca broker removed. Use Warrior scanner instead.",
        "redirect": "/api/scanner/warrior/status",
        "presets": ["warrior_gaps", "warrior_momentum", "warrior_a_grade"]
    }


@router.get("/ALPACA/criteria")
async def alpaca_get_criteria():
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca broker removed. Use Warrior scanner config instead.",
        "redirect": "/api/scanner/warrior/status"
    }


@router.post("/ALPACA/criteria")
async def alpaca_update_criteria(request: UpdateCriteriaRequest = None):
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca broker removed. Edit config/warrior_config.json instead.",
        "success": False
    }


@router.get("/ALPACA/scan")
async def alpaca_run_scan(preset: str = "warrior_momentum"):
    """DEPRECATED: Alpaca removed - forwards to Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca removed. Use /api/scanner/warrior/scan instead.",
        "redirect": "/api/scanner/warrior/scan"
    }


@router.get("/ALPACA/top")
async def alpaca_get_top(limit: int = 10):
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca removed. Use /api/scanner/warrior/a-grade instead.",
        "redirect": "/api/scanner/warrior/a-grade"
    }


@router.post("/ALPACA/check")
async def alpaca_check_symbols(request: CheckSymbolsRequest = None):
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca removed. Use /api/scanner/warrior/scan instead.",
        "redirect": "/api/scanner/warrior/scan"
    }


@router.post("/ALPACA/check_OLD_DISABLED")
async def alpaca_check_symbols_disabled(request: CheckSymbolsRequest = None):
    """OLD DISABLED: Alpaca check symbols - kept for reference"""
    if True:  # Always disabled
        return {"deprecated": True, "message": "Use Warrior scanner"}

    scanner = get_momentum_scanner()
    results = []

    for symbol in request.symbols:
        symbol = symbol.upper().strip()
        if not symbol:
            continue

        scanner.add_symbol_to_universe(symbol)
        stock = scanner.check_criteria(symbol)

        if stock:
            results.append({
                "symbol": stock.symbol,
                "qualified": True,
                "price": stock.price,
                "change_pct": stock.change_pct,
                "volume": stock.volume,
                "avg_volume": stock.avg_volume,
                "rvol": stock.rvol,
                "float_shares": stock.float_shares,
                "spread_pct": stock.spread_pct,
                "momentum_score": stock.momentum_score,
                "catalyst": stock.catalyst,
                "meets_criteria": {
                    "price": CRITERIA['min_price'] <= stock.price <= CRITERIA['max_price'],
                    "volume": stock.volume >= CRITERIA['min_volume'],
                    "spread": stock.spread_pct <= CRITERIA['max_spread_pct'],
                    "float": stock.float_shares <= CRITERIA['max_float'] if stock.float_shares > 0 else "N/A",
                    "momentum": abs(stock.change_pct) >= CRITERIA['min_momentum_pct'],
                }
            })
        else:
            quote = scanner.get_quote(symbol)
            if quote:
                price = float(quote.get('last', 0) or quote.get('bid', 0) or 0)
                bid = float(quote.get('bid', 0) or 0)
                ask = float(quote.get('ask', 0) or 0)
                volume = int(quote.get('volume', 0) or 0)
                prev_close = float(quote.get('prev_close', 0) or price)
                spread_pct = ((ask - bid) / price * 100) if price > 0 and ask > 0 and bid > 0 else 0
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                float_shares = scanner.get_float(symbol)

                fail_reasons = []
                if price < CRITERIA['min_price']:
                    fail_reasons.append(f"price ${price:.2f} < ${CRITERIA['min_price']}")
                if price > CRITERIA['max_price']:
                    fail_reasons.append(f"price ${price:.2f} > ${CRITERIA['max_price']}")
                if volume < CRITERIA['min_volume']:
                    fail_reasons.append(f"volume {volume:,} < {CRITERIA['min_volume']:,}")
                if spread_pct > CRITERIA['max_spread_pct']:
                    fail_reasons.append(f"spread {spread_pct:.1f}% > {CRITERIA['max_spread_pct']}%")
                if float_shares > 0 and float_shares > CRITERIA['max_float']:
                    fail_reasons.append(f"float {float_shares/1_000_000:.1f}M > {CRITERIA['max_float']/1_000_000:.0f}M")
                if abs(change_pct) < CRITERIA['min_momentum_pct']:
                    fail_reasons.append(f"momentum {change_pct:+.1f}% < {CRITERIA['min_momentum_pct']}%")

                results.append({
                    "symbol": symbol,
                    "qualified": False,
                    "price": price,
                    "change_pct": change_pct,
                    "volume": volume,
                    "float_shares": float_shares,
                    "spread_pct": spread_pct,
                    "fail_reasons": fail_reasons,
                })
            else:
                results.append({
                    "symbol": symbol,
                    "qualified": False,
                    "error": "No quote data available"
                })

    return {
        "checked": len(results),
        "qualified": sum(1 for r in results if r.get('qualified')),
        "results": results,
        "criteria_used": CRITERIA
    }


@router.get("/ALPACA/analyze/{symbol}")
async def alpaca_analyze_symbol(symbol: str):
    """DEPRECATED: Alpaca removed - use AI trader analyze"""
    return {
        "deprecated": True,
        "symbol": symbol.upper(),
        "message": "Alpaca removed. Use /api/schwab/ai/trader/analyze/{symbol} instead.",
        "redirect": f"/api/schwab/ai/trader/analyze/{symbol.upper()}"
    }


@router.get("/ALPACA/analyze_OLD_DISABLED/{symbol}")
async def alpaca_analyze_symbol_disabled(symbol: str):
    """OLD DISABLED: kept for reference"""
    return {"deprecated": True}

    quote = scanner.get_quote(symbol)
    if not quote:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    price = float(quote.get('last', 0) or quote.get('bid', 0) or 0)
    bid = float(quote.get('bid', 0) or 0)
    ask = float(quote.get('ask', 0) or 0)
    volume = int(quote.get('volume', 0) or 0)
    prev_close = float(quote.get('prev_close', 0) or price)
    high = float(quote.get('high', price) or price)
    low = float(quote.get('low', price) or price)

    spread = ask - bid if ask > 0 and bid > 0 else 0
    spread_pct = (spread / price * 100) if price > 0 else 0
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
    range_pct = ((high - low) / low * 100) if low > 0 else 0

    bars = scanner.get_bars(symbol, "1Day", 20)
    avg_volume = volume
    if bars:
        volumes = [b.get('volume', 0) for b in bars if b.get('volume', 0) > 0]
        avg_volume = int(sum(volumes) / len(volumes)) if volumes else volume

    rvol = (volume / avg_volume) if avg_volume > 0 else 1.0
    float_shares = scanner.get_float(symbol)

    criteria_check = {
        "price_in_range": CRITERIA['min_price'] <= price <= CRITERIA['max_price'],
        "volume_ok": volume >= CRITERIA['min_volume'],
        "spread_ok": spread_pct <= CRITERIA['max_spread_pct'],
        "float_ok": float_shares <= CRITERIA['max_float'] if float_shares > 0 else True,
        "momentum_ok": abs(change_pct) >= CRITERIA['min_momentum_pct'],
        "rvol_ok": rvol >= CRITERIA['min_rvol'],
    }

    all_criteria_met = all(criteria_check.values())

    score = 0
    score += min(30, abs(change_pct) * 0.6)
    score += min(25, rvol * 5)
    if float_shares > 0:
        score += min(20, (volume / float_shares) * 20)
    if 2.0 <= price <= 10.0:
        score += 15
    elif 1.0 <= price <= 15.0:
        score += 10
    if spread_pct < 0.3:
        score += 10
    elif spread_pct < 0.5:
        score += 7
    elif spread_pct < 1.0:
        score += 4

    return {
        "symbol": symbol,
        "analysis": {
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "spread_pct": round(spread_pct, 2),
            "prev_close": prev_close,
            "change_pct": round(change_pct, 2),
            "high": high,
            "low": low,
            "range_pct": round(range_pct, 2),
            "volume": volume,
            "avg_volume": avg_volume,
            "rvol": round(rvol, 2),
            "float_shares": float_shares,
            "float_rotation": round(volume / float_shares, 2) if float_shares > 0 else 0,
        },
        "criteria_check": criteria_check,
        "all_criteria_met": all_criteria_met,
        "momentum_score": round(min(100, score), 0),
        "recommendation": "QUALIFIED" if all_criteria_met and score >= 60 else "WATCH" if score >= 40 else "SKIP",
        "criteria_used": CRITERIA
    }


@router.post("/ALPACA/start")
async def alpaca_start_continuous(interval: int = 60):
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca removed. Use /api/scanner/warrior/start instead.",
        "redirect": "/api/scanner/warrior/start"
    }


@router.post("/ALPACA/stop")
async def alpaca_stop_continuous():
    """DEPRECATED: Alpaca removed - use Warrior scanner"""
    return {
        "deprecated": True,
        "message": "Alpaca removed. Use /api/scanner/warrior/stop instead.",
        "redirect": "/api/scanner/warrior/stop"
    }


# ============ Split Tracker Endpoints ============

@router.get("/split/{symbol}")
async def get_split_info(symbol: str):
    """Get split momentum index for a symbol"""
    try:
        from .split_tracker import check_split_momentum
        result = check_split_momentum(symbol.upper())
        return result
    except Exception as e:
        logger.error(f"Split check error for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "has_split": False,
            "smi_score": 0,
            "smi_signal": "UNKNOWN",
            "error": str(e)
        }


@router.get("/split/{symbol}/history")
async def get_split_history(symbol: str):
    """Get full split history for a symbol"""
    try:
        from .split_tracker import get_split_tracker
        tracker = get_split_tracker()
        splits = tracker.fetch_split_history(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "split_count": len(splits),
            "splits": [s.to_dict() for s in splits]
        }
    except Exception as e:
        logger.error(f"Split history error for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "split_count": 0,
            "splits": [],
            "error": str(e)
        }


@router.get("/splits/stats")
async def get_split_stats():
    """Get aggregate split momentum statistics"""
    try:
        from .split_tracker import get_split_tracker
        tracker = get_split_tracker()
        tracker.update_momentum_stats()
        return {
            "success": True,
            "stats": tracker.get_stats()
        }
    except Exception as e:
        logger.error(f"Split stats error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/splits/scan")
async def scan_watchlist_splits():
    """Scan watchlist for recent splits and get SMI scores"""
    try:
        from .split_tracker import get_split_tracker
        import httpx

        tracker = get_split_tracker()

        # Get watchlist
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:9100/api/worklist", timeout=5.0)
            if response.status_code != 200:
                return {"success": False, "error": "Could not fetch watchlist"}

            data = response.json()
            symbols = [item.get('symbol') for item in data.get('data', [])]

        results = []
        for symbol in symbols:
            smi = tracker.get_split_momentum_index(symbol)
            if smi.get('has_split'):
                results.append(smi)

        # Sort by days since split (most recent first)
        results.sort(key=lambda x: x.get('days_since_split', 999))

        return {
            "success": True,
            "scanned": len(symbols),
            "with_splits": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Split scan error: {e}")
        return {"success": False, "error": str(e)}


# ============ Pattern Correlation Endpoints ============

@router.post("/patterns/record/{symbol}")
async def record_mover_pattern(symbol: str):
    """Record a moving stock for pattern analysis"""
    try:
        from .pattern_correlator import get_pattern_correlator
        import httpx

        correlator = get_pattern_correlator()

        # Get quote
        async with httpx.AsyncClient() as client:
            quote_resp = await client.get(f"http://localhost:9100/api/price/{symbol.upper()}", timeout=5.0)
            quote = quote_resp.json() if quote_resp.status_code == 200 else {}

            # Get additional context
            context = {}

            # Float
            float_resp = await client.get(f"http://localhost:9100/api/stock/float/{symbol.upper()}", timeout=5.0)
            if float_resp.status_code == 200:
                float_data = float_resp.json()
                context['float'] = float_data.get('float')

            # SPY for market context
            spy_resp = await client.get("http://localhost:9100/api/price/SPY", timeout=3.0)
            if spy_resp.status_code == 200:
                spy = spy_resp.json()
                context['spy_change'] = spy.get('change_percent', 0)
                if context['spy_change'] > 0.5:
                    context['market_trend'] = 'BULLISH'
                elif context['spy_change'] < -0.5:
                    context['market_trend'] = 'BEARISH'
                else:
                    context['market_trend'] = 'NEUTRAL'

        # Record it
        record = await correlator.record_mover(symbol.upper(), quote, context)

        return {
            "success": True,
            "record_id": record.record_id,
            "symbol": symbol.upper(),
            "price": record.price,
            "change_percent": record.change_percent,
            "message": "Mover recorded for pattern analysis"
        }
    except Exception as e:
        logger.error(f"Record pattern error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/patterns/outcome/{record_id}")
async def update_pattern_outcome(record_id: str, outcome_15min: float = None,
                                  outcome_30min: float = None, outcome_1hr: float = None,
                                  outcome_eod: float = None, max_gain: float = None,
                                  max_loss: float = None):
    """Update outcome for a recorded pattern"""
    try:
        from .pattern_correlator import get_pattern_correlator
        correlator = get_pattern_correlator()

        await correlator.update_outcome(record_id, {
            "outcome_15min": outcome_15min,
            "outcome_30min": outcome_30min,
            "outcome_1hr": outcome_1hr,
            "outcome_eod": outcome_eod,
            "max_gain": max_gain,
            "max_loss": max_loss
        })

        return {"success": True, "record_id": record_id, "message": "Outcome updated"}
    except Exception as e:
        logger.error(f"Update outcome error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/patterns/correlations")
async def get_pattern_correlations():
    """Get correlation analysis between factors and outcomes"""
    try:
        from .pattern_correlator import get_pattern_correlator
        correlator = get_pattern_correlator()
        return correlator.analyze_correlations()
    except Exception as e:
        logger.error(f"Correlations error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/patterns/predict/{symbol}")
async def get_pattern_prediction(symbol: str):
    """Get prediction score based on historical patterns"""
    try:
        from .pattern_correlator import get_pattern_correlator
        import httpx

        correlator = get_pattern_correlator()

        # Get current quote
        async with httpx.AsyncClient() as client:
            quote_resp = await client.get(f"http://localhost:9100/api/price/{symbol.upper()}", timeout=5.0)
            quote = quote_resp.json() if quote_resp.status_code == 200 else {}

        # Get context from worklist if available
        context = {}
        async with httpx.AsyncClient() as client:
            worklist_resp = await client.get("http://localhost:9100/api/worklist", timeout=5.0)
            if worklist_resp.status_code == 200:
                data = worklist_resp.json()
                for item in data.get('data', []):
                    if item.get('symbol') == symbol.upper():
                        context['rel_volume'] = item.get('rel_volume', 1.0)
                        context['vwap_extension'] = item.get('vwap_extension', 0)
                        context['macd_signal'] = item.get('macd_signal', 'UNKNOWN')
                        context['news_heat'] = item.get('news_heat', 'COLD')
                        break

        return correlator.get_prediction_score(symbol.upper(), quote, context)
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        return {"symbol": symbol.upper(), "score": 50, "signal": "UNKNOWN", "error": str(e)}


@router.get("/patterns/stats")
async def get_pattern_stats():
    """Get pattern tracking statistics"""
    try:
        from .pattern_correlator import get_pattern_correlator
        correlator = get_pattern_correlator()
        return {
            "success": True,
            **correlator.get_stats()
        }
    except Exception as e:
        logger.error(f"Pattern stats error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/patterns/recent")
async def get_recent_patterns(limit: int = 20):
    """Get recent recorded patterns"""
    try:
        from .pattern_correlator import get_pattern_correlator
        correlator = get_pattern_correlator()
        return {
            "success": True,
            "records": correlator.get_recent_records(limit)
        }
    except Exception as e:
        logger.error(f"Recent patterns error: {e}")
        return {"success": False, "records": [], "error": str(e)}


@router.post("/patterns/record-watchlist")
async def record_watchlist_movers():
    """Record all current watchlist items that are moving significantly"""
    try:
        from .pattern_correlator import get_pattern_correlator
        import httpx

        correlator = get_pattern_correlator()
        recorded = []

        async with httpx.AsyncClient() as client:
            # Get watchlist
            worklist_resp = await client.get("http://localhost:9100/api/worklist", timeout=5.0)
            if worklist_resp.status_code != 200:
                return {"success": False, "error": "Could not fetch watchlist"}

            data = worklist_resp.json()

            # Get market context
            spy_resp = await client.get("http://localhost:9100/api/price/SPY", timeout=3.0)
            spy_change = 0
            market_trend = "NEUTRAL"
            if spy_resp.status_code == 200:
                spy = spy_resp.json()
                spy_change = spy.get('change_percent', 0)
                if spy_change > 0.5:
                    market_trend = 'BULLISH'
                elif spy_change < -0.5:
                    market_trend = 'BEARISH'

            # Record movers (>10% change)
            for item in data.get('data', []):
                change_pct = item.get('change_percent', 0)
                if abs(change_pct) >= 10:  # Only record significant movers
                    quote = {
                        'price': item.get('price'),
                        'change_percent': change_pct,
                        'volume': item.get('volume'),
                        'bid': item.get('bid'),
                        'ask': item.get('ask'),
                        'high': item.get('high'),
                        'low': item.get('low')
                    }
                    context = {
                        'float': item.get('float'),
                        'rel_volume': item.get('rel_volume', 1.0),
                        'vwap': item.get('vwap'),
                        'vwap_extension': item.get('vwap_extension', 0),
                        'macd_signal': item.get('macd_signal', 'UNKNOWN'),
                        'has_news': item.get('has_news', False),
                        'news_heat': item.get('news_heat', 'COLD'),
                        'spy_change': spy_change,
                        'market_trend': market_trend
                    }

                    record = await correlator.record_mover(item['symbol'], quote, context)
                    recorded.append({
                        "symbol": item['symbol'],
                        "record_id": record.record_id,
                        "change_percent": change_pct
                    })

        return {
            "success": True,
            "recorded_count": len(recorded),
            "recorded": recorded
        }
    except Exception as e:
        logger.error(f"Record watchlist error: {e}")
        return {"success": False, "error": str(e)}


# ============ HFT Scalper Endpoints ============

class ScalperConfigUpdate(BaseModel):
    """Request model for scalper config updates"""
    enabled: Optional[bool] = None
    paper_mode: Optional[bool] = None
    account_size: Optional[float] = None
    risk_percent: Optional[float] = None
    use_risk_based_sizing: Optional[bool] = None
    max_position_size: Optional[float] = None
    max_shares: Optional[int] = None
    min_shares: Optional[int] = None
    min_spike_percent: Optional[float] = None
    min_volume_surge: Optional[float] = None
    max_spread_percent: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    profit_target_percent: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    trailing_stop_percent: Optional[float] = None
    max_hold_seconds: Optional[int] = None
    max_daily_loss: Optional[float] = None
    max_daily_trades: Optional[int] = None
    cooldown_after_loss: Optional[int] = None
    # AI filter settings
    use_chronos_filter: Optional[bool] = None
    chronos_min_prob_up: Optional[float] = None
    use_order_flow_filter: Optional[bool] = None
    min_buy_pressure: Optional[float] = None
    use_regime_gating: Optional[bool] = None
    valid_regimes: Optional[list] = None
    use_scalp_fade_filter: Optional[bool] = None
    use_pullback_confirmation: Optional[bool] = None


@router.get("/scalper/status")
async def get_scalper_status():
    """Get HFT scalper status"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return scalper.get_status()
    except Exception as e:
        logger.error(f"Scalper status error: {e}")
        return {"is_running": False, "enabled": False, "error": str(e)}


@router.post("/scalper/start")
async def start_scalper(symbols: str = None):
    """Start the HFT scalper

    Args:
        symbols: Comma-separated list of symbols (e.g., "SPY,QQQ,AAPL")
    """
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()

        # Parse symbols from comma-separated string
        symbol_list = []
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

        # Get watchlist if no symbols provided
        if not symbol_list:
            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get("http://localhost:9100/api/worklist", timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Handle both list response and wrapped response
                        if isinstance(data, list):
                            symbol_list = [item.get('symbol') for item in data if item.get('symbol')]
                        elif isinstance(data, dict) and 'data' in data:
                            symbol_list = [item.get('symbol') for item in data['data'] if item.get('symbol')]
                except Exception as e:
                    logger.warning(f"Could not fetch worklist: {e}")

        # Use default symbols if still empty
        if not symbol_list:
            symbol_list = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
            logger.info(f"Using default symbols: {symbol_list}")

        scalper.start(symbol_list)

        return {
            "status": "started",
            "is_running": scalper.is_running,
            "enabled": scalper.config.enabled,
            "paper_mode": scalper.config.paper_mode,
            "watchlist_count": len(scalper.config.watchlist),
            "symbols": symbol_list[:10],  # Show first 10 symbols
            "message": "Scalper monitoring started (set enabled=true to trade)"
        }
    except Exception as e:
        logger.error(f"Start scalper error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


@router.post("/scalper/stop")
async def stop_scalper():
    """Stop the HFT scalper"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper.stop()
        return {
            "status": "stopped",
            "is_running": False,
            "open_positions": len(scalper.open_positions)
        }
    except Exception as e:
        logger.error(f"Stop scalper error: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/scalper/enable")
async def enable_scalper(paper_mode: bool = True):
    """Enable scalper trading (paper or live)"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper.update_config(enabled=True, paper_mode=paper_mode)

        mode = "PAPER" if paper_mode else "LIVE"
        logger.warning(f"HFT SCALPER ENABLED - {mode} MODE")

        return {
            "status": "enabled",
            "enabled": True,
            "paper_mode": paper_mode,
            "mode": mode,
            "warning": "LIVE trading active!" if not paper_mode else "Paper trading mode"
        }
    except Exception as e:
        logger.error(f"Enable scalper error: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/scalper/disable")
async def disable_scalper():
    """Disable scalper trading"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper.update_config(enabled=False)
        return {"status": "disabled", "enabled": False}
    except Exception as e:
        logger.error(f"Disable scalper error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/scalper/config")
async def get_scalper_config():
    """Get scalper configuration"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return {
            "success": True,
            "config": scalper.config.to_dict()
        }
    except Exception as e:
        logger.error(f"Get config error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/scalper/config")
async def update_scalper_config(request: ScalperConfigUpdate):
    """Update scalper configuration"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()

        updates = {k: v for k, v in request.dict().items() if v is not None}
        scalper.update_config(**updates)

        return {
            "success": True,
            "updated": updates,
            "config": scalper.config.to_dict()
        }
    except Exception as e:
        logger.error(f"Update config error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/scalper/positions")
async def get_scalper_positions():
    """Get open scalper positions"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return {
            "count": len(scalper.open_positions),
            "positions": scalper.get_open_positions()
        }
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return {"count": 0, "positions": [], "error": str(e)}


@router.get("/scalper/trades")
async def get_scalper_trades(limit: int = 50):
    """Get scalper trade history"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        trades = scalper.get_trade_history(limit)
        return {
            "count": len(trades),
            "trades": trades
        }
    except Exception as e:
        logger.error(f"Get trades error: {e}")
        return {"count": 0, "trades": [], "error": str(e)}


@router.get("/scalper/stats")
async def get_scalper_stats():
    """Get scalper trading statistics"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return scalper.get_stats()
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        return {"message": "Error getting stats", "error": str(e)}


@router.post("/scalper/watchlist/add/{symbol}")
async def add_to_scalper_watchlist(symbol: str):
    """Add symbol to scalper watchlist"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper.add_to_watchlist(symbol)
        return {
            "status": "added",
            "symbol": symbol.upper(),
            "watchlist_count": len(scalper.config.watchlist)
        }
    except Exception as e:
        logger.error(f"Add to watchlist error: {e}")
        return {"status": "error", "error": str(e)}


@router.delete("/scalper/watchlist/{symbol}")
async def remove_from_scalper_watchlist(symbol: str):
    """Remove symbol from scalper watchlist"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        scalper.remove_from_watchlist(symbol)
        return {
            "status": "removed",
            "symbol": symbol.upper(),
            "watchlist_count": len(scalper.config.watchlist)
        }
    except Exception as e:
        logger.error(f"Remove from watchlist error: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/scalper/watchlist")
async def get_scalper_watchlist():
    """Get scalper watchlist"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return {
            "count": len(scalper.config.watchlist),
            "symbols": scalper.config.watchlist
        }
    except Exception as e:
        logger.error(f"Get watchlist error: {e}")
        return {"count": 0, "symbols": [], "error": str(e)}


@router.post("/scalper/reset")
async def reset_scalper_daily():
    """Reset daily stats for fresh start"""
    try:
        from .hft_scalper import get_hft_scalper
        scalper = get_hft_scalper()
        return scalper.reset_daily()
    except Exception as e:
        logger.error(f"Reset daily error: {e}")
        return {"reset": False, "error": str(e)}


# ============ Trade Correlation Report Endpoints ============

@router.get("/scalper/correlation-report")
async def get_correlation_report():
    """
    Get trade correlation report analyzing which secondary triggers predict winners.

    Returns analysis of:
    - Time of day patterns
    - Day of week patterns
    - Entry/exit signal performance
    - Technical indicator correlations
    - Symbol performance
    - Actionable recommendations
    """
    try:
        from .correlation_report import generate_correlation_report
        report = generate_correlation_report()
        return {
            "success": True,
            **report
        }
    except Exception as e:
        logger.error(f"Correlation report error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/scalper/correlation-report/text")
async def get_correlation_report_text():
    """Get correlation report as formatted text"""
    try:
        from .correlation_report import generate_correlation_report, print_report
        import io
        import sys

        report = generate_correlation_report()

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        print_report(report)
        text_output = buffer.getvalue()
        sys.stdout = old_stdout

        return {
            "success": True,
            "text": text_output
        }
    except Exception as e:
        logger.error(f"Correlation report text error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/scalper/correlation-report/recommendations")
async def get_correlation_recommendations():
    """Get just the recommendations from correlation analysis"""
    try:
        from .correlation_report import generate_correlation_report
        report = generate_correlation_report()
        return {
            "success": True,
            "summary": report.get('summary', {}),
            "recommendations": report.get('recommendations', [])
        }
    except Exception as e:
        logger.error(f"Correlation recommendations error: {e}")
        return {"success": False, "error": str(e)}


# ============ Order Flow Analysis Endpoints ============

@router.get("/order-flow/{symbol}")
async def get_order_flow(symbol: str):
    """
    Get order flow analysis for a symbol.

    Returns bid/ask imbalance, buy pressure, and entry recommendation.
    """
    try:
        from .order_flow_analyzer import get_order_flow_signal
        signal = await get_order_flow_signal(symbol.upper())
        return {
            "success": True,
            **signal.to_dict()
        }
    except Exception as e:
        logger.error(f"Order flow error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/order-flow/batch/{symbols}")
async def get_order_flow_batch(symbols: str):
    """
    Get order flow analysis for multiple symbols (comma-separated).

    Example: /api/scanner/order-flow/batch/AAPL,TSLA,NVDA
    """
    try:
        from .order_flow_analyzer import get_order_flow_signal
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        results = {}
        for symbol in symbol_list[:10]:  # Limit to 10 symbols
            signal = await get_order_flow_signal(symbol)
            results[symbol] = signal.to_dict()

        return {
            "success": True,
            "count": len(results),
            "signals": results
        }
    except Exception as e:
        logger.error(f"Order flow batch error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/order-flow/summary")
async def get_order_flow_summary():
    """Get order flow summary for all tracked symbols"""
    try:
        from .order_flow_analyzer import get_order_flow_analyzer
        analyzer = get_order_flow_analyzer()

        summaries = []
        for symbol in analyzer.history.keys():
            summaries.append(analyzer.get_summary(symbol))

        # Sort by buy pressure
        summaries.sort(key=lambda x: x.get('avg_buy_pressure', 0), reverse=True)

        return {
            "success": True,
            "count": len(summaries),
            "summaries": summaries
        }
    except Exception as e:
        logger.error(f"Order flow summary error: {e}")
        return {"success": False, "error": str(e)}


# ============ Borrow Status Endpoints ============

@router.get("/borrow-status/{symbol}")
async def get_borrow_status(symbol: str):
    """
    Get borrow status (ETB/HTB) for a symbol.

    Returns short interest data and borrow classification.
    """
    try:
        from .borrow_status import get_borrow_status as get_status
        status = await get_status(symbol.upper())
        return {
            "success": True,
            **status
        }
    except Exception as e:
        logger.error(f"Borrow status error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/borrow-status/batch/{symbols}")
async def get_borrow_status_batch(symbols: str):
    """
    Get borrow status for multiple symbols (comma-separated).

    Example: /api/scanner/borrow-status/batch/AAPL,GME,SOUN
    """
    try:
        from .borrow_status import get_borrow_tracker
        tracker = get_borrow_tracker()
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        results = await tracker.get_batch_status(symbol_list[:10])

        return {
            "success": True,
            "count": len(results),
            "statuses": {s: info.to_dict() for s, info in results.items()}
        }
    except Exception as e:
        logger.error(f"Borrow status batch error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/borrow-status/htb-list")
async def get_htb_list():
    """Get list of HTB (Hard to Borrow) symbols from cache"""
    try:
        from .borrow_status import get_borrow_tracker
        tracker = get_borrow_tracker()
        htb_symbols = tracker.get_htb_symbols()
        return {
            "success": True,
            "count": len(htb_symbols),
            "htb_symbols": htb_symbols
        }
    except Exception as e:
        logger.error(f"HTB list error: {e}")
        return {"success": False, "error": str(e)}


# ============ Backtest Validation Endpoints ============

@router.get("/backtest/validate")
async def validate_scalper_params(symbols: str = "SOUN,AAPL,TSLA", days: int = 30):
    """
    Validate current scalper parameters using walkforward analysis.

    Tests parameters on historical data to detect overfitting.
    """
    try:
        from .pybroker_walkforward import validate_current_params
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        report = validate_current_params(symbols=symbol_list, days=days)
        return {
            "success": True,
            **report
        }
    except Exception as e:
        logger.error(f"Backtest validation error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/backtest/walkforward")
async def run_walkforward(symbols: str = "SPY", days: int = 60):
    """
    Run walkforward analysis on specified symbols.
    """
    try:
        from .pybroker_walkforward import run_walkforward_test
        from datetime import datetime, timedelta
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        result = run_walkforward_test(
            symbols=symbol_list,
            start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            initial_cash=1000.0
        )
        return {
            "success": result.get('success', False),
            "metrics": result.get('metrics', {}),
            "note": result.get('note', '')
        }
    except Exception as e:
        logger.error(f"Walkforward error: {e}")
        return {"success": False, "error": str(e)}


# ============ ATR Stops Endpoints ============

@router.get("/atr-stops/{symbol}")
async def get_atr_stops(symbol: str, entry_price: float = 0):
    """
    Get ATR-based dynamic stop levels for a symbol.

    If entry_price is 0, uses current market price.
    """
    try:
        from .atr_stops import get_atr_calculator
        calculator = get_atr_calculator()

        if entry_price <= 0:
            # Get current price
            from schwab_market_data import get_schwab_market_data
            schwab = get_schwab_market_data()
            quote = schwab.get_quote(symbol.upper())
            if quote:
                entry_price = quote.get('last', 0) or quote.get('price', 0)

        if entry_price <= 0:
            return {"success": False, "error": "Could not determine entry price"}

        stops = calculator.calculate_stops(symbol.upper(), entry_price)
        if stops:
            return {
                "success": True,
                **stops.to_dict()
            }
        else:
            return {"success": False, "error": "ATR calculation failed"}

    except Exception as e:
        logger.error(f"ATR stops error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============ News Auto-Trader Endpoints ============

@router.get("/news-trader/status")
async def get_news_trader_status():
    """Get news auto-trader status and configuration"""
    try:
        from .news_auto_trader import get_news_auto_trader
        trader = get_news_auto_trader()
        return {"success": True, **trader.get_status()}
    except Exception as e:
        logger.error(f"News trader status error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/news-trader/start")
async def start_news_trader(paper_mode: bool = True):
    """
    Start the news auto-trader.

    Args:
        paper_mode: If True, trades are simulated (default True for safety)
    """
    try:
        from .news_auto_trader import start_news_auto_trader
        result = await start_news_auto_trader(paper_mode=paper_mode)
        return {"success": True, "message": "News auto-trader started", **result}
    except Exception as e:
        logger.error(f"News trader start error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/news-trader/stop")
async def stop_news_trader():
    """Stop the news auto-trader"""
    try:
        from .news_auto_trader import stop_news_auto_trader
        result = await stop_news_auto_trader()
        return {"success": True, "message": "News auto-trader stopped", **result}
    except Exception as e:
        logger.error(f"News trader stop error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/news-trader/config")
async def get_news_trader_config():
    """Get news auto-trader configuration"""
    try:
        from .news_auto_trader import get_news_auto_trader
        trader = get_news_auto_trader()
        return {"success": True, "config": trader.config.to_dict()}
    except Exception as e:
        logger.error(f"News trader config error: {e}")
        return {"success": False, "error": str(e)}


class NewsTraderConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    paper_mode: Optional[bool] = None
    min_news_confidence: Optional[float] = None
    min_news_urgency: Optional[str] = None
    require_chronos: Optional[bool] = None
    min_chronos_score: Optional[float] = None
    require_qlib: Optional[bool] = None
    min_qlib_score: Optional[float] = None
    require_order_flow: Optional[bool] = None
    min_buy_pressure: Optional[float] = None
    max_concurrent_trades: Optional[int] = None
    symbol_cooldown_minutes: Optional[int] = None
    max_daily_trades: Optional[int] = None
    auto_add_to_watchlist: Optional[bool] = None


@router.post("/news-trader/config")
async def update_news_trader_config(update: NewsTraderConfigUpdate):
    """Update news auto-trader configuration"""
    try:
        from .news_auto_trader import get_news_auto_trader
        trader = get_news_auto_trader()

        # Only update fields that were provided
        updates = {k: v for k, v in update.dict().items() if v is not None}
        new_config = trader.update_config(**updates)

        return {"success": True, "config": new_config}
    except Exception as e:
        logger.error(f"News trader config update error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/news-trader/candidates")
async def get_news_trader_candidates(limit: int = 50):
    """Get recent trade candidates evaluated by the news auto-trader"""
    try:
        from .news_auto_trader import get_news_auto_trader
        trader = get_news_auto_trader()
        return {"success": True, "candidates": trader.get_candidates(limit)}
    except Exception as e:
        logger.error(f"News trader candidates error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/news-trader/trades")
async def get_news_trader_trades(limit: int = 20):
    """Get trades executed by the news auto-trader"""
    try:
        from .news_auto_trader import get_news_auto_trader
        trader = get_news_auto_trader()
        return {"success": True, "trades": trader.get_executed_trades(limit)}
    except Exception as e:
        logger.error(f"News trader trades error: {e}")
        return {"success": False, "error": str(e)}


# ============ MACD Analyzer Endpoints ============

@router.get("/macd/{symbol}")
async def get_macd_analysis(symbol: str):
    """Get MACD analysis for a symbol"""
    try:
        from .macd_analyzer import analyze_macd
        result = analyze_macd(symbol.upper())
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"MACD analysis error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/macd/phase2/{symbol}")
async def check_macd_phase2_entry(symbol: str):
    """Check if MACD conditions favor Phase 2 entry"""
    try:
        from .macd_analyzer import check_phase2_conditions
        result = check_phase2_conditions(symbol.upper())
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"MACD Phase 2 check error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.get("/macd/batch/{symbols}")
async def get_macd_batch(symbols: str):
    """Get MACD analysis for multiple symbols (comma-separated)"""
    try:
        from .macd_analyzer import get_macd_analyzer
        analyzer = get_macd_analyzer()

        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        results = {}

        for sym in symbol_list[:10]:  # Limit to 10 symbols
            signal = analyzer.analyze(sym)
            if signal:
                results[sym] = signal.to_dict()
            else:
                results[sym] = {"error": "Could not analyze"}

        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"MACD batch error: {e}")
        return {"success": False, "error": str(e)}


# ============ Two-Phase Strategy Endpoints ============

@router.get("/two-phase/status")
async def get_two_phase_status():
    """Get status of all two-phase strategy positions"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        positions = {}
        for symbol, pos in strategy.positions.items():
            positions[symbol] = {
                "phase": pos.phase.value,
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "target1": pos.target1_price,
                "high_since_entry": pos.high_since_entry,
                "phase2_entry": pos.phase2_entry,
                "macd_signal": pos.macd_signal,
                "momentum_score": pos.momentum_score
            }

        return {
            "success": True,
            "active_positions": len(positions),
            "positions": positions
        }
    except Exception as e:
        logger.error(f"Two-phase status error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/two-phase/evaluate/{symbol}")
async def evaluate_phase2_opportunity(symbol: str, current_price: float = 0.0, volume_ratio: float = 1.0):
    """Evaluate Phase 2 entry opportunity for a symbol"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        # If no price provided, try to get it
        if current_price <= 0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol.upper())
                current_price = ticker.fast_info.get('lastPrice', 0) or ticker.history(period='1d')['Close'].iloc[-1]
            except:
                pass

        result = strategy.evaluate_phase2_opportunity(symbol.upper(), current_price, volume_ratio)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Two-phase evaluate error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/two-phase/spike/{symbol}")
async def register_spike(symbol: str, spike_high: float, volume_surge: float = 3.0):
    """Register a volume spike to start tracking for Phase 1"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        result = strategy.on_spike_detected(symbol.upper(), spike_high, volume_surge)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Two-phase spike registration error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/two-phase/pullback/{symbol}")
async def register_pullback(symbol: str, pullback_low: float):
    """Register a pullback to prepare for entry"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        result = strategy.on_pullback(symbol.upper(), pullback_low)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Two-phase pullback error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/two-phase/check-entry/{symbol}")
async def check_phase1_entry(symbol: str, current_high: float, last_candle_low: float):
    """Check if Phase 1 entry conditions are met"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        result = strategy.check_entry(symbol.upper(), current_high, last_candle_low)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Two-phase entry check error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/two-phase/check-exit/{symbol}")
async def check_phase_exit(symbol: str, current_price: float, current_high: float):
    """Check exit conditions for active position"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        # Check regular exit
        result = strategy.check_exit(symbol.upper(), current_price, current_high)

        # Also check MACD exit for Phase 2
        if result.get("action") == "HOLD":
            macd_result = strategy.check_phase2_exit_macd(symbol.upper(), current_price)
            if macd_result.get("action") == "SELL":
                result = macd_result

        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Two-phase exit check error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.delete("/two-phase/{symbol}")
async def remove_two_phase_position(symbol: str):
    """Remove a symbol from two-phase tracking"""
    try:
        from .two_phase_strategy import get_two_phase_strategy
        strategy = get_two_phase_strategy()

        if symbol.upper() in strategy.positions:
            del strategy.positions[symbol.upper()]
            return {"success": True, "message": f"Removed {symbol} from tracking"}
        else:
            return {"success": False, "message": f"{symbol} not being tracked"}
    except Exception as e:
        logger.error(f"Two-phase remove error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============ Phase 2 Manager Endpoints ============

@router.get("/phase2/status")
async def get_phase2_status():
    """Get Phase 2 continuation manager status"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        return {"success": True, "data": manager.get_status()}
    except Exception as e:
        logger.error(f"Phase 2 status error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/phase2/start")
async def start_phase2_monitoring():
    """Start Phase 2 monitoring loop"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        result = manager.start_monitoring()
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Phase 2 start error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/phase2/stop")
async def stop_phase2_monitoring():
    """Stop Phase 2 monitoring loop"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        result = manager.stop_monitoring()
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Phase 2 stop error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/phase2/config")
async def get_phase2_config():
    """Get Phase 2 manager configuration"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        return {"success": True, "config": manager.get_config()}
    except Exception as e:
        logger.error(f"Phase 2 config error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/phase2/config")
async def update_phase2_config(config: Dict):
    """Update Phase 2 manager configuration"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        result = manager.update_config(config)
        return {"success": True, "config": result}
    except Exception as e:
        logger.error(f"Phase 2 config update error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/phase2/register/{symbol}")
async def register_for_phase2(symbol: str, exit_price: float, pnl_pct: float):
    """Manually register a symbol for Phase 2 monitoring"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        result = manager.register_phase1_exit(symbol.upper(), exit_price, pnl_pct)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Phase 2 register error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/phase2/check/{symbol}")
async def check_phase2_conditions(symbol: str):
    """Check Phase 2 entry conditions for a symbol"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        result = await manager.check_conditions(symbol.upper())
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Phase 2 check error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/phase2/trades")
async def get_phase2_trades():
    """Get executed Phase 2 trades"""
    try:
        from .phase2_manager import get_phase2_manager
        manager = get_phase2_manager()
        return {"success": True, "trades": manager.executed_trades}
    except Exception as e:
        logger.error(f"Phase 2 trades error: {e}")
        return {"success": False, "error": str(e)}


# ============ Chronos Exit Manager Endpoints ============

@router.get("/chronos-exit/status")
async def get_chronos_exit_status():
    """
    Get Chronos Exit Manager status.

    Shows all monitored positions with their current regime,
    confidence, and momentum status.
    """
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()

        return {
            "success": True,
            "positions": manager.get_position_contexts(),
            "config": manager.get_config()
        }
    except Exception as e:
        logger.error(f"Chronos exit status error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/chronos-exit/config")
async def get_chronos_exit_config():
    """Get Chronos Exit Manager configuration"""
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()
        return {"success": True, "config": manager.get_config()}
    except Exception as e:
        logger.error(f"Chronos exit config error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/chronos-exit/config")
async def update_chronos_exit_config(config: Dict):
    """
    Update Chronos Exit Manager configuration.

    Key settings:
    - use_failed_momentum_exit: Enable Ross Cameron rule
    - momentum_check_seconds: Time before checking momentum (default 30s)
    - expected_gain_30s: Expected gain in that time (default 0.5%)
    - fade_from_high_percent: Exit if price fades this much from high
    """
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()
        manager.update_config(**config)
        return {"success": True, "config": manager.get_config()}
    except Exception as e:
        logger.error(f"Chronos exit config update error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/chronos-exit/check/{symbol}")
async def check_chronos_exit(symbol: str, current_price: float, entry_price: float):
    """
    Manually check Chronos exit signal for a position.

    This is useful for testing before live trading.
    """
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()

        signal = manager.check_exit(symbol.upper(), current_price, entry_price)

        return {
            "success": True,
            "should_exit": signal.should_exit,
            "reason": signal.reason,
            "urgency": signal.urgency,
            "details": signal.details,
            "position_context": manager.get_position_contexts().get(symbol.upper(), {})
        }
    except Exception as e:
        logger.error(f"Chronos exit check error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/chronos-exit/register/{symbol}")
async def register_chronos_position(symbol: str, entry_price: float):
    """Manually register a position for Chronos monitoring"""
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()

        ctx = manager.register_position(symbol.upper(), entry_price)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "entry_regime": ctx.entry_regime,
            "entry_confidence": ctx.entry_confidence,
            "entry_trend_strength": ctx.entry_trend_strength
        }
    except Exception as e:
        logger.error(f"Chronos exit register error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.delete("/chronos-exit/{symbol}")
async def unregister_chronos_position(symbol: str):
    """Remove a position from Chronos monitoring"""
    try:
        from .chronos_exit_manager import get_chronos_exit_manager
        manager = get_chronos_exit_manager()
        manager.unregister_position(symbol.upper())
        return {"success": True, "message": f"Unregistered {symbol}"}
    except Exception as e:
        logger.error(f"Chronos exit unregister error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# ============ HOD Momentum Scanner Endpoints ============

@router.get("/hod/status")
async def get_hod_scanner_status():
    """
    Get HOD Momentum Scanner status.

    Returns tracking count, recent alerts, A/B grade stocks.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        return {
            "success": True,
            **scanner.get_status()
        }
    except Exception as e:
        logger.error(f"HOD scanner status error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/alerts")
async def get_hod_alerts(limit: int = 50):
    """Get recent HOD alerts with grades"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        data = scanner.to_dict()
        return {
            "success": True,
            "count": len(data["alerts"]),
            "alerts": data["alerts"][-limit:],
            "a_grade": data["a_grade_stocks"],
            "b_grade": data["b_grade_stocks"],
            "running": data["running_stocks"]
        }
    except Exception as e:
        logger.error(f"HOD alerts error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/a-grade")
async def get_hod_a_grade():
    """Get A-grade stocks (5/5 Ross criteria) - FULL POSITION worthy"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        return {
            "success": True,
            "grade": "A",
            "description": "5/5 Ross criteria - FULL POSITION",
            "stocks": scanner.get_a_grade_stocks()
        }
    except Exception as e:
        logger.error(f"HOD A-grade error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/b-grade")
async def get_hod_b_grade():
    """Get B-grade stocks (4/5 Ross criteria) - HALF POSITION worthy"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        return {
            "success": True,
            "grade": "B",
            "description": "4/5 Ross criteria - HALF POSITION",
            "stocks": scanner.get_b_grade_stocks()
        }
    except Exception as e:
        logger.error(f"HOD B-grade error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/running")
async def get_hod_running():
    """Get actively running stocks (multiple HOD breaks)"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        return {
            "success": True,
            "description": "Stocks with multiple HOD breaks in 5 min",
            "stocks": scanner.get_running_stocks()
        }
    except Exception as e:
        logger.error(f"HOD running error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/grade/{symbol}")
async def grade_symbol(symbol: str, has_news: bool = False):
    """
    Grade a symbol based on Ross Cameron's 5 criteria.

    Returns grade (A/B/C) and criteria breakdown.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        result = scanner.grade_symbol(symbol.upper(), has_news)
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"HOD grade error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/hod/add/{symbol}")
async def add_to_hod_tracker(symbol: str):
    """Add a symbol to HOD tracking"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        import httpx

        scanner = get_hod_scanner()

        # Get quote data
        async with httpx.AsyncClient() as client:
            quote_resp = await client.get(f"http://localhost:9100/api/price/{symbol.upper()}", timeout=5.0)
            quote = quote_resp.json() if quote_resp.status_code == 200 else {}

            # Get float
            float_data = {}
            try:
                float_resp = await client.get(f"http://localhost:9100/api/stock/float/{symbol.upper()}", timeout=5.0)
                if float_resp.status_code == 200:
                    float_data = float_resp.json()
            except:
                pass

        # Prepare data
        data = {
            'price': quote.get('price', quote.get('last', 0)),
            'open': quote.get('open', 0),
            'high': quote.get('high', 0),
            'low': quote.get('low', 0),
            'previous_close': quote.get('prev_close', quote.get('close', 0)),
            'volume': quote.get('volume', 0),
            'avg_volume': quote.get('avg_volume', 0),
            'float_shares': float_data.get('float', 0)
        }

        scanner.add_symbol(symbol.upper(), data)

        return {
            "success": True,
            "symbol": symbol.upper(),
            "tracking_count": len(scanner.trackers),
            "data_loaded": data
        }
    except Exception as e:
        logger.error(f"HOD add error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.delete("/hod/{symbol}")
async def remove_from_hod_tracker(symbol: str):
    """Remove a symbol from HOD tracking"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        scanner.remove_symbol(symbol.upper())
        return {
            "success": True,
            "symbol": symbol.upper(),
            "tracking_count": len(scanner.trackers)
        }
    except Exception as e:
        logger.error(f"HOD remove error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/hod/update/{symbol}")
async def update_hod_price(symbol: str, price: float, volume: int = 0, has_news: bool = False):
    """
    Update price for a symbol and check for HOD break.

    Returns HOD alert if new high detected.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()

        alert = scanner.update_price(symbol.upper(), price, volume)

        if alert:
            return {
                "success": True,
                "hod_alert": True,
                "symbol": alert.symbol,
                "price": alert.price,
                "change_pct": alert.change_pct,
                "grade": alert.grade,
                "criteria_met": alert.criteria_met,
                "strategy": alert.strategy_name
            }
        else:
            return {
                "success": True,
                "hod_alert": False,
                "symbol": symbol.upper(),
                "price": price
            }
    except Exception as e:
        logger.error(f"HOD update error for {symbol}: {e}")
        return {"success": False, "error": str(e)}


@router.post("/hod/sync-watchlist")
async def sync_hod_from_watchlist():
    """
    Sync HOD scanner with current watchlist.

    Loads all watchlist symbols into HOD tracker with their data.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        import httpx

        scanner = get_hod_scanner()
        added = []

        async with httpx.AsyncClient() as client:
            # Get watchlist
            resp = await client.get("http://localhost:9100/api/worklist", timeout=10.0)
            if resp.status_code != 200:
                return {"success": False, "error": "Could not fetch watchlist"}

            data = resp.json()

            for item in data.get('data', []):
                symbol = item.get('symbol')
                if symbol:
                    scanner.add_symbol(symbol, {
                        'price': item.get('price', 0),
                        'open': item.get('open', 0),
                        'high': item.get('high', 0),
                        'low': item.get('low', 0),
                        'previous_close': item.get('prev_close', item.get('close', 0)),
                        'volume': item.get('volume', 0),
                        'avg_volume': item.get('avg_volume', 0),
                        'float_shares': item.get('float', 0)
                    })
                    added.append(symbol)

        return {
            "success": True,
            "synced_count": len(added),
            "symbols": added,
            "tracking_count": len(scanner.trackers)
        }
    except Exception as e:
        logger.error(f"HOD sync error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/config")
async def get_hod_config():
    """Get HOD scanner configuration (Ross Cameron's 5 criteria thresholds)"""
    try:
        from .hod_momentum_scanner import get_hod_scanner
        scanner = get_hod_scanner()
        return {
            "success": True,
            "ross_criteria": {
                "description": "Ross Cameron's 5 Indicators of High Demand & Low Supply",
                "1_rvol": f">= {scanner.min_relative_volume}x (5x Relative Volume)",
                "2_change": f">= {scanner.min_change_pct}% (Already up 10%)",
                "3_news": "News Event catalyst (checked separately)",
                "4_price": f"${scanner.min_price} - ${scanner.max_price} (Price range)",
                "5_float": f"< {scanner.max_float/1_000_000:.0f}M shares (Low float)"
            },
            "grading": {
                "A": "5/5 criteria → FULL POSITION",
                "B": "4/5 criteria → HALF POSITION",
                "C": "3 or less → SCALP ONLY"
            },
            "filters": {
                "min_price": scanner.min_price,
                "max_price": scanner.max_price,
                "min_change_pct": scanner.min_change_pct,
                "min_relative_volume": scanner.min_relative_volume,
                "max_float": scanner.max_float,
                "alert_cooldown_seconds": scanner.alert_cooldown_seconds
            }
        }
    except Exception as e:
        logger.error(f"HOD config error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/hod/dtd-format")
async def get_hod_dtd_format(limit: int = 50):
    """
    Get HOD scanner data in Day Trade Dash format.

    Columns match DTD "Small Cap - High of Day Momentum" scanner:
    - Time (with seconds)
    - Symbol/News
    - Price
    - Volume (K/M format)
    - Float (M format)
    - Relative Volume (Daily Rate)
    - Relative Volume (5 min %)
    - Gap(%)
    - Change From Close(%)
    - Short Interest
    - Strategy Name
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        from datetime import datetime

        scanner = get_hod_scanner()

        def format_volume(vol):
            """Format volume as K/M"""
            if vol >= 1_000_000:
                return f"{vol/1_000_000:.2f}M"
            elif vol >= 1_000:
                return f"{vol/1_000:.2f}K"
            return str(vol)

        def format_float(flt):
            """Format float as M"""
            if flt > 0:
                return f"{flt/1_000_000:.2f}M"
            return "N/A"

        def format_short_interest(si):
            """Format short interest as K"""
            if si > 0:
                return f"{si/1_000:.2f}K"
            return "N/A"

        # Get alerts and format in DTD style
        alerts = scanner.alerts[-limit:]
        dtd_rows = []

        for alert in reversed(alerts):  # Most recent first
            # Calculate 5-min relative volume (estimate from alert count)
            rvol_5min = alert.alert_count_5min * 100 if alert.alert_count_5min else 0

            row = {
                "time": alert.timestamp.strftime("%I:%M:%S %p").lower(),
                "symbol": alert.symbol,
                "has_news": alert.has_news,
                "price": round(alert.price, 2),
                "volume": format_volume(alert.volume),
                "volume_raw": alert.volume,
                "float": format_float(alert.float_shares),
                "float_raw": alert.float_shares,
                "rel_vol_daily": round(alert.relative_volume, 2),
                "rel_vol_5min_pct": round(rvol_5min, 2),
                "gap_pct": round(alert.gap_pct, 2),
                "change_from_close_pct": round(alert.change_pct, 2),
                "short_interest": format_short_interest(alert.short_interest),
                "short_interest_raw": alert.short_interest,
                "strategy_name": alert.strategy_name,
                "grade": alert.grade,
                "criteria_met": alert.criteria_met
            }
            dtd_rows.append(row)

        # Also get current tracker data for live view
        live_data = []
        for symbol, tracker in scanner.trackers.items():
            if tracker.current_price > 0 and tracker.change_pct >= 5.0:
                # Determine strategy for live data
                strategy = scanner._classify_strategy(tracker)
                grade, criteria_met, _ = scanner._grade_stock(tracker)

                live_row = {
                    "time": datetime.now().strftime("%I:%M:%S %p").lower(),
                    "symbol": symbol,
                    "has_news": False,
                    "price": round(tracker.current_price, 2),
                    "volume": format_volume(tracker.volume),
                    "volume_raw": tracker.volume,
                    "float": format_float(tracker.float_shares),
                    "float_raw": tracker.float_shares,
                    "rel_vol_daily": round(tracker.relative_volume, 2),
                    "rel_vol_5min_pct": round(tracker.hod_count_in_window(5) * 100, 2),
                    "gap_pct": round(tracker.gap_pct, 2),
                    "change_from_close_pct": round(tracker.change_pct, 2),
                    "short_interest": format_short_interest(tracker.short_interest),
                    "short_interest_raw": tracker.short_interest,
                    "strategy_name": strategy,
                    "grade": grade,
                    "criteria_met": criteria_met
                }
                live_data.append(live_row)

        # Sort live data by change % descending
        live_data.sort(key=lambda x: x["change_from_close_pct"], reverse=True)

        return {
            "success": True,
            "format": "Day Trade Dash",
            "scanner_name": "Small Cap - High of Day Momentum",
            "columns": [
                "Time", "Symbol/News", "Price", "Volume", "Float",
                "Rel Vol (Daily)", "Rel Vol (5min %)", "Gap(%)",
                "Change From Close(%)", "Short Interest", "Strategy Name"
            ],
            "alerts": dtd_rows,
            "live": live_data[:limit],
            "alert_count": len(dtd_rows),
            "live_count": len(live_data)
        }
    except Exception as e:
        logger.error(f"HOD DTD format error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.post("/hod/scan-finviz")
async def scan_hod_from_finviz():
    """
    Scan for HOD candidates using Finviz as data source.

    Fetches top gainers from Finviz and adds them to HOD tracker
    with all the data needed for DTD-style display.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        from .finviz_scanner import get_finviz_scanner

        scanner = get_hod_scanner()
        finviz = get_finviz_scanner()

        # Get top gainers from Finviz (synchronous method)
        gainers = finviz.get_top_gainers(min_change=5.0, max_price=20.0)

        added = []
        for stock in gainers[:30]:  # Top 30
            # ScanResult is a dataclass with attributes, not a dict
            symbol = stock.symbol if hasattr(stock, 'symbol') else stock.get("symbol", "")
            if not symbol:
                continue

            # Access attributes from ScanResult dataclass
            price = stock.price if hasattr(stock, 'price') else stock.get("price", 0)
            change_pct = stock.change_pct if hasattr(stock, 'change_pct') else stock.get("change_pct", 0)
            volume = stock.volume if hasattr(stock, 'volume') else stock.get("volume", 0)
            avg_volume = stock.avg_volume if hasattr(stock, 'avg_volume') else stock.get("avg_volume", 0)

            # Calculate previous close from price and change_pct
            prev_close = price / (1 + change_pct/100) if change_pct != -100 else price

            data = {
                "price": price,
                "previous_close": prev_close,
                "open": price,  # Approximate
                "high": price,  # Will be updated by live data
                "low": price * 0.95,  # Approximate
                "volume": volume,
                "avg_volume": avg_volume,
                "float_shares": 0,  # Finviz doesn't provide this in basic scan
                "short_interest": 0  # Finviz doesn't provide this in basic scan
            }

            scanner.add_symbol(symbol, data)
            added.append(symbol)

        return {
            "success": True,
            "scanned": len(gainers),
            "added_to_tracker": added,
            "tracker_total": len(scanner.trackers)
        }
    except Exception as e:
        logger.error(f"HOD Finviz scan error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/hod/scan-schwab")
async def scan_hod_from_schwab():
    """
    Scan for HOD candidates using Schwab movers API.
    Schwab provides real-time price, volume, and change data.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        import sys
        sys.path.insert(0, 'C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2')
        from schwab_market_data import get_schwab_movers, get_all_movers

        scanner = get_hod_scanner()

        # Get gainers from Schwab (SPX and NASDAQ combined)
        all_movers = get_all_movers()
        gainers = all_movers.get('gainers', [])

        # Filter for small cap criteria: $1-$20, 5%+ change
        filtered = [m for m in gainers if 1.0 <= m.get('price', 0) <= 20.0 and m.get('change_pct', 0) >= 5.0]

        added = []
        for stock in filtered[:30]:  # Top 30
            symbol = stock.get('symbol', '')
            if not symbol:
                continue

            price = stock.get('price', 0)
            change_pct = stock.get('change_pct', 0)
            volume = stock.get('volume', 0)

            # Calculate previous close from price and change_pct
            prev_close = price / (1 + change_pct/100) if change_pct != -100 else price

            data = {
                "price": price,
                "previous_close": prev_close,
                "open": price,
                "high": price,
                "low": price * 0.95,
                "volume": volume,
                "avg_volume": 0,  # Will be enriched by yfinance
                "float_shares": 0,
                "short_interest": 0
            }

            scanner.add_symbol(symbol, data)
            added.append({
                "symbol": symbol,
                "price": price,
                "change_pct": change_pct,
                "volume": volume
            })

        return {
            "success": True,
            "source": "Schwab",
            "total_gainers": len(gainers),
            "filtered": len(filtered),
            "added_to_tracker": len(added),
            "symbols": added,
            "tracker_total": len(scanner.trackers)
        }
    except Exception as e:
        logger.error(f"HOD Schwab scan error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


@router.post("/hod/enrich")
async def enrich_hod_data():
    """
    Enrich HOD tracker data with float, short interest, and avg_volume from yfinance.
    Call this after scan-finviz to get complete Day Trade Dash data.
    """
    try:
        from .hod_momentum_scanner import get_hod_scanner
        import yfinance as yf

        scanner = get_hod_scanner()

        enriched = []
        failed = []

        for symbol, tracker in list(scanner.trackers.items())[:30]:  # Limit to 30 for API rate limits
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Update tracker with enriched data
                if 'floatShares' in info and info['floatShares']:
                    tracker.float_shares = info['floatShares']

                if 'shortRatio' in info and info['shortRatio']:
                    # Convert short ratio to approximate short interest
                    tracker.short_interest = info.get('sharesShort', 0)

                if 'averageVolume' in info and info['averageVolume']:
                    tracker.avg_volume = info['averageVolume']

                enriched.append({
                    "symbol": symbol,
                    "float_shares": tracker.float_shares,
                    "short_interest": tracker.short_interest,
                    "avg_volume": tracker.avg_volume,
                    "rel_vol": tracker.relative_volume
                })

            except Exception as e:
                failed.append({"symbol": symbol, "error": str(e)})

        return {
            "success": True,
            "enriched_count": len(enriched),
            "failed_count": len(failed),
            "enriched": enriched,
            "failed": failed
        }
    except Exception as e:
        logger.error(f"HOD enrich error: {e}")
        return {"success": False, "error": str(e)}


# ============ Top Gappers Scanner Endpoints ============

@router.get("/gappers/status")
async def get_gappers_status():
    """Get Top Gappers scanner status"""
    try:
        from .top_gappers_scanner import get_gappers_scanner
        scanner = get_gappers_scanner()
        return {
            "success": True,
            **scanner.get_status()
        }
    except Exception as e:
        logger.error(f"Gappers status error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/gappers/scan")
async def run_gappers_scan():
    """
    Run Top Gappers scan.

    Scans Schwab movers, FinViz, and Yahoo gainers.
    Returns stocks graded by Ross Cameron's 5 criteria.
    """
    try:
        from .top_gappers_scanner import run_gappers_scan
        gappers = await run_gappers_scan()
        return {
            "success": True,
            "count": len(gappers),
            "gappers": [g.to_dict() for g in gappers],
            "a_grade": [g.to_dict() for g in gappers if g.grade == "A"],
            "b_grade": [g.to_dict() for g in gappers if g.grade == "B"]
        }
    except Exception as e:
        logger.error(f"Gappers scan error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/gappers/results")
async def get_gappers_results():
    """Get latest gappers scan results"""
    try:
        from .top_gappers_scanner import get_gappers_scanner
        scanner = get_gappers_scanner()
        return {
            "success": True,
            **scanner.to_dict()
        }
    except Exception as e:
        logger.error(f"Gappers results error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/gappers/a-grade")
async def get_gappers_a_grade():
    """Get A-grade gappers (5/5 Ross criteria) - FULL POSITION worthy"""
    try:
        from .top_gappers_scanner import get_gappers_scanner
        scanner = get_gappers_scanner()
        a_grade = scanner.get_a_grade()
        return {
            "success": True,
            "grade": "A",
            "description": "5/5 Ross criteria - FULL POSITION",
            "count": len(a_grade),
            "stocks": [g.to_dict() for g in a_grade]
        }
    except Exception as e:
        logger.error(f"Gappers A-grade error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/gappers/b-grade")
async def get_gappers_b_grade():
    """Get B-grade gappers (4/5 Ross criteria) - HALF POSITION worthy"""
    try:
        from .top_gappers_scanner import get_gappers_scanner
        scanner = get_gappers_scanner()
        b_grade = scanner.get_b_grade()
        return {
            "success": True,
            "grade": "B",
            "description": "4/5 Ross criteria - HALF POSITION",
            "count": len(b_grade),
            "stocks": [g.to_dict() for g in b_grade]
        }
    except Exception as e:
        logger.error(f"Gappers B-grade error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/gappers/top3")
async def get_gappers_top3():
    """
    Get top 3 leading percentage gainers.

    Ross: "I typically focus on the top 2-3 leading percentage gainers each day,
    as these will be the most obvious."
    """
    try:
        from .top_gappers_scanner import get_gappers_scanner
        scanner = get_gappers_scanner()
        top3 = scanner.get_top_3()
        return {
            "success": True,
            "description": "Top 3 leading % gainers - the most obvious plays",
            "count": len(top3),
            "stocks": [g.to_dict() for g in top3]
        }
    except Exception as e:
        logger.error(f"Gappers top3 error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/gappers/add-to-watchlist")
async def add_gappers_to_watchlist(grade: str = "A"):
    """
    Add A-grade (or B-grade) gappers to watchlist.

    Args:
        grade: "A" for A-grade only, "B" for B-grade, "AB" for both
    """
    try:
        from .top_gappers_scanner import get_gappers_scanner
        import httpx

        scanner = get_gappers_scanner()
        added = []

        # Get stocks by grade
        stocks = []
        if "A" in grade.upper():
            stocks.extend(scanner.get_a_grade())
        if "B" in grade.upper():
            stocks.extend(scanner.get_b_grade())

        async with httpx.AsyncClient() as client:
            for stock in stocks:
                # Add to dashboard watchlist
                try:
                    await client.post(
                        "http://localhost:9100/api/worklist/add",
                        json={"symbol": stock.symbol},
                        timeout=3.0
                    )
                except:
                    pass

                # Add to scalper watchlist
                try:
                    await client.post(
                        f"http://localhost:9100/api/scanner/scalper/watchlist/add/{stock.symbol}",
                        timeout=3.0
                    )
                except:
                    pass

                # Add to HOD tracker
                try:
                    await client.post(
                        f"http://localhost:9100/api/scanner/hod/add/{stock.symbol}",
                        timeout=3.0
                    )
                except:
                    pass

                added.append({
                    "symbol": stock.symbol,
                    "grade": stock.grade,
                    "change_pct": stock.change_pct
                })

        return {
            "success": True,
            "added_count": len(added),
            "added": added
        }
    except Exception as e:
        logger.error(f"Gappers add to watchlist error: {e}")
        return {"success": False, "error": str(e)}


# ============ FinViz Elite Scanner ============

@router.get("/finviz/status")
async def finviz_status():
    """Get FinViz scanner status"""
    scanner = get_finviz_scanner()
    return {
        "configured": bool(scanner.token),
        "running": is_finviz_scanner_running(),
        "last_scan": scanner.last_scan.isoformat() if scanner.last_scan else None,
        "candidates_count": len(scanner.candidates)
    }


@router.post("/finviz/start")
async def finviz_start(interval: int = 60, min_change: float = 10.0, auto_add: bool = True):
    """Start FinViz auto-scanner that auto-adds movers to worklist"""
    scanner = get_finviz_scanner()
    if not scanner.token:
        return {"success": False, "error": "FinViz Elite token not configured"}

    success = start_finviz_scanner(interval=interval, min_change=min_change, auto_add=auto_add)
    return {
        "success": success,
        "message": f"FinViz scanner started (interval={interval}s, min_change={min_change}%, auto_add={auto_add})" if success else "Scanner already running",
        "status": {
            "running": is_finviz_scanner_running(),
            "interval": interval,
            "min_change": min_change,
            "auto_add": auto_add
        }
    }


@router.post("/finviz/stop")
async def finviz_stop():
    """Stop FinViz auto-scanner"""
    success = stop_finviz_scanner()
    return {
        "success": success,
        "message": "FinViz scanner stopped",
        "running": is_finviz_scanner_running()
    }


@router.get("/finviz/movers")
async def finviz_movers(min_change: float = 5.0, max_price: float = 20.0):
    """Scan FinViz for momentum movers"""
    scanner = get_finviz_scanner()
    if not scanner.token:
        return {"error": "FinViz Elite token not configured", "candidates": []}

    try:
        movers = await scanner.scan_movers(min_change=min_change, max_price=max_price)
        return {
            "success": True,
            "count": len(movers),
            "results": [m.to_dict() for m in movers]
        }
    except Exception as e:
        logger.error(f"FinViz movers scan error: {e}")
        return {"success": False, "error": str(e), "results": []}


@router.get("/finviz/gappers")
async def finviz_gappers(min_gap: float = 5.0, max_price: float = 20.0):
    """Scan FinViz for gap plays"""
    scanner = get_finviz_scanner()
    if not scanner.token:
        return {"error": "FinViz Elite token not configured", "gappers": []}

    try:
        gappers = await scanner.scan_gappers(min_gap=min_gap, max_price=max_price)
        return {
            "success": True,
            "count": len(gappers),
            "results": [g.to_dict() for g in gappers]
        }
    except Exception as e:
        logger.error(f"FinViz gappers scan error: {e}")
        return {"success": False, "error": str(e), "results": []}


@router.get("/finviz/top-plays")
async def finviz_top_plays(limit: int = 10):
    """Get top momentum plays with news enrichment"""
    scanner = get_finviz_scanner()
    if not scanner.token:
        return {"error": "FinViz Elite token not configured", "plays": []}

    try:
        plays = await scanner.get_top_plays(limit=limit)
        return {
            "success": True,
            "count": len(plays),
            "results": [p.to_dict() for p in plays]
        }
    except Exception as e:
        logger.error(f"FinViz top plays error: {e}")
        return {"success": False, "error": str(e), "results": []}


@router.post("/finviz/sync-to-worklist")
async def finviz_sync_to_worklist(min_score: float = 15.0, max_add: int = 5):
    """Sync top FinViz plays to scalper watchlist"""
    scanner = get_finviz_scanner()
    if not scanner.token:
        return {"error": "FinViz Elite token not configured"}

    try:
        result = await scanner.sync_to_scalper_watchlist(min_score=min_score, max_add=max_add)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"FinViz sync error: {e}")
        return {"success": False, "error": str(e)}


logger.info("Scanner API routes initialized (Penny, Warrior, Split Tracker, Pattern Correlator, HFT Scalper, Correlation Report, Order Flow, Borrow Status, Backtest, ATR Stops, News Auto-Trader, MACD Analyzer, Two-Phase Strategy, Phase 2 Manager, Chronos Exit Manager, HOD Momentum Scanner, Top Gappers Scanner, FinViz Elite)")
