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
    """Get Alpaca momentum scanner status"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")
    scanner = get_momentum_scanner()
    return scanner.get_status()


@router.get("/ALPACA/presets")
async def alpaca_scanner_presets():
    """Get Alpaca scanner presets"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")
    return {
        "presets": list(SCANNER_PRESETS.keys()),
        "criteria": CRITERIA
    }


@router.get("/ALPACA/criteria")
async def alpaca_get_criteria():
    """Get current Alpaca trading criteria"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")
    return {
        "criteria": CRITERIA,
        "presets": list(SCANNER_PRESETS.keys())
    }


@router.post("/ALPACA/criteria")
async def alpaca_update_criteria(request: UpdateCriteriaRequest):
    """Update Alpaca trading criteria"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    updates = {}
    if request.min_price is not None:
        CRITERIA['min_price'] = request.min_price
        updates['min_price'] = request.min_price
    if request.max_price is not None:
        CRITERIA['max_price'] = request.max_price
        updates['max_price'] = request.max_price
    if request.min_volume is not None:
        CRITERIA['min_volume'] = request.min_volume
        updates['min_volume'] = request.min_volume
    if request.max_spread_pct is not None:
        CRITERIA['max_spread_pct'] = request.max_spread_pct
        updates['max_spread_pct'] = request.max_spread_pct
    if request.max_float is not None:
        CRITERIA['max_float'] = request.max_float
        updates['max_float'] = request.max_float
    if request.min_gap_pct is not None:
        CRITERIA['min_gap_pct'] = request.min_gap_pct
        updates['min_gap_pct'] = request.min_gap_pct
    if request.min_momentum_pct is not None:
        CRITERIA['min_momentum_pct'] = request.min_momentum_pct
        updates['min_momentum_pct'] = request.min_momentum_pct
    if request.min_rvol is not None:
        CRITERIA['min_rvol'] = request.min_rvol
        updates['min_rvol'] = request.min_rvol

    return {
        "success": True,
        "updated": updates,
        "current_criteria": CRITERIA
    }


@router.get("/ALPACA/scan")
async def alpaca_run_scan(preset: str = "warrior_momentum"):
    """Run Alpaca momentum scan with preset"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    scanner = get_momentum_scanner()

    # Apply preset if specified
    if preset in SCANNER_PRESETS:
        for key, value in SCANNER_PRESETS[preset].items():
            CRITERIA[key] = value

    result = scanner.run_full_scan()
    return result


@router.get("/ALPACA/top")
async def alpaca_get_top(limit: int = 10):
    """Get current top momentum stocks from Alpaca scanner"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    scanner = get_momentum_scanner()
    return {
        "top_stocks": scanner.get_top_stocks(limit),
        "last_scan": scanner.last_scan_time
    }


@router.post("/ALPACA/check")
async def alpaca_check_symbols(request: CheckSymbolsRequest):
    """
    Check specific symbols against Alpaca trading criteria.
    Use this to compare with Day Trade Dash scanner results.
    """
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

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
    """
    Get detailed analysis of a single symbol via Alpaca.
    Perfect for comparing with Day Trade Dash scanner output.
    """
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    scanner = get_momentum_scanner()
    symbol = symbol.upper()
    scanner.add_symbol_to_universe(symbol)

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
    """Start continuous Alpaca scanning"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    scanner = get_momentum_scanner()
    scanner.start_continuous_scan(interval)
    return {
        "success": True,
        "message": f"Continuous scanning started (every {interval}s)"
    }


@router.post("/ALPACA/stop")
async def alpaca_stop_continuous():
    """Stop continuous Alpaca scanning"""
    if not HAS_ALPACA_SCANNER:
        raise HTTPException(status_code=503, detail="Alpaca scanner not available")

    scanner = get_momentum_scanner()
    scanner.stop_continuous_scan()
    return {
        "success": True,
        "message": "Continuous scanning stopped"
    }


logger.info("Scanner API routes initialized (includes Penny, Warrior & Alpaca scanners)")
