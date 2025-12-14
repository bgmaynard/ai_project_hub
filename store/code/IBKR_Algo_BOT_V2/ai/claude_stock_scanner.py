"""
Claude AI Stock Scanner
=======================
Comprehensive stock scanner with customizable presets that scans the market
and feeds results directly to Claude AI for analysis and watchlist management.

Scanner Types:
- Momentum (gap up/down, high volume)
- Breakout (new highs, range breakouts)
- Reversal (oversold bounces, overbought pullbacks)
- After-Hours (pre/post market movers)
- Custom (user-defined criteria)
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)


class ScannerType(Enum):
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    AFTER_HOURS = "after_hours"
    GAP = "gap"
    VOLUME_SURGE = "volume_surge"
    NEW_HIGH = "new_high"
    NEW_LOW = "new_low"
    CUSTOM = "custom"


@dataclass
class ScannerPreset:
    """Scanner preset configuration"""
    id: str
    name: str
    scanner_type: ScannerType
    description: str
    criteria: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[str] = None
    results_count: int = 0


@dataclass
class ScanResult:
    """Individual scan result"""
    symbol: str
    name: str
    price: float
    change_percent: float
    volume: int
    avg_volume: int
    volume_ratio: float
    market_cap: float
    sector: str
    scanner_type: str
    score: float
    signals: List[str]
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class ClaudeStockScanner:
    """
    AI-powered stock scanner that scans the entire market based on
    configurable criteria and feeds results to Claude for analysis.
    """

    def __init__(self):
        self.data_path = Path("store/scanner")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Scanner results
        self.scan_results: Dict[str, List[ScanResult]] = {}
        self.last_scan_time: Optional[datetime] = None

        # Full stock universe (will be populated)
        self.stock_universe: List[Dict] = []
        self.universe_loaded = False

        # Scanner presets
        self.presets: Dict[str, ScannerPreset] = {}
        self._setup_default_presets()

        # Claude AI
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self.ai_available = False

        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.ai_available = True
        except ImportError:
            pass

        logger.info("[SCANNER] ClaudeStockScanner initialized")

    def _setup_default_presets(self):
        """Setup default scanner presets - LOOSENED for better results"""
        presets = [
            ScannerPreset(
                id="momentum_warrior",
                name="Momentum Warrior",
                scanner_type=ScannerType.MOMENTUM,
                description="High momentum stocks - any price movement with decent volume",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "min_volume": 0,  # No minimum - use volume ratio instead
                    "min_change_percent": 0.5,  # Just 0.5% move
                    "min_volume_ratio": 0.5,  # Even below average ok
                }
            ),
            ScannerPreset(
                id="gap_and_go",
                name="Gap and Go",
                scanner_type=ScannerType.GAP,
                description="Stocks with any gap from previous close",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "min_gap_percent": 0.5,  # Any gap 0.5%+
                    "max_gap_percent": 50,  # Allow big gaps
                    "min_volume_ratio": 0.3,
                }
            ),
            ScannerPreset(
                id="breakout_hunter",
                name="Breakout Hunter",
                scanner_type=ScannerType.BREAKOUT,
                description="Stocks near recent highs",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "near_52_week_high_percent": 20,  # Within 20% of high
                    "min_volume_ratio": 0.3,
                }
            ),
            ScannerPreset(
                id="oversold_bounce",
                name="Oversold Bounce",
                scanner_type=ScannerType.REVERSAL,
                description="Stocks with lower RSI showing any bounce",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "max_rsi": 70,  # Most stocks qualify
                    "min_change_percent": 0,  # Any change ok
                }
            ),
            ScannerPreset(
                id="volume_explosion",
                name="Volume Explosion",
                scanner_type=ScannerType.VOLUME_SURGE,
                description="Above average volume activity",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "min_volume_ratio": 0.8,  # Just slightly below average
                }
            ),
            ScannerPreset(
                id="after_hours_movers",
                name="After Hours Movers",
                scanner_type=ScannerType.AFTER_HOURS,
                description="Stocks with any movement",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "min_ah_change_percent": 0,  # Any movement
                    "min_volume_ratio": 0.1,
                }
            ),
            ScannerPreset(
                id="new_52_week_high",
                name="New 52-Week Highs",
                scanner_type=ScannerType.NEW_HIGH,
                description="Stocks near 52-week highs",
                criteria={
                    "min_price": 1,
                    "max_price": 2000,
                    "near_52_week_high_percent": 15,  # Within 15% of high
                    "min_volume_ratio": 0.3,
                }
            ),
        ]

        for preset in presets:
            self.presets[preset.id] = preset

    async def load_stock_universe(self) -> int:
        """
        Load the full tradeable stock universe.
        Uses hardcoded popular stocks as Schwab doesn't have a universe API.
        """
        logger.info("[SCANNER] Loading stock universe...")

        try:
            # Try to get stock universe from broker
            from unified_broker import get_unified_broker
            broker = get_unified_broker()

            # Use default popular stocks - Schwab doesn't have a universe API
            pass  # Fall through to hardcoded list

        except Exception as e:
            logger.warning(f"[SCANNER] Universe load failed: {e}")

        # Fallback to hardcoded popular stocks
        self.stock_universe = self._get_fallback_universe()
        self.universe_loaded = True
        logger.info(f"[SCANNER] Using fallback universe of {len(self.stock_universe)} stocks")
        return len(self.stock_universe)

    def _get_fallback_universe(self) -> List[Dict]:
        """Fallback list of popular stocks if API fails"""
        symbols = [
            # Tech Giants
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "IBM", "QCOM",
            # Semiconductors
            "AVGO", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "MU", "ON",
            # Software/Cloud
            "NOW", "SNOW", "PLTR", "DDOG", "CRWD", "ZS", "PANW", "FTNT",
            # E-commerce/Internet
            "SHOP", "PYPL", "SQ", "COIN", "HOOD", "ABNB", "UBER", "LYFT",
            # Biotech/Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "GILD",
            "MRNA", "BNTX", "VRTX", "REGN", "BIIB", "AMGN", "ISRG",
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
            "V", "MA", "AXP",
            # Consumer
            "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD",
            "DIS", "NFLX", "CMCSA", "T", "VZ",
            # Industrial/Energy
            "XOM", "CVX", "COP", "SLB", "BA", "CAT", "DE", "GE",
            "LMT", "RTX", "HON", "UPS", "FDX",
            # EV/Clean Energy
            "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR",
            # Hot Momentum Stocks
            "SMCI", "ARM", "MSTR", "IONQ", "RGTI", "QUBT", "RKLB", "LUNR",
            # SPACs/Growth
            "SOFI", "CHPT", "PLUG", "SPCE", "DKNG", "PENN",
            # ETFs for reference
            "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV"
        ]

        return [{"symbol": s, "name": s, "exchange": "NASDAQ", "tradable": True} for s in symbols]

    async def get_stock_data(self, symbol: str, session=None) -> Optional[Dict]:
        """Get current data for a single stock - uses main API for reliable data"""
        try:
            import aiohttp

            # Use provided session or create a temporary one
            should_close = False
            if session is None:
                connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
                session = aiohttp.ClientSession(connector=connector)
                should_close = True

            try:
                # Try the main price API which works reliably
                async with session.get(
                    f"http://localhost:9100/api/price/{symbol}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("data"):
                            d = data["data"]
                            current_price = d.get("last", 0) or d.get("price", 0)
                            if current_price <= 0:
                                return None

                            # Calculate change percent from close
                            close_price = d.get("close", current_price)
                            if close_price > 0:
                                change_pct = ((current_price - close_price) / close_price) * 100
                            else:
                                change_pct = d.get("change_percent", 0) or d.get("change", 0)

                            return {
                                "symbol": symbol,
                                "price": current_price,
                                "change_percent": round(change_pct, 2),
                                "volume": d.get("volume", 0),
                                "avg_volume": d.get("volume", 1000000),
                                "volume_ratio": 1.0,  # Default to 1.0
                                "high_52w": d.get("high", current_price),
                                "low_52w": d.get("low", current_price),
                                "rsi": 50,  # Default RSI
                                "market_cap": 0,
                            }
            finally:
                if should_close:
                    await session.close()
        except Exception as e:
            logger.debug(f"Error getting data for {symbol}: {e}")

        return None

    async def run_scan(self, preset_id: str, max_results: int = 50) -> List[ScanResult]:
        """
        Run a scanner with specific preset.
        Uses connection pooling to prevent resource exhaustion.
        """
        import aiohttp

        if preset_id not in self.presets:
            logger.error(f"[SCANNER] Unknown preset: {preset_id}")
            return []

        preset = self.presets[preset_id]
        criteria = preset.criteria

        logger.info(f"[SCANNER] Running scan: {preset.name}")

        # Load universe if needed
        if not self.universe_loaded:
            await self.load_stock_universe()

        results = []
        scan_count = 0

        # Scan stocks in batches with a SHARED session (connection pooling)
        batch_size = 10  # Reduced batch size to prevent resource exhaustion
        symbols = [s["symbol"] for s in self.stock_universe]

        # Create a single session with connection limits for the entire scan
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                # Get data for batch in parallel using shared session
                tasks = [self.get_stock_data(symbol, session) for symbol in batch]
                batch_data = await asyncio.gather(*tasks, return_exceptions=True)

                for symbol, data in zip(batch, batch_data):
                    if isinstance(data, Exception) or data is None:
                        continue

                    scan_count += 1

                    # Apply criteria
                    if not self._matches_criteria(data, criteria, preset.scanner_type):
                        continue

                    # Calculate score
                    score = self._calculate_score(data, preset.scanner_type)

                    # Generate signals
                    signals = self._generate_signals(data, preset.scanner_type)

                    results.append(ScanResult(
                        symbol=symbol,
                        name=symbol,
                        price=data["price"],
                        change_percent=data["change_percent"],
                        volume=data["volume"],
                        avg_volume=data["avg_volume"],
                        volume_ratio=data["volume_ratio"],
                        market_cap=data.get("market_cap", 0),
                        sector="",
                        scanner_type=preset.scanner_type.value,
                        score=score,
                        signals=signals,
                        timestamp=datetime.now().isoformat()
                    ))

                # Yield control and add small delay between batches
                await asyncio.sleep(0.2)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:max_results]

        # Update preset
        preset.last_run = datetime.now().isoformat()
        preset.results_count = len(results)

        # Store results
        self.scan_results[preset_id] = results
        self.last_scan_time = datetime.now()

        # Save results
        self._save_results(preset_id, results)

        logger.info(f"[SCANNER] {preset.name}: Scanned {scan_count} stocks, found {len(results)} matches")

        return results

    def _matches_criteria(self, data: Dict, criteria: Dict, scanner_type: ScannerType) -> bool:
        """Check if stock matches the criteria - LENIENT VERSION"""
        price = data.get("price", 0)
        volume = data.get("volume", 0)
        change_pct = data.get("change_percent", 0)
        volume_ratio = data.get("volume_ratio", 1.0)  # Default to 1.0 instead of 0
        rsi = data.get("rsi", 50)

        # Skip stocks with no price data
        if price <= 0:
            return False

        # Price filter
        if "min_price" in criteria and criteria["min_price"] > 0 and price < criteria["min_price"]:
            return False
        if "max_price" in criteria and price > criteria["max_price"]:
            return False

        # Volume filter - skip if 0 requirement
        if "min_volume" in criteria and criteria["min_volume"] > 0 and volume < criteria["min_volume"]:
            return False

        # Volume ratio filter - be lenient, treat missing as 1.0
        if "min_volume_ratio" in criteria and criteria["min_volume_ratio"] > 0:
            effective_ratio = volume_ratio if volume_ratio > 0 else 1.0
            if effective_ratio < criteria["min_volume_ratio"]:
                return False

        # Change percent filter - only apply if > 0
        if "min_change_percent" in criteria and criteria["min_change_percent"] > 0:
            if abs(change_pct) < criteria["min_change_percent"]:
                return False

        # Gap filter
        if "min_gap_percent" in criteria and criteria["min_gap_percent"] > 0:
            if abs(change_pct) < criteria["min_gap_percent"]:
                return False
        if "max_gap_percent" in criteria and abs(change_pct) > criteria["max_gap_percent"]:
            return False

        # RSI filter
        if "max_rsi" in criteria and rsi > criteria["max_rsi"]:
            return False
        if "min_rsi" in criteria and rsi < criteria["min_rsi"]:
            return False

        # 52-week high filter - made more lenient
        if criteria.get("at_52_week_high"):
            high_52w = data.get("high_52w", 0)
            if high_52w > 0 and price < high_52w * 0.90:  # Within 10% of high (was 2%)
                return False

        if "near_52_week_high_percent" in criteria:
            high_52w = data.get("high_52w", 0)
            threshold = criteria["near_52_week_high_percent"]
            if high_52w > 0:
                pct_from_high = ((high_52w - price) / high_52w) * 100
                if pct_from_high > threshold:
                    return False

        return True

    def _calculate_score(self, data: Dict, scanner_type: ScannerType) -> float:
        """Calculate a score for ranking"""
        score = 50.0  # Base score

        change_pct = abs(data.get("change_percent", 0))
        volume_ratio = data.get("volume_ratio", 1)
        rsi = data.get("rsi", 50)

        # Volume contribution
        score += min(volume_ratio * 10, 30)

        # Change contribution
        score += min(change_pct * 5, 25)

        # Scanner-specific adjustments
        if scanner_type == ScannerType.REVERSAL:
            # Prefer stocks with lower RSI (more oversold)
            if rsi < 30:
                score += 20
            elif rsi < 40:
                score += 10

        elif scanner_type == ScannerType.BREAKOUT:
            # Prefer stocks near 52-week high
            price = data.get("price", 0)
            high_52w = data.get("high_52w", price)
            if high_52w > 0:
                pct_from_high = ((high_52w - price) / high_52w) * 100
                if pct_from_high < 2:
                    score += 20
                elif pct_from_high < 5:
                    score += 10

        return round(min(score, 100), 1)

    def _generate_signals(self, data: Dict, scanner_type: ScannerType) -> List[str]:
        """Generate signal descriptions"""
        signals = []

        change_pct = data.get("change_percent", 0)
        volume_ratio = data.get("volume_ratio", 1)
        rsi = data.get("rsi", 50)

        if change_pct > 5:
            signals.append("Strong upward momentum")
        elif change_pct > 2:
            signals.append("Positive momentum")
        elif change_pct < -5:
            signals.append("Strong downward momentum")
        elif change_pct < -2:
            signals.append("Negative momentum")

        if volume_ratio > 3:
            signals.append("Extremely high volume")
        elif volume_ratio > 2:
            signals.append("Very high volume")
        elif volume_ratio > 1.5:
            signals.append("Above average volume")

        if rsi < 30:
            signals.append("Oversold (RSI < 30)")
        elif rsi > 70:
            signals.append("Overbought (RSI > 70)")

        price = data.get("price", 0)
        high_52w = data.get("high_52w", 0)
        if high_52w > 0 and price >= high_52w * 0.98:
            signals.append("Near 52-week high")

        return signals

    def _save_results(self, preset_id: str, results: List[ScanResult]):
        """Save scan results to file"""
        try:
            file_path = self.data_path / f"scan_{preset_id}.json"
            data = {
                "preset_id": preset_id,
                "timestamp": datetime.now().isoformat(),
                "count": len(results),
                "results": [r.to_dict() for r in results]
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[SCANNER] Error saving results: {e}")

    async def run_all_scans(self) -> Dict[str, List[ScanResult]]:
        """Run all enabled scanner presets"""
        all_results = {}

        for preset_id, preset in self.presets.items():
            if preset.enabled:
                results = await self.run_scan(preset_id)
                all_results[preset_id] = results

        return all_results

    async def claude_analyze_scan_results(self, preset_id: str,
                                          add_to_watchlist: bool = True) -> Dict:
        """Have Claude analyze scan results and optionally add to watchlist"""
        if preset_id not in self.scan_results:
            return {"error": "No scan results for this preset"}

        results = self.scan_results[preset_id]
        if not results:
            return {"error": "Empty scan results"}

        if not self.ai_available:
            # Return top 5 without AI analysis
            top_symbols = [r.symbol for r in results[:5]]
            return {
                "selected": top_symbols,
                "analysis": "AI not available - returning top 5 by score"
            }

        # Prepare data for Claude
        results_text = "\n".join([
            f"- {r.symbol}: ${r.price:.2f}, {r.change_percent:+.1f}%, "
            f"Vol {r.volume_ratio:.1f}x, Score: {r.score}, Signals: {', '.join(r.signals)}"
            for r in results[:20]
        ])

        preset = self.presets[preset_id]

        prompt = f"""Analyze these stock scanner results and select the TOP 5-10 best trading opportunities.

SCANNER: {preset.name}
DESCRIPTION: {preset.description}

SCAN RESULTS:
{results_text}

Please analyze each stock and select the best opportunities based on:
1. Momentum strength and sustainability
2. Volume confirmation
3. Risk/reward potential
4. Overall setup quality

Return your response in JSON format:
{{
    "selected": [
        {{"symbol": "SYMBOL", "action": "BUY/WATCH", "reason": "Brief reason", "priority": 1-10}},
        ...
    ],
    "market_context": "Brief market assessment",
    "risk_notes": "Any warnings or risks to consider"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())

                # Add to watchlist if requested
                if add_to_watchlist:
                    from .claude_watchlist_manager import get_watchlist_manager
                    manager = get_watchlist_manager()

                    for sel in result.get("selected", []):
                        symbol = sel.get("symbol", "").upper()
                        if symbol and sel.get("action") in ["BUY", "WATCH"]:
                            manager.add_to_working_list(
                                symbol=symbol,
                                reason=f"Scanner: {preset.name} - {sel.get('reason', '')}",
                                added_by="scanner"
                            )

                return result

        except Exception as e:
            logger.error(f"[SCANNER] Claude analysis error: {e}")

        return {"error": str(e)}

    def get_presets(self) -> List[Dict]:
        """Get all scanner presets"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "type": p.scanner_type.value,
                "description": p.description,
                "enabled": p.enabled,
                "last_run": p.last_run,
                "results_count": p.results_count
            }
            for p in self.presets.values()
        ]

    def get_results(self, preset_id: str) -> List[Dict]:
        """Get latest results for a preset"""
        if preset_id in self.scan_results:
            return [r.to_dict() for r in self.scan_results[preset_id]]
        return []


# Global instance
_scanner: Optional[ClaudeStockScanner] = None


def get_stock_scanner() -> ClaudeStockScanner:
    """Get or create the global scanner instance"""
    global _scanner
    if _scanner is None:
        _scanner = ClaudeStockScanner()
    return _scanner


async def run_momentum_scan():
    """Convenience function to run momentum scan"""
    scanner = get_stock_scanner()
    return await scanner.run_scan("momentum_warrior")


if __name__ == "__main__":
    async def test():
        scanner = get_stock_scanner()
        await scanner.load_stock_universe()

        # Run momentum scan
        results = await scanner.run_scan("momentum_warrior", max_results=10)

        print(f"\nFound {len(results)} momentum stocks:")
        for r in results:
            print(f"  {r.symbol}: ${r.price:.2f} ({r.change_percent:+.1f}%) "
                  f"Vol: {r.volume_ratio:.1f}x, Score: {r.score}")

    asyncio.run(test())
