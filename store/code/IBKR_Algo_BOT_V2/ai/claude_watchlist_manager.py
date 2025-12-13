"""
Claude AI Watchlist Manager
============================
AI-powered watchlist management that can:
- Scan for after-hours momentum stocks
- Update watchlists dynamically
- Train AI models on selected stocks
- Build optimized working lists based on performance
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MomentumStock:
    """Stock with momentum metrics"""
    symbol: str
    name: str
    current_price: float
    change_percent: float
    volume: int
    avg_volume: int
    volume_ratio: float  # volume / avg_volume
    after_hours_change: float
    momentum_score: float
    sector: str = ""
    market_cap: float = 0
    reason: str = ""


@dataclass
class WorkingListEntry:
    """Entry in the working list with performance tracking"""
    symbol: str
    added_at: str
    added_by: str  # "claude_ai", "user", "scanner"
    reason: str
    ai_score: float
    win_rate: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    last_signal: str = ""
    active: bool = True


class ClaudeWatchlistManager:
    """
    AI-powered watchlist management system.
    Claude can analyze, update, and optimize watchlists.
    """

    def __init__(self):
        self.data_path = Path("store/watchlists")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Working list file
        self.working_list_file = self.data_path / "ai_working_list.json"
        self.momentum_scan_file = self.data_path / "momentum_scan.json"

        # Load existing working list
        self.working_list: Dict[str, WorkingListEntry] = {}
        self._load_working_list()

        # NASDAQ symbols for scanning (top 200 by volume)
        self.nasdaq_universe = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
            "AVGO", "COST", "NFLX", "AMD", "QCOM", "ADBE", "PEP", "CSCO",
            "INTC", "TMUS", "CMCSA", "INTU", "TXN", "AMGN", "HON", "AMAT",
            "ISRG", "BKNG", "SBUX", "VRTX", "GILD", "ADI", "MDLZ", "LRCX",
            "ADP", "REGN", "PANW", "KLAC", "SNPS", "CDNS", "ASML", "MRVL",
            "PYPL", "MNST", "ORLY", "FTNT", "CHTR", "CTAS", "MAR", "NXPI",
            "CPRT", "PCAR", "KDP", "MELI", "WDAY", "ADSK", "AEP", "PAYX",
            "ROST", "ABNB", "ODFL", "MCHP", "KHC", "DXCM", "IDXX", "CEG",
            "LULU", "EXC", "FAST", "EA", "VRSK", "CTSH", "CSGP", "BKR",
            "FANG", "GEHC", "XEL", "TEAM", "ANSS", "ON", "DLTR", "ZS",
            "DDOG", "WBD", "GFS", "ILMN", "ALGN", "MDB", "TTWO", "DASH",
            "CRWD", "SPLK", "ROKU", "OKTA", "ZM", "DOCU", "SNOW", "COIN",
            "PLTR", "RIVN", "LCID", "SOFI", "HOOD", "RBLX", "U", "PATH"
        ]

        # Claude AI client
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self.ai_available = False

        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.ai_available = True
                logger.info("[WATCHLIST] Claude AI initialized for watchlist management")
        except ImportError:
            logger.warning("[WATCHLIST] anthropic package not installed")

        logger.info("[WATCHLIST] ClaudeWatchlistManager initialized")

    def _load_working_list(self):
        """Load working list from file"""
        if self.working_list_file.exists():
            try:
                with open(self.working_list_file, 'r') as f:
                    data = json.load(f)
                    for symbol, entry_data in data.items():
                        self.working_list[symbol] = WorkingListEntry(**entry_data)
                logger.info(f"[WATCHLIST] Loaded {len(self.working_list)} symbols from working list")
            except Exception as e:
                logger.error(f"[WATCHLIST] Error loading working list: {e}")

    def _save_working_list(self):
        """Save working list to file"""
        try:
            data = {symbol: asdict(entry) for symbol, entry in self.working_list.items()}
            with open(self.working_list_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"[WATCHLIST] Saved {len(self.working_list)} symbols to working list")
        except Exception as e:
            logger.error(f"[WATCHLIST] Error saving working list: {e}")

    async def scan_after_hours_momentum(self, min_change: float = 2.0,
                                        min_volume_ratio: float = 1.5) -> List[MomentumStock]:
        """
        Scan NASDAQ stocks for after-hours momentum.
        Returns stocks with significant after-hours movement and volume.
        """
        logger.info("[WATCHLIST] Scanning for after-hours momentum...")

        momentum_stocks = []

        try:
            from alpaca_market_data import get_alpaca_market_data
            market_data = get_alpaca_market_data()

            for symbol in self.nasdaq_universe:
                try:
                    # Get latest quote (includes after-hours)
                    quote = market_data.get_latest_quote(symbol)
                    if not quote:
                        continue

                    # Get daily bars for volume comparison
                    bars = market_data.get_bars(symbol, timeframe="1Day", limit=5)
                    if not bars or len(bars) < 2:
                        continue

                    current_price = quote.get("price", 0)
                    prev_close = bars[-2].get("close", current_price) if len(bars) >= 2 else current_price

                    # Calculate after-hours change
                    if prev_close > 0:
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                    else:
                        change_pct = 0

                    # Calculate volume ratio
                    current_volume = bars[-1].get("volume", 0) if bars else 0
                    avg_volume = sum(b.get("volume", 0) for b in bars[:-1]) / max(len(bars) - 1, 1)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

                    # Filter by momentum criteria
                    if abs(change_pct) >= min_change and volume_ratio >= min_volume_ratio:
                        # Calculate momentum score
                        momentum_score = (abs(change_pct) * 10) + (volume_ratio * 20)

                        momentum_stocks.append(MomentumStock(
                            symbol=symbol,
                            name=symbol,  # Could fetch company name if needed
                            current_price=current_price,
                            change_percent=change_pct,
                            volume=current_volume,
                            avg_volume=int(avg_volume),
                            volume_ratio=round(volume_ratio, 2),
                            after_hours_change=change_pct,
                            momentum_score=round(momentum_score, 2),
                            reason=f"AH move: {change_pct:+.1f}%, Vol: {volume_ratio:.1f}x avg"
                        ))

                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
                    continue

            # Sort by momentum score
            momentum_stocks.sort(key=lambda x: x.momentum_score, reverse=True)

            # Save scan results
            scan_data = {
                "timestamp": datetime.now().isoformat(),
                "criteria": {"min_change": min_change, "min_volume_ratio": min_volume_ratio},
                "results": [asdict(s) for s in momentum_stocks]
            }
            with open(self.momentum_scan_file, 'w') as f:
                json.dump(scan_data, f, indent=2)

            logger.info(f"[WATCHLIST] Found {len(momentum_stocks)} stocks with after-hours momentum")

        except Exception as e:
            logger.error(f"[WATCHLIST] Error in momentum scan: {e}")

        return momentum_stocks

    async def claude_analyze_and_select(self, momentum_stocks: List[MomentumStock],
                                        max_selections: int = 10) -> List[str]:
        """
        Have Claude AI analyze momentum stocks and select the best ones.
        """
        if not momentum_stocks:
            return []

        if not self.ai_available:
            # Fallback: just return top N by momentum score
            return [s.symbol for s in momentum_stocks[:max_selections]]

        # Prepare data for Claude
        stock_data = "\n".join([
            f"- {s.symbol}: Price ${s.current_price:.2f}, Change {s.change_percent:+.1f}%, "
            f"Volume {s.volume_ratio:.1f}x avg, Momentum Score: {s.momentum_score}"
            for s in momentum_stocks[:30]  # Limit to top 30 for analysis
        ])

        prompt = f"""Analyze these NASDAQ stocks showing after-hours momentum and select the TOP {max_selections} for our trading watchlist.

STOCKS WITH AFTER-HOURS MOMENTUM:
{stock_data}

SELECTION CRITERIA:
1. Strong momentum (high % change with volume confirmation)
2. Sustainable move (not just a spike)
3. Trading opportunity potential for next session
4. Avoid extreme volatility that could be news-driven reversals

For each selection, explain briefly why it's a good candidate.

Return your response in this JSON format:
{{
    "selections": [
        {{"symbol": "SYMBOL", "reason": "Brief reason for selection", "priority": 1-10}},
        ...
    ],
    "market_outlook": "Brief assessment of overall after-hours sentiment"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse Claude's response
            response_text = response.content[0].text

            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                selections = result.get("selections", [])

                # Add to working list
                selected_symbols = []
                for sel in selections[:max_selections]:
                    symbol = sel.get("symbol", "").upper()
                    if symbol:
                        selected_symbols.append(symbol)

                        # Add to working list
                        self.working_list[symbol] = WorkingListEntry(
                            symbol=symbol,
                            added_at=datetime.now().isoformat(),
                            added_by="claude_ai",
                            reason=sel.get("reason", "AI selected for momentum"),
                            ai_score=sel.get("priority", 5) * 10,
                            active=True
                        )

                self._save_working_list()
                logger.info(f"[WATCHLIST] Claude selected {len(selected_symbols)} stocks: {selected_symbols}")

                return selected_symbols

        except Exception as e:
            logger.error(f"[WATCHLIST] Claude analysis error: {e}")

        # Fallback
        return [s.symbol for s in momentum_stocks[:max_selections]]

    async def train_on_working_list(self) -> Dict:
        """
        Train AI models on all symbols in the working list.
        """
        if not self.working_list:
            return {"status": "error", "message": "Working list is empty"}

        active_symbols = [s for s, e in self.working_list.items() if e.active]

        if not active_symbols:
            return {"status": "error", "message": "No active symbols in working list"}

        logger.info(f"[WATCHLIST] Training AI on {len(active_symbols)} symbols: {active_symbols}")

        try:
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            predictor = get_alpaca_predictor()

            results = {"trained": [], "failed": [], "details": {}}

            for symbol in active_symbols:
                try:
                    # Train model for this symbol
                    result = await predictor.train_model(symbol)

                    if result.get("success") or result.get("accuracy", 0) > 0:
                        results["trained"].append(symbol)
                        results["details"][symbol] = {
                            "accuracy": result.get("accuracy", 0),
                            "status": "success"
                        }

                        # Update working list entry
                        if symbol in self.working_list:
                            self.working_list[symbol].ai_score = result.get("accuracy", 0) * 100

                    else:
                        results["failed"].append(symbol)
                        results["details"][symbol] = {"status": "failed", "error": result.get("error", "Unknown")}

                except Exception as e:
                    results["failed"].append(symbol)
                    results["details"][symbol] = {"status": "error", "error": str(e)}

            self._save_working_list()

            return {
                "status": "success",
                "trained_count": len(results["trained"]),
                "failed_count": len(results["failed"]),
                "results": results
            }

        except Exception as e:
            logger.error(f"[WATCHLIST] Training error: {e}")
            return {"status": "error", "message": str(e)}

    async def rank_and_filter_working_list(self) -> List[Dict]:
        """
        Rank symbols in working list by AI predictions and performance.
        Returns sorted list with recommendations.
        """
        if not self.working_list:
            return []

        ranked_list = []

        try:
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            predictor = get_alpaca_predictor()

            for symbol, entry in self.working_list.items():
                if not entry.active:
                    continue

                try:
                    # Get prediction for this symbol
                    prediction = await predictor.predict(symbol)

                    if prediction:
                        ranked_list.append({
                            "symbol": symbol,
                            "signal": prediction.get("signal", "HOLD"),
                            "confidence": prediction.get("confidence", 0),
                            "ai_score": entry.ai_score,
                            "win_rate": entry.win_rate,
                            "total_trades": entry.total_trades,
                            "total_pnl": entry.total_pnl,
                            "added_by": entry.added_by,
                            "reason": entry.reason,
                            "combined_score": (prediction.get("confidence", 0) * 0.5) + (entry.ai_score * 0.3) + (entry.win_rate * 0.2)
                        })
                except Exception as e:
                    logger.debug(f"Error getting prediction for {symbol}: {e}")
                    ranked_list.append({
                        "symbol": symbol,
                        "signal": "HOLD",
                        "confidence": 0,
                        "ai_score": entry.ai_score,
                        "win_rate": entry.win_rate,
                        "combined_score": entry.ai_score * 0.5
                    })

            # Sort by combined score
            ranked_list.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        except Exception as e:
            logger.error(f"[WATCHLIST] Ranking error: {e}")

        return ranked_list

    def add_to_working_list(self, symbol: str, reason: str = "Manual add",
                           added_by: str = "user") -> bool:
        """Add a symbol to the working list"""
        symbol = symbol.upper()

        self.working_list[symbol] = WorkingListEntry(
            symbol=symbol,
            added_at=datetime.now().isoformat(),
            added_by=added_by,
            reason=reason,
            ai_score=50.0,
            active=True
        )

        self._save_working_list()
        logger.info(f"[WATCHLIST] Added {symbol} to working list: {reason}")
        return True

    def remove_from_working_list(self, symbol: str) -> bool:
        """Remove a symbol from the working list"""
        symbol = symbol.upper()

        if symbol in self.working_list:
            del self.working_list[symbol]
            self._save_working_list()
            logger.info(f"[WATCHLIST] Removed {symbol} from working list")
            return True
        return False

    def get_working_list(self) -> List[Dict]:
        """Get current working list as list of dicts"""
        return [
            {**asdict(entry), "symbol": symbol}
            for symbol, entry in self.working_list.items()
        ]

    def update_performance(self, symbol: str, won: bool, pnl: float):
        """Update performance metrics for a symbol after a trade"""
        symbol = symbol.upper()

        if symbol in self.working_list:
            entry = self.working_list[symbol]
            entry.total_trades += 1
            entry.total_pnl += pnl

            # Recalculate win rate
            if won:
                wins = int(entry.win_rate * (entry.total_trades - 1)) + 1
            else:
                wins = int(entry.win_rate * (entry.total_trades - 1))

            entry.win_rate = wins / entry.total_trades if entry.total_trades > 0 else 0
            entry.last_signal = datetime.now().isoformat()

            self._save_working_list()

    async def sync_to_platform_watchlist(self, watchlist_name: str = "AI_Working_List") -> Dict:
        """
        Sync the AI working list to the platform's watchlist system.
        Also updates the main worklist so symbols appear in the dashboard.
        """
        try:
            # Get active symbols
            active_symbols = [s for s, e in self.working_list.items() if e.active]

            if not active_symbols:
                return {"status": "error", "message": "No active symbols to sync"}

            # Save to watchlist file format
            watchlist_path = Path("store/watchlists") / f"{watchlist_name}.json"
            watchlist_path.parent.mkdir(parents=True, exist_ok=True)

            watchlist_data = {
                "name": watchlist_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "symbols": active_symbols,
                "metadata": {
                    "source": "claude_ai_watchlist_manager",
                    "count": len(active_symbols)
                }
            }

            with open(watchlist_path, 'w') as f:
                json.dump(watchlist_data, f, indent=2)

            # ALSO sync to the main platform worklist (database watchlist manager)
            try:
                from watchlist_manager import get_watchlist_manager as get_platform_wm
                platform_wm = get_platform_wm()
                default_watchlist = platform_wm.get_default_watchlist()

                # Add all active symbols to default watchlist
                platform_wm.add_symbols(default_watchlist['watchlist_id'], active_symbols)
                logger.info(f"[WATCHLIST] Added {len(active_symbols)} symbols to platform default watchlist")
            except Exception as e:
                logger.warning(f"[WATCHLIST] Could not sync to platform watchlist manager: {e}")

            logger.info(f"[WATCHLIST] Synced {len(active_symbols)} symbols to {watchlist_name}")

            return {
                "status": "success",
                "watchlist": watchlist_name,
                "symbols": active_symbols,
                "count": len(active_symbols)
            }

        except Exception as e:
            logger.error(f"[WATCHLIST] Sync error: {e}")
            return {"status": "error", "message": str(e)}

    async def execute_full_workflow(self, min_change: float = 2.0,
                                    max_selections: int = 10,
                                    train_models: bool = True) -> Dict:
        """
        Execute the full workflow:
        1. Scan for after-hours momentum
        2. Have Claude select the best stocks
        3. Add to working list
        4. Train AI models
        5. Sync to platform watchlist
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }

        # Step 1: Scan for momentum
        logger.info("[WATCHLIST] Step 1: Scanning for after-hours momentum...")
        momentum_stocks = await self.scan_after_hours_momentum(min_change=min_change)
        results["steps"]["scan"] = {
            "found": len(momentum_stocks),
            "top_movers": [asdict(s) for s in momentum_stocks[:10]]
        }

        if not momentum_stocks:
            results["status"] = "no_momentum_found"
            return results

        # Step 2: Claude selects best stocks
        logger.info("[WATCHLIST] Step 2: Claude AI analyzing and selecting stocks...")
        selected = await self.claude_analyze_and_select(momentum_stocks, max_selections)
        results["steps"]["selection"] = {
            "selected": selected,
            "count": len(selected)
        }

        # Step 3: Train models (optional)
        if train_models and selected:
            logger.info("[WATCHLIST] Step 3: Training AI models on selected stocks...")
            training_results = await self.train_on_working_list()
            results["steps"]["training"] = training_results

        # Step 4: Sync to platform watchlist
        logger.info("[WATCHLIST] Step 4: Syncing to platform watchlist...")
        sync_result = await self.sync_to_platform_watchlist()
        results["steps"]["sync"] = sync_result

        # Step 5: Get final ranked list
        logger.info("[WATCHLIST] Step 5: Ranking working list...")
        ranked = await self.rank_and_filter_working_list()
        results["steps"]["ranking"] = {
            "top_picks": ranked[:5] if ranked else []
        }

        results["status"] = "success"
        results["working_list_count"] = len(self.working_list)

        return results


# Global instance
_watchlist_manager: Optional[ClaudeWatchlistManager] = None


def get_watchlist_manager() -> ClaudeWatchlistManager:
    """Get or create the global watchlist manager"""
    global _watchlist_manager
    if _watchlist_manager is None:
        _watchlist_manager = ClaudeWatchlistManager()
    return _watchlist_manager


async def run_momentum_workflow():
    """Convenience function to run the full momentum workflow"""
    manager = get_watchlist_manager()
    return await manager.execute_full_workflow()


if __name__ == "__main__":
    async def test():
        manager = get_watchlist_manager()
        result = await manager.execute_full_workflow(min_change=1.5, max_selections=5, train_models=False)
        print(json.dumps(result, indent=2))

    asyncio.run(test())
